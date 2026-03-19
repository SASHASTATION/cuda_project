#include "common.cuh"
#include "stages.h"
#include <cuda_fp16.h>

__global__ void kernel_cast_f32_to_f16(const float* src, __half* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = __float2half(src[i]);
}

__global__ void kernel_predict_fp16(const __half* X_fp16, const __half* w_fp16, float* pred, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int D2 = D / 2;
    const half2* x2 = reinterpret_cast<const half2*>(X_fp16 + (size_t)i * D);
    const half2* w2 = reinterpret_cast<const half2*>(w_fp16);

    float dot = 0.0f;
    for (int j = 0; j < D2; ++j) {
        half2 p = __hmul2(x2[j], w2[j]);
        dot += __low2float(p) + __high2float(p);
    }
    pred[i] = 1.0f / (1.0f + expf(-dot));
}

__global__ void kernel_gradient_mixed(const __half* X_fp16, const float* errors, __half* grad_fp16,
                                      float loss_scale, int N, int D) {
    int feature = blockIdx.x;
    if (feature >= D) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += errors[i] * __half2float(X_fp16[(size_t)i * D + feature]);
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        float g = (sdata[0] / (float)N) * loss_scale;
        grad_fp16[feature] = __float2half(g);
    }
}

__global__ void kernel_adam_mixed(float* w_master, __half* w_fp16, const __half* grad_fp16,
                                  float* m, float* v,
                                  float lr, float beta1, float beta2,
                                  float eps, float bc1, float bc2,
                                  float inv_loss_scale, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = __half2float(grad_fp16[j]) * inv_loss_scale;
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;

    float m_hat = mj / bc1;
    float v_hat = vj / bc2;
    float new_w = w_master[j] - lr * m_hat / (sqrtf(v_hat) + eps);

    w_master[j] = new_w;
    w_fp16[j] = __float2half(new_w);
    m[j] = mj;
    v[j] = vj;
}

StageResult run_mixed_adam_stage() {
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    __half* X_fp16 = nullptr;
    __half* w_fp16 = nullptr;
    __half* grad_fp16 = nullptr;
    float* w_master = nullptr;
    float* m = nullptr;
    float* v = nullptr;

    CUDA_CHECK(cudaMalloc(&X_fp16, (size_t)N_SAMPLES * N_FEATURES * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&w_fp16, N_FEATURES * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&grad_fp16, N_FEATURES * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&w_master, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v, N_FEATURES * sizeof(float)));

    int total = N_SAMPLES * N_FEATURES;
    int cast_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_n = (N_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_cast_f32_to_f16<<<cast_blocks, BLOCK_SIZE>>>(ds.X, X_fp16, total);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaMemset(w_master, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(m, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(v, 0, N_FEATURES * sizeof(float)));
    kernel_cast_f32_to_f16<<<opt_blocks, BLOCK_SIZE>>>(w_master, w_fp16, N_FEATURES);
    CUDA_KERNEL_CHECK();

    const float lr = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const float loss_scale = 1024.0f;
    const float inv_loss_scale = 1.0f / loss_scale;

    StageResult out;
    out.key = "mixed_adam";
    out.name = "Mixed Adam";
    out.opt_mem_bytes = 2 * N_FEATURES * sizeof(float) + N_FEATURES * sizeof(float) + N_FEATURES * sizeof(__half);

    print_header("4. Mixed Precision Adam");
    GpuTimer timer;

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        timer.tic();

        kernel_predict_fp16<<<blocks_n, BLOCK_SIZE>>>(X_fp16, w_fp16, ds.pred, N_SAMPLES, N_FEATURES);
        CUDA_KERNEL_CHECK();
        kernel_errors<<<blocks_n, BLOCK_SIZE>>>(ds.pred, ds.y, ds.errors, N_SAMPLES);
        CUDA_KERNEL_CHECK();
        kernel_gradient_mixed<<<N_FEATURES, BLOCK_SIZE>>>(X_fp16, ds.errors, grad_fp16, loss_scale, N_SAMPLES, N_FEATURES);
        CUDA_KERNEL_CHECK();

        int t = epoch + 1;
        float bc1 = 1.0f - powf(beta1, (float)t);
        float bc2 = 1.0f - powf(beta2, (float)t);

        kernel_adam_mixed<<<opt_blocks, BLOCK_SIZE>>>(w_master, w_fp16, grad_fp16, m, v,
                                                      lr, beta1, beta2, eps, bc1, bc2,
                                                      inv_loss_scale, N_FEATURES);
        CUDA_KERNEL_CHECK();

        float ms = timer.toc_ms();
        out.total_ms += ms;

        kernel_predict_fp16<<<blocks_n, BLOCK_SIZE>>>(X_fp16, w_fp16, ds.pred, N_SAMPLES, N_FEATURES);
        CUDA_KERNEL_CHECK();
        float loss = compute_loss(ds, N_SAMPLES);
        float acc = compute_accuracy(ds, N_SAMPLES);
        out.history.push_back({epoch, loss, acc, ms, -1});

        if (epoch % 5 == 0 || epoch == N_EPOCHS - 1) {
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n", epoch, loss, acc * 100.0f, ms);
        }
    }

    CUDA_CHECK(cudaFree(X_fp16));
    CUDA_CHECK(cudaFree(w_fp16));
    CUDA_CHECK(cudaFree(grad_fp16));
    CUDA_CHECK(cudaFree(w_master));
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaFree(v));
    free_dataset(ds);
    return out;
}

#ifdef STAGE4_STANDALONE
int main() {
    auto result = run_mixed_adam_stage();
    const auto& last = result.history.back();
    printf("\nFinal: loss=%.6f acc=%.2f%% total=%.2f ms\n", last.loss, last.acc * 100.0f, result.total_ms);
    return 0;
}
#endif
