#include "common.cuh"
#include "stages.h"

__global__ void kernel_adam_update(float* w, const float* grad, float* m, float* v,
                                   float lr, float beta1, float beta2,
                                   float eps, float bc1, float bc2, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = grad[j];
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;

    float m_hat = mj / bc1;
    float v_hat = vj / bc2;
    w[j] -= lr * m_hat / (sqrtf(v_hat) + eps);

    m[j] = mj;
    v[j] = vj;
}

StageResult run_adam_stage() {
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    float* m = nullptr;
    float* v = nullptr;
    CUDA_CHECK(cudaMalloc(&m, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(m, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(v, 0, N_FEATURES * sizeof(float)));

    const int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float lr = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;

    StageResult out;
    out.key = "adam";
    out.name = "Adam";
    out.opt_mem_bytes = 2 * N_FEATURES * sizeof(float);

    print_header("2. Adam");
    GpuTimer timer;

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        timer.tic();
        forward_backward(ds, N_SAMPLES, N_FEATURES);

        int t = epoch + 1;
        float bc1 = 1.0f - powf(beta1, (float)t);
        float bc2 = 1.0f - powf(beta2, (float)t);

        kernel_adam_update<<<opt_blocks, BLOCK_SIZE>>>(ds.w, ds.grad, m, v, lr, beta1, beta2, eps, bc1, bc2, N_FEATURES);
        CUDA_KERNEL_CHECK();
        float ms = timer.toc_ms();
        out.total_ms += ms;

        float loss = 0.0f, acc = 0.0f;
        evaluate_model(ds, N_SAMPLES, N_FEATURES, loss, acc);
        out.history.push_back({epoch, loss, acc, ms, -1});

        if (epoch % 5 == 0 || epoch == N_EPOCHS - 1) {
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n", epoch, loss, acc * 100.0f, ms);
        }
    }

    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaFree(v));
    free_dataset(ds);
    return out;
}

#ifdef STAGE2_STANDALONE
int main() {
    auto result = run_adam_stage();
    const auto& last = result.history.back();
    printf("\nFinal: loss=%.6f acc=%.2f%% total=%.2f ms\n", last.loss, last.acc * 100.0f, result.total_ms);
    return 0;
}
#endif
