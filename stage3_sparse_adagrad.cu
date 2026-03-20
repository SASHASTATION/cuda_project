#include "common.cuh"
#include "stages.h"

constexpr float GRAD_THRESHOLD = 1e-6f;
// kernel_extract_sparse — это фильтрация S^(k) = { j : |g_j| > τ }
__global__ void kernel_extract_sparse(const float* grad, int* idx, float* val, int* nnz, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = grad[j];
    if (fabsf(g) > GRAD_THRESHOLD) {
        int pos = atomicAdd(nnz, 1);
        idx[pos] = j;
        val[pos] = g;
    }
}

__global__ void kernel_sparse_adagrad_update(float* w, float* G2,
                                             const int* idx, const float* val,
                                             int nnz, float lr, float eps) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;

    int j = idx[k];
    float g = val[k];
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}

StageResult run_sparse_adagrad_stage() {
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    float* G2 = nullptr;
    int* sparse_idx = nullptr;
    float* sparse_val = nullptr;
    int* nnz_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&G2, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sparse_idx, N_FEATURES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sparse_val, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nnz_dev, sizeof(int)));
    CUDA_CHECK(cudaMemset(G2, 0, N_FEATURES * sizeof(float)));

    const int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float lr = 0.10f;
    const float eps = 1e-8f;

    StageResult out;
    out.key = "sparse_adagrad";
    out.name = "Sparse Adagrad";
    out.opt_mem_bytes = N_FEATURES * sizeof(float);

    print_header("3. Sparse Adagrad");
    GpuTimer timer;

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        timer.tic();
        forward_backward(ds, N_SAMPLES, N_FEATURES);

        CUDA_CHECK(cudaMemset(nnz_dev, 0, sizeof(int)));
        kernel_extract_sparse<<<opt_blocks, BLOCK_SIZE>>>(ds.grad, sparse_idx, sparse_val, nnz_dev, N_FEATURES);
        CUDA_KERNEL_CHECK();

        int nnz = 0;
        CUDA_CHECK(cudaMemcpy(&nnz, nnz_dev, sizeof(int), cudaMemcpyDeviceToHost));

        if (nnz > 0) {
            int sparse_blocks = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_sparse_adagrad_update<<<sparse_blocks, BLOCK_SIZE>>>(ds.w, G2, sparse_idx, sparse_val, nnz, lr, eps);
            CUDA_KERNEL_CHECK();
        }

        float ms = timer.toc_ms();
        out.total_ms += ms;

        float loss = 0.0f, acc = 0.0f;
        evaluate_model(ds, N_SAMPLES, N_FEATURES, loss, acc);
        out.history.push_back({epoch, loss, acc, ms, nnz});

        if (epoch % 5 == 0 || epoch == N_EPOCHS - 1) {
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f  (nnz=%d)\n", epoch, loss, acc * 100.0f, ms, nnz);
        }
    }

    CUDA_CHECK(cudaFree(G2));
    CUDA_CHECK(cudaFree(sparse_idx));
    CUDA_CHECK(cudaFree(sparse_val));
    CUDA_CHECK(cudaFree(nnz_dev));
    free_dataset(ds);
    return out;
}

#ifdef STAGE3_STANDALONE
int main() {
    auto result = run_sparse_adagrad_stage();
    const auto& last = result.history.back();
    printf("\nFinal: loss=%.6f acc=%.2f%% total=%.2f ms\n", last.loss, last.acc * 100.0f, result.total_ms);
    return 0;
}
#endif
