#include "common.cuh"
#include "stages.h"

__global__ void kernel_adagrad_update(float* w, const float* grad, float* G2, float lr, float eps, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = grad[j];
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}

StageResult run_adagrad_stage() {
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    float* G2 = nullptr;
    CUDA_CHECK(cudaMalloc(&G2, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(G2, 0, N_FEATURES * sizeof(float)));

    const int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float lr = 0.10f;
    const float eps = 1e-8f;

    StageResult out;
    out.key = "adagrad";
    out.name = "Adagrad";
    out.opt_mem_bytes = N_FEATURES * sizeof(float);

    print_header("1. Adagrad");
    GpuTimer timer;

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        timer.tic();
        forward_backward(ds, N_SAMPLES, N_FEATURES);
        kernel_adagrad_update<<<opt_blocks, BLOCK_SIZE>>>(ds.w, ds.grad, G2, lr, eps, N_FEATURES);
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

    CUDA_CHECK(cudaFree(G2));
    free_dataset(ds);
    return out;
}

#ifdef STAGE1_STANDALONE
int main() {
    auto result = run_adagrad_stage();
    const auto& last = result.history.back();
    printf("\nFinal: loss=%.6f acc=%.2f%% total=%.2f ms\n", last.loss, last.acc * 100.0f, result.total_ms);
    return 0;
}
#endif
