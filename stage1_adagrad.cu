// ============================================================
//  Этап 1 — Adagrad + Memory Coalescing
//
//  Техники:
//    • SoA (Structure of Arrays) раскладка состояния оптимизатора
//    • Coalesced memory access — потоки warp'а читают соседние адреса
//    • Один вспомогательный массив G² (накопленные квадраты градиентов)
//
//  Формула обновления:
//    G²[j] += grad[j]²
//    w[j]  -= lr * grad[j] / (sqrt(G²[j]) + eps)
// ============================================================

#include "common.cuh"

// ============================================================
//  Ядро Adagrad
//  Каждый поток обновляет один вес — идеальный coalesced доступ:
//  потоки 0..31 в warp'е читают w[0..31], grad[0..31], G2[0..31]
//  — все массивы в SoA, обращения к соседним float'ам
// ============================================================
__global__ void kernel_adagrad_update(float* w,          // веса [D]
                                      const float* grad, // градиенты [D]
                                      float* G2,         // накопленные квадраты [D]
                                      float lr, float eps,
                                      int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    // --- SoA: три массива лежат раздельно в памяти ---
    // warp из 32 потоков читает 32 последовательных float'а
    // → одна транзакция 128 байт (coalesced)

    float g = grad[j];                     // чтение градиента (coalesced)
    float g2 = G2[j];                      // чтение накопленного квадрата

    g2 += g * g;                           // обновляем аккумулятор

    w[j] -= lr * g / (sqrtf(g2) + eps);   // обновляем вес
    G2[j] = g2;                            // записываем обратно (coalesced)
}


int main() {
    // --- подготовка данных ---
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    // --- состояние оптимизатора: один массив G² ---
    float* d_G2;
    CUDA_CHECK(cudaMalloc(&d_G2, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G2, 0, N_FEATURES * sizeof(float)));

    // гиперпараметры
    float lr  = 0.1f;
    float eps = 1e-8f;

    int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    print_header("Stage 1: Adagrad (SoA + Coalesced Access)");
    GpuTimer timer;
    float total_ms = 0.0f;

    // --- цикл обучения ---
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        timer.tic();

        // forward + backward (предсказания → ошибки → градиенты)
        forward_backward(ds, N_SAMPLES, N_FEATURES);

        // обновление весов Adagrad
        kernel_adagrad_update<<<opt_blocks, BLOCK_SIZE>>>(
            ds.d_w, ds.d_grad, d_G2, lr, eps, N_FEATURES);

        timer.toc();
        float epoch_ms = timer.ms();
        total_ms += epoch_ms;

        // метрики каждые 5 эпох
        if (epoch % 5 == 0 || epoch == N_EPOCHS - 1) {
            float loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N_SAMPLES);
            float acc  = compute_accuracy(ds.d_pred, ds.d_y, N_SAMPLES);
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                   epoch, loss, acc * 100.0f, epoch_ms);
        }
    }

    printf("  ------|------------|----------|----------\n");
    printf("  Total training time: %.2f ms\n", total_ms);
    printf("  Avg time per epoch:  %.2f ms\n", total_ms / N_EPOCHS);

    // --- память оптимизатора ---
    size_t opt_mem = N_FEATURES * sizeof(float); // только G²
    printf("  Optimizer memory:    %.2f KB (G² only)\n", opt_mem / 1024.0f);

    // --- очистка ---
    CUDA_CHECK(cudaFree(d_G2));
    free_dataset(ds);
    printf("====================================================\n\n");

    return 0;
}
