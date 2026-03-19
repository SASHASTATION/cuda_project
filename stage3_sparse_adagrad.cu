// ============================================================
//  Этап 3 — Sparse Adagrad
//
//  Сценарий: текст, рекомендации — большинство фичей = 0.
//  Обновляем только веса с ненулевыми градиентами.
//
//  Техники:
//    • Sparse-представление данных (CSR-подобное)
//    • Sparse-градиент: (index, value) пары
//    • atomicAdd для безопасного параллельного обновления
//    • Сравнение dense vs sparse Adagrad на разреженных данных
//
//  Датасет: те же 100k×1000, но ~10% ненулевых фичей на сэмпл
// ============================================================

#include "common.cuh"

// ============================================================
//  Генерация РАЗРЕЖЕННЫХ данных
//  Каждый сэмпл имеет ~SPARSITY_RATIO ненулевых фичей
// ============================================================
#define SPARSITY_RATIO 0.10f   // 10% ненулевых

__global__ void kernel_generate_sparse_data(float* X, float* y,
                                            float* w_true,
                                            int N, int D,
                                            unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    float dot = 0.0f;
    for (int j = 0; j < D; j++) {
        float val = 0.0f;
        // с вероятностью SPARSITY_RATIO генерируем ненулевое значение
        if (curand_uniform(&rng) < SPARSITY_RATIO) {
            val = curand_normal(&rng) * rsqrtf((float)D * SPARSITY_RATIO);
        }
        X[idx * D + j] = val;
        dot += val * w_true[j];
    }

    float prob = 1.0f / (1.0f + expf(-dot));
    y[idx] = (curand_uniform(&rng) < prob) ? 1.0f : 0.0f;
}

// ============================================================
//  Ядро: определить ненулевые градиенты
//  Порог: |grad[j]| > threshold → включаем в sparse-набор
//  Записываем индексы и значения + общее число ненулевых
// ============================================================
#define GRAD_THRESHOLD 1e-7f

__global__ void kernel_extract_sparse_grad(const float* grad,
                                           int* sparse_idx,
                                           float* sparse_val,
                                           int* nnz,      // device-счётчик
                                           int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = grad[j];
    if (fabsf(g) > GRAD_THRESHOLD) {
        // atomicAdd возвращает старое значение — это наш слот
        int pos = atomicAdd(nnz, 1);
        sparse_idx[pos] = j;
        sparse_val[pos] = g;
    }
}

// ============================================================
//  Sparse Adagrad: обновляем только веса из sparse-набора
//  Каждый поток обрабатывает один ненулевой градиент.
//  atomicAdd не нужен здесь: каждый индекс уникален в наборе.
// ============================================================
__global__ void kernel_sparse_adagrad_update(float* w,
                                             float* G2,
                                             const int* sparse_idx,
                                             const float* sparse_val,
                                             int nnz,
                                             float lr, float eps)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;

    int j   = sparse_idx[k];   // индекс фичи
    float g = sparse_val[k];   // значение градиента

    // обновляем только задействованные веса
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}

// ============================================================
//  Dense Adagrad для сравнения (из Stage 1)
// ============================================================
__global__ void kernel_adagrad_dense(float* w, const float* grad,
                                     float* G2, float lr, float eps, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g = grad[j];
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}


int main() {
    // --- подготовка разреженных данных ---
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);

    // генерируем w_true
    float* d_w_true;
    CUDA_CHECK(cudaMalloc(&d_w_true, N_FEATURES * sizeof(float)));
    int blocks_d = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_init_true_weights<<<blocks_d, BLOCK_SIZE>>>(d_w_true, N_FEATURES, 123456UL);

    // генерируем разреженные данные
    int blocks_n = (N_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_generate_sparse_data<<<blocks_n, BLOCK_SIZE>>>(
        ds.d_X, ds.d_y, d_w_true, N_SAMPLES, N_FEATURES, 654321UL);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_w_true));
    CUDA_CHECK(cudaMemset(ds.d_w, 0, N_FEATURES * sizeof(float)));

    // --- буферы для sparse-представления ---
    int*   d_sparse_idx;
    float* d_sparse_val;
    int*   d_nnz;
    CUDA_CHECK(cudaMalloc(&d_sparse_idx, N_FEATURES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sparse_val, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nnz,        sizeof(int)));

    // --- состояние оптимизатора ---
    float* d_G2;
    CUDA_CHECK(cudaMalloc(&d_G2, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G2, 0, N_FEATURES * sizeof(float)));

    float lr  = 0.1f;
    float eps = 1e-8f;
    int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ============================================================
    //  RUN 1: Dense Adagrad (baseline на разреженных данных)
    // ============================================================
    print_header("Stage 3a: Dense Adagrad (baseline, sparse data)");
    GpuTimer timer;
    float total_dense_ms = 0.0f;

    // сброс весов и G²
    CUDA_CHECK(cudaMemset(ds.d_w, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G2,   0, N_FEATURES * sizeof(float)));

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        timer.tic();
        forward_backward(ds, N_SAMPLES, N_FEATURES);
        kernel_adagrad_dense<<<opt_blocks, BLOCK_SIZE>>>(
            ds.d_w, ds.d_grad, d_G2, lr, eps, N_FEATURES);
        timer.toc();
        float epoch_ms = timer.ms();
        total_dense_ms += epoch_ms;

        if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
            float loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N_SAMPLES);
            float acc  = compute_accuracy(ds.d_pred, ds.d_y, N_SAMPLES);
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                   epoch, loss, acc * 100.0f, epoch_ms);
        }
    }
    printf("  Total: %.2f ms | Avg: %.2f ms/epoch\n",
           total_dense_ms, total_dense_ms / N_EPOCHS);

    // ============================================================
    //  RUN 2: Sparse Adagrad
    // ============================================================
    print_header("Stage 3b: Sparse Adagrad (sparse data)");
    float total_sparse_ms = 0.0f;

    // сброс весов и G²
    CUDA_CHECK(cudaMemset(ds.d_w, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_G2,   0, N_FEATURES * sizeof(float)));

    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        timer.tic();

        // forward + backward (общий для обоих вариантов)
        forward_backward(ds, N_SAMPLES, N_FEATURES);

        // --- извлекаем ненулевые градиенты ---
        CUDA_CHECK(cudaMemset(d_nnz, 0, sizeof(int)));
        kernel_extract_sparse_grad<<<opt_blocks, BLOCK_SIZE>>>(
            ds.d_grad, d_sparse_idx, d_sparse_val, d_nnz, N_FEATURES);

        // читаем количество ненулевых на CPU
        int h_nnz;
        CUDA_CHECK(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

        // --- sparse обновление (только nnz потоков!) ---
        if (h_nnz > 0) {
            int sparse_blocks = (h_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_sparse_adagrad_update<<<sparse_blocks, BLOCK_SIZE>>>(
                ds.d_w, d_G2, d_sparse_idx, d_sparse_val,
                h_nnz, lr, eps);
        }

        timer.toc();
        float epoch_ms = timer.ms();
        total_sparse_ms += epoch_ms;

        if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
            float loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N_SAMPLES);
            float acc  = compute_accuracy(ds.d_pred, ds.d_y, N_SAMPLES);
            printf("  %5d | %10.6f | %6.2f%%  | %8.2f  (nnz=%d/%d)\n",
                   epoch, loss, acc * 100.0f, epoch_ms, h_nnz, N_FEATURES);
        }
    }

    printf("  Total: %.2f ms | Avg: %.2f ms/epoch\n",
           total_sparse_ms, total_sparse_ms / N_EPOCHS);

    // --- сравнение ---
    printf("\n  === Сравнение (update-step only) ===\n");
    printf("  Dense  total: %8.2f ms\n", total_dense_ms);
    printf("  Sparse total: %8.2f ms\n", total_sparse_ms);
    printf("  Speedup: %.2fx\n", total_dense_ms / total_sparse_ms);
    printf("  (Основной выигрыш — в ядре обновления весов;\n");
    printf("   forward/backward по-прежнему dense)\n");

    // --- очистка ---
    CUDA_CHECK(cudaFree(d_G2));
    CUDA_CHECK(cudaFree(d_sparse_idx));
    CUDA_CHECK(cudaFree(d_sparse_val));
    CUDA_CHECK(cudaFree(d_nnz));
    free_dataset(ds);
    printf("====================================================\n\n");

    return 0;
}
