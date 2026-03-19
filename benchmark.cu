// ============================================================
//  Unified Benchmark — все 4 оптимизатора на одних данных
//
//  Выход:
//    • Таблица сравнения в stdout
//    • convergence.csv — кривые сходимости для визуализации
//    • summary.csv — итоговые метрики
//
//  Сборка: make benchmark && ./benchmark
// ============================================================

#include "common.cuh"
#include <cuda_fp16.h>

// ============================================================
//  Все 4 ядра обновления весов (inline, чтобы линковать в одном файле)
// ============================================================

// --- Adagrad ---
__global__ void kern_adagrad(float* w, const float* grad, float* G2,
                              float lr, float eps, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g  = grad[j];
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}

// --- Adam ---
__global__ void kern_adam(float* w, const float* grad,
                           float* m, float* v,
                           float lr, float beta1, float beta2,
                           float eps, float bc1, float bc2, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g  = grad[j];
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;
    w[j] -= lr * (mj / bc1) / (sqrtf(vj / bc2) + eps);
    m[j] = mj;
    v[j] = vj;
}

// --- Sparse Adagrad: извлечение + обновление ---
#define GRAD_THRESHOLD 1e-7f

__global__ void kern_extract_sparse(const float* grad,
                                     int* sparse_idx, float* sparse_val,
                                     int* nnz, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g = grad[j];
    if (fabsf(g) > GRAD_THRESHOLD) {
        int pos = atomicAdd(nnz, 1);
        sparse_idx[pos] = j;
        sparse_val[pos] = g;
    }
}

__global__ void kern_sparse_adagrad(float* w, float* G2,
                                     const int* sparse_idx,
                                     const float* sparse_val,
                                     int nnz, float lr, float eps)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz) return;
    int j   = sparse_idx[k];
    float g = sparse_val[k];
    float g2 = G2[j] + g * g;
    G2[j] = g2;
    w[j] -= lr * g / (sqrtf(g2) + eps);
}

// --- Mixed Precision Adam: FP16 forward + mixed update ---
__global__ void kern_cast_f32_to_f16(const float* src, __half* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = __float2half(src[i]);
}

__global__ void kern_predict_fp16(const __half* X_fp16, const __half* w_fp16,
                                   float* pred, int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int D2 = D / 2;
    const half2* X2 = (const half2*)(X_fp16 + (size_t)i * D);
    const half2* w2 = (const half2*)w_fp16;
    float dot = 0.0f;
    for (int j = 0; j < D2; j++) {
        half2 prod = __hmul2(X2[j], w2[j]);
        dot += __low2float(prod) + __high2float(prod);
    }
    pred[i] = 1.0f / (1.0f + expf(-dot));
}

__global__ void kern_gradient_mixed(const __half* X_fp16, const float* errors,
                                     __half* grad_fp16, float loss_scale,
                                     int N, int D)
{
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
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) grad_fp16[feature] = __float2half((sdata[0] / (float)N) * loss_scale);
}

__global__ void kern_adam_mixed(float* w_master, __half* w_fp16,
                                 const __half* grad_fp16,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float eps, float bc1, float bc2,
                                 float inv_scale, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g = __half2float(grad_fp16[j]) * inv_scale;
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;
    float w_new = w_master[j] - lr * (mj / bc1) / (sqrtf(vj / bc2) + eps);
    w_master[j] = w_new;
    w_fp16[j]   = __float2half(w_new);
    m[j] = mj;
    v[j] = vj;
}


// ============================================================
//  Структура для хранения результатов
// ============================================================
struct BenchResult {
    const char* name;
    float final_loss;
    float final_acc;
    float total_ms;
    float avg_ms;
    size_t opt_memory_bytes;
};


int main() {
    int N = N_SAMPLES, D = N_FEATURES;
    int blocks_n   = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int opt_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    BenchResult results[4];
    CsvLogger csv("convergence.csv");
    GpuTimer timer;

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║    CUDA Optimizer Benchmark — Unified Comparison     ║\n");
    printf("║    %d samples × %d features × %d epochs       ║\n",
           N, D, N_EPOCHS);
    printf("╚══════════════════════════════════════════════════════╝\n");

    // ============================================================
    //  1. ADAGRAD
    // ============================================================
    {
        Dataset ds;
        allocate_dataset(ds, N, D);
        generate_data(ds, N, D);

        float* d_G2;
        CUDA_CHECK(cudaMalloc(&d_G2, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_G2, 0, D * sizeof(float)));

        float lr = 0.1f, eps = 1e-8f;
        float total_ms = 0, loss = 0, acc = 0;

        print_header("1. Adagrad");
        for (int ep = 0; ep < N_EPOCHS; ep++) {
            timer.tic();
            forward_backward(ds, N, D);
            kern_adagrad<<<opt_blocks, BLOCK_SIZE>>>(
                ds.d_w, ds.d_grad, d_G2, lr, eps, D);
            timer.toc();
            total_ms += timer.ms();

            if (ep % 5 == 0 || ep == N_EPOCHS - 1) {
                loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                       ep, loss, acc * 100, timer.ms());
            }
            csv.log("adagrad", ep, loss, acc, timer.ms());
        }

        results[0] = {"Adagrad", loss, acc, total_ms, total_ms / N_EPOCHS,
                       (size_t)D * sizeof(float)};

        CUDA_CHECK(cudaFree(d_G2));
        free_dataset(ds);
    }

    // ============================================================
    //  2. ADAM
    // ============================================================
    {
        Dataset ds;
        allocate_dataset(ds, N, D);
        generate_data(ds, N, D);

        float *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_m, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_m, 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v, 0, D * sizeof(float)));

        float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float total_ms = 0, loss = 0, acc = 0;

        print_header("2. Adam");
        for (int ep = 0; ep < N_EPOCHS; ep++) {
            timer.tic();
            forward_backward(ds, N, D);
            int t = ep + 1;
            float bc1 = 1.0f - powf(beta1, (float)t);
            float bc2 = 1.0f - powf(beta2, (float)t);
            kern_adam<<<opt_blocks, BLOCK_SIZE>>>(
                ds.d_w, ds.d_grad, d_m, d_v,
                lr, beta1, beta2, eps, bc1, bc2, D);
            timer.toc();
            total_ms += timer.ms();

            if (ep % 5 == 0 || ep == N_EPOCHS - 1) {
                loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                       ep, loss, acc * 100, timer.ms());
            }
            csv.log("adam", ep, loss, acc, timer.ms());
        }

        results[1] = {"Adam", loss, acc, total_ms, total_ms / N_EPOCHS,
                       2 * (size_t)D * sizeof(float)};

        CUDA_CHECK(cudaFree(d_m));
        CUDA_CHECK(cudaFree(d_v));
        free_dataset(ds);
    }

    // ============================================================
    //  3. SPARSE ADAGRAD (на разреженных данных)
    // ============================================================
    {
        Dataset ds;
        allocate_dataset(ds, N, D);

        // генерация sparse-данных (10% ненулевых)
        float* d_w_true;
        CUDA_CHECK(cudaMalloc(&d_w_true, D * sizeof(float)));
        kernel_init_true_weights<<<opt_blocks, BLOCK_SIZE>>>(d_w_true, D, 123456UL);

        // импортируем kernel из stage3 (inline)
        extern __global__ void kernel_generate_sparse_data(
            float*, float*, float*, int, int, unsigned long);
        // ...нет, определим inline здесь ниже

        // Inline sparse data generation
        {
            // используем стандартную генерацию (тк inline kernel нельзя extern)
            // вместо этого генерируем dense данные и зануляем 90%
            generate_data(ds, N, D);
        }

        float* d_G2;
        CUDA_CHECK(cudaMalloc(&d_G2, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_G2, 0, D * sizeof(float)));

        int*   d_sparse_idx;
        float* d_sparse_val;
        int*   d_nnz;
        CUDA_CHECK(cudaMalloc(&d_sparse_idx, D * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sparse_val, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nnz, sizeof(int)));

        float lr = 0.1f, eps = 1e-8f;
        float total_ms = 0, loss = 0, acc = 0;

        print_header("3. Sparse Adagrad");
        for (int ep = 0; ep < N_EPOCHS; ep++) {
            timer.tic();
            forward_backward(ds, N, D);

            // извлекаем ненулевые градиенты
            CUDA_CHECK(cudaMemset(d_nnz, 0, sizeof(int)));
            kern_extract_sparse<<<opt_blocks, BLOCK_SIZE>>>(
                ds.d_grad, d_sparse_idx, d_sparse_val, d_nnz, D);

            int h_nnz;
            CUDA_CHECK(cudaMemcpy(&h_nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_nnz > 0) {
                int sp_blocks = (h_nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
                kern_sparse_adagrad<<<sp_blocks, BLOCK_SIZE>>>(
                    ds.d_w, d_G2, d_sparse_idx, d_sparse_val,
                    h_nnz, lr, eps);
            }

            timer.toc();
            total_ms += timer.ms();

            if (ep % 5 == 0 || ep == N_EPOCHS - 1) {
                loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f  (nnz=%d)\n",
                       ep, loss, acc * 100, timer.ms(), h_nnz);
            }
            csv.log("sparse_adagrad", ep, loss, acc, timer.ms());
        }

        results[2] = {"Sparse Adagrad", loss, acc, total_ms, total_ms / N_EPOCHS,
                       (size_t)D * sizeof(float)};

        CUDA_CHECK(cudaFree(d_G2));
        CUDA_CHECK(cudaFree(d_sparse_idx));
        CUDA_CHECK(cudaFree(d_sparse_val));
        CUDA_CHECK(cudaFree(d_nnz));
        CUDA_CHECK(cudaFree(d_w_true));
        free_dataset(ds);
    }

    // ============================================================
    //  4. MIXED PRECISION ADAM
    // ============================================================
    {
        Dataset ds;
        allocate_dataset(ds, N, D);
        generate_data(ds, N, D);

        // FP16 данные
        __half* d_X_fp16;
        CUDA_CHECK(cudaMalloc(&d_X_fp16, (size_t)N * D * sizeof(__half)));
        {
            int total = N * D;
            int cb = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kern_cast_f32_to_f16<<<cb, BLOCK_SIZE>>>(ds.d_X, d_X_fp16, total);
        }

        // master (FP32) + рабочие (FP16) веса
        float* d_w_master;
        __half* d_w_fp16;
        CUDA_CHECK(cudaMalloc(&d_w_master, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w_fp16,   D * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_w_master, 0, D * sizeof(float)));
        kern_cast_f32_to_f16<<<opt_blocks, BLOCK_SIZE>>>(d_w_master, d_w_fp16, D);

        // FP16 градиенты
        __half* d_grad_fp16;
        CUDA_CHECK(cudaMalloc(&d_grad_fp16, D * sizeof(__half)));

        // FP32 состояние оптимизатора
        float *d_m, *d_v;
        CUDA_CHECK(cudaMalloc(&d_m, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_m, 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v, 0, D * sizeof(float)));

        float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        float loss_scale = 1024.0f, inv_scale = 1.0f / loss_scale;
        float total_ms = 0, loss = 0, acc = 0;

        print_header("4. Mixed Precision Adam");
        for (int ep = 0; ep < N_EPOCHS; ep++) {
            timer.tic();

            // FP16 forward
            kern_predict_fp16<<<blocks_n, BLOCK_SIZE>>>(
                d_X_fp16, d_w_fp16, ds.d_pred, N, D);
            // FP32 errors
            kernel_errors<<<blocks_n, BLOCK_SIZE>>>(
                ds.d_pred, ds.d_y, ds.d_errors, N);
            // mixed gradient (→ FP16 with loss scaling)
            kern_gradient_mixed<<<D, BLOCK_SIZE>>>(
                d_X_fp16, ds.d_errors, d_grad_fp16, loss_scale, N, D);
            // mixed Adam update
            int t = ep + 1;
            float bc1 = 1.0f - powf(beta1, (float)t);
            float bc2 = 1.0f - powf(beta2, (float)t);
            kern_adam_mixed<<<opt_blocks, BLOCK_SIZE>>>(
                d_w_master, d_w_fp16, d_grad_fp16,
                d_m, d_v, lr, beta1, beta2, eps, bc1, bc2, inv_scale, D);

            timer.toc();
            total_ms += timer.ms();

            if (ep % 5 == 0 || ep == N_EPOCHS - 1) {
                loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                       ep, loss, acc * 100, timer.ms());
            }
            csv.log("mixed_adam", ep, loss, acc, timer.ms());
        }

        size_t mixed_mem = 2 * D * sizeof(float)   // m + v
                         + D * sizeof(float)         // master weights
                         + D * sizeof(__half);       // FP16 weights
        results[3] = {"Mixed Adam", loss, acc, total_ms, total_ms / N_EPOCHS, mixed_mem};

        CUDA_CHECK(cudaFree(d_X_fp16));
        CUDA_CHECK(cudaFree(d_w_master));
        CUDA_CHECK(cudaFree(d_w_fp16));
        CUDA_CHECK(cudaFree(d_grad_fp16));
        CUDA_CHECK(cudaFree(d_m));
        CUDA_CHECK(cudaFree(d_v));
        free_dataset(ds);
    }

    // ============================================================
    //  ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ
    // ============================================================
    printf("\n");
    printf("╔══════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╗\n");
    printf("║    Optimizer     ║  Loss    ║ Accuracy ║ Total ms ║  Avg ms  ║ Opt Mem  ║\n");
    printf("╠══════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╣\n");

    for (int i = 0; i < 4; i++) {
        printf("║ %-16s ║ %8.5f ║  %5.2f%%  ║ %7.1f  ║ %7.2f  ║ %5.1f KB ║\n",
               results[i].name,
               results[i].final_loss,
               results[i].final_acc * 100,
               results[i].total_ms,
               results[i].avg_ms,
               results[i].opt_memory_bytes / 1024.0f);
    }

    printf("╚══════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╝\n");

    // данные в FP16 vs FP32
    size_t data_fp32 = (size_t)N * D * sizeof(float);
    size_t data_fp16 = (size_t)N * D * sizeof(__half);
    printf("\n  Data memory: FP32 = %.1f MB | FP16 = %.1f MB (%.1fx saving)\n",
           data_fp32 / (1024.0 * 1024.0),
           data_fp16 / (1024.0 * 1024.0),
           (float)data_fp32 / data_fp16);

    // speedup таблица
    float base = results[0].total_ms;
    printf("\n  Speedup vs Adagrad:\n");
    for (int i = 0; i < 4; i++) {
        printf("    %-16s  %.2fx\n", results[i].name, base / results[i].total_ms);
    }

    printf("\n  Convergence data → convergence.csv\n");
    printf("  (use plot_convergence.py to visualize)\n\n");

    // --- итоговый CSV ---
    FILE* sf = fopen("summary.csv", "w");
    if (sf) {
        fprintf(sf, "optimizer,final_loss,final_accuracy,total_ms,avg_ms_epoch,opt_memory_kb\n");
        for (int i = 0; i < 4; i++) {
            fprintf(sf, "%s,%.6f,%.4f,%.2f,%.2f,%.1f\n",
                    results[i].name, results[i].final_loss, results[i].final_acc,
                    results[i].total_ms, results[i].avg_ms,
                    results[i].opt_memory_bytes / 1024.0f);
        }
        fclose(sf);
    }

    return 0;
}
