// ============================================================
//  Этап 4 — Mixed Precision Adam (FP16 + FP32)
//
//  Схема Mixed Precision Training:
//    1. Master weights в FP32 — точное накопление обновлений
//    2. Рабочие веса в FP16 — для forward/backward (2x экономия памяти)
//    3. Loss scaling — предотвращает underflow мелких FP16-градиентов
//    4. Состояние оптимизатора (m, v) всегда в FP32
//
//  Техники:
//    • half2 intrinsics — 2 FP16 операции за 1 инструкцию
//    • FP32 → FP16 cast kernels
//    • Loss scaling + unscaling
//    • Сравнение точности и скорости с чистым FP32 Adam
//
//  Требования: compute capability >= 5.3, рекомендуется >= 7.0
// ============================================================

#include "common.cuh"
#include <cuda_fp16.h>

// ============================================================
//  FP32 Adam kernel (baseline, тот же что в Stage 2)
// ============================================================
__global__ void kernel_adam_fp32(float* w, const float* grad,
                                 float* m, float* v,
                                 float lr, float beta1, float beta2,
                                 float eps, float bc1, float bc2,
                                 int D)
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

// ============================================================
//  FP32 → FP16 конвертация
// ============================================================
__global__ void kernel_cast_fp32_to_fp16(const float* src, __half* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = __float2half(src[i]);
}

// ============================================================
//  Forward pass в FP16
//  Данные и веса в FP16, half2 для попарной обработки фичей.
//  Результат (предсказание) в FP32 — для стабильного вычисления loss.
// ============================================================
__global__ void kernel_predict_fp16(const __half* X_fp16,
                                    const __half* w_fp16,
                                    float* pred,
                                    int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // half2: обрабатываем по 2 фичи за инструкцию
    // требует D чётное (1000 — ОК)
    int D2 = D / 2;
    const half2* X2 = (const half2*)(X_fp16 + (size_t)i * D);
    const half2* w2 = (const half2*)w_fp16;

    float dot = 0.0f;
    for (int j = 0; j < D2; j++) {
        half2 x_pair = X2[j];  // загружаем 2 фичи за раз (4 байта vs 8 для FP32)
        half2 w_pair = w2[j];
        // __hmul2: умножает обе половины параллельно
        half2 prod = __hmul2(x_pair, w_pair);
        // аккумулируем в FP32 для точности
        dot += __low2float(prod) + __high2float(prod);
    }

    // sigmoid в FP32
    pred[i] = 1.0f / (1.0f + expf(-dot));
}

// ============================================================
//  Градиент: X_fp16 * errors → grad_fp16
//  Ошибки в FP32, данные в FP16, результат в FP16 с loss scaling
// ============================================================
__global__ void kernel_gradient_mixed(const __half* X_fp16,
                                      const float* errors,
                                      __half* grad_fp16,
                                      float loss_scale,
                                      int N, int D)
{
    int feature = blockIdx.x;
    if (feature >= D) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // редукция по сэмплам для одной фичи
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float x_val = __half2float(X_fp16[(size_t)i * D + feature]);
        sum += errors[i] * x_val;
    }
    sdata[tid] = sum;
    __syncthreads();

    // shared memory reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        // loss scaling: масштабируем градиент перед записью в FP16
        // это сдвигает мелкие значения из зоны underflow
        float g = (sdata[0] / (float)N) * loss_scale;
        grad_fp16[feature] = __float2half(g);
    }
}

// ============================================================
//  Mixed Precision Adam update
//  Вход: FP16 градиенты (scaled) → FP32 для вычислений
//  Обновляем master weights (FP32) и рабочие веса (FP16)
// ============================================================
__global__ void kernel_adam_mixed(float* w_master,       // FP32 master [D]
                                  __half* w_fp16,        // FP16 рабочие [D]
                                  const __half* grad_fp16, // FP16 scaled grad [D]
                                  float* m, float* v,    // FP32 моменты [D]
                                  float lr, float beta1, float beta2,
                                  float eps, float bc1, float bc2,
                                  float inv_loss_scale,  // 1/loss_scale для unscaling
                                  int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    // FP16 → FP32 + loss unscaling
    float g = __half2float(grad_fp16[j]) * inv_loss_scale;

    // Adam update — целиком в FP32
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;

    float m_hat = mj / bc1;
    float v_hat = vj / bc2;

    float w_new = w_master[j] - lr * m_hat / (sqrtf(v_hat) + eps);

    // записываем оба представления
    w_master[j] = w_new;                  // FP32 для точности
    w_fp16[j]   = __float2half(w_new);    // FP16 для следующего forward pass

    m[j] = mj;
    v[j] = vj;
}


int main() {
    int N = N_SAMPLES;
    int D = N_FEATURES;
    int blocks_n   = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int opt_blocks = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float lr = 0.001f, beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    // ============================================================
    //  ЧАСТЬ A: FP32 Adam (baseline)
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

        print_header("Stage 4a: FP32 Adam (baseline)");
        GpuTimer timer;
        float total_ms = 0.0f;
        float final_loss = 0, final_acc = 0;

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            timer.tic();
            forward_backward(ds, N, D);

            int t = epoch + 1;
            float bc1 = 1.0f - powf(beta1, (float)t);
            float bc2 = 1.0f - powf(beta2, (float)t);

            kernel_adam_fp32<<<opt_blocks, BLOCK_SIZE>>>(
                ds.d_w, ds.d_grad, d_m, d_v,
                lr, beta1, beta2, eps, bc1, bc2, D);

            timer.toc();
            total_ms += timer.ms();

            if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
                final_loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                final_acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                       epoch, final_loss, final_acc * 100.0f, timer.ms());
            }
        }
        printf("  Total: %.2f ms | Avg: %.2f ms/epoch\n", total_ms, total_ms / N_EPOCHS);
        printf("  Memory (weights): %zu KB (FP32)\n", D * sizeof(float) / 1024);
        printf("  Memory (optim):   %zu KB (m + v FP32)\n", 2 * D * sizeof(float) / 1024);

        float fp32_total = total_ms;
        float fp32_loss  = final_loss;
        float fp32_acc   = final_acc;

        CUDA_CHECK(cudaFree(d_m));
        CUDA_CHECK(cudaFree(d_v));

        // ============================================================
        //  ЧАСТЬ B: Mixed Precision Adam
        // ============================================================

        // перегенерируем данные для честного сравнения
        generate_data(ds, N, D);

        // FP16 копия данных X
        __half* d_X_fp16;
        CUDA_CHECK(cudaMalloc(&d_X_fp16, (size_t)N * D * sizeof(__half)));
        {
            int total = N * D;
            int cast_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
            kernel_cast_fp32_to_fp16<<<cast_blocks, BLOCK_SIZE>>>(ds.d_X, d_X_fp16, total);
        }

        // FP16 рабочие веса + FP32 master weights
        __half* d_w_fp16;
        float* d_w_master;
        CUDA_CHECK(cudaMalloc(&d_w_fp16,   D * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_w_master, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_w_master, 0, D * sizeof(float)));
        kernel_cast_fp32_to_fp16<<<opt_blocks, BLOCK_SIZE>>>(d_w_master, d_w_fp16, D);

        // FP16 градиенты
        __half* d_grad_fp16;
        CUDA_CHECK(cudaMalloc(&d_grad_fp16, D * sizeof(__half)));

        // FP32 состояние оптимизатора
        float *d_m2, *d_v2;
        CUDA_CHECK(cudaMalloc(&d_m2, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v2, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_m2, 0, D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v2, 0, D * sizeof(float)));

        // loss scaling — статический (динамический сложнее, но принцип тот же)
        float loss_scale     = 1024.0f;
        float inv_loss_scale = 1.0f / loss_scale;

        print_header("Stage 4b: Mixed Precision Adam (FP16 + FP32)");
        float total_mixed_ms = 0.0f;
        float mixed_loss = 0, mixed_acc = 0;

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            timer.tic();

            // 1. Forward в FP16
            kernel_predict_fp16<<<blocks_n, BLOCK_SIZE>>>(
                d_X_fp16, d_w_fp16, ds.d_pred, N, D);

            // 2. Ошибки (FP32)
            kernel_errors<<<blocks_n, BLOCK_SIZE>>>(
                ds.d_pred, ds.d_y, ds.d_errors, N);

            // 3. Градиент: X_fp16 * errors → grad_fp16 (с loss scaling)
            kernel_gradient_mixed<<<D, BLOCK_SIZE>>>(
                d_X_fp16, ds.d_errors, d_grad_fp16, loss_scale, N, D);

            // 4. Mixed Adam update
            int t = epoch + 1;
            float bc1 = 1.0f - powf(beta1, (float)t);
            float bc2 = 1.0f - powf(beta2, (float)t);

            kernel_adam_mixed<<<opt_blocks, BLOCK_SIZE>>>(
                d_w_master, d_w_fp16, d_grad_fp16,
                d_m2, d_v2,
                lr, beta1, beta2, eps, bc1, bc2,
                inv_loss_scale, D);

            timer.toc();
            total_mixed_ms += timer.ms();

            if (epoch % 10 == 0 || epoch == N_EPOCHS - 1) {
                mixed_loss = compute_loss(ds.d_pred, ds.d_y, ds.d_losses, N);
                mixed_acc  = compute_accuracy(ds.d_pred, ds.d_y, N);
                printf("  %5d | %10.6f | %6.2f%%  | %8.2f\n",
                       epoch, mixed_loss, mixed_acc * 100.0f, timer.ms());
            }
        }
        printf("  Total: %.2f ms | Avg: %.2f ms/epoch\n",
               total_mixed_ms, total_mixed_ms / N_EPOCHS);

        size_t mem_weights_fp16 = D * sizeof(__half);
        size_t mem_master_fp32  = D * sizeof(float);
        size_t mem_optim        = 2 * D * sizeof(float);
        printf("  Memory (FP16 weights): %zu KB\n", mem_weights_fp16 / 1024);
        printf("  Memory (FP32 master):  %zu KB\n", mem_master_fp32 / 1024);
        printf("  Memory (optim m+v):    %zu KB\n", mem_optim / 1024);

        // ============================================================
        //  СРАВНЕНИЕ
        // ============================================================
        printf("\n  ╔══════════════════════════════════════════════╗\n");
        printf("  ║         FP32 vs Mixed Precision Adam         ║\n");
        printf("  ╠════════════════╦═══════════╦═════════════════╣\n");
        printf("  ║                ║   FP32    ║  Mixed (FP16)   ║\n");
        printf("  ╠════════════════╬═══════════╬═════════════════╣\n");
        printf("  ║ Total time     ║ %7.1f ms ║ %7.1f ms       ║\n",
               fp32_total, total_mixed_ms);
        printf("  ║ Final loss     ║ %9.6f ║ %9.6f       ║\n",
               fp32_loss, mixed_loss);
        printf("  ║ Final accuracy ║   %5.2f%%  ║   %5.2f%%        ║\n",
               fp32_acc * 100, mixed_acc * 100);
        printf("  ║ Speedup        ║   1.00x   ║   %.2fx          ║\n",
               fp32_total / total_mixed_ms);
        printf("  ╠════════════════╬═══════════╬═════════════════╣\n");

        size_t fp32_data_mem = (size_t)N * D * sizeof(float);
        size_t fp16_data_mem = (size_t)N * D * sizeof(__half);
        printf("  ║ Data memory    ║ %5zu MB  ║ %5zu MB         ║\n",
               fp32_data_mem / (1024*1024), fp16_data_mem / (1024*1024));
        printf("  ╚════════════════╩═══════════╩═════════════════╝\n");

        printf("\n  Ключевые наблюдения:\n");
        printf("    • FP16 data = 2x меньше GPU RAM → больше batch / больше модель\n");
        printf("    • half2: 2 элемента за 1 инструкцию → выше throughput\n");
        printf("    • Loss scaling сохраняет мелкие градиенты от underflow\n");
        printf("    • Master weights в FP32 → точность не деградирует\n");

        // --- очистка ---
        CUDA_CHECK(cudaFree(d_X_fp16));
        CUDA_CHECK(cudaFree(d_w_fp16));
        CUDA_CHECK(cudaFree(d_w_master));
        CUDA_CHECK(cudaFree(d_grad_fp16));
        CUDA_CHECK(cudaFree(d_m2));
        CUDA_CHECK(cudaFree(d_v2));
        free_dataset(ds);
    }

    printf("====================================================\n\n");
    return 0;
}
