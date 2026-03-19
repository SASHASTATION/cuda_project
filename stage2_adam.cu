// ============================================================
//  Этап 2 — Adam (Adaptive Moment Estimation)
//
//  Техники:
//    • Два вспомогательных массива: m (первый момент), v (второй момент)
//    • Bias correction — компенсация нулевой инициализации
//    • Тот же coalesced SoA-паттерн что в Adagrad
//
//  Формулы:
//    m[j] = β₁ * m[j] + (1 - β₁) * grad[j]
//    v[j] = β₂ * v[j] + (1 - β₂) * grad[j]²
//    m̂ = m[j] / (1 - β₁ᵗ)
//    v̂ = v[j] / (1 - β₂ᵗ)
//    w[j] -= lr * m̂ / (sqrt(v̂) + eps)
// ============================================================

#include "common.cuh"

// ============================================================
//  Ядро Adam
//  Каждый поток обновляет один вес.
//  Память: w, grad, m, v — четыре SoA-массива,
//  все coalesced при последовательных потоках.
// ============================================================
__global__ void kernel_adam_update(float* w,            // веса [D]
                                   const float* grad,   // градиенты [D]
                                   float* m,            // первый момент [D]
                                   float* v,            // второй момент [D]
                                   float lr,
                                   float beta1, float beta2,
                                   float eps,
                                   float bc1,           // 1 - beta1^t (предвычислено)
                                   float bc2,           // 1 - beta2^t
                                   int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;

    float g = grad[j];

    // --- экспоненциально скользящие средние ---
    float mj = beta1 * m[j] + (1.0f - beta1) * g;       // momentum (сглаженный градиент)
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;   // скользящий квадрат градиента

    // --- bias correction ---
    // на первых шагах m и v сильно занижены (инициализированы нулями)
    // деление на (1 - βᵗ) компенсирует этот сдвиг
    float m_hat = mj / bc1;
    float v_hat = vj / bc2;

    // --- обновление веса ---
    w[j] -= lr * m_hat / (sqrtf(v_hat) + eps);

    // --- запись состояния обратно ---
    m[j] = mj;
    v[j] = vj;
}


int main() {
    // --- подготовка данных ---
    Dataset ds;
    allocate_dataset(ds, N_SAMPLES, N_FEATURES);
    generate_data(ds, N_SAMPLES, N_FEATURES);

    // --- состояние оптимизатора: два массива m и v ---
    float *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_m, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m, 0, N_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v, 0, N_FEATURES * sizeof(float)));

    // гиперпараметры (стандартные из статьи Kingma & Ba, 2014)
    float lr    = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-8f;

    int opt_blocks = (N_FEATURES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    print_header("Stage 2: Adam (Momentum + Adaptive LR)");
    GpuTimer timer;
    float total_ms = 0.0f;

    // --- цикл обучения ---
    for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
        timer.tic();

        // forward + backward
        forward_backward(ds, N_SAMPLES, N_FEATURES);

        // bias correction коэффициенты
        int t = epoch + 1;
        float bc1 = 1.0f - powf(beta1, (float)t);
        float bc2 = 1.0f - powf(beta2, (float)t);

        // обновление весов Adam
        kernel_adam_update<<<opt_blocks, BLOCK_SIZE>>>(
            ds.d_w, ds.d_grad, d_m, d_v,
            lr, beta1, beta2, eps, bc1, bc2,
            N_FEATURES);

        timer.toc();
        float epoch_ms = timer.ms();
        total_ms += epoch_ms;

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
    size_t opt_mem = 2 * N_FEATURES * sizeof(float); // m + v
    printf("  Optimizer memory:    %.2f KB (m + v)\n", opt_mem / 1024.0f);

    // --- очистка ---
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_v));
    free_dataset(ds);

    printf("\n  Почему Adam > Adagrad:\n");
    printf("    • momentum (m) сглаживает шумные градиенты\n");
    printf("    • экспоненциальный v не растёт бесконечно (в отличие от G² Adagrad)\n");
    printf("    • bias correction убирает смещение на первых шагах\n");
    printf("  Известные слабости Adam:\n");
    printf("    • может не сойтись к оптимуму (Reddi et al., 2018)\n");
    printf("    • чувствителен к eps и lr на поздних этапах\n");
    printf("====================================================\n\n");

    return 0;
}
