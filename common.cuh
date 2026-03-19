#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// ============================================================
//  Параметры бенчмарка (единые для всех этапов)
// ============================================================
#define N_SAMPLES   100000   // количество точек
#define N_FEATURES  1000     // количество фичей
#define N_EPOCHS    50       // эпохи обучения
#define BLOCK_SIZE  256      // потоков в блоке

// ============================================================
//  Макрос проверки ошибок CUDA
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
//  Таймер на CUDA events
// ============================================================
struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic()  { CUDA_CHECK(cudaEventRecord(start, 0)); }
    void toc()  { CUDA_CHECK(cudaEventRecord(stop, 0));
                   CUDA_CHECK(cudaEventSynchronize(stop)); }
    float ms()  { float t; cudaEventElapsedTime(&t, start, stop); return t; }
};

// ============================================================
//  Генерация синтетических данных на GPU
//  X ~ N(0, 1/sqrt(D)),  y = sigma(X * w_true) > 0.5
// ============================================================
__global__ void kernel_generate_data(float* X, float* y,
                                     float* w_true,
                                     int N, int D, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // каждый поток — один сэмпл
    curandState rng;
    curand_init(seed, idx, 0, &rng);

    float dot = 0.0f;
    for (int j = 0; j < D; j++) {
        float val = curand_normal(&rng) * rsqrtf((float)D);
        X[idx * D + j] = val;                    // row-major: coalesced при чтении по фичам
        dot += val * w_true[j];
    }

    // бинарная метка через сигмоиду + шум
    float prob = 1.0f / (1.0f + expf(-dot));
    float noise = curand_uniform(&rng);
    y[idx] = (noise < prob) ? 1.0f : 0.0f;
}

__global__ void kernel_init_true_weights(float* w_true, int D, unsigned long seed) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    curandState rng;
    curand_init(seed + 42, j, 0, &rng);
    w_true[j] = curand_normal(&rng) * 0.5f;
}

// ============================================================
//  Forward pass: предсказание = sigmoid(X * w)
//  Каждый поток — один сэмпл
// ============================================================
__global__ void kernel_predict(const float* X, const float* w,
                               float* pred, int N, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dot = 0.0f;
    for (int j = 0; j < D; j++) {
        dot += X[i * D + j] * w[j];
    }
    pred[i] = 1.0f / (1.0f + expf(-dot));
}

// ============================================================
//  Ошибки: error[i] = pred[i] - y[i]
// ============================================================
__global__ void kernel_errors(const float* pred, const float* y,
                              float* errors, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    errors[i] = pred[i] - y[i];
}

// ============================================================
//  Градиент: grad[j] = (1/N) * sum_i error[i] * X[i,j]
//  Один блок — одна фича, потоки блока делят сэмплы между собой
//  Shared-memory редукция внутри блока
// ============================================================
__global__ void kernel_gradient(const float* X, const float* errors,
                                float* grad, int N, int D)
{
    int feature = blockIdx.x;              // какую фичу считаем
    if (feature >= D) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // каждый поток суммирует свою порцию сэмплов
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += errors[i] * X[i * D + feature]; // coalesced: соседние блоки читают соседние столбцы
    }
    sdata[tid] = sum;
    __syncthreads();

    // параллельная редукция
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        grad[feature] = sdata[0] / (float)N;
    }
}

// ============================================================
//  Binary Cross-Entropy loss (поэлементно)
// ============================================================
__global__ void kernel_bce_loss(const float* pred, const float* y,
                                float* losses, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float p = fmaxf(fminf(pred[i], 1.0f - 1e-7f), 1e-7f);
    losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
}

// ============================================================
//  Подсчёт accuracy
// ============================================================
__global__ void kernel_accuracy(const float* pred, const float* y,
                                int* correct, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int label_pred = (pred[i] >= 0.5f) ? 1 : 0;
    int label_true = (int)y[i];
    if (label_pred == label_true) atomicAdd(correct, 1);
}

// ============================================================
//  Вычислить loss (CPU-редукция для простоты)
// ============================================================
float compute_loss(const float* d_pred, const float* d_y,
                   float* d_losses, int N)
{
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_bce_loss<<<blocks, BLOCK_SIZE>>>(d_pred, d_y, d_losses, N);

    // копируем на CPU и считаем среднее
    float* h_losses = (float*)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_losses, d_losses, N * sizeof(float), cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (int i = 0; i < N; i++) total += h_losses[i];
    free(h_losses);
    return (float)(total / N);
}

// ============================================================
//  Вычислить accuracy
// ============================================================
float compute_accuracy(const float* d_pred, const float* d_y, int N) {
    int* d_correct;
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_accuracy<<<blocks, BLOCK_SIZE>>>(d_pred, d_y, d_correct, N);

    int h_correct;
    CUDA_CHECK(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_correct));
    return (float)h_correct / N;
}

// ============================================================
//  Общая подготовка данных (вызывается из main каждого этапа)
// ============================================================
struct Dataset {
    float *d_X, *d_y;
    float *d_w;            // обучаемые веса
    float *d_pred;         // предсказания
    float *d_errors;       // ошибки pred - y
    float *d_grad;         // градиенты
    float *d_losses;       // поэлементные потери
};

void allocate_dataset(Dataset& ds, int N, int D) {
    CUDA_CHECK(cudaMalloc(&ds.d_X,      (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_y,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_w,      D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_pred,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_errors, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_grad,   D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.d_losses, N * sizeof(float)));
}

void free_dataset(Dataset& ds) {
    cudaFree(ds.d_X);      cudaFree(ds.d_y);
    cudaFree(ds.d_w);      cudaFree(ds.d_pred);
    cudaFree(ds.d_errors); cudaFree(ds.d_grad);
    cudaFree(ds.d_losses);
}

void generate_data(Dataset& ds, int N, int D) {
    // создаём «истинные» веса
    float* d_w_true;
    CUDA_CHECK(cudaMalloc(&d_w_true, D * sizeof(float)));

    int blocks_d = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_init_true_weights<<<blocks_d, BLOCK_SIZE>>>(d_w_true, D, 123456UL);

    // генерируем X и y
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_generate_data<<<blocks_n, BLOCK_SIZE>>>(
        ds.d_X, ds.d_y, d_w_true, N, D, 654321UL);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_w_true));

    // инициализируем обучаемые веса нулями
    CUDA_CHECK(cudaMemset(ds.d_w, 0, D * sizeof(float)));
}

// ============================================================
//  Forward + backward pass (общий для dense-оптимизаторов)
// ============================================================
void forward_backward(Dataset& ds, int N, int D) {
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. предсказания
    kernel_predict<<<blocks_n, BLOCK_SIZE>>>(ds.d_X, ds.d_w, ds.d_pred, N, D);

    // 2. ошибки
    kernel_errors<<<blocks_n, BLOCK_SIZE>>>(ds.d_pred, ds.d_y, ds.d_errors, N);

    // 3. градиенты (один блок на фичу)
    kernel_gradient<<<D, BLOCK_SIZE>>>(ds.d_X, ds.d_errors, ds.d_grad, N, D);
}

// ============================================================
//  Печать шапки результатов
// ============================================================
void print_header(const char* name) {
    printf("\n");
    printf("====================================================\n");
    printf("  %s\n", name);
    printf("  Dataset: %d samples x %d features\n", N_SAMPLES, N_FEATURES);
    printf("  Epochs:  %d\n", N_EPOCHS);
    printf("====================================================\n");
    printf("  Epoch |    Loss    | Accuracy |  Time (ms)\n");
    printf("  ------|------------|----------|----------\n");
}
