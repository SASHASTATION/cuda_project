#pragma once

#include "config.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        CUDA_CHECK(cudaGetLastError()); \
    } while (0)

struct GpuTimer {
    cudaEvent_t start{}, stop{};

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic() { CUDA_CHECK(cudaEventRecord(start)); }
    float toc_ms() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

struct Dataset {
    float* X = nullptr;
    float* y = nullptr;
    float* w = nullptr;
    float* pred = nullptr;
    float* errors = nullptr;
    float* grad = nullptr;
    float* losses = nullptr;
};

inline void allocate_dataset(Dataset& ds, int N, int D) {
    CUDA_CHECK(cudaMalloc(&ds.X, (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.y, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.w, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.pred, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.errors, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.grad, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds.losses, N * sizeof(float)));
}

inline void free_dataset(Dataset& ds) {
    cudaFree(ds.X);
    cudaFree(ds.y);
    cudaFree(ds.w);
    cudaFree(ds.pred);
    cudaFree(ds.errors);
    cudaFree(ds.grad);
    cudaFree(ds.losses);
    ds = {};
}

__global__ void kernel_init_true_weights(float* w_true, int D, unsigned long seed) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    curandState st;
    curand_init(seed, j, 0, &st);
    w_true[j] = 0.5f * curand_normal(&st);
}

__global__ void kernel_generate_data(float* X, float* y, const float* w_true, int N, int D, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState st;
    curand_init(seed, i, 0, &st);

    float dot = 0.0f;
    for (int j = 0; j < D; ++j) {
        float x = curand_normal(&st) * rsqrtf((float)D);
        X[(size_t)i * D + j] = x;
        dot += x * w_true[j];
    }

    float p = 1.0f / (1.0f + expf(-dot));
    y[i] = (curand_uniform(&st) < p) ? 1.0f : 0.0f;
}

inline void generate_data(Dataset& ds, int N, int D) {
    float* w_true = nullptr;
    CUDA_CHECK(cudaMalloc(&w_true, D * sizeof(float)));

    int blocks_d = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_init_true_weights<<<blocks_d, BLOCK_SIZE>>>(w_true, D, 1234UL);
    CUDA_KERNEL_CHECK();

    kernel_generate_data<<<blocks_n, BLOCK_SIZE>>>(ds.X, ds.y, w_true, N, D, 5678UL);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(ds.w, 0, D * sizeof(float)));
    CUDA_CHECK(cudaFree(w_true));
}

__global__ void kernel_predict(const float* X, const float* w, float* pred, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dot = 0.0f;
    for (int j = 0; j < D; ++j) {
        dot += X[(size_t)i * D + j] * w[j];
    }
    pred[i] = 1.0f / (1.0f + expf(-dot));
}

__global__ void kernel_errors(const float* pred, const float* y, float* errors, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    errors[i] = pred[i] - y[i];
}

__global__ void kernel_gradient(const float* X, const float* errors, float* grad, int N, int D) {
    int feature = blockIdx.x;
    if (feature >= D) return;

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += errors[i] * X[(size_t)i * D + feature];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) grad[feature] = sdata[0] / (float)N;
}

inline void forward_backward(Dataset& ds, int N, int D) {
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_predict<<<blocks_n, BLOCK_SIZE>>>(ds.X, ds.w, ds.pred, N, D);
    CUDA_KERNEL_CHECK();

    kernel_errors<<<blocks_n, BLOCK_SIZE>>>(ds.pred, ds.y, ds.errors, N);
    CUDA_KERNEL_CHECK();

    kernel_gradient<<<D, BLOCK_SIZE>>>(ds.X, ds.errors, ds.grad, N, D);
    CUDA_KERNEL_CHECK();
}

__global__ void kernel_bce_loss(const float* pred, const float* y, float* losses, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float p = fminf(fmaxf(pred[i], 1e-7f), 1.0f - 1e-7f);
    losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
}

__global__ void kernel_reduce_sum(const float* data, float* total, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? data[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total, sdata[0]);
}

__global__ void kernel_accuracy(const float* pred, const float* y, int* correct, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int pred_label = (pred[i] >= 0.5f) ? 1 : 0;
    int true_label = (y[i] >= 0.5f) ? 1 : 0;
    if (pred_label == true_label) atomicAdd(correct, 1);
}

inline float compute_loss(const Dataset& ds, int N) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float* total = nullptr;
    float host_total = 0.0f;

    CUDA_CHECK(cudaMalloc(&total, sizeof(float)));
    CUDA_CHECK(cudaMemset(total, 0, sizeof(float)));

    kernel_bce_loss<<<blocks, BLOCK_SIZE>>>(ds.pred, ds.y, ds.losses, N);
    CUDA_KERNEL_CHECK();
    kernel_reduce_sum<<<blocks, BLOCK_SIZE>>>(ds.losses, total, N);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(&host_total, total, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(total));
    return host_total / N;
}

inline float compute_accuracy(const Dataset& ds, int N) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int* correct = nullptr;
    int host_correct = 0;

    CUDA_CHECK(cudaMalloc(&correct, sizeof(int)));
    CUDA_CHECK(cudaMemset(correct, 0, sizeof(int)));

    kernel_accuracy<<<blocks, BLOCK_SIZE>>>(ds.pred, ds.y, correct, N);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(&host_correct, correct, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(correct));
    return (float)host_correct / N;
}

inline void evaluate_model(Dataset& ds, int N, int D, float& loss, float& acc) {
    int blocks_n = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_predict<<<blocks_n, BLOCK_SIZE>>>(ds.X, ds.w, ds.pred, N, D);
    CUDA_KERNEL_CHECK();
    loss = compute_loss(ds, N);
    acc = compute_accuracy(ds, N);
}

inline void print_header(const char* title) {
    printf("\n====================================================\n");
    printf("  %s\n", title);
    printf("  Dataset: %d samples x %d features\n", N_SAMPLES, N_FEATURES);
    printf("  Epochs:  %d\n", N_EPOCHS);
    printf("====================================================\n");
    printf("  Epoch |    Loss    | Accuracy |  Time (ms)\n");
    printf("  ------|------------|----------|----------\n");
}
