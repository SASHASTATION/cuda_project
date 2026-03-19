// ============================================================
//  kernels.cu — все CUDA-ядра для проекта оптимизаторов
//
//  Компилируется в shared library: libkernels.so
//  Вызывается из Python через ctypes
//
//  Ядра:
//    1. Forward pass (sigmoid prediction)
//    2. Backward pass (errors + gradient)
//    3. Adagrad update
//    4. Adam update
//    5. Sparse Adagrad (extract + update)
//    6. Mixed Precision Adam (FP16 forward, mixed update)
//    7. Loss / Accuracy computation
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

#define BLOCK 256

// ============================================================
//  Вспомогательные макросы
// ============================================================
#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        return -1;                                                   \
    }                                                                \
} while(0)

#define CUDA_CHECK_VOID(call) do {                                   \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
    }                                                                \
} while(0)

// ============================================================
//  1. Forward pass: pred[i] = sigmoid( dot(X[i], w) )
//     Каждый поток — один сэмпл
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
//  2a. Ошибки: error[i] = pred[i] - y[i]
// ============================================================
__global__ void kernel_errors(const float* pred, const float* y,
                              float* errors, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    errors[i] = pred[i] - y[i];
}

// ============================================================
//  2b. Градиент: grad[j] = (1/N) * Σ_i error[i] * X[i,j]
//      Один блок на фичу, shared-memory редукция
// ============================================================
__global__ void kernel_gradient(const float* X, const float* errors,
                                float* grad, int N, int D)
{
    int feature = blockIdx.x;
    if (feature >= D) return;

    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += errors[i] * X[i * D + feature];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        grad[feature] = sdata[0] / (float)N;
    }
}

// ============================================================
//  3. BCE loss (поэлементно)
// ============================================================
__global__ void kernel_bce(const float* pred, const float* y,
                           float* losses, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float p = fmaxf(fminf(pred[i], 1.0f - 1e-7f), 1e-7f);
    losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
}

// ============================================================
//  4. Accuracy
// ============================================================
__global__ void kernel_accuracy(const float* pred, const float* y,
                                int* correct, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int p = (pred[i] >= 0.5f) ? 1 : 0;
    int t = (int)y[i];
    if (p == t) atomicAdd(correct, 1);
}

// ============================================================
//  5. Редукция суммы (shared memory + atomicAdd)
// ============================================================
__global__ void kernel_reduce_sum(const float* data, float* result, int N)
{
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < N) ? data[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

// ============================================================
//  OPTIMIZER KERNELS
// ============================================================

// --- Adagrad ---
// G²[j] += grad[j]²
// w[j]  -= lr * grad[j] / (sqrt(G²[j]) + eps)
__global__ void kernel_adagrad(float* w, const float* grad, float* G2,
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
// m[j] = β₁·m[j] + (1-β₁)·g
// v[j] = β₂·v[j] + (1-β₂)·g²
// w[j] -= lr · (m̂ / (sqrt(v̂) + eps))
__global__ void kernel_adam(float* w, const float* grad,
                            float* m, float* v,
                            float lr, float beta1, float beta2,
                            float eps, float bc1, float bc2, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g  = grad[j];
    float mj = beta1 * m[j] + (1.0f - beta1) * g;
    float vj = beta2 * v[j] + (1.0f - beta2) * g * g;
    float m_hat = mj / bc1;
    float v_hat = vj / bc2;
    w[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
    m[j] = mj;
    v[j] = vj;
}

// --- Sparse Adagrad: извлечение ненулевых градиентов ---
__global__ void kernel_extract_sparse(const float* grad,
                                      int* sparse_idx, float* sparse_val,
                                      int* nnz, float threshold, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float g = grad[j];
    if (fabsf(g) > threshold) {
        int pos = atomicAdd(nnz, 1);
        sparse_idx[pos] = j;
        sparse_val[pos] = g;
    }
}

// --- Sparse Adagrad: обновление только ненулевых ---
__global__ void kernel_sparse_adagrad(float* w, float* G2,
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

// --- Mixed Precision: FP32 → FP16 cast ---
__global__ void kernel_f32_to_f16(const float* src, __half* dst, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dst[i] = __float2half(src[i]);
}

// --- Mixed Precision: FP16 forward с half2 intrinsics ---
__global__ void kernel_predict_fp16(const __half* X_fp16,
                                    const __half* w_fp16,
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

// --- Mixed Precision: gradient X_fp16 * errors → grad_fp16 ---
__global__ void kernel_gradient_mixed(const __half* X_fp16,
                                      const float* errors,
                                      __half* grad_fp16,
                                      float loss_scale, int N, int D)
{
    int feature = blockIdx.x;
    if (feature >= D) return;

    __shared__ float sdata[BLOCK];
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

    if (tid == 0) {
        grad_fp16[feature] = __float2half((sdata[0] / (float)N) * loss_scale);
    }
}

// --- Mixed Precision Adam update ---
__global__ void kernel_adam_mixed(float* w_master, __half* w_fp16,
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
    float m_hat = mj / bc1;
    float v_hat = vj / bc2;
    float w_new = w_master[j] - lr * m_hat / (sqrtf(v_hat) + eps);

    w_master[j] = w_new;
    w_fp16[j]   = __float2half(w_new);
    m[j] = mj;
    v[j] = vj;
}


// ============================================================
//  C API — вызывается из Python через ctypes
//  Каждая функция возвращает 0 при успехе, -1 при ошибке
// ============================================================
extern "C" {

// --- Forward + Backward ---
int cuda_forward_backward(float* d_X, float* d_w, float* d_y,
                          float* d_pred, float* d_errors, float* d_grad,
                          int N, int D)
{
    int bn = (N + BLOCK - 1) / BLOCK;

    kernel_predict<<<bn, BLOCK>>>(d_X, d_w, d_pred, N, D);
    CUDA_CHECK(cudaGetLastError());

    kernel_errors<<<bn, BLOCK>>>(d_pred, d_y, d_errors, N);
    CUDA_CHECK(cudaGetLastError());

    kernel_gradient<<<D, BLOCK>>>(d_X, d_errors, d_grad, N, D);
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

// --- Compute loss (returns via pointer) ---
int cuda_compute_loss(float* d_pred, float* d_y, float* d_buf,
                      float* h_loss, int N)
{
    int bn = (N + BLOCK - 1) / BLOCK;
    kernel_bce<<<bn, BLOCK>>>(d_pred, d_y, d_buf, N);
    CUDA_CHECK(cudaGetLastError());

    // Reduction
    float* d_total;
    CUDA_CHECK(cudaMalloc(&d_total, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_total, 0, sizeof(float)));
    kernel_reduce_sum<<<bn, BLOCK>>>(d_buf, d_total, N);
    CUDA_CHECK(cudaGetLastError());

    float total;
    CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_total));
    *h_loss = total / N;
    return 0;
}

// --- Compute accuracy (returns via pointer) ---
int cuda_compute_accuracy(float* d_pred, float* d_y, float* h_acc, int N)
{
    int bn = (N + BLOCK - 1) / BLOCK;
    int* d_correct;
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
    kernel_accuracy<<<bn, BLOCK>>>(d_pred, d_y, d_correct, N);
    CUDA_CHECK(cudaGetLastError());

    int correct;
    CUDA_CHECK(cudaMemcpy(&correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_correct));
    *h_acc = (float)correct / N;
    return 0;
}

// --- Adagrad step ---
int cuda_adagrad_step(float* d_w, float* d_grad, float* d_G2,
                      float lr, float eps, int D)
{
    int bd = (D + BLOCK - 1) / BLOCK;
    kernel_adagrad<<<bd, BLOCK>>>(d_w, d_grad, d_G2, lr, eps, D);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// --- Adam step ---
int cuda_adam_step(float* d_w, float* d_grad, float* d_m, float* d_v,
                   float lr, float beta1, float beta2, float eps,
                   float bc1, float bc2, int D)
{
    int bd = (D + BLOCK - 1) / BLOCK;
    kernel_adam<<<bd, BLOCK>>>(d_w, d_grad, d_m, d_v,
                               lr, beta1, beta2, eps, bc1, bc2, D);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// --- Sparse Adagrad: extract + update, returns nnz ---
int cuda_sparse_adagrad_step(float* d_w, float* d_grad, float* d_G2,
                             int* d_sparse_idx, float* d_sparse_val,
                             float lr, float eps, float threshold,
                             int D, int* h_nnz)
{
    int bd = (D + BLOCK - 1) / BLOCK;

    // Allocate and zero nnz counter
    int* d_nnz;
    CUDA_CHECK(cudaMalloc(&d_nnz, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_nnz, 0, sizeof(int)));

    kernel_extract_sparse<<<bd, BLOCK>>>(d_grad, d_sparse_idx, d_sparse_val,
                                          d_nnz, threshold, D);
    CUDA_CHECK(cudaGetLastError());

    int nnz;
    CUDA_CHECK(cudaMemcpy(&nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_nnz));

    if (nnz > 0) {
        int sb = (nnz + BLOCK - 1) / BLOCK;
        kernel_sparse_adagrad<<<sb, BLOCK>>>(d_w, d_G2, d_sparse_idx,
                                              d_sparse_val, nnz, lr, eps);
        CUDA_CHECK(cudaGetLastError());
    }

    *h_nnz = nnz;
    return 0;
}

// --- Mixed precision: cast FP32 → FP16 ---
int cuda_cast_f32_to_f16(float* d_src, void* d_dst, int N)
{
    int bn = (N + BLOCK - 1) / BLOCK;
    kernel_f32_to_f16<<<bn, BLOCK>>>(d_src, (__half*)d_dst, N);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// --- Mixed precision forward + backward ---
int cuda_mixed_forward_backward(void* d_X_fp16, void* d_w_fp16,
                                float* d_y, float* d_pred,
                                float* d_errors, void* d_grad_fp16,
                                float loss_scale, int N, int D)
{
    int bn = (N + BLOCK - 1) / BLOCK;

    kernel_predict_fp16<<<bn, BLOCK>>>((__half*)d_X_fp16, (__half*)d_w_fp16,
                                        d_pred, N, D);
    CUDA_CHECK(cudaGetLastError());

    kernel_errors<<<bn, BLOCK>>>(d_pred, d_y, d_errors, N);
    CUDA_CHECK(cudaGetLastError());

    kernel_gradient_mixed<<<D, BLOCK>>>((__half*)d_X_fp16, d_errors,
                                         (__half*)d_grad_fp16, loss_scale, N, D);
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

// --- Mixed Adam step ---
int cuda_mixed_adam_step(float* d_w_master, void* d_w_fp16,
                         void* d_grad_fp16, float* d_m, float* d_v,
                         float lr, float beta1, float beta2, float eps,
                         float bc1, float bc2, float inv_scale, int D)
{
    int bd = (D + BLOCK - 1) / BLOCK;
    kernel_adam_mixed<<<bd, BLOCK>>>(d_w_master, (__half*)d_w_fp16,
                                     (__half*)d_grad_fp16, d_m, d_v,
                                     lr, beta1, beta2, eps, bc1, bc2,
                                     inv_scale, D);
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// --- Synchronize (for timing) ---
int cuda_sync()
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

// --- Memory management wrappers ---
int cuda_malloc(void** ptr, size_t size)
{
    CUDA_CHECK(cudaMalloc(ptr, size));
    return 0;
}

int cuda_free(void* ptr)
{
    CUDA_CHECK(cudaFree(ptr));
    return 0;
}

int cuda_memcpy_h2d(void* dst, const void* src, size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

int cuda_memcpy_d2h(void* dst, const void* src, size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}

int cuda_memset_zero(void* ptr, size_t size)
{
    CUDA_CHECK(cudaMemset(ptr, 0, size));
    return 0;
}

} // extern "C"
