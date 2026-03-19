"""
cuda_ops.py — Python-обёртка для CUDA-ядер (ctypes)

Загружает libkernels.so и предоставляет удобный API:
  - GPU memory allocation / data transfer
  - Forward/backward pass
  - Optimizer steps (Adagrad, Adam, Sparse Adagrad, Mixed Adam)
  - Loss / accuracy computation
"""

import ctypes
import os
import numpy as np
from ctypes import c_float, c_int, c_size_t, c_void_p, POINTER, byref

# ============================================================
#  Загрузка shared library
# ============================================================
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libkernels.so")

if not os.path.exists(_lib_path):
    raise RuntimeError(
        f"libkernels.so не найден в {_dir}.\n"
        f"Соберите: make lib  (или nvcc -shared -o libkernels.so kernels.cu ...)"
    )

_lib = ctypes.CDLL(_lib_path)


def _check(ret):
    """Проверка возвращаемого кода CUDA-функции"""
    if ret != 0:
        raise RuntimeError(f"CUDA error (code {ret})")


# ============================================================
#  Низкоуровневые обёртки
# ============================================================
def gpu_malloc(nbytes: int) -> c_void_p:
    """Выделить память на GPU, вернуть device pointer"""
    ptr = c_void_p()
    _check(_lib.cuda_malloc(byref(ptr), c_size_t(nbytes)))
    return ptr


def gpu_free(ptr: c_void_p):
    _check(_lib.cuda_free(ptr))


def to_gpu(arr: np.ndarray) -> c_void_p:
    """Скопировать numpy array на GPU"""
    arr = np.ascontiguousarray(arr)
    nbytes = arr.nbytes
    ptr = gpu_malloc(nbytes)
    _check(_lib.cuda_memcpy_h2d(ptr, arr.ctypes.data_as(c_void_p), c_size_t(nbytes)))
    return ptr


def from_gpu(ptr: c_void_p, shape, dtype=np.float32) -> np.ndarray:
    """Скопировать данные с GPU в numpy array"""
    arr = np.empty(shape, dtype=dtype)
    _check(_lib.cuda_memcpy_d2h(arr.ctypes.data_as(c_void_p), ptr,
                                 c_size_t(arr.nbytes)))
    return arr


def gpu_zeros(nbytes: int) -> c_void_p:
    """Выделить и обнулить GPU-память"""
    ptr = gpu_malloc(nbytes)
    _check(_lib.cuda_memset_zero(ptr, c_size_t(nbytes)))
    return ptr


def sync():
    """cudaDeviceSynchronize"""
    _check(_lib.cuda_sync())


# ============================================================
#  Высокоуровневый API
# ============================================================

def forward_backward(d_X, d_w, d_y, d_pred, d_errors, d_grad, N, D):
    """Forward pass + ошибки + градиент (dense, FP32)"""
    _check(_lib.cuda_forward_backward(d_X, d_w, d_y, d_pred, d_errors, d_grad,
                                       c_int(N), c_int(D)))


def compute_loss(d_pred, d_y, d_buf, N) -> float:
    """BCE loss (GPU reduction)"""
    loss = c_float()
    _check(_lib.cuda_compute_loss(d_pred, d_y, d_buf, byref(loss), c_int(N)))
    return loss.value


def compute_accuracy(d_pred, d_y, N) -> float:
    """Accuracy (GPU reduction)"""
    acc = c_float()
    _check(_lib.cuda_compute_accuracy(d_pred, d_y, byref(acc), c_int(N)))
    return acc.value


def adagrad_step(d_w, d_grad, d_G2, lr, eps, D):
    """Один шаг Adagrad"""
    _check(_lib.cuda_adagrad_step(d_w, d_grad, d_G2,
                                   c_float(lr), c_float(eps), c_int(D)))


def adam_step(d_w, d_grad, d_m, d_v, lr, beta1, beta2, eps, t, D):
    """Один шаг Adam с bias correction"""
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    _check(_lib.cuda_adam_step(d_w, d_grad, d_m, d_v,
                                c_float(lr), c_float(beta1), c_float(beta2),
                                c_float(eps), c_float(bc1), c_float(bc2),
                                c_int(D)))


def sparse_adagrad_step(d_w, d_grad, d_G2, d_sp_idx, d_sp_val,
                        lr, eps, threshold, D) -> int:
    """Один шаг Sparse Adagrad. Возвращает число ненулевых градиентов."""
    nnz = c_int()
    _check(_lib.cuda_sparse_adagrad_step(
        d_w, d_grad, d_G2, d_sp_idx, d_sp_val,
        c_float(lr), c_float(eps), c_float(threshold),
        c_int(D), byref(nnz)))
    return nnz.value


def cast_f32_to_f16(d_src, d_dst, N):
    """FP32 → FP16 конвертация на GPU"""
    _check(_lib.cuda_cast_f32_to_f16(d_src, d_dst, c_int(N)))


def mixed_forward_backward(d_X_fp16, d_w_fp16, d_y, d_pred,
                           d_errors, d_grad_fp16, loss_scale, N, D):
    """Forward + backward в mixed precision"""
    _check(_lib.cuda_mixed_forward_backward(
        d_X_fp16, d_w_fp16, d_y, d_pred, d_errors, d_grad_fp16,
        c_float(loss_scale), c_int(N), c_int(D)))


def mixed_adam_step(d_w_master, d_w_fp16, d_grad_fp16, d_m, d_v,
                    lr, beta1, beta2, eps, t, inv_scale, D):
    """Один шаг Mixed Precision Adam"""
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    _check(_lib.cuda_mixed_adam_step(
        d_w_master, d_w_fp16, d_grad_fp16, d_m, d_v,
        c_float(lr), c_float(beta1), c_float(beta2), c_float(eps),
        c_float(bc1), c_float(bc2), c_float(inv_scale), c_int(D)))
