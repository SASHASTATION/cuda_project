#!/usr/bin/env python3
"""
benchmark.py — Единый бенчмарк всех 4 CUDA-оптимизаторов

Генерирует данные в NumPy, копирует на GPU, запускает обучение
через CUDA-ядра, собирает метрики, строит графики.

Запуск:
    python3 benchmark.py              # все оптимизаторы
    python3 benchmark.py --optimizer adagrad  # только один
    python3 benchmark.py --no-plot    # без графиков
"""

import argparse
import time
import csv
import sys
import numpy as np

import cuda_ops as cu

# ============================================================
#  Параметры бенчмарка
# ============================================================
N_SAMPLES  = 100_000
N_FEATURES = 1_000
N_EPOCHS   = 50
SPARSITY   = 0.10  # для sparse Adagrad: 10% ненулевых фичей


# ============================================================
#  Генерация данных (NumPy — легко проверить и отладить)
# ============================================================
def generate_data(N, D, sparse=False, seed=42):
    """
    Синтетический датасет для логистической регрессии.

    X ~ N(0, 1/sqrt(D)),  y = 1{sigmoid(X @ w_true) + noise > 0.5}
    """
    rng = np.random.RandomState(seed)

    # Истинные веса
    w_true = rng.randn(D).astype(np.float32) * 0.5

    # Фичи
    X = rng.randn(N, D).astype(np.float32) / np.sqrt(D)

    if sparse:
        # Зануляем (1-SPARSITY) фичей для имитации разреженных данных
        mask = rng.rand(N, D) < SPARSITY
        X *= mask.astype(np.float32)
        # Подкорректируем масштаб для сохранения дисперсии
        X *= 1.0 / np.sqrt(SPARSITY)

    # Метки
    logits = X @ w_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    noise = rng.rand(N).astype(np.float32)
    y = (noise < probs).astype(np.float32)

    return X, y


def verify_data(X, y):
    """Базовая проверка данных перед обучением"""
    assert X.dtype == np.float32
    assert y.dtype == np.float32
    assert np.all(np.isfinite(X)), "X contains NaN/Inf!"
    assert np.all((y == 0) | (y == 1)), "y must be 0 or 1!"
    balance = y.mean()
    print(f"    Data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"    Class balance: {balance:.1%} positive")
    print(f"    X range: [{X.min():.3f}, {X.max():.3f}], mean={X.mean():.4f}")


# ============================================================
#  Общие GPU-буферы для dense оптимизаторов
# ============================================================
class GpuBuffers:
    """Управление GPU-памятью для обучения"""

    def __init__(self, N, D):
        self.N, self.D = N, D
        sz_f = 4  # sizeof(float)
        self.d_pred   = cu.gpu_zeros(N * sz_f)
        self.d_errors = cu.gpu_zeros(N * sz_f)
        self.d_grad   = cu.gpu_zeros(D * sz_f)
        self.d_buf    = cu.gpu_zeros(N * sz_f)  # для loss reduction

    def free(self):
        for attr in ['d_pred', 'd_errors', 'd_grad', 'd_buf']:
            cu.gpu_free(getattr(self, attr))


# ============================================================
#  Тренировочный цикл — общий для всех dense-оптимизаторов
# ============================================================
def train_loop(name, d_X, d_w, d_y, bufs, optimizer_step, N, D, n_epochs):
    """
    Универсальный цикл обучения.

    optimizer_step(d_w, d_grad, epoch) — вызывается после backward pass
    Возвращает список dict с метриками по эпохам.
    """
    history = []

    for epoch in range(n_epochs):
        t0 = time.perf_counter()

        # Forward + Backward
        cu.forward_backward(d_X, d_w, d_y,
                            bufs.d_pred, bufs.d_errors, bufs.d_grad,
                            N, D)

        # Optimizer step
        extra = optimizer_step(d_w, bufs.d_grad, epoch)

        cu.sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Метрики: пересчитываем предсказания с обновлёнными весами
        cu.forward_backward(d_X, d_w, d_y,
                            bufs.d_pred, bufs.d_errors, bufs.d_grad,
                            N, D)
        cu.sync()

        loss = cu.compute_loss(bufs.d_pred, d_y, bufs.d_buf, N)
        acc  = cu.compute_accuracy(bufs.d_pred, d_y, N)

        record = {
            'epoch': epoch, 'loss': loss, 'acc': acc,
            'time_ms': elapsed_ms, 'stage': name,
        }
        if extra:
            record.update(extra)
        history.append(record)

        # Печать каждые 5 эпох
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            extra_str = ""
            if 'nnz' in record:
                extra_str = f"  (nnz={record['nnz']}/{D})"
            print(f"    Epoch {epoch:3d} | loss={loss:.6f} | "
                  f"acc={acc*100:6.2f}% | {elapsed_ms:7.2f} ms{extra_str}")

    return history


# ============================================================
#  1. Adagrad
# ============================================================
def run_adagrad(X, y, n_epochs=N_EPOCHS):
    N, D = X.shape
    print("\n" + "="*60)
    print("  1. Adagrad (SoA + Coalesced Access)")
    print("="*60)
    verify_data(X, y)

    d_X = cu.to_gpu(X)
    d_y = cu.to_gpu(y)
    d_w = cu.gpu_zeros(D * 4)
    d_G2 = cu.gpu_zeros(D * 4)
    bufs = GpuBuffers(N, D)

    lr, eps = 0.1, 1e-8

    def step(d_w, d_grad, epoch):
        cu.adagrad_step(d_w, d_grad, d_G2, lr, eps, D)
        return None

    history = train_loop("adagrad", d_X, d_w, d_y, bufs, step, N, D, n_epochs)

    bufs.free()
    cu.gpu_free(d_G2)
    cu.gpu_free(d_w)
    cu.gpu_free(d_y)
    cu.gpu_free(d_X)

    return history


# ============================================================
#  2. Adam
# ============================================================
def run_adam(X, y, n_epochs=N_EPOCHS):
    N, D = X.shape
    print("\n" + "="*60)
    print("  2. Adam (Momentum + Adaptive LR + Bias Correction)")
    print("="*60)
    verify_data(X, y)

    d_X = cu.to_gpu(X)
    d_y = cu.to_gpu(y)
    d_w = cu.gpu_zeros(D * 4)
    d_m = cu.gpu_zeros(D * 4)
    d_v = cu.gpu_zeros(D * 4)
    bufs = GpuBuffers(N, D)

    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8

    def step(d_w, d_grad, epoch):
        cu.adam_step(d_w, d_grad, d_m, d_v, lr, beta1, beta2, eps, epoch + 1, D)
        return None

    history = train_loop("adam", d_X, d_w, d_y, bufs, step, N, D, n_epochs)

    bufs.free()
    cu.gpu_free(d_m)
    cu.gpu_free(d_v)
    cu.gpu_free(d_w)
    cu.gpu_free(d_y)
    cu.gpu_free(d_X)

    return history


# ============================================================
#  3. Sparse Adagrad
# ============================================================
def run_sparse_adagrad(X, y, n_epochs=N_EPOCHS):
    N, D = X.shape
    print("\n" + "="*60)
    print("  3. Sparse Adagrad (sparse gradient + atomicAdd)")
    print("="*60)
    verify_data(X, y)

    d_X = cu.to_gpu(X)
    d_y = cu.to_gpu(y)
    d_w = cu.gpu_zeros(D * 4)
    d_G2 = cu.gpu_zeros(D * 4)
    d_sp_idx = cu.gpu_malloc(D * 4)  # int[D]
    d_sp_val = cu.gpu_malloc(D * 4)  # float[D]
    bufs = GpuBuffers(N, D)

    lr, eps, threshold = 0.1, 1e-8, 1e-7

    def step(d_w, d_grad, epoch):
        nnz = cu.sparse_adagrad_step(d_w, d_grad, d_G2, d_sp_idx, d_sp_val,
                                      lr, eps, threshold, D)
        return {'nnz': nnz}

    history = train_loop("sparse_adagrad", d_X, d_w, d_y, bufs, step, N, D, n_epochs)

    bufs.free()
    cu.gpu_free(d_G2)
    cu.gpu_free(d_sp_idx)
    cu.gpu_free(d_sp_val)
    cu.gpu_free(d_w)
    cu.gpu_free(d_y)
    cu.gpu_free(d_X)

    return history


# ============================================================
#  4. Mixed Precision Adam
# ============================================================
def run_mixed_adam(X, y, n_epochs=N_EPOCHS):
    N, D = X.shape
    print("\n" + "="*60)
    print("  4. Mixed Precision Adam (FP16 data + FP32 master weights)")
    print("="*60)
    verify_data(X, y)

    d_X_f32 = cu.to_gpu(X)
    d_y     = cu.to_gpu(y)

    # FP16 копия данных
    d_X_fp16 = cu.gpu_malloc(N * D * 2)  # sizeof(half) = 2
    cu.cast_f32_to_f16(d_X_f32, d_X_fp16, N * D)
    cu.gpu_free(d_X_f32)  # FP32 данные больше не нужны

    # Веса: master (FP32) + рабочие (FP16)
    d_w_master = cu.gpu_zeros(D * 4)
    d_w_fp16   = cu.gpu_zeros(D * 2)

    # FP16 градиенты
    d_grad_fp16 = cu.gpu_malloc(D * 2)

    # FP32 состояние оптимизатора
    d_m = cu.gpu_zeros(D * 4)
    d_v = cu.gpu_zeros(D * 4)

    # Буферы для метрик (pred, errors в FP32)
    d_pred   = cu.gpu_zeros(N * 4)
    d_errors = cu.gpu_zeros(N * 4)
    d_buf    = cu.gpu_zeros(N * 4)

    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    loss_scale = 1024.0
    inv_scale  = 1.0 / loss_scale

    history = []

    for epoch in range(n_epochs):
        t0 = time.perf_counter()

        # Mixed forward + backward
        cu.mixed_forward_backward(d_X_fp16, d_w_fp16, d_y,
                                  d_pred, d_errors, d_grad_fp16,
                                  loss_scale, N, D)

        # Mixed Adam update
        cu.mixed_adam_step(d_w_master, d_w_fp16, d_grad_fp16,
                          d_m, d_v, lr, beta1, beta2, eps,
                          epoch + 1, inv_scale, D)

        cu.sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Пересчёт метрик с обновлёнными весами
        cu.mixed_forward_backward(d_X_fp16, d_w_fp16, d_y,
                                  d_pred, d_errors, d_grad_fp16,
                                  loss_scale, N, D)
        cu.sync()

        loss = cu.compute_loss(d_pred, d_y, d_buf, N)
        acc  = cu.compute_accuracy(d_pred, d_y, N)

        history.append({
            'epoch': epoch, 'loss': loss, 'acc': acc,
            'time_ms': elapsed_ms, 'stage': 'mixed_adam',
        })

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch:3d} | loss={loss:.6f} | "
                  f"acc={acc*100:6.2f}% | {elapsed_ms:7.2f} ms")

    # Очистка
    for ptr in [d_X_fp16, d_y, d_w_master, d_w_fp16, d_grad_fp16,
                d_m, d_v, d_pred, d_errors, d_buf]:
        cu.gpu_free(ptr)

    return history


# ============================================================
#  Вывод итоговой таблицы
# ============================================================
def print_summary(all_history):
    """Красивая таблица сравнения"""
    print("\n")
    print("╔══════════════════╦══════════╦══════════╦══════════╦══════════╗")
    print("║    Optimizer     ║  Loss    ║ Accuracy ║ Total ms ║  Avg ms  ║")
    print("╠══════════════════╬══════════╬══════════╬══════════╬══════════╣")

    summaries = []
    for name, hist in all_history.items():
        last = hist[-1]
        total_ms = sum(r['time_ms'] for r in hist)
        avg_ms   = total_ms / len(hist)
        summaries.append((name, last['loss'], last['acc'], total_ms, avg_ms))
        print(f"║ {name:16s} ║ {last['loss']:8.5f} ║  {last['acc']*100:5.2f}%  "
              f"║ {total_ms:7.1f}  ║ {avg_ms:7.2f}  ║")

    print("╚══════════════════╩══════════╩══════════╩══════════╩══════════╝")

    # Speedup
    if len(summaries) > 1:
        base_ms = summaries[0][3]
        print("\n  Speedup vs", summaries[0][0] + ":")
        for name, _, _, total, _ in summaries:
            if total > 0:
                print(f"    {name:20s} {base_ms / total:.2f}x")


# ============================================================
#  Сохранение в CSV
# ============================================================
def save_csv(all_history, filename="convergence.csv"):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stage', 'epoch', 'loss', 'accuracy', 'time_ms'])
        for name, hist in all_history.items():
            for r in hist:
                writer.writerow([name, r['epoch'], f"{r['loss']:.6f}",
                                f"{r['acc']:.4f}", f"{r['time_ms']:.2f}"])
    print(f"\n  Convergence data → {filename}")


# ============================================================
#  Визуализация
# ============================================================
def plot_results(all_history, output="convergence.png"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib не найден — пропускаю графики")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CUDA Optimizer Benchmark — Convergence Comparison',
                 fontsize=14, fontweight='bold')

    colors = {
        'adagrad':        '#3B82F6',
        'adam':            '#10B981',
        'sparse_adagrad': '#F59E0B',
        'mixed_adam':      '#8B5CF6',
    }
    labels = {
        'adagrad':        'Adagrad',
        'adam':            'Adam',
        'sparse_adagrad': 'Sparse Adagrad',
        'mixed_adam':      'Mixed Precision Adam',
    }

    for name, hist in all_history.items():
        epochs = [r['epoch'] for r in hist]
        losses = [r['loss'] for r in hist]
        accs   = [r['acc'] * 100 for r in hist]
        times  = [r['time_ms'] for r in hist]
        cum_t  = np.cumsum(times).tolist()

        c = colors.get(name, '#666')
        l = labels.get(name, name)

        axes[0, 0].plot(epochs, losses, color=c, label=l, linewidth=1.5)
        axes[0, 1].plot(epochs, accs,   color=c, label=l, linewidth=1.5)
        axes[1, 0].plot(epochs, times,  color=c, label=l, linewidth=1.5, alpha=0.7)
        axes[1, 1].plot(epochs, cum_t,  color=c, label=l, linewidth=1.5)

    for ax, title, ylabel in [
        (axes[0, 0], 'Training Loss (BCE)',         'Loss'),
        (axes[0, 1], 'Accuracy (%)',                'Accuracy %'),
        (axes[1, 0], 'Time per Epoch (ms)',         'ms'),
        (axes[1, 1], 'Cumulative Training Time',    'ms (total)'),
    ]:
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"  Графики → {output}")


# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CUDA Optimizer Benchmark')
    parser.add_argument('--optimizer', '-o',
                        choices=['adagrad', 'adam', 'sparse_adagrad', 'mixed_adam', 'all'],
                        default='all', help='Какой оптимизатор запустить')
    parser.add_argument('--epochs', '-e', type=int, default=N_EPOCHS)
    parser.add_argument('--samples', '-n', type=int, default=N_SAMPLES)
    parser.add_argument('--features', '-d', type=int, default=N_FEATURES)
    parser.add_argument('--no-plot', action='store_true', help='Без графиков')
    args = parser.parse_args()

    N, D = args.samples, args.features

    print("\n╔══════════════════════════════════════════════════════╗")
    print(f"║    CUDA Optimizer Benchmark                          ║")
    print(f"║    {N} samples × {D} features × {args.epochs} epochs           ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Генерируем данные ОДИН РАЗ — для честного сравнения
    print("\n  Генерация данных...")
    X_dense, y = generate_data(N, D, sparse=False, seed=42)
    X_sparse, y_sparse = generate_data(N, D, sparse=True, seed=42)

    all_history = {}
    runners = {
        'adagrad':        lambda: run_adagrad(X_dense, y, args.epochs),
        'adam':            lambda: run_adam(X_dense, y, args.epochs),
        'sparse_adagrad': lambda: run_sparse_adagrad(X_sparse, y_sparse, args.epochs),
        'mixed_adam':     lambda: run_mixed_adam(X_dense, y, args.epochs),
    }

    if args.optimizer == 'all':
        for name in ['adagrad', 'adam', 'sparse_adagrad', 'mixed_adam']:
            all_history[name] = runners[name]()
    else:
        all_history[args.optimizer] = runners[args.optimizer]()

    # Итоги
    print_summary(all_history)
    save_csv(all_history)

    if not args.no_plot:
        plot_results(all_history)


if __name__ == '__main__':
    main()
