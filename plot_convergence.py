#!/usr/bin/env python3
"""
Визуализация кривых сходимости из convergence.csv

Запуск: python3 plot_convergence.py
Вход:   convergence.csv (генерируется benchmark)
Выход:  convergence.png — 3 графика (loss, accuracy, time, cumulative time)
"""

import csv
import sys

# ============================================================
#  Попробуем matplotlib; если нет — выведем ASCII-таблицу
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')  # без GUI
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib не найден — выведу текстовый отчёт")

def load_csv(filename):
    """Загрузить convergence.csv → dict[stage] → list of (epoch, loss, acc, ms)"""
    data = {}
    try:
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stage = row['stage']
                if stage not in data:
                    data[stage] = []
                data[stage].append({
                    'epoch': int(row['epoch']),
                    'loss':  float(row['loss']),
                    'acc':   float(row['accuracy']),
                    'ms':    float(row['time_ms']),
                })
    except FileNotFoundError:
        print(f"Файл {filename} не найден. Запустите ./benchmark сначала.")
        sys.exit(1)
    return data

def plot_convergence(data, output='convergence.png'):
    """3 subplot'а: loss, accuracy, time per epoch, cumulative time"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CUDA Optimizer Benchmark — Convergence Comparison',
                 fontsize=14, fontweight='bold')

    colors = {
        'adagrad':        '#3B82F6',
        'adam':            '#10B981',
        'sparse_adagrad': '#F59E0B',

    }
    labels = {
        'adagrad':        'Adagrad',
        'adam':            'Adam',
        'sparse_adagrad': 'Sparse Adagrad',
 
    }

    for stage, records in data.items():
        epochs = [r['epoch'] for r in records]
        losses = [r['loss'] for r in records]
        accs   = [r['acc'] * 100 for r in records]
        times  = [r['ms'] for r in records]
        cum_t  = []
        s = 0
        for t in times:
            s += t
            cum_t.append(s)

        c = colors.get(stage, '#666')
        l = labels.get(stage, stage)

        # loss
        axes[0, 0].plot(epochs, losses, color=c, label=l, linewidth=1.5)
        # accuracy
        axes[0, 1].plot(epochs, accs, color=c, label=l, linewidth=1.5)
        # time per epoch
        axes[1, 0].plot(epochs, times, color=c, label=l, linewidth=1.5, alpha=0.7)
        # cumulative time
        axes[1, 1].plot(epochs, cum_t, color=c, label=l, linewidth=1.5)

    axes[0, 0].set_title('Training Loss (BCE)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Accuracy (%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy %')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Time per Epoch (ms)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('ms')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Cumulative Training Time (ms)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ms (total)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Графики сохранены → {output}")

def print_ascii(data):
    """Fallback: ASCII-таблица финальных метрик"""
    print("\n  Stage             | Final Loss | Final Acc  | Total Time")
    print("  ------------------|------------|------------|----------")
    for stage, records in data.items():
        last = records[-1]
        total = sum(r['ms'] for r in records)
        print(f"  {stage:18s}| {last['loss']:10.6f} | {last['acc']*100:8.2f}%  | {total:8.1f} ms")

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else 'convergence.csv'
    data = load_csv(filename)

    if HAS_MPL:
        plot_convergence(data)
    else:
        print_ascii(data)
