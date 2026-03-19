# CUDA Optimizer Project

Серия GPU-оптимизаторов для градиентного спуска — от Adagrad до Mixed Precision Adam.
Каждый этап демонстрирует конкретные CUDA-техники с измеримым приростом производительности.

## Архитектура проекта

```
kernels.cu          — ВСЕ CUDA-ядра (компилируется в libkernels.so)
cuda_ops.py         — Python-обёртка (ctypes) для вызова ядер
benchmark.py        — Генерация данных, обучение, метрики, графики
Makefile            — Сборка и запуск
```

**Принцип разделения:**
- **CUDA (kernels.cu)** — только GPU-вычисления: forward pass, backward pass, оптимизаторы
- **Python (benchmark.py)** — всё остальное: данные (NumPy), цикл обучения, метрики, визуализация

Это упрощает отладку: данные генерируются в NumPy (можно проверить глазами),
а CUDA отвечает только за то, что должно быть быстрым.

## Быстрый старт

```bash
# Указать compute capability вашей GPU
# 61=GTX 1060/1070/1080, 70=V100, 75=RTX 2xxx, 80=A100, 86=RTX 30xx, 89=RTX 40xx
make bench ARCH=86

# Только один оптимизатор
make bench-adam ARCH=86

# С кастомными параметрами
python3 benchmark.py --epochs 100 --samples 50000 --features 500
```

## Бенчмарк

Единая задача для честного сравнения:
- **Логистическая регрессия** на синтетическом датасете
- 100 000 сэмплов × 1 000 фичей
- Binary Cross-Entropy loss
- 50 эпох полного batch gradient descent

Данные генерируются в NumPy один раз и передаются всем оптимизаторам.

## Этапы

### Этап 1 — Adagrad + Memory Coalescing

| Техника | Описание |
|---------|----------|
| SoA layout | Отдельные массивы w[], grad[], G² — вместо AoS |
| Coalesced access | Потоки warp'а читают соседние float'ы → одна транзакция |

**Формула:** `G²[j] += grad[j]²; w[j] -= lr * grad[j] / (√G²[j] + ε)`

### Этап 2 — Adam

| Техника | Описание |
|---------|----------|
| Momentum (m) | Экспоненциальное сглаживание градиента |
| Adaptive LR (v) | Не растёт бесконечно как G² в Adagrad |
| Bias correction | Компенсация нулевой инициализации m и v |

**Формула:** `m = β₁·m + (1-β₁)·g; v = β₂·v + (1-β₂)·g²; w -= lr·(m̂)/(√v̂ + ε)`

### Этап 3 — Sparse Adagrad

| Техника | Описание |
|---------|----------|
| Sparse gradient | COO-представление (index, value) для ненулевых элементов |
| atomicAdd | Безопасная параллельная запись в счётчик nnz |
| Selective update | Обновляем только K весов из D (K << D) |

### Этап 4 — Mixed Precision Adam

| Техника | Описание |
|---------|----------|
| FP16 weights | 2x экономия памяти для данных и рабочих весов |
| FP32 master | Точное накопление обновлений |
| half2 intrinsics | 2 операции за 1 инструкцию |
| Loss scaling | Предотвращает underflow мелких FP16-градиентов |

## Ожидаемые результаты

| Метрика | Adagrad | Adam | Sparse Adagrad | Mixed Adam |
|---------|---------|------|----------------|------------|
| Сходимость | Медленная | Быстрая | Как Adagrad | Как Adam |
| Optimizer memory | 1× (G²) | 2× (m + v) | 1× (G²) | 2× + master |
| Data memory | baseline | baseline | baseline | ~0.5× (FP16) |

## Требования

- CUDA Toolkit >= 11.0
- GPU compute capability >= 5.3 (для FP16), рекомендуется >= 7.0
- Python 3.8+ с NumPy
- matplotlib (опционально, для графиков)
- ~500 MB GPU RAM
