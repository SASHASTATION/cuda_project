# CUDA Optimizer Project

Серия GPU-оптимизаторов для градиентного спуска — от Adagrad до Mixed Precision Adam.
Каждый этап демонстрирует конкретные CUDA-техники с измеримым приростом производительности.

## Бенчмарк

Единая задача на всех этапах для честного сравнения:
- **Логистическая регрессия** на синтетическом датасете
- 100 000 сэмплов × 1 000 фичей
- Binary Cross-Entropy loss
- 50 эпох полного batch gradient descent

## Структура проекта

```
common.cuh                  — общие ядра: генерация данных, forward/backward, таймер
stage1_adagrad.cu           — Adagrad + SoA layout + coalesced access
stage2_adam.cu              — Adam + bias correction + сравнение с Adagrad
stage3_sparse_adagrad.cu    — Sparse Adagrad + atomicAdd + sparse vs dense
stage4_mixed_adam.cu        — Mixed Precision Adam (FP16 + FP32) + loss scaling
Makefile                    — сборка и запуск
```

## Сборка

```bash
# Указать compute capability вашей GPU (по умолчанию 70 = Volta)
# 70=V100, 75=RTX 2xxx, 80=A100, 86=RTX 30xx, 89=RTX 40xx, 90=H100
make all ARCH=86

# Запуск всех этапов
make run

# Только один этап
make stage1_adagrad && ./stage1_adagrad
```

## Этапы

### Этап 1 — Adagrad + Memory Coalescing

| Техника | Описание |
|---------|----------|
| SoA layout | Отдельные массивы w[], grad[], G² — вместо AoS |
| Coalesced access | Потоки warp'а читают соседние float'ы → одна транзакция |
| Базовый профиль | Точка отсчёта для Nsight Compute |

### Этап 2 — Adam

| Техника | Описание |
|---------|----------|
| Momentum (m) | Экспоненциальное сглаживание градиента — фильтрует шум |
| Adaptive LR (v) | Не растёт бесконечно как G² в Adagrad |
| Bias correction | Компенсация нулевой инициализации m и v |

### Этап 3 — Sparse Adagrad

| Техника | Описание |
|---------|----------|
| Sparse gradient | COO-представление (index, value) для ненулевых элементов |
| atomicAdd | Безопасная параллельная запись в общий массив |
| Selective update | Обновляем только K весов из D (K << D) |

### Этап 4 — Mixed Precision Adam

| Техника | Описание |
|---------|----------|
| FP16 weights | 2x экономия памяти для данных и рабочих весов |
| FP32 master | Точное накопление обновлений |
| half2 intrinsics | 2 операции за 1 инструкцию |
| Loss scaling | Предотвращает underflow мелких FP16-градиентов |

## Профилирование

```bash
# Nsight Compute — детальный анализ ядер
make profile_stage1
ncu-ui profile_adagrad.ncu-rep

# Nsight Systems — timeline всего приложения
nsys profile --stats=true ./stage1_adagrad
```

## Ожидаемые результаты

| Метрика | Adagrad | Adam | Sparse Adagrad | Mixed Adam |
|---------|---------|------|----------------|------------|
| Сходимость | Медленная | Быстрая | Как Adagrad | Как Adam |
| Optimizer memory | 1× (G²) | 2× (m + v) | 1× (G²) | 2× + master |
| Data memory | baseline | baseline | baseline | ~0.5× |
| Update kernel | baseline | ~1× | < 1× (sparse) | ~0.7× (half2) |

## Требования

- CUDA Toolkit >= 11.0
- GPU compute capability >= 5.3 (для FP16), рекомендуется >= 7.0
- ~500 MB GPU RAM
