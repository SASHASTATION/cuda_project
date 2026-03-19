# ============================================================
#  CUDA Optimizer Project — Makefile
#
#  Сборка:
#    make all       — собрать все 4 этапа
#    make stage1    — только Adagrad
#    make run       — собрать и запустить всё последовательно
#    make clean     — удалить бинарники
# ============================================================

NVCC      = nvcc
NVCC_FLAGS = -O2 -std=c++17 --expt-relaxed-constexpr

# Compute capability:
#   5.3  — минимум для FP16
#   7.0  — Volta (tensor cores, half2 math)
#   8.0  — Ampere
#   8.6  — RTX 30xx
#   8.9  — RTX 40xx
#   9.0  — H100
# Укажите свою архитектуру ниже:
ARCH ?= 70
GPU_ARCH = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

TARGETS = stage1_adagrad stage2_adam stage3_sparse_adagrad stage4_mixed_adam benchmark

.PHONY: all clean run bench plot

all: $(TARGETS)

stage1_adagrad: stage1_adagrad.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ $<

stage2_adam: stage2_adam.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ $<

stage3_sparse_adagrad: stage3_sparse_adagrad.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ $<

stage4_mixed_adam: stage4_mixed_adam.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ $<

benchmark: benchmark.cu common.cuh
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ $<

# Запуск всех этапов последовательно
run: all
	@echo ""
	@echo "╔══════════════════════════════════════════════╗"
	@echo "║   CUDA Optimizer Benchmark — Full Suite      ║"
	@echo "╚══════════════════════════════════════════════╝"
	@echo ""
	./stage1_adagrad
	./stage2_adam
	./stage3_sparse_adagrad
	./stage4_mixed_adam
	@echo ""
	@echo "Все этапы завершены."

clean:
	rm -f $(TARGETS) convergence.csv summary.csv convergence.png

# Единый бенчмарк: все 4 оптимизатора + сравнительная таблица
bench: benchmark
	./benchmark

# Визуализация: бенчмарк + графики
plot: bench
	python3 plot_convergence.py

# Профилирование через Nsight Compute (stage 1 как baseline)
profile_stage1: stage1_adagrad
	ncu --set full -o profile_adagrad ./stage1_adagrad

profile_stage4: stage4_mixed_adam
	ncu --set full -o profile_mixed_adam ./stage4_mixed_adam
