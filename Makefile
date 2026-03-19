# ============================================================
#  CUDA Optimizer Project — Makefile
#
#  Основная цель — shared library libkernels.so,
#  которую вызывает Python-обёртка benchmark.py
#
#  Использование:
#    make lib                — собрать libkernels.so
#    make bench              — собрать + запустить бенчмарк
#    make bench ARCH=86      — для RTX 30xx
#    make clean              — удалить артефакты
# ============================================================

NVCC       = nvcc
NVCC_FLAGS = -O2 -std=c++17 --expt-relaxed-constexpr

# Compute capability (по умолчанию 70 = Volta/V100):
#   61 = GTX 1060/1070/1080
#   70 = V100
#   75 = RTX 2xxx
#   80 = A100
#   86 = RTX 30xx
#   89 = RTX 40xx
#   90 = H100
ARCH ?= 70
GPU_ARCH = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

.PHONY: lib bench plot clean

# Собрать shared library
lib: libkernels.so

libkernels.so: kernels.cu
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) \
		--compiler-options '-fPIC' \
		-shared -o $@ $<

# Бенчмарк: все 4 оптимизатора
bench: lib
	python3 benchmark.py

# Один оптимизатор
bench-adagrad: lib
	python3 benchmark.py -o adagrad

bench-adam: lib
	python3 benchmark.py -o adam

bench-sparse: lib
	python3 benchmark.py -o sparse_adagrad

bench-mixed: lib
	python3 benchmark.py -o mixed_adam

# Бенчмарк + графики
plot: lib
	python3 benchmark.py

clean:
	rm -f libkernels.so convergence.csv convergence.png
