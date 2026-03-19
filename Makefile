NVCC       = nvcc
NVCC_FLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr
ARCH       ?= 70
GPU_ARCH   = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

COMMON_OBJS = stage1_adagrad.o stage2_adam.o stage3_sparse_adagrad.o stage4_mixed_adam.o

all: benchmark stage1_adagrad stage2_adam stage3_sparse_adagrad stage4_mixed_adam

benchmark: benchmark.cu $(COMMON_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ benchmark.cu $(COMMON_OBJS)

stage1_adagrad: stage1_adagrad.cu
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE1_STANDALONE -o $@ stage1_adagrad.cu

stage2_adam: stage2_adam.cu
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE2_STANDALONE -o $@ stage2_adam.cu

stage3_sparse_adagrad: stage3_sparse_adagrad.cu
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE3_STANDALONE -o $@ stage3_sparse_adagrad.cu

stage4_mixed_adam: stage4_mixed_adam.cu
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE4_STANDALONE -o $@ stage4_mixed_adam.cu

stage1_adagrad.o: stage1_adagrad.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -dc -o $@ stage1_adagrad.cu

stage2_adam.o: stage2_adam.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -dc -o $@ stage2_adam.cu

stage3_sparse_adagrad.o: stage3_sparse_adagrad.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -dc -o $@ stage3_sparse_adagrad.cu

stage4_mixed_adam.o: stage4_mixed_adam.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -dc -o $@ stage4_mixed_adam.cu

clean:
	rm -f *.o benchmark stage1_adagrad stage2_adam stage3_sparse_adagrad stage4_mixed_adam convergence.csv
