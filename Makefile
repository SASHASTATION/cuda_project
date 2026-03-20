NVCC       = nvcc
NVCC_FLAGS ?= -O2 -std=c++17 --expt-relaxed-constexpr -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0
ARCH       ?= 70
GPU_ARCH = -gencode arch=compute_$(ARCH),code=sm_$(ARCH) \
           -gencode arch=compute_$(ARCH),code=compute_$(ARCH)

all: benchmark stage1_adagrad stage2_adam stage3_sparse_adagrad 

benchmark: benchmark.cu stage1_adagrad.cu stage2_adam.cu stage3_sparse_adagrad.cu  common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -o $@ benchmark.cu

stage1_adagrad: stage1_adagrad.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE1_STANDALONE -o $@ stage1_adagrad.cu

stage2_adam: stage2_adam.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE2_STANDALONE -o $@ stage2_adam.cu

stage3_sparse_adagrad: stage3_sparse_adagrad.cu common.cuh stages.h config.h
	$(NVCC) $(NVCC_FLAGS) $(GPU_ARCH) -DSTAGE3_STANDALONE -o $@ stage3_sparse_adagrad.cu


clean:
	rm -f *.o benchmark stage1_adagrad stage2_adam stage3_sparse_adagrad  convergence.csv
