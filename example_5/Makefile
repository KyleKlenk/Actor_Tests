# Makefile for example_3
NVCC=/home/kklenk/miniforge3/envs/research-env/bin/nvcc

LIBS=-L/home/kklenk/miniforge3/envs/research-env/lib -L/home/kklenk/.local/caf/lib -lcaf_core -lcaf_io -lcaf_net -lcaf_cuda -lcuda
INCLUDES=-I/home/kklenk/.local/caf/include

# Embed rpath so the binary finds caf-cuda and CUDA libs at runtime
LDFLAGS=-Wl,-rpath,/home/kklenk/.local/caf/lib -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib

# Auto-detect GPU compute capability (e.g. "5.2" -> "sm_52")
NVCC_ARCH := sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

all: cuda actors

cuda:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -arch=$(NVCC_ARCH) --cubin matmul_kernel.cu -o matmul.cubin

actors:
	g++ main.cpp -std=c++20 $(INCLUDES) $(LDFLAGS) $(LIBS) -o main

clean:
	rm -f main matmul.cubin

.PHONY: cuda actors
