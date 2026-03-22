NVCC=/home/kklenk/miniforge3/envs/research-env/bin/nvcc
CXX=/home/kklenk/miniforge3/envs/research-env/bin/g++
CXXFLAGS=-std=c++20 -include cstddef -include cerrno -include climits

LIBS=-L/home/kklenk/miniforge3/envs/research-env/lib \
		 -L/home/kklenk/.local/caf/lib \
		 -lcaf_core -lcaf_io -lcaf_net -lcaf_cuda -lcuda

CUDA_INCLUDES=-I/home/kklenk/miniforge3/envs/research-env/targets/x86_64-linux/include
INCLUDES=-I/home/kklenk/.local/caf/include \
		 -I/home/kklenk/miniforge3/envs/research-env/include \
		 $(CUDA_INCLUDES)

LDFLAGS=-Wl,-rpath,/home/kklenk/.local/caf/lib \
		 -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib
NVCC_ARCH := sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

.PHONY: all example_1 example_2 example_3 example_4 example_5 clean

all: example_1 example_2 example_3 example_4 example_5

example_1:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin ./example_1/matmul_kernel.cu -o matmul.cubin
	$(CXX) -g ./example_1/main.cpp $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o bin_1

example_2:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin ./example_2/matmul_kernel.cu -o matmul.cubin
	$(CXX) -g ./example_2/main.cpp $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o bin_2

example_3:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin ./example_3/matmul_kernel.cu -o matmul.cubin
	$(CXX) -g ./example_3/main.cpp $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o bin_3

example_4:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin ./example_4/matmul_kernel.cu -o matmul.cubin
	$(CXX) -g ./example_4/main.cpp $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o bin_4

example_5:
	@echo "[NVCC] Building for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin ./example_5/matmul_kernel.cu -o matmul.cubin
	$(CXX) -g ./example_5/main.cpp $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o bin_5	
	

clean:
	rm -f matmul.cubin
	rm -f bin_1 bin_2 2>/dev/null || true