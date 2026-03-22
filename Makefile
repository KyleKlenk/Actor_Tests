NVCC=/home/kklenk/miniforge3/envs/research-env/bin/nvcc -Wno-deprecated-gpu-targets
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

.PHONY: all clean

all: bin_1 bin_2 bin_3 bin_4 bin_5

matmul_1.cubin: ./example_1/matmul_kernel.cu
	@echo "[NVCC] Building example_1 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

matmul_2.cubin: ./example_2/matmul_kernel.cu
	@echo "[NVCC] Building example_2 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

matmul_3.cubin: ./example_3/matmul_kernel.cu
	@echo "[NVCC] Building example_3 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

matmul_4.cubin: ./example_4/matmul_kernel.cu
	@echo "[NVCC] Building example_4 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

matmul_5.cubin: ./example_5/matmul_kernel.cu
	@echo "[NVCC] Building example_5 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

bin_1: ./example_1/main.cpp matmul_1.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_2: ./example_2/main.cpp matmul_2.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_3: ./example_3/main.cpp matmul_3.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_4: ./example_4/main.cpp matmul_4.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_5: ./example_5/main.cpp matmul_5.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -f matmul_1.cubin matmul_2.cubin matmul_3.cubin matmul_4.cubin matmul_5.cubin
	rm -f bin_1 bin_2 bin_3 bin_4 bin_5