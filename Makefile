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

all: bin_1 bin_2 bin_3 bin_4 bin_5 bin_6 bin_7 bin_8 bin_9 bin_10 bin_11

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

matmul_6.cubin: ./example_6/matmul_kernel.cu
	@echo "[NVCC] Building example_6 cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

trivial_7.cubin: ./example_7/trivial_kernel.cu
	@echo "[NVCC] Building example_7 trivial cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

matmul_7.cubin: ./example_7/matmul_kernel.cu
	@echo "[NVCC] Building example_7 matmul cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

delay_8.cubin: ./example_8/delay_kernel.cu
	@echo "[NVCC] Building example_8 delay cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

delay_9.cubin: ./example_9/delay_kernel.cu
	@echo "[NVCC] Building example_9 delay cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

delay_10.cubin: ./example_10/delay_kernel.cu
	@echo "[NVCC] Building example_10 delay cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

delay_11.cubin: ./example_11/delay_kernel.cu
	@echo "[NVCC] Building example_11 delay cubin for $(NVCC_ARCH)"
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

bin_6: ./example_6/main.cpp matmul_6.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_7: ./example_7/main.cpp trivial_7.cubin matmul_7.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_8: ./example_8/main.cpp delay_8.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_9: ./example_9/main.cpp delay_9.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_10: ./example_10/main.cpp delay_10.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

bin_11: ./example_11/main.cpp delay_11.cubin
	$(CXX) -g $< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -f matmul_1.cubin matmul_2.cubin matmul_3.cubin matmul_4.cubin matmul_5.cubin
	rm -f matmul_6.cubin trivial_7.cubin matmul_7.cubin delay_8.cubin delay_9.cubin delay_10.cubin delay_11.cubin
	rm -f bin_1 bin_2 bin_3 bin_4 bin_5 bin_6 bin_7 bin_8 bin_9 bin_10 bin_11