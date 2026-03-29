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

BIN_DIR:=bin
EXS:=1 2 3 4 5 6 7 8 9 10 11 12
MATMUL_EXS:=1 2 3 4 5 6 7
DELAY_EXS:=8 9 10 11 12
TRIVIAL_EXS:=7

.PHONY: all clean

BINS:=$(addprefix $(BIN_DIR)/bin_,$(EXS))

all: $(BINS)

# cubin pattern rules
matmul_%.cubin: ./example_%/matmul_kernel.cu
	@echo "[NVCC] Building example_$* matmul cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

trivial_%.cubin: ./example_%/trivial_kernel.cu
	@echo "[NVCC] Building example_$* trivial cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

delay_%.cubin: ./example_%/delay_kernel.cu
	@echo "[NVCC] Building example_$* delay cubin for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $< -o $@

# small macro to generate bin targets (creates $(BIN_DIR) when needed)
define BIN_RULE
$(BIN_DIR)/bin_$1: ./example_$1/main.cpp $2
	@mkdir -p $(BIN_DIR)
	$(CXX) -g $$< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $$@
endef

$(eval $(call BIN_RULE,1,matmul_1.cubin))
$(eval $(call BIN_RULE,2,matmul_2.cubin))
$(eval $(call BIN_RULE,3,matmul_3.cubin))
$(eval $(call BIN_RULE,4,matmul_4.cubin))
$(eval $(call BIN_RULE,5,matmul_5.cubin))
$(eval $(call BIN_RULE,6,matmul_6.cubin))
$(eval $(call BIN_RULE,7,matmul_7.cubin trivial_7.cubin))
$(eval $(call BIN_RULE,8,delay_8.cubin))
$(eval $(call BIN_RULE,9,delay_9.cubin))
$(eval $(call BIN_RULE,10,delay_10.cubin))
$(eval $(call BIN_RULE,11,delay_11.cubin))
$(eval $(call BIN_RULE,12,delay_12.cubin))

clean:
	rm -f matmul_*.cubin trivial_7.cubin delay_*.cubin
	rm -rf $(BIN_DIR)