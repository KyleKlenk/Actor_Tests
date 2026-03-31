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
NVCC_ARCH ?= sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')

ifeq ($(strip $(NVCC_ARCH)),sm_)
$(error Unable to detect GPU compute capability. Run make with NVCC_ARCH=sm_XX, for example: make NVCC_ARCH=sm_86 example_1)
endif

BIN_DIR:=bin
KERNEL_SRC_DIR:=kernels/src
CUBIN_DIR:=kernels/cubin
COMMON_HEADERS:=common/kernel_paths.hpp

EXS:=1 2 3 4 5 6 7 8 9 10 11 12 13 14
EXAMPLE_TARGETS:=$(addprefix example_,$(EXS))
LEGACY_BIN_TARGETS:=$(addprefix bin_,$(EXS))

MATMUL_VERBOSE_CUBIN:=$(CUBIN_DIR)/matmul_verbose.cubin
MATMUL_QUIET_CUBIN:=$(CUBIN_DIR)/matmul_quiet.cubin
TRIVIAL_CUBIN:=$(CUBIN_DIR)/trivial.cubin
DELAY_CUBIN:=$(CUBIN_DIR)/delay.cubin
MONTE_CARLO_CUBIN:=$(CUBIN_DIR)/monte_carlo.cubin
ALL_CUBINS:=$(MATMUL_VERBOSE_CUBIN) $(MATMUL_QUIET_CUBIN) $(TRIVIAL_CUBIN) $(DELAY_CUBIN) $(MONTE_CARLO_CUBIN)

.PHONY: all clean cubins $(EXAMPLE_TARGETS) $(LEGACY_BIN_TARGETS)

all: $(EXAMPLE_TARGETS)

cubins: $(ALL_CUBINS)

$(BIN_DIR) $(CUBIN_DIR):
	@mkdir -p $@

$(MATMUL_VERBOSE_CUBIN): $(KERNEL_SRC_DIR)/matmul_verbose.cu | $(CUBIN_DIR)
	@echo "[NVCC] Building $@ for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $(CUDA_INCLUDES) $< -o $@

$(MATMUL_QUIET_CUBIN): $(KERNEL_SRC_DIR)/matmul_quiet.cu | $(CUBIN_DIR)
	@echo "[NVCC] Building $@ for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $(CUDA_INCLUDES) $< -o $@

$(TRIVIAL_CUBIN): $(KERNEL_SRC_DIR)/trivial.cu | $(CUBIN_DIR)
	@echo "[NVCC] Building $@ for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $(CUDA_INCLUDES) $< -o $@

$(DELAY_CUBIN): $(KERNEL_SRC_DIR)/delay.cu | $(CUBIN_DIR)
	@echo "[NVCC] Building $@ for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $(CUDA_INCLUDES) $< -o $@

$(MONTE_CARLO_CUBIN): $(KERNEL_SRC_DIR)/monte_carlo.cu | $(CUBIN_DIR)
	@echo "[NVCC] Building $@ for $(NVCC_ARCH)"
	$(NVCC) -g -arch=$(NVCC_ARCH) --cubin $(CUDA_INCLUDES) $< -o $@

define EXAMPLE_RULE
example_$1: $(BIN_DIR)/bin_$1

bin_$1: $(BIN_DIR)/bin_$1

$(BIN_DIR)/bin_$1: ./example_$1/main.cpp $(COMMON_HEADERS) $2 | $(BIN_DIR)
	@echo "[CXX] Building $$@"
	$(CXX) -g $$< $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $(LIBS) -o $$@
endef

$(eval $(call EXAMPLE_RULE,1,$(MATMUL_VERBOSE_CUBIN)))
$(eval $(call EXAMPLE_RULE,2,$(MATMUL_VERBOSE_CUBIN)))
$(eval $(call EXAMPLE_RULE,3,$(MATMUL_VERBOSE_CUBIN)))
$(eval $(call EXAMPLE_RULE,4,$(MATMUL_VERBOSE_CUBIN)))
$(eval $(call EXAMPLE_RULE,5,$(MATMUL_VERBOSE_CUBIN)))
$(eval $(call EXAMPLE_RULE,6,$(MATMUL_QUIET_CUBIN)))
$(eval $(call EXAMPLE_RULE,7,$(MATMUL_QUIET_CUBIN) $(TRIVIAL_CUBIN)))
$(eval $(call EXAMPLE_RULE,8,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,9,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,10,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,11,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,12,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,13,$(DELAY_CUBIN)))
$(eval $(call EXAMPLE_RULE,14,$(MONTE_CARLO_CUBIN)))

clean:
	rm -f ./*.cubin
	rm -rf $(BIN_DIR) $(CUBIN_DIR)