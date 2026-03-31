# Actor_Tests

This directory is organized around three roles:

- `example_N/`: one self-contained CAF/CUDA example per folder, with only the C++ driver code for that example.
- `kernels/src/`: shared CUDA kernel sources, deduplicated across examples.
- `kernels/cubin/`: generated CUBIN files produced by the top-level Makefile.

The top-level Makefile builds only what each example needs:

- `make example_1`
- `make example_7`
- `make example_14`
- `make cubins`
- `make clean`

If GPU architecture auto-detection fails on your machine, pass it explicitly, for example:

- `make NVCC_ARCH=sm_86 example_1`