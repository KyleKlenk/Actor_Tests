// monte_carlo_kernel.cu
//
// Monte Carlo estimation of π via unit-circle hit counting.
//
// Each thread generates a strided slice of `total_samples` random points
// (x, y) ∈ [0,1]².  A hit is counted when x²+y² ≤ 1.  Local hit counts
// are reduced within each warp using __shfl_down_sync, then one lane per
// warp atomically adds to the global output counter.
//
// Build:
//   nvcc -arch=<sm_xx> --cubin monte_carlo_kernel.cu -o monte_carlo_14.cubin
//
// Arguments:
//   seed          (int, in)  — PRNG seed; use distinct values per batch
//   total_samples (int, in)  — total number of (x,y) pairs to generate
//   out_count     (int, out) — accumulates number of points inside the circle

#include <cuda.h>
#include <curand_kernel.h>

extern "C" __global__
void monteCarloKernel(int seed, int total_samples, int* out_count) {
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = gridDim.x  * blockDim.x;

    // Initialize per-thread PRNG state.
    curandState_t rng;
    curand_init(seed, tid, 0, &rng);

    // Strided loop: each thread takes every n_threads-th sample.
    long long local_count = 0;
    for (int i = tid; i < total_samples; i += n_threads) {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        if (x * x + y * y <= 1.0f)
            local_count++;
    }

    // Warp-level reduction (32-wide).
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    // Lane 0 of each warp contributes to the global counter.
    if ((threadIdx.x & 31) == 0)
        atomicAdd(out_count, (int)local_count);
}
