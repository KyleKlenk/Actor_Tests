#include <cuda.h>
#include <cstdio>

extern "C" __global__
void delayKernel(int delay_seconds, int* out) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row == 0 && col == 0) {
        long long delay_cycles = 1500000000ULL * delay_seconds;
        printf("[GPU] Kernel started. Delaying for %lld cycles...\n", delay_cycles);
        long long start_clock = clock64();
        long long clock_offset = 0;

        while (clock_offset < delay_cycles) {
            clock_offset = clock64() - start_clock;
        }

        printf("[GPU] Kernel finished delay.\n");
        if (out != nullptr) {
            *out = 42;
        }
    }
}
