// Minimal kernel: single-thread element copy.
// Used by test 7 to measure pure dispatch/framework overhead with
// negligible GPU compute time (1 thread, 1 read, 1 write).

#include <cuda.h>

extern "C" __global__
void trivial_kernel(const int* in_data, int* out_data) {
    out_data[0] = in_data[0];
}
