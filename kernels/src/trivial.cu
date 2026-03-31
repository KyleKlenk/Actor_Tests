#include <cuda.h>

extern "C" __global__
void trivial_kernel(const int* in_data, int* out_data) {
    out_data[0] = in_data[0];
}