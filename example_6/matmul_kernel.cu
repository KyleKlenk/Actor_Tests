// Noiseless matrix multiplication kernel (no printf).
// Used by test 6 (multi-GPU) and test 7 (overhead scaling) so that actor
// output is uncluttered and measurements are accurate.

#include <cuda.h>

extern "C" __global__
void matrixMul(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
