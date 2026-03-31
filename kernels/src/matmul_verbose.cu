#include <cuda.h>
#include <cstdio>

extern "C" __global__
void matrixMul(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int tempVal = 0;
        for (int k = 0; k < N; ++k) {
            tempVal += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = tempVal;

        printf("[GPU] Thread(%d,%d) computed result: %d\n", row, col, tempVal);
    }
}