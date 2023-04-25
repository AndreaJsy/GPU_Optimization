#include <stdio.h>
#define N 1024
#define BLOCK_SIZE 256

// PROBLEM: matrixMult_atomic matrixMult_atomic.cu
// ptxas error   : Entry function '_Z23matrix_multiply_atomicsPfS_S_' uses too much shared data (0x80000 bytes, 0xc000 max)
 
__global__ void matrix_multiply_atomics(float *a, float *b, float *c) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    // flattened index of the current element in the output matrix

    float sum = 0.0;

    for (int k = 0; k < N; k += BLOCK_SIZE) {
        __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];
        // shared memory array used to store a submatrix of a and b.

        for (int i = 0; i < BLOCK_SIZE; i++) {
            s_a[i][tid] = a[(k + i) * N + idx];
            s_b[i][tid] = b[(idx * N) + (k + i)];
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                sum += s_a[i][j] * s_b[j][i];
            }
        }

        __syncthreads();
    }

    atomicAdd(&c[idx], sum);
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*) malloc(N * N * sizeof(float));
    b = (float*) malloc(N * N * sizeof(float));
    c = (float*) calloc(N * N, sizeof(float));

    cudaMalloc((void**) &d_a, N * N * sizeof(float));
    cudaMalloc((void**) &d_b, N * N * sizeof(float));
    cudaMalloc((void**) &d_c, N * N * sizeof(float));

    // Initialize matrices a and b with some values

    cudaMemcpy(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(N / BLOCK_SIZE, 1, 1);

    matrix_multiply_atomics<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrix c

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
