#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 16
 
__global__ void matrix_multiply(float *a, float *b, float *c) {
    int i = blockIdx.x * BLOCK_SIZE;
    int j = blockIdx.y * BLOCK_SIZE;
    int k = threadIdx.x;

    __shared__ float s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_b[BLOCK_SIZE][BLOCK_SIZE];

    float temp = 0.0;

    for (int jj = 0; jj < BLOCK_SIZE; jj++) {
        s_a[k][jj] = a[(i + k) * N + (j + jj)];
        s_b[k][jj] = b[(j + jj) * N + (i + k)];
    }

    __syncthreads();

    for (int kk = 0; kk < BLOCK_SIZE; kk++) {
        for (int jj = 0; jj < BLOCK_SIZE; jj++) {
            temp += s_a[k][kk] * s_b[jj][kk];
        }
    }

    c[(i + k) * N + j] += temp;
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
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE, 1);

    matrix_multiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

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
