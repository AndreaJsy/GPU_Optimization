#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 16
#define NUM_STREAMS 4
 
__global__ void matrix_multiply(float *a, float *b, float *c, int start_idx, int end_idx) {
    for (int idx = start_idx; idx < end_idx; idx++) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + idx];
            }
            c[i * N + idx] = sum;
        }
    }
}

void matrix_multiply_task(float *a, float *b, float *c, int start_idx, int end_idx, cudaStream_t stream) {
    float *d_a, *d_b, *d_c;
    int size = N * (end_idx - start_idx) * sizeof(float);

    cudaMalloc((void**) &d_a, N * N * sizeof(float));
    cudaMalloc((void**) &d_b, N * (end_idx - start_idx) * sizeof(float));
    cudaMalloc((void**) &d_c, N * (end_idx - start_idx) * sizeof(float));

    cudaMemcpyAsync(d_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b + start_idx * N, size, cudaMemcpyHostToDevice, stream);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (end_idx - start_idx + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiply<<<dimGrid, dimBlock, 0, stream>>>(d_a, d_b, d_c, start_idx, end_idx);

    cudaMemcpyAsync(c + start_idx * N, d_c, size, cudaMemcpyDeviceToHost, stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    float *a, *b, *c;
    cudaStream_t streams[NUM_STREAMS];

    a = (float*) malloc(N * N * sizeof(float));
    b = (float*) malloc(N * N * sizeof(float));
    c = (float*) calloc(N * N, sizeof(float));

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Initialize matrices a and b with some values

    int chunk_size = N / NUM_STREAMS;

    for (int i = 0; i < NUM_STREAMS; i++) {
        int start_idx = i * chunk_size;
        int end_idx = (i + 1) * chunk_size;
        if (i == NUM_STREAMS - 1) {
            end_idx = N;
        }
        matrix_multiply_task(a, b, c, start_idx, end_idx, streams[i]);
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Print matrix c

    free(a);
    free(b);
    free(c);

    return 0;
}
