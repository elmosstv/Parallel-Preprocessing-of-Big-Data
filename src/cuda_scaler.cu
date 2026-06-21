#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

struct timespec ts_start, ts_end;
double time_spent;

void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}


// Kernel: Calculate partial stats for the current block
// Grid: 1D blocks of threads. Thread 'j' handles column 'j'.
__global__ void block_stats_kernel(const double* __restrict__ block, 
                                   double* d_sum, double* d_sum_sq, 
                                   double* d_min, double* d_max, 
                                   int rows, int D) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < D) {
        double sum = 0.0;
        double sum_sq = 0.0;
        double c_min = INFINITY;
        double c_max = -INFINITY;

        for (int r = 0; r < rows; r++) {
            double val = block[r * D + col];
            sum += val;
            sum_sq += val * val;
            if (val < c_min) c_min = val;
            if (val > c_max) c_max = val;
        }

        d_sum[col] = sum;
        d_sum_sq[col] = sum_sq;
        d_min[col] = c_min;
        d_max[col] = c_max;
    }
}

// Kernel: StandardScaler (Embarrassingly Parallel)
// 1 Thread = 1 Matrix Element
__global__ void standard_scale_kernel(double* block, 
                                      const double* __restrict__ mean, 
                                      const double* __restrict__ inv_std, 
                                      long total_elements, int D) 
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        int col = i % D;
        block[i] = (block[i] - mean[col]) * inv_std[col];
    }
}

// Kernel: MinMaxScaler (Embarrassingly Parallel)
// 1 Thread = 1 Matrix Element
__global__ void minmax_scale_kernel(double* block, 
                                    const double* __restrict__ scaled_min, 
                                    const double* __restrict__ inv_range, 
                                    long total_elements, int D) 
{
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elements) {
        int col = i % D;
        block[i] = block[i] * inv_range[col] - scaled_min[col];
    }
}


// ---------------------------------------------------------
// HOST FUNCTIONS (CPU)
// ---------------------------------------------------------

int main(int argc, char *argv[])
{
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <N> <D> <mode> [block_rows]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    long        N           = atol(argv[3]);
    int         D           = atoi(argv[4]);
    const char *mode        = argv[5];
    long        block_rows  = (argc == 7) ? atol(argv[6]) : 100000;

    if (N <= 0 || D <= 0 || block_rows <= 0) die("N, D, and block_rows must be positive");
    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0) die("mode must be 'standard' or 'minmax'");

    printf("Input  : %s\n", input_file);
    printf("Output : %s\n", output_file);
    printf("N=%ld  D=%d  mode=%s  block_rows=%ld\n", N, D, mode, block_rows);

    FILE *fin  = fopen(input_file, "rb");
    if (!fin)  die("Failed to open input file");
    FILE *fout = fopen(output_file, "wb");
    if (!fout) die("Failed to open output file");

    size_t block_bytes = (size_t)block_rows * D * sizeof(double);
    size_t d_bytes = D * sizeof(double);

    // 1. PINNED HOST MEMORY ALLOCATION 
    double *h_block;
    CHECK_CUDA(cudaMallocHost((void**)&h_block, block_bytes));
    
    double *mean = (double*)malloc(d_bytes);
    double *min  = (double*)malloc(d_bytes);
    double *max  = (double*)malloc(d_bytes);
    double *std  = (double*)malloc(d_bytes);
    double *sum_sq = (double*)malloc(d_bytes);

    for (int j = 0; j < D; j++) {
        mean[j] = 0.0; sum_sq[j] = 0.0;
        min[j] = INFINITY; max[j] = -INFINITY;
    }

    // 2. DEVICE MEMORY ALLOCATION
    double *d_block, *d_sum, *d_sum_sq, *d_min, *d_max;
    CHECK_CUDA(cudaMalloc((void**)&d_block, block_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_sum, d_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_sum_sq, d_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_min, d_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_max, d_bytes));

    double *h_block_sum = (double*)malloc(d_bytes);
    double *h_block_sum_sq = (double*)malloc(d_bytes);
    double *h_block_min = (double*)malloc(d_bytes);
    double *h_block_max = (double*)malloc(d_bytes);

    //  COMPUTE STATISTICS
    printf("\n[Phase 1] Computing statistics via GPU...\n");
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    long rows_left = N;
    int grid_D = (D + 255) / 256; 
    dim3 threads_D(256);

    while (rows_left > 0) {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        size_t current_block_bytes = (size_t)rows_read * D * sizeof(double);

        // Read from disk to pinned memory
        if (fread(h_block, sizeof(double), rows_read * D, fin) != (size_t)(rows_read * D)) die("EOF in Phase 1");

        // Copy to GPU
        CHECK_CUDA(cudaMemcpy(d_block, h_block, current_block_bytes, cudaMemcpyHostToDevice));

        // Launch Kernel
        block_stats_kernel<<<grid_D, threads_D>>>(d_block, d_sum, d_sum_sq, d_min, d_max, rows_read, D);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Bring partial block stats back to CPU
        CHECK_CUDA(cudaMemcpy(h_block_sum, d_sum, d_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_block_sum_sq, d_sum_sq, d_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_block_min, d_min, d_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_block_max, d_max, d_bytes, cudaMemcpyDeviceToHost));

        // CPU aggregates partials into global statistics
        for (int j = 0; j < D; j++) {
            mean[j]   += h_block_sum[j];
            sum_sq[j] += h_block_sum_sq[j];
            if (h_block_min[j] < min[j]) min[j] = h_block_min[j];
            if (h_block_max[j] > max[j]) max[j] = h_block_max[j];
        }
        rows_left -= rows_read;
    }

    // Finalize global statistics
    for (int j = 0; j < D; j++) {
        mean[j] /= N;
        double var = (sum_sq[j] / N) - (mean[j] * mean[j]);
        if (var < 0.0) var = 0.0;
        std[j] = sqrt(var);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("Statistics computed in %.3f seconds\n", (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9);

    
    // APPLY SCALING
    rewind(fin);
    printf("\n[Phase 2] Applying %s scaling via GPU...\n", mode);
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // Pre-calculate constants to avoid GPU division
    double *h_const1 = (double*)malloc(d_bytes); // holds inv_std OR scaled_min
    double *h_const2 = (double*)malloc(d_bytes); // holds mean OR inv_range
    double *d_const1, *d_const2;
    CHECK_CUDA(cudaMalloc((void**)&d_const1, d_bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_const2, d_bytes));

    if (strcmp(mode, "standard") == 0) {
        for (int j = 0; j < D; j++) {
            h_const1[j] = (std[j] > 0.0) ? 1.0 / std[j] : 0.0;
            h_const2[j] = mean[j];
        }
    } else {
        for (int j = 0; j < D; j++) {
            double range = max[j] - min[j];
            if (range > 0.0) {
                h_const1[j] = min[j] / range; // scaled_min
                h_const2[j] = 1.0 / range;    // inv_range
            } else {
                h_const1[j] = 0.0; h_const2[j] = 0.0;
            }
        }
    }

    // Send constants to GPU once
    CHECK_CUDA(cudaMemcpy(d_const1, h_const1, d_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_const2, h_const2, d_bytes, cudaMemcpyHostToDevice));

    rows_left = N;
    int threads_per_block = 256;

    while (rows_left > 0) {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        long total_elements = rows_read * D;
        size_t current_block_bytes = total_elements * sizeof(double);
        
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        if (fread(h_block, sizeof(double), total_elements, fin) != (size_t)total_elements) die("EOF in Phase 2");

        // H2D Transfer
        CHECK_CUDA(cudaMemcpy(d_block, h_block, current_block_bytes, cudaMemcpyHostToDevice));

        // Launch  Kernel
        if (strcmp(mode, "standard") == 0) {
            standard_scale_kernel<<<num_blocks, threads_per_block>>>(d_block, d_const2, d_const1, total_elements, D);
        } else {
            minmax_scale_kernel<<<num_blocks, threads_per_block>>>(d_block, d_const1, d_const2, total_elements, D);
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // D2H Transfer
        CHECK_CUDA(cudaMemcpy(h_block, d_block, current_block_bytes, cudaMemcpyDeviceToHost));

        // Write scaled block to disk
        if (fwrite(h_block, sizeof(double), total_elements, fout) != (size_t)total_elements) die("Write Error in Phase 2");
        
        rows_left -= rows_read;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    printf("Scaling applied in %.3f seconds\n", (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9);

    fclose(fin); fclose(fout);
    cudaFreeHost(h_block);
    cudaFree(d_block); cudaFree(d_sum); cudaFree(d_sum_sq); cudaFree(d_min); cudaFree(d_max);
    cudaFree(d_const1); cudaFree(d_const2);
    free(mean); free(min); free(max); free(std); free(sum_sq);
    free(h_block_sum); free(h_block_sum_sq); free(h_block_min); free(h_block_max);
    free(h_const1); free(h_const2);

    return 0;
}