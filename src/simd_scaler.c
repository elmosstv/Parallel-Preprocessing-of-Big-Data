#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
  #include <malloc.h>  /* _aligned_malloc, _aligned_free on MinGW/MSVC */
  #define aligned_malloc(align, size) _aligned_malloc((size), (align))
  #define aligned_free(ptr)           _aligned_free((ptr))
#else
  /* Linux/Mac: aligned_alloc requires size to be a multiple of alignment */
  static inline void *aligned_malloc(size_t align, size_t size) {
      size_t aligned_size = (size + align - 1) & ~(align - 1);
      return aligned_alloc(align, aligned_size);
  }
  #define aligned_free(ptr) free((ptr))
#endif
#include <immintrin.h>

struct timespec ts_start, ts_end;
double time_spent;

void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void *alloc_aligned(size_t n_elements, size_t elem_size)
{
    size_t size = n_elements * elem_size;
    void *ptr = aligned_malloc(32, size);
    if (!ptr) die("aligned_malloc failed");
    return ptr;
}

static void compute_stats(FILE *fin, double *mean, double *min, double *max, double *std,  double *var, long N, int D, long block_rows)
{
    for (int j = 0; j < D; j++) {
        mean[j] = 0.0;
        min[j]  =  INFINITY;
        max[j]  = -INFINITY;
        std[j]  = 0.0;
        var[j]  = 0.0;
    }

    double *block  = alloc_aligned((size_t)block_rows * D, sizeof(double));
    double *sum_sq = alloc_aligned(D, sizeof(double));
    /* alloc_aligned does not zero — memset manually */
    memset(sum_sq, 0, D * sizeof(double));

    long rows_left = N;
    while (rows_left > 0) {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
        if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 1");
        // PITHANI ALLAGI TOY LOOP GIA EPITAXINSI PREPI NA DOKIMASTI STO CLUSTER
        for (long i = 0; i < rows_read; i++) {
            double *row = block + (size_t)i * D;
            for (int j = 0; j < D; j++) {
                double val = row[j];
                mean[j]   += val;
                sum_sq[j] += val * val;
                if (val < min[j]) min[j] = val;
                if (val > max[j]) max[j] = val;
            }
        }
        rows_left -= rows_read;
    }
    aligned_free(block);

    for (int j = 0; j < D; j++) {
        mean[j] /= N;
        var[j]   = sum_sq[j] / N - mean[j] * mean[j];
        if (var[j] < 0.0) var[j] = 0.0;
        std[j]   = sqrt(var[j]);
    }
    aligned_free(sum_sq);
}

static void apply_StandardScaler(FILE *fin, FILE *fout,
                                  double *mean, double *std,
                                  long N, int D, long block_rows)
{
    double *block = alloc_aligned((size_t)block_rows * D, sizeof(double));
    double *inv_std = alloc_aligned(D, sizeof(double));
    for (int j = 0; j < D; j++)
        inv_std[j] = (std[j] > 0.0) ? 1.0 / std[j] : 0.0;

    long rows_left = N;
    while (rows_left > 0) {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
        if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 2 (standard)");
        for (long i = 0; i < rows_read; i++) {
            double *row = block + (size_t)i * D;
            for (int j = 0; j < D; j++)
                row[j] = (row[j] - mean[j]) * inv_std[j];
        }

        size_t w = fwrite(block, sizeof(double), (size_t)rows_read * D, fout);
        if (w != (size_t)rows_read * D) die("Write error in phase 2 (standard)");
        rows_left -= rows_read;
    }

    aligned_free(inv_std);
    aligned_free(block);
}

static void apply_MinMaxScaler(FILE *fin, FILE *fout, double *min, double *max,long N, int D, long block_rows)
{
    double *block = alloc_aligned((size_t)block_rows * D, sizeof(double));
    double *inv_range = alloc_aligned(D, sizeof(double));
    double *scaled_min = alloc_aligned(D, sizeof(double));
    for (int j = 0; j < D; j++) {
        double range = max[j] - min[j];
        if (range > 0.0) {
            inv_range[j]  = 1.0 / range;
            scaled_min[j] = min[j] / range;
        } else {
            inv_range[j]  = 0.0;
            scaled_min[j] = 0.0;
        }
    }

    long rows_left = N;
    while (rows_left > 0) {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
        if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 2 (minmax)");

        for (long i = 0; i < rows_read; i++) {
            double *row = block + (size_t)i * D;
            for (int j = 0; j < D; j++)
                row[j] = row[j] * inv_range[j] - scaled_min[j];
        }

        size_t w = fwrite(block, sizeof(double), (size_t)rows_read * D, fout);
        if (w != (size_t)rows_read * D) die("Write error in phase 2 (minmax)");
        rows_left -= rows_read;
    }

    aligned_free(inv_range);
    aligned_free(scaled_min);
    aligned_free(block);
}

int main(int argc, char *argv[])
{
    if (argc < 6 || argc > 7) {
        fprintf(stderr,
            "Usage: %s <input_file> <output_file> <N> <D> <mode> [block_rows]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_file  = argv[1];
    const char *output_file = argv[2];
    long        N           = atol(argv[3]);
    int         D           = atoi(argv[4]);
    const char *mode        = argv[5];
    long        block_rows  = (argc == 7) ? atol(argv[6]) : 100000;

    if (N <= 0 || D <= 0 || block_rows <= 0)
        die("N, D, and block_rows must be positive integers");
    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0)
        die("mode must be 'standard' or 'minmax'");

    printf("Input  : %s\n", input_file);
    printf("Output : %s\n", output_file);
    printf("N=%ld  D=%d  mode=%s  block_rows=%ld\n", N, D, mode, block_rows);
    printf("Block size: %.2f MB\n",
           (double)block_rows * D * sizeof(double) / (1024.0 * 1024.0));

    FILE *fin  = fopen(input_file, "rb");
    if (!fin)  die("Failed to open input file");
    FILE *fout = fopen(output_file, "wb");
    if (!fout) die("Failed to open output file");

    double *mean = alloc_aligned(D, sizeof(double));
    double *min  = alloc_aligned(D, sizeof(double));
    double *max  = alloc_aligned(D, sizeof(double));
    double *std  = alloc_aligned(D, sizeof(double));
    double *var  = alloc_aligned(D, sizeof(double));

    printf("\n[Phase 1] Computing statistics...\n");
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    compute_stats(fin, mean, min, max, std, var, N, D, block_rows);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    time_spent = (ts_end.tv_sec  - ts_start.tv_sec) +
                 (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    printf("Statistics computed in %.3f seconds\n", time_spent);

    rewind(fin);

    printf("\n[Phase 2] Applying %s scaling...\n", mode);
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    if (strcmp(mode, "standard") == 0)
        apply_StandardScaler(fin, fout, mean, std, N, D, block_rows);
    else
        apply_MinMaxScaler(fin, fout, min, max, N, D, block_rows);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    time_spent = (ts_end.tv_sec  - ts_start.tv_sec) +
                 (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    printf("Scaling applied in %.3f seconds\n", time_spent);

    fclose(fin);
    fclose(fout);
    aligned_free(mean); aligned_free(min); aligned_free(max);
    aligned_free(std);  aligned_free(var);
    return 0;
}