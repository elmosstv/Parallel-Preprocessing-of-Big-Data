#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void compute_stats(FILE *fin, double *mean, double *min, double *max, double *std, double *var, long N, int D, long block_rows)
{
  for(int j=0; j<D; j++){
    mean[j] = 0.0;
    min[j] = INFINITY;
    max[j] = -INFINITY;
    std[j] = 0.0;
    var[j] = 0.0;
  }

  double *block = malloc((size_t)block_rows * D * sizeof(double));
  if (!block) die("Memory allocation failed");
  double *sum_sq = calloc(D , sizeof(double));
  if (!sum_sq) die("Memory allocation failed");

  long rows_left = N;
  while(rows_left > 0)
  {
    long rows_read = rows_left < block_rows ? rows_left : block_rows;
    size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
    if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 1");
    
    // OpenMP Παραλληλισμός με Array Reductions για αποφυγή Race Conditions
    #pragma omp parallel for default(none) shared(block, D, rows_read) \
        reduction(+:mean[0:D], sum_sq[0:D]) \
        reduction(min:min[0:D]) reduction(max:max[0:D])
    for(int i = 0; i<rows_read; i++){
      for(int j=0; j<D; j++){
        double val = block[i * D + j];
        mean[j] += val;
        sum_sq[j] += val * val; 
        if (val < min[j]) min[j] = val;
        if (val > max[j]) max[j] = val;
      }
    }
    rows_left -= rows_read;
  }
  free(block);

  for(int j=0; j<D; j++){
    mean[j] /= N;
    var[j] = sum_sq[j]/N - mean[j] * mean[j];
    std[j] = sqrt(var[j]);
  }
  free(sum_sq);
}

static void apply_StandardScaler(FILE *fin, FILE *fout, double *mean, double *std, long N, int D, long block_rows)
{
  double *block = malloc((size_t)block_rows * D * sizeof(double));
  if (!block) die("Memory allocation failed");

  long rows_left = N;
  while(rows_left > 0){
    long rows_read = rows_left < block_rows ? rows_left : block_rows;
    size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
    if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 2");

    // OpenMP Παραλληλισμός (Χωρίς Reductions, πλήρως ανεξάρτητο)
    #pragma omp parallel for default(none) shared(block, D, rows_read, mean, std)
    for(int i = 0; i<rows_read; i++){
      for(int j=0; j<D; j++){
        double val = block[i * D + j];
        block[i * D + j] =(std[j] > 0) ?((val - mean[j]) / std[j]) : 0.0;
      }
    }
    size_t w = fwrite(block, sizeof(double), (size_t)rows_read * D, fout);
    if (w != (size_t)rows_read * D) die("Write error in phase 2 (standard)");
    rows_left -= rows_read;
  }
  free(block);
}

static void apply_MinMaxScaler(FILE *fin, FILE *fout, double *min, double *max, long N, int D, long block_rows)
{
  double *block = malloc((size_t)block_rows * D * sizeof(double));
  if (!block) die("Memory allocation failed");

  long rows_left = N;
  while(rows_left > 0){
    long rows_read = rows_left < block_rows ? rows_left : block_rows;
    size_t n = fread(block, sizeof(double), (size_t)rows_read * D, fin);
    if (n != (size_t)rows_read * D) die("Unexpected EOF in phase 2");

    // OpenMP Παραλληλισμός (Χωρίς Reductions, πλήρως ανεξάρτητο)
    #pragma omp parallel for default(none) shared(block, D, rows_read, min, max)
    for(int i = 0; i<rows_read; i++){
      for(int j=0; j<D; j++){
        double val = block[i * D + j];
        block[i * D + j] =(max[j] != min[j]) ?((val - min[j]) / (max[j] - min[j])) : 0.0;
      }
    }
    size_t w = fwrite(block, sizeof(double), (size_t)rows_read * D, fout);
    if (w != (size_t)rows_read * D) die("Write error in phase 2 (minmax)");
    rows_left -= rows_read;
  }
  free(block);
}

int main(int argc, char *argv[])
{
    if(argc < 6 || argc > 7){
      fprintf(stderr, "Usage: %s <input_file> <output_file> <num_samples> <num_features> <mode> [block_rows]\n", argv[0]);
      return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];  
    long N = atol(argv[3]);
    int D = atoi(argv[4]);
    const char *mode = argv[5];
    long block_rows = (argc == 7) ? atol(argv[6]) : 100000; // Default block size

    if (N <= 0 || D <= 0 || block_rows <= 0)
        die("N, D, and block_rows must be positive integers");
    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0)
        die("mode must be 'standard' or 'minmax'");
 
    printf("Input  : %s\n", input_file);
    printf("Output : %s\n", output_file);
    printf("N=%ld  D=%d  mode=%s  block_rows=%ld\n", (long)N, (int)D, mode, (long)block_rows);
    printf("Block size: %.2f MB\n",
           (double)block_rows * D * sizeof(double) / (1024.0 * 1024.0));

    FILE *fin = fopen(input_file, "rb");
    if (!fin) die("Failed to open input file");
    
    FILE *fout = fopen(output_file, "wb");
    if (!fout) die("Failed to open output file");

    double *mean = malloc(D * sizeof(double));
    double *min = malloc(D * sizeof(double));
    double *max = malloc(D * sizeof(double));
    double *std = malloc(D * sizeof(double));
    double *var = malloc(D * sizeof(double));
    if (!mean || !min || !max || !std || !var)
        die("Memory allocation failed");  

    double start_time, end_time;

    /*Φάση Υπολογισμού Στατιστικών*/
    printf("Computing statistics...\n");
    start_time = omp_get_wtime();
    
    compute_stats(fin, mean, min, max, std, var, N, D, block_rows);
    
    end_time = omp_get_wtime();
    printf("Statistics computed in %f seconds\n", end_time - start_time);
    
    rewind(fin); 

    /*Φάση Εφαρμογής Scaler*/
    printf("\n[Phase 2] Applying %s scaling...\n", mode);
    start_time = omp_get_wtime();
    
    if (strcmp(mode, "standard") == 0){
        apply_StandardScaler(fin, fout, mean, std, N, D, block_rows);
    }
    else if (strcmp(mode, "minmax") == 0){
        apply_MinMaxScaler(fin, fout, min, max, N, D, block_rows);
    }

    end_time = omp_get_wtime();
    printf("Scaling applied in %f seconds\n", end_time - start_time);  

    fclose(fin);
    fclose(fout);
    free(mean);
    free(min);
    free(max);
    free(std);
    free(var);

    return 0;
}