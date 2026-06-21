#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Η συνάρτηση die δέχεται πλέον το rank και καλεί την MPI_Abort 
για να τερματίσει όλες τις διεργασίες σε περίπτωση σφάλματος.*/
void die(const char *msg, int rank) {
    if (rank == 0) {
        fprintf(stderr, "Error: %s\n", msg);
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

/* Data Partitioning: υπολογίζει τις γραμμές (chunk) που αναλαμβάνει κάθε διεργασία.*/
void get_local_range(long N, int rank, int size, long *local_start, long *local_N) {
    long rows_per_proc = N / size;
    long remainder = N % size;

    if (rank < remainder) {
        *local_start = rank * (rows_per_proc + 1);
        *local_N = rows_per_proc + 1;
    } else {
        *local_start = rank * rows_per_proc + remainder;
        *local_N = rows_per_proc;
    }
}

/* Χρήση POSIX FILE* αντί για MPI_File */
static void compute_stats_mpi(FILE *fin, double *global_mean, double *global_min, double *global_max, double *global_std, double *global_var, long local_N, long N_total, int D, long block_rows, MPI_Offset start_offset, int rank)
{
    /* Δέσμευση μνήμης για τα ΤΟΠΙΚΑ στατιστικά της κάθε διεργασίας */
    double *loc_sum = calloc(D, sizeof(double));
    double *loc_sum_sq = calloc(D, sizeof(double));
    double *loc_min = malloc(D * sizeof(double));
    double *loc_max = malloc(D * sizeof(double));

    if (!loc_sum || !loc_sum_sq || !loc_min || !loc_max) die("Memory allocation failed (stats)", rank);

    for(int j=0; j<D; j++){
        loc_min[j] = INFINITY;
        loc_max[j] = -INFINITY;
    }

    double *block = malloc((size_t)block_rows * D * sizeof(double));
    if (!block) die("Memory allocation failed (block)", rank);

    long rows_left = local_N;
    MPI_Offset current_offset = start_offset;

    while(rows_left > 0)
    {
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        int count = (int)(rows_read * D);

        // Ανάγνωση με fseek/fread στο κατάλληλο offset
        fseek(fin, (long)current_offset, SEEK_SET);
        fread(block, sizeof(double), count, fin);
        
        for(int i = 0; i < rows_read; i++){
            for(int j = 0; j < D; j++){
                double val = block[i * D + j];
                loc_sum[j] += val; // Υπολογισμός τοπικών αθροισμάτων
                loc_sum_sq[j] += val * val; 
                if (val < loc_min[j]) loc_min[j] = val;
                if (val > loc_max[j]) loc_max[j] = val;
            }
        }
        rows_left -= rows_read;
        current_offset += count * sizeof(double); // Ενημέρωση του offset
    }
    free(block);

    // Δέσμευση μνήμης για τα καθολικά (global) αθροίσματα
    double *global_sum = malloc(D * sizeof(double));
    double *global_sum_sq = malloc(D * sizeof(double));
    if (!global_sum || !global_sum_sq) die("Memory allocation failed (global stats)", rank);

    // Συλλογικές επικοινωνίες MPI_Allreduce για να συγκεντρωθούν τα τοπικά στατιστικά σε ολικά
    MPI_Allreduce(loc_sum, global_sum, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(loc_sum_sq, global_sum_sq, D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(loc_min, global_min, D, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(loc_max, global_max, D, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // Υπολογισμός τελικών στατιστικών χρησιμοποιώντας το N_total
    for(int j=0; j<D; j++){
        global_mean[j] = global_sum[j] / N_total;
        global_var[j] = global_sum_sq[j] / N_total - global_mean[j] * global_mean[j];
        if (global_var[j] < 0) global_var[j] = 0.0;
        global_std[j] = sqrt(global_var[j]);
    }

    free(loc_sum);
    free(loc_sum_sq);
    free(loc_min);
    free(loc_max);
    free(global_sum);
    free(global_sum_sq);
}

static void apply_StandardScaler_mpi(FILE *fin, FILE *fout, double *mean, double *std, long local_N, int D, long block_rows, MPI_Offset start_offset, int rank)
{
    double *block = malloc((size_t)block_rows * D * sizeof(double));
    if (!block) die("Memory allocation failed (block phase 2)", rank);

    long rows_left = local_N;
    MPI_Offset current_offset = start_offset;

    while(rows_left > 0){
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        int count = (int)(rows_read * D);

        // Ανάγνωση αρχικού block από το offset της διεργασίας
        fseek(fin, (long)current_offset, SEEK_SET);
        fread(block, sizeof(double), count, fin);

        for(int i = 0; i < rows_read; i++){
            for(int j = 0; j < D; j++){
                double val = block[i * D + j];
                block[i * D + j] = (std[j] > 0) ? ((val - mean[j]) / std[j]) : 0.0;
            }
        }
        
        // Εγγραφή block στο αρχείο εξόδου, στο ίδιο ακριβώς offset
        fseek(fout, (long)current_offset, SEEK_SET);
        fwrite(block, sizeof(double), count, fout);
        
        rows_left -= rows_read;
        current_offset += count * sizeof(double);
    }
    free(block);
}

static void apply_MinMaxScaler_mpi(FILE *fin, FILE *fout, double *min, double *max, long local_N, int D, long block_rows, MPI_Offset start_offset, int rank)
{
    double *block = malloc((size_t)block_rows * D * sizeof(double));
    if (!block) die("Memory allocation failed (block phase 2)", rank);

    long rows_left = local_N;
    MPI_Offset current_offset = start_offset;

    while(rows_left > 0){
        long rows_read = rows_left < block_rows ? rows_left : block_rows;
        int count = (int)(rows_read * D);

        // Ανάγνωση αρχικού block από το offset της διεργασίας
        fseek(fin, (long)current_offset, SEEK_SET);
        fread(block, sizeof(double), count, fin);

        for(int i = 0; i < rows_read; i++){
            for(int j = 0; j < D; j++){
                double val = block[i * D + j];
                block[i * D + j] = (max[j] != min[j]) ? ((val - min[j]) / (max[j] - min[j])) : 0.0;
            }
        }
        
        // Εγγραφή block στο αρχείο εξόδου, στο ίδιο ακριβώς offset
        fseek(fout, (long)current_offset, SEEK_SET);
        fwrite(block, sizeof(double), count, fout);
        
        rows_left -= rows_read;
        current_offset += count * sizeof(double);
    }
    free(block);
}

int main(int argc, char *argv[])
{
    // Αρχικοποίηση του περιβάλλοντος MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 6 || argc > 7){
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <procs> %s <input> <output> <N> <D> <mode> [block_rows]\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];  
    long N = atol(argv[3]);
    int D = atoi(argv[4]);
    const char *mode = argv[5];
    long block_rows = (argc == 7) ? atol(argv[6]) : 100000;

    if (N <= 0 || D <= 0 || block_rows <= 0) die("N, D, and block_rows must be positive integers", rank);
    if (strcmp(mode, "standard") != 0 && strcmp(mode, "minmax") != 0) die("mode must be 'standard' or 'minmax'", rank);
 
    if (rank == 0) {
        printf("--- MPI Scaler ---\n");
        printf("MPI Size: %d processes\n", size);
        printf("Input   : %s\n", input_file);
        printf("Output  : %s\n", output_file);
        printf("N=%ld  D=%d  mode=%s  block_rows=%ld\n", N, D, mode, block_rows);
        printf("Block size: %.2f MB\n\n", (double)block_rows * D * sizeof(double) / (1024.0 * 1024.0));
    }

    /* Κατανομή του φόρτου εργασίας (data partitioning) */
    long local_start, local_N;
    get_local_range(N, rank, size, &local_start, &local_N);

    // Το starting offset (σε bytes) 
    MPI_Offset start_offset = (MPI_Offset)local_start * D * sizeof(double);

    // Άνοιγμα του αρχείου εισόδου με κλασικό C File I/O
    FILE *fin = fopen(input_file, "rb");
    if (!fin) die("Failed to open input file", rank);

    double *global_mean = malloc(D * sizeof(double));
    double *global_min = malloc(D * sizeof(double));
    double *global_max = malloc(D * sizeof(double));
    double *global_std = malloc(D * sizeof(double));
    double *global_var = malloc(D * sizeof(double));
    
    if (!global_mean || !global_min || !global_max || !global_std || !global_var) die("Memory allocation failed (globals)", rank);  

    double start_time, end_time;

    /* Φάση 1: Υπολογισμός Στατιστικών */
    if (rank == 0) printf("Computing statistics...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    compute_stats_mpi(fin, global_mean, global_min, global_max, global_std, global_var, local_N, N, D, block_rows, start_offset, rank);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0) printf("Statistics computed in %f seconds\n", end_time - start_time);
    
    /* Φάση 2: Εφαρμογή Scaler */
    
    // Η διεργασία 0 δημιουργεί το αρχείο εξόδου, για να βεβαιωθούμε ότι υπάρχει άδειο
    if (rank == 0) {
        FILE *f = fopen(output_file, "wb");
        if (!f) die("Failed to create output file", rank);
        fclose(f);
    }
    
    // Περιμένουμε όλοι να ολοκληρωθεί η δημιουργία
    MPI_Barrier(MPI_COMM_WORLD);

    // Όλες οι διεργασίες ανοίγουν το αρχείο εξόδου για παράλληλη ενημέρωση (r+b)
    FILE *fout = fopen(output_file, "r+b");
    if (!fout) die("Failed to open output file for update", rank);

    if (rank == 0) printf("\n[Phase 2] Applying %s scaling...\n", mode);
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    if (strcmp(mode, "standard") == 0){
        apply_StandardScaler_mpi(fin, fout, global_mean, global_std, local_N, D, block_rows, start_offset, rank);
    }
    else if (strcmp(mode, "minmax") == 0){
        apply_MinMaxScaler_mpi(fin, fout, global_min, global_max, local_N, D, block_rows, start_offset, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0) printf("Scaling applied in %f seconds\n", end_time - start_time);  

    // Κλείσιμο των αρχείων
    fclose(fin);
    fclose(fout);

    free(global_mean);
    free(global_min);
    free(global_max);
    free(global_std);
    free(global_var);

    MPI_Finalize();
    return 0;
}