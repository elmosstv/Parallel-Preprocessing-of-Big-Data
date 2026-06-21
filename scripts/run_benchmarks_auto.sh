#!/bin/bash

# ==============================================================================
# ΡΥΘΜΙΣΕΙΣ BENCHMARK
# ==============================================================================
DATASET="data_1M_32.bin"
N=1000000
D=32
MODE="standard"
RUNS=3   # Πόσες φορές θα εκτελεστεί το κάθε πείραμα
RESULTS_FILE="best_results_${N}_${D}.txt"

echo "=== ΕΝΑΡΞΗ ΑΥΤΟΜΑΤΟΠΟΙΗΜΕΝΟΥ BENCHMARKING ($RUNS ΕΚΤΕΛΕΣΕΙΣ) ===" | tee $RESULTS_FILE
echo "Dataset: $DATASET (N=$N, D=$D, Mode=$MODE)" | tee -a $RESULTS_FILE
echo "Βρίσκουμε τον ΑΠΟΛΥΤΑ ΕΛΑΧΙΣΤΟ χρόνο (Minimum Time)..." | tee -a $RESULTS_FILE
echo "==================================================================" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# ==============================================================================
# ΣΥΝΑΡΤΗΣΗ: Τρέχει μια εντολή $RUNS φορές, βρίσκει το ελάχιστο, σώζει το output
# ==============================================================================
run_and_keep_min() {
    local title="$1"
    shift
    local cmd=("$@")

    # Το -n αποτρέπει την αλλαγή γραμμής για να φτιάξουμε "μπάρα φόρτωσης"
    echo -n ">> Εκτέλεση: $title "
    echo "--- $title ---" >> $RESULTS_FILE

    local min_time=999999.0
    local best_log="best_run.tmp"

    for i in $(seq 1 $RUNS); do
        # Τυπώνει σε ποιο run βρισκόμαστε (π.χ. [1/5]) δίπλα στο όνομα
        echo -n "[$i/$RUNS] "
        
        local current_log="current_run.tmp"
        
        # Εκτέλεση με </dev/null για να αποτρέψουμε το deadlock του mpirun στο Bash!
        "${cmd[@]}" > "$current_log" 2>&1 </dev/null

        local t1=$(grep "Statistics computed in" "$current_log" | awk '{print $4}')
        local t2=$(grep "Scaling applied in" "$current_log" | awk '{print $4}')

        if [ -z "$t1" ]; then t1=0; fi
        if [ -z "$t2" ]; then t2=0; fi

        local total_time=$(awk -v t1="$t1" -v t2="$t2" 'BEGIN {print t1 + t2}')
        local is_less=$(awk -v tot="$total_time" -v min="$min_time" 'BEGIN {if (tot < min && tot > 0) print 1; else print 0}')

        if [ "$is_less" -eq 1 ]; then
            min_time=$total_time
            cp "$current_log" "$best_log"
        fi
    done

    # Αλλαγή γραμμής και τύπωμα του αποτελέσματος
    cat "$best_log" >> $RESULTS_FILE
    echo "-" >> $RESULTS_FILE
    echo "   Καλύτερος χρόνος: $min_time s"
    
    rm -f current_run.tmp best_run.tmp
}

# ==============================================================================
# ΕΚΤΕΛΕΣΗ ΟΛΩΝ ΤΩΝ ΠΕΙΡΑΜΑΤΩΝ
# ==============================================================================

run_and_keep_min "SERIAL" ./scaler_serial $DATASET out_serial.bin $N $D $MODE

run_and_keep_min "SIMD" ./scaler_simd $DATASET out_simd.bin $N $D $MODE

for t in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$t
    run_and_keep_min "OPENMP ($t Threads)" ./scaler_omp $DATASET out_omp.bin $N $D $MODE
done

for p in 1 2 4 8 16; do
    run_and_keep_min "MPI ($p Processes)" mpirun --mca io romio321 -np $p ./scaler_mpi $DATASET out_mpi.bin $N $D $MODE
done

run_and_keep_min "CUDA GPU" ./scaler_cuda $DATASET out_cuda.bin $N $D $MODE

echo "========================================="
echo "Τα benchmarks ολοκληρώθηκαν με επιτυχία!"
echo "Το καθαρό αποτέλεσμα με τους καλύτερους χρόνους βρίσκεται στο: $RESULTS_FILE"