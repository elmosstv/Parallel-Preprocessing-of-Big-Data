# Παράλληλη Προεπεξεργασία Μεγάλου Όγκου Δεδομένων

Υλοποίηση παράλληλου feature scaler σε 5 εκδόσεις: Serial, SIMD, OpenMP, MPI και CUDA.

---

## Απαιτήσεις

| Εργαλείο | Χρήση |
|---|---|
| `gcc` (≥ 9) | Serial, SIMD, OpenMP |
| `mpicc` / OpenMPI | MPI |
| `nvcc` / CUDA Toolkit | CUDA |
| `python3` + `numpy`, `scikit-learn` | Παραγωγή δεδομένων & Επαλήθευση |

Εγκατάσταση Python εξαρτήσεων:
```bash
pip install numpy scikit-learn
```

---

## Δομή Project

```
.
├── src/
│   ├── serial_scaler.c
│   ├── simd_scaler.c
│   ├── omp_scaler.c
│   ├── mpi_scaler.c
│   └── cuda_scaler.cu
├── scripts/
│   ├── generate_data.py
│   ├── generate_all_datasets.sh
│   ├── run_benchmarks_auto.sh
│   ├── verify_scaler.py
│   └── verify_all_outputs.sh
├── Makefile
└── README.md
```

---

## 1. Μεταγλώττιση

Για να μεταγλωττιστούν **όλες** οι υλοποιήσεις μαζί:

```bash
make all
```

Ή μεμονωμένα:

```bash
make scaler_serial
make scaler_simd
make scaler_omp
make scaler_mpi
make scaler_cuda
```

Για διαγραφή των εκτελέσιμων:

```bash
make clean
```

> **Σημείωση για CUDA:** Το `NVCC_FLAGS` έχει οριστεί για αρχιτεκτονική `sm_70` (Volta). Αν χρησιμοποιείτε διαφορετική GPU, τροποποιήστε αντίστοιχα το `Makefile` (π.χ. `sm_86` για Ampere).

---

## 2. Παραγωγή Δεδομένων

### Μεμονωμένο dataset

```bash
python3 scripts/generate_data.py \
    --samples <N> \
    --features <D> \
    --output <output.bin> \
    --dtype float64
```

Παράδειγμα (Small dataset ~256 MB):

```bash
python3 scripts/generate_data.py \
    --samples 1000000 \
    --features 32 \
    --output data_1M_32.bin \
    --dtype float64
```

### Όλα τα datasets μαζί

```bash
bash scripts/generate_all_datasets.sh
```

Δημιουργεί 4 αρχεία (απαιτεί **~65 GB** ελεύθερο χώρο):

| Αρχείο | N | D | Μέγεθος |
|---|---|---|---|
| `data_1M_32.bin` | 1,000,000 | 32 | ~256 MB |
| `data_5M_64.bin` | 5,000,000 | 64 | ~2.56 GB |
| `data_10M_128.bin` | 10,000,000 | 128 | ~10.24 GB |
| `data_50M_128.bin` | 50,000,000 | 128 | ~51.2 GB |

---

## 3. Εκτέλεση

Σύνταξη για όλες τις υλοποιήσεις:

```
./<εκτελέσιμο> <input.bin> <output.bin> <N> <D> <mode> [block_rows]
```

| Παράμετρος | Περιγραφή |
|---|---|
| `input.bin` | Αρχείο εισόδου (raw binary, float64, row-major) |
| `output.bin` | Αρχείο εξόδου |
| `N` | Αριθμός γραμμών (samples) |
| `D` | Αριθμός στηλών (features) |
| `mode` | `standard` ή `minmax` |
| `block_rows` | *(Προαιρετικό)* Γραμμές ανά block — default: `100000` |

### Serial

```bash
./scaler_serial data_1M_32.bin out_serial.bin 1000000 32 standard
```

### SIMD

```bash
./scaler_simd data_1M_32.bin out_simd.bin 1000000 32 standard
```

### OpenMP

Ορισμός αριθμού threads πριν την εκτέλεση:

```bash
export OMP_NUM_THREADS=8
./scaler_omp data_1M_32.bin out_omp.bin 1000000 32 standard
```

### MPI

```bash
mpirun --mca io romio321 -np 4 ./scaler_mpi data_1M_32.bin out_mpi.bin 1000000 32 standard
```

Αντικαταστήστε το `-np 4` με τον αριθμό processes που θέλετε (1, 2, 4, 8, 16).

### CUDA

```bash
./scaler_cuda data_1M_32.bin out_cuda.bin 1000000 32 standard
```

### Χρήση MinMax αντί Standard

Σε οποιαδήποτε υλοποίηση αντικαταστήστε `standard` με `minmax`:

```bash
./scaler_serial data_1M_32.bin out_minmax.bin 1000000 32 minmax
```

---

## 4. Αυτοματοποιημένα Benchmarks

Το script εκτελεί όλες τις υλοποιήσεις 3 φορές, κρατά τον **ελάχιστο χρόνο** για κάθε μία και αποθηκεύει τα αποτελέσματα σε αρχείο κειμένου.

Επεξεργαστείτε τις ρυθμίσεις στην αρχή του αρχείου αν θέλετε να αλλάξετε dataset ή αριθμό επαναλήψεων, και στη συνέχεια:

```bash
bash scripts/run_benchmarks_auto.sh
```

Τα αποτελέσματα αποθηκεύονται αυτόματα στο:

```
best_results_<N>_<D>.txt
```

---

## 5. Επαλήθευση Ορθότητας

### Μεμονωμένη επαλήθευση

Σύγκριση ενός αρχείου εξόδου με το αποτέλεσμα αναφοράς της scikit-learn:

```bash
python3 scripts/verify_scaler.py \
    --input data_1M_32.bin \
    --cpp-output out_serial.bin \
    --samples 1000000 \
    --features 32 \
    --mode standard
```

### Μαζική επαλήθευση όλων των υλοποιήσεων

Βεβαιωθείτε ότι τα αρχεία εξόδου ακολουθούν την ονομαστική σύμβαση `out_<impl>_<N>_<D>.bin` (π.χ. `out_serial_1000000_32.bin`), και στη συνέχεια:

```bash
bash scripts/verify_all_outputs.sh
```

Το script εκτυπώνει για κάθε υλοποίηση:

```
Max Absolute Error:  X.XXXXXXXXXXe-XX
Mean Absolute Error: X.XXXXXXXXXXe-XX
SUCCESS / WARNING
```

Αποτέλεσμα θεωρείται **σωστό** αν το `Max Absolute Error < 1e-9`.

---

## Γρήγορη Αναφορά Εντολών

```bash
# Build
make all

# Παραγωγή small dataset
python3 scripts/generate_data.py --samples 1000000 --features 32 --output data_1M_32.bin --dtype float64

# Εκτέλεση
./scaler_serial  data_1M_32.bin out_serial.bin  1000000 32 standard
./scaler_simd    data_1M_32.bin out_simd.bin    1000000 32 standard
OMP_NUM_THREADS=8 ./scaler_omp data_1M_32.bin out_omp.bin 1000000 32 standard
mpirun -np 4 ./scaler_mpi data_1M_32.bin out_mpi.bin 1000000 32 standard
./scaler_cuda    data_1M_32.bin out_cuda.bin    1000000 32 standard

# Benchmarks
bash scripts/run_benchmarks_auto.sh

# Επαλήθευση
python3 scripts/verify_scaler.py --input data_1M_32.bin --cpp-output out_serial.bin --samples 1000000 --features 32 --mode standard
```