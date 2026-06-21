# Compilers
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Common Flags
CFLAGS = -O3 -Wall
LDFLAGS = -lm

# Specific Flags
OMP_FLAGS = -fopenmp
SIMD_FLAGS = -mavx2 -mfma -fopt-info-vec-optimized
NVCC_FLAGS = -O3 -arch=sm_70

# Directories
SRC_DIR = src

# Executables
TARGETS = scaler_serial scaler_omp scaler_simd scaler_mpi scaler_cuda

.PHONY: all clean

all: $(TARGETS)

scaler_serial: $(SRC_DIR)/serial_scaler.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

scaler_omp: $(SRC_DIR)/omp_scaler.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) $< -o $@ $(LDFLAGS)

scaler_simd: $(SRC_DIR)/simd_scaler.c
	$(CC) $(CFLAGS) $(SIMD_FLAGS) $< -o $@ $(LDFLAGS)

scaler_mpi: $(SRC_DIR)/mpi_scaler.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)

scaler_cuda: $(SRC_DIR)/cuda_scaler.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)