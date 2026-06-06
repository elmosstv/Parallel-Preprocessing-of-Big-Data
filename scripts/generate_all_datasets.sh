#!/bin/bash

echo "Έναρξη Παραγωγής Datasets..."
echo "Προσοχή: Βεβαιωθείτε ότι έχετε τουλάχιστον 65GB ελεύθερο χώρο στον δίσκο!"
echo ""

# 1. Small (~256 MB)
echo "[1/4] Δημιουργία Small Dataset (N=1,000,000 | D=32)"
python3 scripts/generate_data.py --samples 1000000 --features 32 --output data_1M_32.bin --dtype float64
echo "Το Small Dataset ολοκληρώθηκε."
echo "---------------------------------------------------"

# 2. Medium (~2.56 GB)
echo "[2/4] Δημιουργία Medium Dataset (N=5,000,000 | D=64)"
python3 scripts/generate_data.py --samples 5000000 --features 64 --output data_5M_64.bin --dtype float64
echo "Το Medium Dataset ολοκληρώθηκε."
echo "---------------------------------------------------"

# 3. Large (~10.24 GB)
echo "[3/4] Δημιουργία Large Dataset (N=10,000,000 | D=128)"
python3 scripts/generate_data.py --samples 10000000 --features 128 --output data_10M_128.bin --dtype float64
echo "Το Large Dataset ολοκληρώθηκε."
echo "---------------------------------------------------"

# # 4. Very Large (~51.2 GB)
# echo "[4/4] Δημιουργία Very Large Dataset (N=50,000,000 | D=128)"
# echo "Αυτό θα πάρει αρκετή ώρα..."
# python3 scripts/generate_data.py --samples 50000000 --features 128 --output data_50M_128.bin --dtype float64
# echo "Το Very Large Dataset ολοκληρώθηκε."
# echo "---------------------------------------------------"

echo "Όλα τα αρχεία δημιουργήθηκαν με επιτυχία!"