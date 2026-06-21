#!/usr/bin/env python3
import argparse
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    parser = argparse.ArgumentParser(description="Verify C++ scaler output against scikit-learn.")
    parser.add_argument("--input", type=str, required=True, help="Original input binary file")
    parser.add_argument("--cpp-output", type=str, required=True, help="Scaled output from C++")
    parser.add_argument("--samples", type=int, required=True, help="Number of rows N")
    parser.add_argument("--features", type=int, required=True, help="Number of columns D")
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "minmax"])
    
    args = parser.parse_args()

    print(f"Loading ORIGINAL data: N={args.samples}, D={args.features}...")
    # Φορτώνουμε ΜΟΝΟ το αρχικό αρχείο (10 GB)
    X_original = np.fromfile(args.input, dtype=np.float64).reshape(args.samples, args.features)
    
    print(f"Applying scikit-learn {args.mode} scaler...")
    if args.mode == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    # Φτιάχνουμε τον reference πίνακα (10 GB) - Peak Memory: 20 GB
    X_ref = scaler.fit_transform(X_original)

    # Διαγράφουμε τον αρχικό πίνακα για ελευθέρωση 10 GB!
    del X_original
    gc.collect()

    print("Loading C++ output...")
    # Φορτώνουμε τον πίνακα της C++ (10 GB) - Peak Memory ξανά 20 GB
    X_cpp = np.fromfile(args.cpp_output, dtype=np.float64).reshape(args.samples, args.features)

    print("Comparing results (In-Place)...")
    # In-place αφαίρεση (X_ref = X_ref - X_cpp). Δεν δημιουργεί νέο πίνακα!
    X_ref -= X_cpp
    # In-place απόλυτη τιμή (X_ref = |X_ref|)
    np.abs(X_ref, out=X_ref)
    
    max_err = np.max(X_ref)
    mean_err = np.mean(X_ref)

    print("\n--- Error Report ---")
    print(f"Max Absolute Error:  {max_err:.10e}")
    print(f"Mean Absolute Error: {mean_err:.10e}")

    if max_err < 1e-9:
        print("\nSUCCESS: Your C++ scaler matches the scikit-learn reference perfectly!")
    else:
        print("\nWARNING: Errors detected. Check your math or indexing in the C++ code.")

if __name__ == "__main__":
    main()