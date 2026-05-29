#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    parser = argparse.ArgumentParser(description="Verify C++ scaler output against scikit-learn.")
    parser.add_argument("--input", type=str, required=True, help="Original input binary file")
    parser.add_argument("--cpp-output", type=str, required=True, help="Scaled output from C++")
    parser.add_argument("--samples", type=int, required=True, help="Number of rows N")
    parser.add_argument("--features", type=int, required=True, help="Number of columns D")
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "minmax"])
    
    args = parser.parse_args()

    print(f"Loading data: N={args.samples}, D={args.features}...")
    
    # Load original data
    X_original = np.fromfile(args.input, dtype=np.float64).reshape(args.samples, args.features)
    
    # Load C++ output
    X_cpp = np.fromfile(args.cpp_output, dtype=np.float64).reshape(args.samples, args.features)

    print(f"Applying scikit-learn {args.mode} scaler...")
    if args.mode == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
        
    X_ref = scaler.fit_transform(X_original)

    print("Comparing results...")
    # Calculate Absolute Errors
    abs_diff = np.abs(X_ref - X_cpp)
    max_err = np.max(abs_diff)
    mean_err = np.mean(abs_diff)

    print("\n--- Error Report ---")
    print(f"Max Absolute Error:  {max_err:.10e}")
    print(f"Mean Absolute Error: {mean_err:.10e}")

    # A good threshold for floating point math differences is usually around 1e-10
    if max_err < 1e-9:
        print("\nSUCCESS: Your C++ scaler matches the scikit-learn reference perfectly!")
    else:
        print("\nWARNING: Errors detected. Check your math or indexing in the C++ code.")

if __name__ == "__main__":
    main()