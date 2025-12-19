import pandas as pd
import numpy as np
import sys
import os
import itertools
from datetime import datetime

# Add src to path to import fence_optimization
sys.path.append(os.path.join(os.path.dirname(__file__)))

from fence_optimization import load_data, generate_candidates, optimize_layout, CLEANED_DATA_PATH, DEFAULT_MIN_SAMPLES

def run_sensitivity_analysis():
    print("Starting Sensitivity Analysis for Problem 3...")
    
    # Parameters
    R_values = [150, 200, 300]
    alpha_values = [0.81, 0.9, 0.99]
    rf_values = [45, 50, 55] # Maps to eps_meters
    M_values = [450, 500, 550]
    
    results = []
    
    # Load data once
    points = load_data(CLEANED_DATA_PATH)
    
    # Iterate over r_f (requires re-clustering)
    for rf in rf_values:
        print(f"\nProcessing r_f (eps) = {rf} m...")
        # Re-cluster
        clusters = generate_candidates(points, eps_meters=rf, min_samples=DEFAULT_MIN_SAMPLES)
        
        # Iterate over M (requires filtering candidates)
        for M in M_values:
            print(f"  Processing M = {M}...")
            if len(clusters) > M:
                candidates = clusters.sort_values(by='weight', ascending=False).head(M).copy()
            else:
                candidates = clusters.copy()
            
            # Demand points are assumed to be the candidates (cluster centers) for this problem formulation
            demand_points = candidates.copy()
            candidate_points = candidates.copy()
            
            # Iterate over R and Alpha (optimization parameters)
            for R in R_values:
                for alpha in alpha_values:
                    # Optimize
                    # print(f"    Running for R={R}, alpha={alpha}...")
                    selected_fences, final_ratio, _ = optimize_layout(
                        demand_points, candidate_points, r_fence=R, alpha=alpha
                    )
                    
                    fence_count = len(selected_fences)
                    
                    results.append({
                        'R (m)': R,
                        'Alpha': alpha,
                        'r_f (m)': rf,
                        'M': M,
                        'Fence Count (|S|)': fence_count,
                        'Coverage Rate (CR)': final_ratio
                    })
                    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Output file
    output_path = os.path.join("data/output", "q3_sensitivity_analysis.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"\nSensitivity Analysis Completed. Results saved to {output_path}")
    print("\nFirst 10 rows of results:")
    print(results_df.head(10))
    
    # Print summary statistics or specific pivot if needed?
    # The user asked to "output |S| and CR for each parameter group".
    # The CSV contains exactly that.

if __name__ == "__main__":
    run_sensitivity_analysis()
