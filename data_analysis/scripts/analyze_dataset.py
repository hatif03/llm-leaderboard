#!/usr/bin/env python3
"""
Script to analyze the dataset for duplicates, missing values, and feature engineering opportunities.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_dataset():
    """Analyze the combined dataset for duplicates and missing values."""
    
    # Load the combined dataset
    print("Loading dataset...")
    df = pd.read_csv('combined_data/all_llm_data.csv')
    
    print(f"Total records: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Focus on benchmark results for duplicate analysis
    benchmark_results = df[df['data_type'] == 'benchmark_result'].copy()
    print(f"\nBenchmark results: {len(benchmark_results)}")
    
    # Analyze duplicates
    print("\n=== DUPLICATE ANALYSIS ===")
    model_benchmark_combinations = benchmark_results.groupby(['model_id', 'benchmark_id']).size()
    duplicates = model_benchmark_combinations[model_benchmark_combinations > 1]
    
    print(f"Unique model-benchmark combinations: {len(model_benchmark_combinations)}")
    print(f"Combinations with duplicates: {len(duplicates)}")
    print(f"Total duplicate entries: {duplicates.sum() - len(duplicates)}")
    
    if len(duplicates) > 0:
        print("\nTop 10 most duplicated combinations:")
        print(duplicates.sort_values(ascending=False).head(10))
        
        # Show example of duplicates
        print("\nExample of duplicate entries:")
        example_model = duplicates.index[0][0]
        example_benchmark = duplicates.index[0][1]
        example_data = benchmark_results[
            (benchmark_results['model_id'] == example_model) & 
            (benchmark_results['benchmark_id'] == example_benchmark)
        ][['model_id', 'benchmark_id', 'provider_id', 'score', 'normalized_score', 'is_self_reported']]
        print(example_data)
    
    # Analyze missing values
    print("\n=== MISSING VALUES ANALYSIS ===")
    missing_analysis = df.isnull().sum()
    missing_percentage = (missing_analysis / len(df)) * 100
    
    print("Columns with missing values:")
    missing_data = pd.DataFrame({
        'Missing Count': missing_analysis,
        'Missing Percentage': missing_percentage
    }).sort_values('Missing Count', ascending=False)
    
    # Show only columns with missing values
    missing_data = missing_data[missing_data['Missing Count'] > 0]
    print(missing_data.head(20))
    
    # Focus on key numerical columns
    key_columns = ['param_count', 'training_tokens', 'score', 'normalized_score']
    print(f"\nKey numerical columns analysis:")
    for col in key_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            zero_count = (df[col] == 0).sum() if df[col].dtype in ['int64', 'float64'] else 0
            print(f"{col}: {missing} missing, {zero_count} zeros")
    
    # Analyze cost-related columns
    print(f"\nCost-related columns:")
    cost_columns = [col for col in df.columns if 'cost' in col.lower() or 'cents' in col.lower()]
    for col in cost_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"{col}: {missing} missing")
    
    # Analyze license data
    print(f"\nLicense analysis:")
    if 'license_id' in df.columns:
        license_counts = df['license_id'].value_counts()
        print(f"Unique licenses: {len(license_counts)}")
        print("Top 10 licenses:")
        print(license_counts.head(10))
    
    # Analyze release dates
    print(f"\nRelease date analysis:")
    if 'release_date' in df.columns:
        release_dates = df['release_date'].dropna()
        print(f"Records with release dates: {len(release_dates)}")
        if len(release_dates) > 0:
            print(f"Date range: {release_dates.min()} to {release_dates.max()}")
    
    return df, benchmark_results

if __name__ == "__main__":
    df, benchmark_results = analyze_dataset()
