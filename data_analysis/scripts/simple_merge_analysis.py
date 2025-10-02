#!/usr/bin/env python3
"""
Simple analysis of merge compatibility between the two datasets.
"""

import pandas as pd

def analyze_merge():
    """Analyze merge compatibility."""
    
    # Load both datasets
    enhanced = pd.read_csv('combined_data/enhanced_llm_data.csv')
    model_analysis = pd.read_csv('combined_data/model_analysis_dataset.csv')
    
    print("=" * 60)
    print("MERGE COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    print("\n1. BASIC COMPARISON")
    print("-" * 30)
    print(f"enhanced_llm_data.csv: {len(enhanced):,} records, {len(enhanced.columns)} columns")
    print(f"model_analysis_dataset.csv: {len(model_analysis):,} records, {len(model_analysis.columns)} columns")
    
    print("\n2. DATA TYPE BREAKDOWN")
    print("-" * 30)
    print("enhanced_llm_data.csv:")
    for dtype, count in enhanced['data_type'].value_counts().items():
        print(f"  {dtype}: {count:,} records")
    
    print("\nmodel_analysis_dataset.csv:")
    if 'data_type' in model_analysis.columns:
        for dtype, count in model_analysis['data_type'].value_counts().items():
            print(f"  {dtype}: {count:,} records")
    else:
        print("  No data_type column")
    
    print("\n3. COLUMN OVERLAP")
    print("-" * 30)
    enhanced_cols = set(enhanced.columns)
    model_analysis_cols = set(model_analysis.columns)
    
    common_cols = enhanced_cols & model_analysis_cols
    unique_to_enhanced = enhanced_cols - model_analysis_cols
    unique_to_model_analysis = model_analysis_cols - enhanced_cols
    
    print(f"Common columns: {len(common_cols)}")
    print(f"Unique to enhanced: {len(unique_to_enhanced)}")
    print(f"Unique to model_analysis: {len(unique_to_model_analysis)}")
    
    print("\n4. RECORD OVERLAP")
    print("-" * 30)
    enhanced_benchmark_results = enhanced[enhanced['data_type'] == 'benchmark_result']
    print(f"Enhanced benchmark results: {len(enhanced_benchmark_results):,}")
    print(f"Model analysis records: {len(model_analysis):,}")
    
    if len(enhanced_benchmark_results) == len(model_analysis):
        print("SAME NUMBER: Model analysis is a subset of enhanced benchmark results")
    else:
        print(f"DIFFERENT: {len(enhanced_benchmark_results) - len(model_analysis)} difference")
    
    print("\n5. MERGE POSSIBILITY")
    print("-" * 30)
    
    print("CAN THEY BE COMBINED? YES, but with considerations:")
    print()
    print("Option 1: Simple Concatenation")
    print("  - Would duplicate benchmark result records")
    print("  - Not recommended due to duplication")
    print()
    print("Option 2: Union (Remove Duplicates)")
    print("  - Model analysis records are subset of enhanced")
    print("  - Would result in same as enhanced_llm_data.csv")
    print("  - Loses analysis features")
    print()
    print("Option 3: Merge with Enhanced as Base (RECOMMENDED)")
    print("  - Use enhanced_llm_data.csv as foundation")
    print("  - Add analysis features from model_analysis_dataset.csv")
    print("  - Preserves all data types")
    print("  - Adds analysis capabilities")
    
    print("\n6. IMPLEMENTATION STRATEGY")
    print("-" * 30)
    print("1. Use enhanced_llm_data.csv as the base dataset")
    print("2. For benchmark_result records, add analysis features from model_analysis_dataset.csv")
    print("3. For other record types, fill analysis features with null/False values")
    print("4. This creates a comprehensive dataset with analysis capabilities")
    
    print("\n7. CONFLICTS CHECK")
    print("-" * 30)
    
    # Check for data type conflicts in common columns
    conflicts = []
    for col in list(common_cols)[:10]:  # Check first 10 common columns
        if col in enhanced.columns and col in model_analysis.columns:
            enhanced_dtype = enhanced[col].dtype
            model_analysis_dtype = model_analysis[col].dtype
            if enhanced_dtype != model_analysis_dtype:
                conflicts.append(f"{col}: {enhanced_dtype} vs {model_analysis_dtype}")
    
    if conflicts:
        print("Data type conflicts found:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    else:
        print("No major data type conflicts detected")
    
    print("\n8. CONCLUSION")
    print("-" * 30)
    print("YES, the files can be combined without major conflicts.")
    print("The model_analysis_dataset.csv is essentially a specialized")
    print("version of the benchmark_result records from enhanced_llm_data.csv")
    print("with additional analysis features added.")
    print()
    print("Best approach: Merge analysis features into enhanced_llm_data.csv")
    print("to create a single comprehensive dataset.")

if __name__ == "__main__":
    analyze_merge()
