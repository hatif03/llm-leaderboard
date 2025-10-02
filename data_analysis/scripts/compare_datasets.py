#!/usr/bin/env python3
"""
Compare enhanced_llm_data.csv and model_analysis_dataset.csv
"""

import pandas as pd

def compare_datasets():
    """Compare the two main datasets."""
    
    # Load both datasets
    enhanced = pd.read_csv('combined_data/enhanced_llm_data.csv')
    model_analysis = pd.read_csv('combined_data/model_analysis_dataset.csv')
    
    print("=" * 60)
    print("DATASET COMPARISON")
    print("=" * 60)
    
    print("\n1. BASIC STATISTICS")
    print("-" * 30)
    print(f"enhanced_llm_data.csv:")
    print(f"  Records: {len(enhanced):,}")
    print(f"  Columns: {len(enhanced.columns)}")
    print(f"  File size: {(len(enhanced) * len(enhanced.columns)):,} cells")
    
    print(f"\nmodel_analysis_dataset.csv:")
    print(f"  Records: {len(model_analysis):,}")
    print(f"  Columns: {len(model_analysis.columns)}")
    print(f"  File size: {(len(model_analysis) * len(model_analysis.columns)):,} cells")
    
    print("\n2. DATA TYPE BREAKDOWN")
    print("-" * 30)
    print("enhanced_llm_data.csv:")
    if 'data_type' in enhanced.columns:
        print(enhanced['data_type'].value_counts().to_string())
    else:
        print("  No data_type column")
    
    print(f"\nmodel_analysis_dataset.csv:")
    if 'data_type' in model_analysis.columns:
        print(model_analysis['data_type'].value_counts().to_string())
    else:
        print("  No data_type column")
    
    print("\n3. PURPOSE & FOCUS")
    print("-" * 30)
    print("enhanced_llm_data.csv:")
    print("  - Contains ALL data types (models, benchmarks, organizations, etc.)")
    print("  - Comprehensive dataset with deduplication and feature engineering")
    print("  - Suitable for general analysis across all entity types")
    
    print("\nmodel_analysis_dataset.csv:")
    print("  - Focused specifically on MODEL-BENCHMARK combinations")
    print("  - Optimized for performance analysis and model comparisons")
    print("  - Contains enriched features for model evaluation")
    
    print("\n4. KEY DIFFERENCES")
    print("-" * 30)
    
    # Check what's unique to each dataset
    enhanced_cols = set(enhanced.columns)
    model_analysis_cols = set(model_analysis.columns)
    
    unique_to_enhanced = enhanced_cols - model_analysis_cols
    unique_to_model_analysis = model_analysis_cols - enhanced_cols
    
    print(f"Columns unique to enhanced_llm_data.csv ({len(unique_to_enhanced)}):")
    for col in sorted(unique_to_enhanced):
        print(f"  - {col}")
    
    print(f"\nColumns unique to model_analysis_dataset.csv ({len(unique_to_model_analysis)}):")
    for col in sorted(unique_to_model_analysis):
        print(f"  - {col}")
    
    print("\n5. SAMPLE DATA STRUCTURE")
    print("-" * 30)
    print("enhanced_llm_data.csv sample columns:")
    print(enhanced.columns[:10].tolist())
    
    print("\nmodel_analysis_dataset.csv sample columns:")
    print(model_analysis.columns[:10].tolist())
    
    print("\n6. USE CASES")
    print("-" * 30)
    print("Use enhanced_llm_data.csv when you need:")
    print("  - Complete dataset with all entity types")
    print("  - Organization, provider, license information")
    print("  - Benchmark definitions and metadata")
    print("  - General data exploration")
    
    print("\nUse model_analysis_dataset.csv when you need:")
    print("  - Model performance analysis")
    print("  - Benchmark result comparisons")
    print("  - Model capability analysis")
    print("  - Performance-cost analysis")
    print("  - Model ranking and evaluation")
    
    print("\n7. FEATURE ENGINEERING DIFFERENCES")
    print("-" * 30)
    
    # Check for analysis-specific features
    analysis_features = [col for col in model_analysis.columns if any(x in col.lower() for x in ['performance', 'is_', 'tier', 'ratio', 'category'])]
    print(f"Analysis-specific features in model_analysis_dataset.csv ({len(analysis_features)}):")
    for feature in sorted(analysis_features)[:15]:  # Show first 15
        print(f"  - {feature}")
    if len(analysis_features) > 15:
        print(f"  ... and {len(analysis_features) - 15} more")
    
    print("\n8. RECORD COUNT BREAKDOWN")
    print("-" * 30)
    if 'data_type' in enhanced.columns:
        print("enhanced_llm_data.csv record breakdown:")
        for dtype, count in enhanced['data_type'].value_counts().items():
            print(f"  {dtype}: {count:,} records")
    
    print(f"\nmodel_analysis_dataset.csv:")
    print(f"  All records are model-benchmark combinations: {len(model_analysis):,} records")
    
    return enhanced, model_analysis

if __name__ == "__main__":
    enhanced, model_analysis = compare_datasets()
