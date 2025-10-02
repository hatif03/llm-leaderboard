#!/usr/bin/env python3
"""
Merge enhanced_llm_data.csv and model_analysis_dataset.csv into a single comprehensive dataset.
"""

import pandas as pd
import numpy as np

def merge_datasets():
    """Merge the two datasets into a single comprehensive dataset."""
    
    print("Loading datasets...")
    enhanced = pd.read_csv('combined_data/enhanced_llm_data.csv')
    model_analysis = pd.read_csv('combined_data/model_analysis_dataset.csv')
    
    print(f"Enhanced dataset: {len(enhanced):,} records, {len(enhanced.columns)} columns")
    print(f"Model analysis dataset: {len(model_analysis):,} records, {len(model_analysis.columns)} columns")
    
    # Identify analysis features to add
    enhanced_cols = set(enhanced.columns)
    model_analysis_cols = set(model_analysis.columns)
    
    # Get analysis-specific features (not in enhanced dataset)
    analysis_features = model_analysis_cols - enhanced_cols
    print(f"\nAnalysis features to add: {len(analysis_features)}")
    
    # Create a copy of enhanced dataset
    merged_df = enhanced.copy()
    
    print("\nAdding analysis features...")
    
    # For each analysis feature, add it to the merged dataset
    for feature in analysis_features:
        if feature in model_analysis.columns:
            # Initialize the column with appropriate default values
            if model_analysis[feature].dtype == 'bool':
                merged_df[feature] = False
            elif model_analysis[feature].dtype in ['int64', 'float64']:
                merged_df[feature] = np.nan
            else:
                merged_df[feature] = None
    
    print("Merging analysis data for benchmark results...")
    
    # Create a mapping from model_analysis for benchmark results
    benchmark_analysis = model_analysis[model_analysis['data_type'] == 'benchmark_result'].copy()
    
    # Create a key for matching records
    if 'model_id' in merged_df.columns and 'benchmark_id' in merged_df.columns:
        # Add analysis features to benchmark result records
        for idx, row in merged_df[merged_df['data_type'] == 'benchmark_result'].iterrows():
            model_id = row['model_id']
            benchmark_id = row['benchmark_id']
            
            # Find matching record in model_analysis
            matching_record = benchmark_analysis[
                (benchmark_analysis['model_id'] == model_id) & 
                (benchmark_analysis['benchmark_id'] == benchmark_id)
            ]
            
            if not matching_record.empty:
                # Update analysis features
                for feature in analysis_features:
                    if feature in matching_record.columns:
                        merged_df.at[idx, feature] = matching_record[feature].iloc[0]
    
    # Save the merged dataset
    output_file = 'combined_data/merged_llm_dataset.csv'
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nMerged dataset saved to: {output_file}")
    print(f"Final dataset: {len(merged_df):,} records, {len(merged_df.columns)} columns")
    
    # Show summary
    print("\n" + "="*60)
    print("MERGED DATASET SUMMARY")
    print("="*60)
    
    print("\nData type breakdown:")
    for dtype, count in merged_df['data_type'].value_counts().items():
        print(f"  {dtype}: {count:,} records")
    
    print(f"\nTotal columns: {len(merged_df.columns)}")
    print(f"Original enhanced columns: {len(enhanced.columns)}")
    print(f"Added analysis columns: {len(analysis_features)}")
    
    # Show some analysis features
    print(f"\nSample analysis features added:")
    analysis_feature_list = sorted(list(analysis_features))
    for feature in analysis_feature_list[:15]:
        print(f"  - {feature}")
    if len(analysis_feature_list) > 15:
        print(f"  ... and {len(analysis_feature_list) - 15} more")
    
    # Check data quality
    print(f"\nData quality check:")
    benchmark_records = merged_df[merged_df['data_type'] == 'benchmark_result']
    print(f"Benchmark result records: {len(benchmark_records):,}")
    
    # Check how many have analysis features populated
    analysis_populated = 0
    for feature in analysis_feature_list[:5]:  # Check first 5 features
        if feature in merged_df.columns:
            non_null_count = merged_df[feature].notna().sum()
            print(f"  {feature}: {non_null_count:,} non-null values")
            if feature in ['is_code', 'is_math', 'is_reasoning']:  # Boolean features
                true_count = (merged_df[feature] == True).sum()
                print(f"    -> {true_count:,} True values")
    
    print(f"\n✓ Successfully merged datasets!")
    print(f"✓ All data types preserved")
    print(f"✓ Analysis features added for benchmark results")
    print(f"✓ Other record types have null/False values for analysis features")
    
    return merged_df

if __name__ == "__main__":
    merged_df = merge_datasets()
