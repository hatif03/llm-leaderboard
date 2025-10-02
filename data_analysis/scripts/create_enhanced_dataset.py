#!/usr/bin/env python3
"""
Script to create an enhanced dataset with deduplication, missing value handling, and feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

def categorize_license(license_id):
    """Categorize license into open/permissive, proprietary, or restricted."""
    if pd.isna(license_id):
        return 'Unknown'
    
    license_id = str(license_id).lower()
    
    # Open & Permissive licenses
    open_licenses = ['apache_2_0', 'mit', 'mit_license', 'modified_mit_license', 'creative_commons_attribution_4_0_license']
    
    # Proprietary licenses
    proprietary_licenses = ['proprietary']
    
    # Restricted/Community licenses
    restricted_licenses = ['llama_3_1_community_license', 'llama_3_2_community_license', 'llama_3_3_community_license_agreement', 
                          'llama_4_community_license_agreement', 'llama3_2', 'gemma', 'deepseek', 'qwen', 'tongyi_qianwen',
                          'mistral_research_license', 'mistral_research_license_(mrl)_for_research;_mistral_commercial_license_for_commercial_use',
                          'jamba_open_model_license', 'mnpl_0_1', 'health_ai_developer_foundations_terms_of_use']
    
    if license_id in open_licenses:
        return 'Open & Permissive'
    elif license_id in proprietary_licenses:
        return 'Proprietary'
    elif license_id in restricted_licenses:
        return 'Restricted/Community'
    else:
        return 'Other'

def extract_temporal_features(release_date):
    """Extract year and month from release date."""
    if pd.isna(release_date):
        return None, None, None
    
    try:
        # Handle different date formats
        if isinstance(release_date, str):
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%Y-%m', '%Y']:
                try:
                    date_obj = datetime.strptime(release_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None, None, None
        else:
            date_obj = release_date
        
        year = date_obj.year
        month = date_obj.month
        year_month = f"{year}-{month:02d}"
        
        return year, month, year_month
    except:
        return None, None, None

def calculate_cost_io_metric(input_cost, output_cost, input_ratio=0.25, output_ratio=0.75):
    """Calculate unified cost metric using weighted average."""
    if pd.isna(input_cost) and pd.isna(output_cost):
        return None
    elif pd.isna(input_cost):
        return output_cost * output_ratio
    elif pd.isna(output_cost):
        return input_cost * input_ratio
    else:
        return input_cost * input_ratio + output_cost * output_ratio

def deduplicate_benchmark_results(df):
    """Deduplicate benchmark results by taking the best score for each model-benchmark combination."""
    print("Deduplicating benchmark results...")
    
    # Focus on benchmark results
    benchmark_results = df[df['data_type'] == 'benchmark_result'].copy()
    other_data = df[df['data_type'] != 'benchmark_result'].copy()
    
    print(f"Original benchmark results: {len(benchmark_results)}")
    
    # Group by model_id and benchmark_id, take the maximum score
    deduplicated = benchmark_results.groupby(['model_id', 'benchmark_id']).agg({
        'score': 'max',
        'normalized_score': 'max',
        'model_benchmark_id': 'first',
        'is_self_reported': 'any',  # True if any is self-reported
        'self_reported_source_link': 'first',
        'verified_by_llmstats': 'any',  # True if any is verified
        'analysis_method': 'first',
        'verification_provider_id': 'first',
        'verification_hardware': 'first',
        'verification_date': 'first',
        'verification_notes': 'first',
        'created_at': 'first',
        'updated_at': 'first',
        'benchmark_name': 'first',
        'organization_id': 'first',
        'provider_id': 'first'  # Keep first provider
    }).reset_index()
    
    # Add back other columns that were lost in aggregation
    deduplicated['data_type'] = 'benchmark_result'
    
    print(f"Deduplicated benchmark results: {len(deduplicated)}")
    print(f"Removed {len(benchmark_results) - len(deduplicated)} duplicate entries")
    
    # Combine with other data
    enhanced_df = pd.concat([other_data, deduplicated], ignore_index=True)
    
    return enhanced_df

def handle_missing_values(df):
    """Handle missing values in key columns."""
    print("Handling missing values...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle param_count - flag as undisclosed if missing or zero
    df_clean['param_count_disclosed'] = ~df_clean['param_count'].isna()
    df_clean['param_count_clean'] = df_clean['param_count'].fillna(0)
    
    # Handle training_tokens - similar approach
    df_clean['training_tokens_disclosed'] = ~df_clean['training_tokens'].isna()
    df_clean['training_tokens_clean'] = df_clean['training_tokens'].fillna(0)
    
    # Handle scores - fill with 0 for missing values
    df_clean['score_clean'] = df_clean['score'].fillna(0)
    df_clean['normalized_score_clean'] = df_clean['normalized_score'].fillna(0)
    
    return df_clean

def engineer_features(df):
    """Create engineered features."""
    print("Engineering features...")
    
    df_enhanced = df.copy()
    
    # 1. License categorization
    print("  - Creating license_type feature...")
    df_enhanced['license_type'] = df_enhanced['license_id'].apply(categorize_license)
    
    # 2. Temporal features
    print("  - Creating temporal features...")
    temporal_features = df_enhanced['release_date'].apply(extract_temporal_features)
    df_enhanced['release_year'] = [x[0] if x[0] is not None else None for x in temporal_features]
    df_enhanced['release_month'] = [x[1] if x[1] is not None else None for x in temporal_features]
    df_enhanced['release_year_month'] = [x[2] if x[2] is not None else None for x in temporal_features]
    
    # 3. Cost metrics (if cost columns exist)
    print("  - Creating cost metrics...")
    cost_columns = [col for col in df_enhanced.columns if 'cost' in col.lower() or 'cents' in col.lower()]
    print(f"    Found cost columns: {cost_columns}")
    
    # Look for input/output cost columns
    input_cost_cols = [col for col in cost_columns if 'input' in col.lower()]
    output_cost_cols = [col for col in cost_columns if 'output' in col.lower()]
    
    if input_cost_cols and output_cost_cols:
        input_cost_col = input_cost_cols[0]
        output_cost_col = output_cost_cols[0]
        df_enhanced['cost_per_million_io_tokens'] = df_enhanced.apply(
            lambda row: calculate_cost_io_metric(row[input_cost_col], row[output_cost_col]), 
            axis=1
        )
    else:
        print("    No input/output cost columns found, skipping cost calculation")
        df_enhanced['cost_per_million_io_tokens'] = None
    
    # 4. Model size categories
    print("  - Creating model size categories...")
    def categorize_model_size(param_count):
        if pd.isna(param_count) or param_count == 0:
            return 'Undisclosed'
        elif param_count < 1:
            return 'Small (<1B)'
        elif param_count < 7:
            return 'Medium (1-7B)'
        elif param_count < 70:
            return 'Large (7-70B)'
        else:
            return 'Very Large (>70B)'
    
    df_enhanced['model_size_category'] = df_enhanced['param_count'].apply(categorize_model_size)
    
    # 5. Performance categories for benchmark results
    print("  - Creating performance categories...")
    def categorize_performance(score):
        if pd.isna(score):
            return 'No Score'
        elif score >= 0.9:
            return 'Excellent (90%+)'
        elif score >= 0.8:
            return 'Very Good (80-89%)'
        elif score >= 0.7:
            return 'Good (70-79%)'
        elif score >= 0.6:
            return 'Fair (60-69%)'
        else:
            return 'Poor (<60%)'
    
    df_enhanced['performance_category'] = df_enhanced['score'].apply(categorize_performance)
    
    return df_enhanced

def create_enhanced_dataset():
    """Create the enhanced dataset with all improvements."""
    print("Creating enhanced LLM dataset...")
    
    # Load the original dataset
    print("Loading original dataset...")
    df = pd.read_csv('combined_data/all_llm_data.csv')
    print(f"Original dataset: {len(df)} records, {len(df.columns)} columns")
    
    # Step 1: Deduplicate benchmark results
    df_dedup = deduplicate_benchmark_results(df)
    
    # Step 2: Handle missing values
    df_clean = handle_missing_values(df_dedup)
    
    # Step 3: Engineer features
    df_enhanced = engineer_features(df_clean)
    
    # Save enhanced dataset
    output_file = 'combined_data/enhanced_llm_data.csv'
    df_enhanced.to_csv(output_file, index=False)
    print(f"\nEnhanced dataset saved to: {output_file}")
    print(f"Enhanced dataset: {len(df_enhanced)} records, {len(df_enhanced.columns)} columns")
    
    # Create summary statistics
    print("\n=== ENHANCED DATASET SUMMARY ===")
    
    # Data type breakdown
    print("\nData type breakdown:")
    print(df_enhanced['data_type'].value_counts())
    
    # License type breakdown
    print("\nLicense type breakdown:")
    print(df_enhanced['license_type'].value_counts())
    
    # Model size breakdown
    print("\nModel size breakdown:")
    print(df_enhanced['model_size_category'].value_counts())
    
    # Performance breakdown (for benchmark results)
    benchmark_results = df_enhanced[df_enhanced['data_type'] == 'benchmark_result']
    if len(benchmark_results) > 0:
        print("\nPerformance category breakdown (benchmark results):")
        print(benchmark_results['performance_category'].value_counts())
    
    # Temporal analysis
    release_years = df_enhanced['release_year'].dropna()
    if len(release_years) > 0:
        print(f"\nRelease year range: {release_years.min()} - {release_years.max()}")
        print("Release year distribution:")
        print(release_years.value_counts().sort_index())
    
    # Missing value summary
    print("\nMissing value summary for key columns:")
    key_columns = ['param_count', 'training_tokens', 'score', 'license_type', 'release_year']
    for col in key_columns:
        if col in df_enhanced.columns:
            missing = df_enhanced[col].isnull().sum()
            print(f"  {col}: {missing} missing ({missing/len(df_enhanced)*100:.1f}%)")
    
    return df_enhanced

if __name__ == "__main__":
    enhanced_df = create_enhanced_dataset()
