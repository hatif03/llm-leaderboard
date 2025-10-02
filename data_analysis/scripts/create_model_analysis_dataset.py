#!/usr/bin/env python3
"""
Create a specialized dataset for high-level model analysis with deduplication and feature engineering.
This focuses on model-benchmark combinations for performance analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_model_analysis_dataset():
    """Create a dataset optimized for model performance analysis."""
    
    print("Creating model analysis dataset...")
    
    # Load the enhanced dataset
    df = pd.read_csv('combined_data/enhanced_llm_data.csv')
    
    # Focus on models and benchmark results
    models = df[df['data_type'] == 'model'].copy()
    benchmark_results = df[df['data_type'] == 'benchmark_result'].copy()
    
    print(f"Models: {len(models)}")
    print(f"Benchmark results: {len(benchmark_results)}")
    
    # Create a comprehensive model-benchmark dataset
    print("\nCreating model-benchmark analysis dataset...")
    
    # Merge benchmark results with model information
    model_benchmark_df = benchmark_results.merge(
        models[['model_id', 'name', 'organization_id', 'organization_name', 
                'param_count', 'param_count_clean', 'param_count_disclosed',
                'training_tokens_clean', 'training_tokens_disclosed',
                'multimodal', 'license_id', 'license_type', 'release_date', 
                'release_year', 'release_month', 'release_year_month',
                'model_size_category', 'description']], 
        on='model_id', 
        how='left'
    )
    
    print(f"Model-benchmark combinations: {len(model_benchmark_df)}")
    
    # Add benchmark information
    benchmark_definitions = df[df['data_type'] == 'benchmark_definition'].copy()
    print(f"Benchmark definitions found: {len(benchmark_definitions)}")
    
    if len(benchmark_definitions) > 0:
        # Check what columns are available
        available_cols = ['benchmark_id', 'name', 'categories', 'modality', 
                         'multilingual', 'max_score', 'language', 'description']
        existing_cols = [col for col in available_cols if col in benchmark_definitions.columns]
        print(f"Available benchmark columns: {existing_cols}")
        
        model_benchmark_df = model_benchmark_df.merge(
            benchmark_definitions[existing_cols], 
            on='benchmark_id', 
            how='left',
            suffixes=('_model', '_benchmark')
        )
        print(f"After merge, columns: {model_benchmark_df.columns.tolist()}")
    else:
        # If no benchmark definitions, create dummy columns
        model_benchmark_df['name_benchmark'] = model_benchmark_df['benchmark_name']
        model_benchmark_df['categories'] = None
        model_benchmark_df['modality'] = 'text'
        model_benchmark_df['multilingual'] = False
        model_benchmark_df['max_score'] = 1.0
        model_benchmark_df['language'] = 'en'
        model_benchmark_df['description_benchmark'] = None
    
    # Create additional analysis features
    print("Creating analysis features...")
    
    # 1. Performance relative to maximum possible score
    # Use the benchmark max_score if available, otherwise default to 1.0
    max_score_col = 'max_score_benchmark' if 'max_score_benchmark' in model_benchmark_df.columns else 'max_score'
    if max_score_col not in model_benchmark_df.columns:
        model_benchmark_df['max_score'] = 1.0
        max_score_col = 'max_score'
    
    model_benchmark_df['performance_ratio'] = model_benchmark_df['score'] / model_benchmark_df[max_score_col]
    
    # 2. Benchmark category flags
    benchmark_categories = ['general', 'code', 'math', 'reasoning', 'language', 'multimodal', 
                           'safety', 'long_context', 'roleplay', 'agents', 'factuality', 'vision']
    
    # Use the benchmark categories column
    categories_col = 'categories_benchmark' if 'categories_benchmark' in model_benchmark_df.columns else 'categories'
    if categories_col not in model_benchmark_df.columns:
        model_benchmark_df['categories'] = None
        categories_col = 'categories'
    
    for category in benchmark_categories:
        model_benchmark_df[f'is_{category}'] = model_benchmark_df[categories_col].apply(
            lambda x: category in str(x) if pd.notna(x) else False
        )
    
    # 3. Model capability flags
    # Use the model multimodal column
    multimodal_col = 'multimodal_x' if 'multimodal_x' in model_benchmark_df.columns else 'multimodal'
    model_benchmark_df['is_multimodal_model'] = model_benchmark_df[multimodal_col].fillna(False)
    
    # Use the model param_count_clean column
    param_col = 'param_count_clean_x' if 'param_count_clean_x' in model_benchmark_df.columns else 'param_count_clean'
    model_benchmark_df['is_large_model'] = model_benchmark_df[param_col] >= 70
    
    # Use the model license_type column
    license_col = 'license_type_x' if 'license_type_x' in model_benchmark_df.columns else 'license_type'
    model_benchmark_df['is_open_source'] = model_benchmark_df[license_col].isin(['Open & Permissive'])
    
    # 4. Temporal features
    # Use the model release_year column
    release_year_col = 'release_year_x' if 'release_year_x' in model_benchmark_df.columns else 'release_year'
    model_benchmark_df['is_recent_model'] = model_benchmark_df[release_year_col] >= 2024
    model_benchmark_df['is_2025_model'] = model_benchmark_df[release_year_col] == 2025
    
    # 5. Performance tiers
    def get_performance_tier(score, max_score):
        if pd.isna(score) or pd.isna(max_score) or max_score == 0:
            return 'No Score'
        
        ratio = score / max_score
        if ratio >= 0.95:
            return 'SOTA (95%+)'
        elif ratio >= 0.9:
            return 'Excellent (90-95%)'
        elif ratio >= 0.8:
            return 'Very Good (80-90%)'
        elif ratio >= 0.7:
            return 'Good (70-80%)'
        elif ratio >= 0.6:
            return 'Fair (60-70%)'
        else:
            return 'Poor (<60%)'
    
    model_benchmark_df['performance_tier'] = model_benchmark_df.apply(
        lambda row: get_performance_tier(row['score'], row[max_score_col]), axis=1
    )
    
    # 6. Model family analysis
    model_benchmark_df['model_family'] = model_benchmark_df['model_id'].apply(
        lambda x: x.split('-')[0] if pd.notna(x) else 'Unknown'
    )
    
    # Save the model analysis dataset
    output_file = 'combined_data/model_analysis_dataset.csv'
    model_benchmark_df.to_csv(output_file, index=False)
    
    print(f"\nModel analysis dataset saved to: {output_file}")
    print(f"Records: {len(model_benchmark_df)}")
    print(f"Columns: {len(model_benchmark_df.columns)}")
    
    # Create summary statistics
    print("\n=== MODEL ANALYSIS DATASET SUMMARY ===")
    
    # Top performing models by average score
    print("\nTop 10 models by average performance:")
    # Use the correct name column
    name_col = 'name_x' if 'name_x' in model_benchmark_df.columns else 'name'
    model_performance = model_benchmark_df.groupby(['model_id', name_col]).agg({
        'score': ['mean', 'count'],
        'performance_ratio': 'mean',
        'organization_name_x': 'first',
        'param_count_clean_x': 'first',
        'license_type_x': 'first'
    }).round(3)
    
    model_performance.columns = ['avg_score', 'benchmark_count', 'avg_performance_ratio', 
                                'organization', 'param_count', 'license_type']
    model_performance = model_performance[model_performance['benchmark_count'] >= 5]  # At least 5 benchmarks
    top_models = model_performance.sort_values('avg_performance_ratio', ascending=False).head(10)
    print(top_models[['organization', 'avg_performance_ratio', 'benchmark_count', 'license_type']])
    
    # Performance by organization
    print("\nPerformance by organization (avg performance ratio):")
    org_performance = model_benchmark_df.groupby('organization_name_x').agg({
        'performance_ratio': ['mean', 'count'],
        'model_id': 'nunique'
    }).round(3)
    org_performance.columns = ['avg_performance_ratio', 'total_benchmarks', 'unique_models']
    org_performance = org_performance[org_performance['total_benchmarks'] >= 10]
    print(org_performance.sort_values('avg_performance_ratio', ascending=False))
    
    # Performance by benchmark category
    print("\nPerformance by benchmark category:")
    category_performance = {}
    for category in benchmark_categories:
        cat_data = model_benchmark_df[model_benchmark_df[f'is_{category}']]
        if len(cat_data) > 0:
            avg_perf = cat_data['performance_ratio'].mean()
            count = len(cat_data)
            category_performance[category] = {'avg_performance': avg_perf, 'count': count}
    
    cat_df = pd.DataFrame(category_performance).T.sort_values('avg_performance', ascending=False)
    print(cat_df)
    
    # License type analysis
    print("\nPerformance by license type:")
    license_performance = model_benchmark_df.groupby('license_type_x').agg({
        'performance_ratio': ['mean', 'count'],
        'model_id': 'nunique'
    }).round(3)
    license_performance.columns = ['avg_performance_ratio', 'total_benchmarks', 'unique_models']
    license_performance = license_performance[license_performance['total_benchmarks'] >= 10]
    print(license_performance.sort_values('avg_performance_ratio', ascending=False))
    
    # Model size analysis
    print("\nPerformance by model size:")
    size_performance = model_benchmark_df.groupby('model_size_category_x').agg({
        'performance_ratio': ['mean', 'count'],
        'model_id': 'nunique'
    }).round(3)
    size_performance.columns = ['avg_performance_ratio', 'total_benchmarks', 'unique_models']
    print(size_performance.sort_values('avg_performance_ratio', ascending=False))
    
    # Temporal analysis
    print("\nPerformance by release year:")
    temporal_performance = model_benchmark_df.groupby('release_year_x').agg({
        'performance_ratio': ['mean', 'count'],
        'model_id': 'nunique'
    }).round(3)
    temporal_performance.columns = ['avg_performance_ratio', 'total_benchmarks', 'unique_models']
    temporal_performance = temporal_performance[temporal_performance['total_benchmarks'] >= 5]
    print(temporal_performance.sort_index())
    
    return model_benchmark_df

if __name__ == "__main__":
    analysis_df = create_model_analysis_dataset()
