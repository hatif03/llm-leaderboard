# Dataset Quick Reference Guide

## 🎯 Which Dataset to Use?

| Use Case | Recommended Dataset | Records | Columns | Key Features |
|----------|-------------------|---------|---------|--------------|
| **General Analysis** | `merged_llm_dataset.csv` | 2,755 | 130 | All data types + analysis features |
| **Model Performance** | `model_analysis_dataset.csv` | 2,190 | 110 | Focused on model-benchmark combinations |
| **Data Exploration** | `enhanced_llm_data.csv` | 2,755 | 66 | Clean data with basic enhancements |
| **Specific Data Types** | Individual CSV files | Varies | Varies | Focused on single entity type |

## 📊 Dataset Comparison

### merged_llm_dataset.csv ⭐ **MAIN DATASET**
- **Best for**: Comprehensive analysis, research, general use
- **Contains**: All data types + 64 analysis features
- **Unique features**: Complete dataset with analysis capabilities

### model_analysis_dataset.csv
- **Best for**: Model performance analysis, benchmarking
- **Contains**: Only model-benchmark combinations
- **Unique features**: Performance ratios, category flags, model capabilities

### enhanced_llm_data.csv
- **Best for**: Data exploration, understanding structure
- **Contains**: All data types with basic enhancements
- **Unique features**: Clean, deduplicated data

## 🔍 Key Columns Reference

### Performance Analysis
- `performance_ratio`: Score relative to maximum possible score
- `performance_tier`: Categorized performance level
- `score`: Raw benchmark score
- `normalized_score`: Normalized benchmark score

### Model Capabilities
- `is_multimodal_model`: Supports multiple modalities
- `is_large_model`: >70B parameters
- `is_open_source`: Open source license
- `is_recent_model`: Released 2024+
- `is_2025_model`: Released in 2025

### Benchmark Categories
- `is_code`: Code-related benchmarks
- `is_math`: Math-related benchmarks
- `is_reasoning`: Reasoning benchmarks
- `is_safety`: Safety benchmarks
- `is_vision`: Vision benchmarks
- `is_language`: Language benchmarks

### Model Information
- `model_id`: Unique model identifier
- `name`: Model display name
- `organization_name`: Creating organization
- `param_count_clean`: Parameter count (cleaned)
- `license_type`: License category
- `release_year`: Release year

### Benchmark Information
- `benchmark_id`: Unique benchmark identifier
- `benchmark_name`: Benchmark display name
- `categories_benchmark`: Benchmark categories
- `modality_benchmark`: Input modality
- `max_score_benchmark`: Maximum possible score

## 🚀 Quick Analysis Examples

### Top 10 Models by Performance
```python
import pandas as pd
df = pd.read_csv('datasets/merged_llm_dataset.csv')

# Filter benchmark results and calculate average performance
top_models = (df[df['data_type'] == 'benchmark_result']
              .groupby(['model_id', 'name'])['performance_ratio']
              .mean()
              .sort_values(ascending=False)
              .head(10))
```

### Performance by Category
```python
# Math performance
math_perf = df[df['is_math'] == True]['performance_ratio'].mean()

# Code performance  
code_perf = df[df['is_code'] == True]['performance_ratio'].mean()

# Safety performance
safety_perf = df[df['is_safety'] == True]['performance_ratio'].mean()
```

### Open Source vs Proprietary
```python
# Open source performance
open_source = df[df['is_open_source'] == True]['performance_ratio'].mean()

# Proprietary performance
proprietary = df[df['license_type'] == 'Proprietary']['performance_ratio'].mean()
```

### Model Size Analysis
```python
# Performance by model size
size_performance = (df[df['data_type'] == 'benchmark_result']
                   .groupby('model_size_category')['performance_ratio']
                   .mean()
                   .sort_values(ascending=False))
```

### Temporal Analysis
```python
# Performance by release year
yearly_performance = (df[df['data_type'] == 'benchmark_result']
                     .groupby('release_year')['performance_ratio']
                     .mean()
                     .sort_index())
```

## 📈 Data Quality Notes

- **Missing Values**: Handled with appropriate defaults (0 for scores, False for flags)
- **Duplicates**: Removed (8 duplicate model-benchmark combinations)
- **Data Types**: Consistent across all datasets
- **Encoding**: UTF-8, compatible with all major tools

## 🔧 Troubleshooting

### Common Issues
1. **Memory**: Use `model_analysis_dataset.csv` for large-scale analysis (smaller file)
2. **Missing Data**: Check `*_disclosed` columns for data availability flags
3. **Performance**: Use category flags (`is_*`) for efficient filtering

### Data Validation
```python
# Check data quality
print(f"Total records: {len(df):,}")
print(f"Missing values: {df.isnull().sum().sum():,}")
print(f"Data types: {df['data_type'].value_counts().to_dict()}")
```
