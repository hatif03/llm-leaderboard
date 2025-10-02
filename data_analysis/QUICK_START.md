# 🚀 Quick Start Guide

## 📁 What's in this folder?

This folder contains everything you need for LLM data analysis:

- **`datasets/`** - All CSV files ready for analysis
- **`scripts/`** - Python scripts for data processing
- **`documentation/`** - Detailed guides and references

## 🎯 Start Here

### For Immediate Analysis
```python
import pandas as pd

# Load the main dataset
df = pd.read_csv('datasets/merged_llm_dataset.csv')

# Quick overview
print(f"Records: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(df['data_type'].value_counts())
```

### For Model Performance Analysis
```python
# Load specialized dataset
df = pd.read_csv('datasets/model_analysis_dataset.csv')

# Top performing models
top_models = (df.groupby('model_id')['performance_ratio']
              .mean()
              .sort_values(ascending=False)
              .head(10))
```

## 📊 Key Datasets

| File | Records | Columns | Best For |
|------|---------|---------|----------|
| `merged_llm_dataset.csv` ⭐ | 2,755 | 130 | **Everything** |
| `model_analysis_dataset.csv` | 2,190 | 110 | Model performance |
| `enhanced_llm_data.csv` | 2,755 | 66 | Data exploration |

## 🔍 Quick Examples

### Top 10 Models
```python
df = pd.read_csv('datasets/merged_llm_dataset.csv')
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
```

### Open Source vs Proprietary
```python
open_source = df[df['is_open_source'] == True]['performance_ratio'].mean()
proprietary = df[df['license_type'] == 'Proprietary']['performance_ratio'].mean()
```

## 📚 Documentation

- **`README.md`** - Complete project overview
- **`documentation/dataset_guide.md`** - Detailed dataset reference
- **`documentation/scripts_guide.md`** - Script documentation

## 🛠️ Regenerate Data

If you need to recreate the datasets:
```bash
cd scripts
python combine_llm_data.py
python create_enhanced_dataset.py
python create_model_analysis_dataset.py
python merge_datasets.py
```

## ✅ Ready to Go!

Everything is organized and ready for your analysis. Start with `merged_llm_dataset.csv` for the most comprehensive dataset!
