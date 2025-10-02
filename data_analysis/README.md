# LLM Data Analysis Project

This folder contains all scripts, datasets, and documentation for the LLM leaderboard data analysis project.

## 📁 Folder Structure

```
llm_data_analysis/
├── scripts/                    # Python scripts for data processing
├── datasets/                   # All generated CSV datasets
├── documentation/              # Documentation and guides
└── README.md                  # This file
```

## 🐍 Scripts (`scripts/`)

### Core Data Processing Scripts

1. **`combine_llm_data.py`**
   - **Purpose**: Initial data extraction and combination
   - **Input**: Raw JSON data from `data/` folder
   - **Output**: Basic CSV files for each data type
   - **Features**: Extracts models, benchmarks, organizations, providers, licenses

2. **`create_enhanced_dataset.py`**
   - **Purpose**: Creates enhanced dataset with deduplication and feature engineering
   - **Input**: Basic CSV files from `combine_llm_data.py`
   - **Output**: `enhanced_llm_data.csv`
   - **Features**: 
     - Deduplicates model-benchmark combinations
     - Handles missing values
     - Creates license categories, temporal features, model size categories

3. **`create_model_analysis_dataset.py`**
   - **Purpose**: Creates specialized dataset for model performance analysis
   - **Input**: Enhanced dataset
   - **Output**: `model_analysis_dataset.csv`
   - **Features**:
     - Focuses on model-benchmark combinations
     - Adds performance analysis features
     - Creates category flags and model capability indicators

4. **`merge_datasets.py`**
   - **Purpose**: Combines enhanced and model analysis datasets
   - **Input**: Both enhanced and model analysis datasets
   - **Output**: `merged_llm_dataset.csv`
   - **Features**: Single comprehensive dataset with all features

### Analysis Scripts

5. **`analyze_dataset.py`**
   - **Purpose**: Analyzes dataset for duplicates and missing values
   - **Input**: Any CSV dataset
   - **Output**: Console analysis report

6. **`compare_datasets.py`**
   - **Purpose**: Compares different datasets
   - **Input**: Multiple CSV datasets
   - **Output**: Detailed comparison report

7. **`simple_merge_analysis.py`**
   - **Purpose**: Analyzes merge compatibility between datasets
   - **Input**: Two datasets to compare
   - **Output**: Merge compatibility report

## 📊 Datasets (`datasets/`)

### Main Datasets

1. **`merged_llm_dataset.csv`** ⭐ **RECOMMENDED**
   - **Records**: 2,755
   - **Columns**: 130
   - **Description**: Complete dataset with all data types and analysis features
   - **Use Case**: General analysis and performance evaluation

2. **`enhanced_llm_data.csv`**
   - **Records**: 2,755
   - **Columns**: 66
   - **Description**: Enhanced dataset with deduplication and basic feature engineering
   - **Use Case**: General data exploration

3. **`model_analysis_dataset.csv`**
   - **Records**: 2,190
   - **Columns**: 110
   - **Description**: Specialized dataset for model performance analysis
   - **Use Case**: Model comparison and performance evaluation

### Individual Data Type Files

4. **`all_llm_data.csv`** - Original combined data (2,763 records, 53 columns)
5. **`models.csv`** - Model information (165 records)
6. **`benchmark_results.csv`** - Benchmark performance results (2,198 records)
7. **`benchmark_definitions.csv`** - Benchmark definitions (340 records)
8. **`organizations.csv`** - Organization information (17 records)
9. **`providers.csv`** - Provider information (20 records)
10. **`licenses.csv`** - License information (23 records)

## 🚀 Quick Start

### For General Analysis
```python
import pandas as pd
df = pd.read_csv('datasets/merged_llm_dataset.csv')
```

### For Model Performance Analysis
```python
import pandas as pd
df = pd.read_csv('datasets/model_analysis_dataset.csv')
```

### For Data Exploration
```python
import pandas as pd
df = pd.read_csv('datasets/enhanced_llm_data.csv')
```

## 🔧 Key Features

### Data Quality Improvements
- ✅ **Deduplication**: Removed 8 duplicate model-benchmark combinations
- ✅ **Missing Value Handling**: Created clean versions of numerical columns
- ✅ **Feature Engineering**: Added 64 analysis features

### Analysis Features
- **Performance Metrics**: `performance_ratio`, `performance_tier`
- **Category Flags**: `is_code`, `is_math`, `is_reasoning`, `is_safety`, etc.
- **Model Capabilities**: `is_multimodal_model`, `is_large_model`, `is_open_source`
- **Temporal Analysis**: `is_recent_model`, `is_2025_model`, `release_year_month`
- **License Classification**: `Open & Permissive`, `Proprietary`, `Restricted/Community`

### Data Types Included
- **Models**: 165 AI/ML models with metadata
- **Benchmark Results**: 2,190 performance scores
- **Benchmark Definitions**: 340 benchmark descriptions
- **Organizations**: 17 companies/organizations
- **Providers**: 20 API providers
- **Licenses**: 23 different license types

## 📈 Usage Examples

### Model Performance Analysis
```python
# Top performing models
top_models = df[df['data_type'] == 'benchmark_result'].groupby('model_id')['performance_ratio'].mean().sort_values(ascending=False)

# Performance by category
math_performance = df[df['is_math'] == True]['performance_ratio'].mean()
code_performance = df[df['is_code'] == True]['performance_ratio'].mean()
```

### License Analysis
```python
# Open source vs proprietary performance
open_source_perf = df[df['is_open_source'] == True]['performance_ratio'].mean()
proprietary_perf = df[df['license_type'] == 'Proprietary']['performance_ratio'].mean()
```

### Temporal Analysis
```python
# Recent model performance
recent_models = df[df['is_recent_model'] == True]['performance_ratio'].mean()
```

## 🎯 Recommended Workflow

1. **Start with**: `merged_llm_dataset.csv` for comprehensive analysis
2. **Use**: `model_analysis_dataset.csv` for detailed model performance analysis
3. **Explore**: Individual CSV files for specific data types
4. **Run**: Analysis scripts for custom processing

## 📝 Notes

- All datasets are derived from the original LLM leaderboard JSON data
- Data has been cleaned, deduplicated, and enhanced for analysis
- Missing values are handled gracefully with appropriate defaults
- All scripts are self-contained and can be run independently

## 🔄 Regenerating Datasets

To regenerate all datasets from scratch:
```bash
cd scripts
python combine_llm_data.py
python create_enhanced_dataset.py
python create_model_analysis_dataset.py
python merge_datasets.py
```

---

**Created**: February 2025  
**Data Source**: LLM Leaderboard Repository  
**Total Records**: 2,755 (merged dataset)  
**Total Features**: 130 (merged dataset)
