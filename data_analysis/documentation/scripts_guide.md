# Scripts Reference Guide

## 🐍 Available Scripts

### 1. `combine_llm_data.py`
**Purpose**: Initial data extraction and combination from JSON files

**Usage**:
```bash
python combine_llm_data.py
```

**Input**: Raw JSON data from `data/` folder
**Output**: Basic CSV files in `combined_data/` folder
- `all_llm_data.csv` - Combined data
- `models.csv` - Model information
- `benchmark_results.csv` - Benchmark results
- `benchmark_definitions.csv` - Benchmark definitions
- `organizations.csv` - Organization data
- `providers.csv` - Provider data
- `licenses.csv` - License data

**Features**:
- Extracts data from all JSON files
- Flattens nested structures
- Creates separate CSV files for each data type

---

### 2. `create_enhanced_dataset.py`
**Purpose**: Creates enhanced dataset with deduplication and feature engineering

**Usage**:
```bash
python create_enhanced_dataset.py
```

**Input**: Basic CSV files from `combine_llm_data.py`
**Output**: `enhanced_llm_data.csv`

**Features**:
- Deduplicates model-benchmark combinations (removes 8 duplicates)
- Handles missing values with clean versions
- Creates license categories (Open & Permissive, Proprietary, Restricted/Community)
- Adds temporal features (release_year, release_month, release_year_month)
- Creates model size categories (Small, Medium, Large, Very Large, Undisclosed)
- Adds performance categories for benchmark results

---

### 3. `create_model_analysis_dataset.py`
**Purpose**: Creates specialized dataset for model performance analysis

**Usage**:
```bash
python create_model_analysis_dataset.py
```

**Input**: Enhanced dataset
**Output**: `model_analysis_dataset.csv`

**Features**:
- Focuses on model-benchmark combinations only
- Merges model and benchmark information
- Adds performance analysis features:
  - `performance_ratio`: Score relative to max possible score
  - `performance_tier`: Categorized performance levels
- Creates category flags: `is_code`, `is_math`, `is_reasoning`, etc.
- Adds model capability flags: `is_multimodal_model`, `is_large_model`, `is_open_source`
- Creates temporal analysis flags: `is_recent_model`, `is_2025_model`

---

### 4. `merge_datasets.py`
**Purpose**: Combines enhanced and model analysis datasets

**Usage**:
```bash
python merge_datasets.py
```

**Input**: `enhanced_llm_data.csv` and `model_analysis_dataset.csv`
**Output**: `merged_llm_dataset.csv`

**Features**:
- Uses enhanced dataset as base (2,755 records)
- Adds 64 analysis features from model analysis dataset
- Preserves all data types
- Populates analysis features for benchmark results
- Sets appropriate defaults for other record types

---

### 5. `analyze_dataset.py`
**Purpose**: Analyzes dataset for duplicates and missing values

**Usage**:
```bash
python analyze_dataset.py
```

**Input**: Any CSV dataset
**Output**: Console analysis report

**Features**:
- Identifies duplicate model-benchmark combinations
- Analyzes missing values in key columns
- Provides data quality summary
- Shows column statistics

---

### 6. `compare_datasets.py`
**Purpose**: Compares different datasets

**Usage**:
```bash
python compare_datasets.py
```

**Input**: Multiple CSV datasets
**Output**: Detailed comparison report

**Features**:
- Compares record counts and column counts
- Identifies common and unique columns
- Analyzes data type compatibility
- Provides merge recommendations

---

### 7. `simple_merge_analysis.py`
**Purpose**: Analyzes merge compatibility between datasets

**Usage**:
```bash
python simple_merge_analysis.py
```

**Input**: Two datasets to compare
**Output**: Merge compatibility report

**Features**:
- Checks for data type conflicts
- Analyzes record overlap
- Provides merge strategy recommendations
- Identifies potential issues

## 🔄 Complete Workflow

To regenerate all datasets from scratch:

```bash
# 1. Extract and combine raw data
python combine_llm_data.py

# 2. Create enhanced dataset
python create_enhanced_dataset.py

# 3. Create model analysis dataset
python create_model_analysis_dataset.py

# 4. Merge datasets
python merge_datasets.py

# 5. Analyze results (optional)
python analyze_dataset.py
python compare_datasets.py
```

## 📊 Script Dependencies

```
combine_llm_data.py
    ↓
create_enhanced_dataset.py
    ↓
create_model_analysis_dataset.py
    ↓
merge_datasets.py
```

## 🛠️ Customization

### Modifying Feature Engineering
Edit `create_enhanced_dataset.py` to:
- Add new license categories
- Modify model size thresholds
- Create additional temporal features

### Adding Analysis Features
Edit `create_model_analysis_dataset.py` to:
- Add new category flags
- Create additional performance metrics
- Modify model capability indicators

### Custom Merging
Edit `merge_datasets.py` to:
- Change merge strategy
- Add custom analysis features
- Modify default values

## ⚠️ Important Notes

1. **Run scripts in order**: Each script depends on outputs from previous scripts
2. **Check file paths**: Ensure scripts can find input files
3. **Memory usage**: Large datasets may require significant memory
4. **Error handling**: Scripts include error handling for missing files
5. **Backup data**: Original data is preserved in `data/` folder

## 🔧 Troubleshooting

### Common Issues
1. **File not found**: Check file paths and run scripts in correct order
2. **Memory error**: Use smaller datasets or increase system memory
3. **Unicode errors**: Scripts handle Unicode characters properly
4. **Data type conflicts**: Scripts include conflict resolution

### Debug Mode
Add debug prints to any script:
```python
print(f"Processing {len(df)} records...")
print(f"Columns: {df.columns.tolist()}")
```
