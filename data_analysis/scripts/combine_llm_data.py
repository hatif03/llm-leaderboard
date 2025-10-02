#!/usr/bin/env python3
"""
Script to combine all LLM data from the repository into a single CSV file.
This script extracts data from:
- Model information (organizations/*/models/*/model.json)
- Benchmark results (organizations/*/models/*/benchmarks.json)
- Benchmark definitions (benchmarks/*.json)
- Organization information (organizations/*/organization.json)
- Provider information (providers/*/provider.json)
- License information (licenses/*.json)
"""

import json
import csv
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional

def load_json_file(file_path: str) -> Optional[Dict]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_all_json_files(directory: str, pattern: str) -> List[str]:
    """Get all JSON files matching the pattern in the directory."""
    search_pattern = os.path.join(directory, pattern)
    return glob.glob(search_pattern, recursive=True)

def extract_models_data(data_dir: str) -> List[Dict]:
    """Extract all model data from organization directories."""
    models_data = []
    organizations_dir = os.path.join(data_dir, "organizations")
    
    if not os.path.exists(organizations_dir):
        print(f"Organizations directory not found: {organizations_dir}")
        return models_data
    
    for org_dir in os.listdir(organizations_dir):
        org_path = os.path.join(organizations_dir, org_dir)
        if not os.path.isdir(org_path):
            continue
            
        # Load organization info
        org_file = os.path.join(org_path, "organization.json")
        org_data = load_json_file(org_file)
        
        models_dir = os.path.join(org_path, "models")
        if not os.path.exists(models_dir):
            continue
            
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            # Load model data
            model_file = os.path.join(model_path, "model.json")
            model_data = load_json_file(model_file)
            
            if model_data and org_data:
                # Combine model and organization data
                combined_data = {
                    **model_data,
                    "organization_name": org_data.get("name", ""),
                    "organization_website": org_data.get("website", ""),
                    "organization_description": org_data.get("description", ""),
                    "organization_country": org_data.get("country", ""),
                    "data_type": "model"
                }
                models_data.append(combined_data)
    
    return models_data

def extract_benchmark_results(data_dir: str) -> List[Dict]:
    """Extract all benchmark results from model directories."""
    benchmark_results = []
    organizations_dir = os.path.join(data_dir, "organizations")
    
    if not os.path.exists(organizations_dir):
        return benchmark_results
    
    for org_dir in os.listdir(organizations_dir):
        org_path = os.path.join(organizations_dir, org_dir)
        if not os.path.isdir(org_path):
            continue
            
        models_dir = os.path.join(org_path, "models")
        if not os.path.exists(models_dir):
            continue
            
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
                
            # Load benchmark results
            benchmarks_file = os.path.join(model_path, "benchmarks.json")
            benchmarks_data = load_json_file(benchmarks_file)
            
            if benchmarks_data and isinstance(benchmarks_data, list):
                for result in benchmarks_data:
                    result["organization_id"] = org_dir
                    result["data_type"] = "benchmark_result"
                    benchmark_results.append(result)
    
    return benchmark_results

def extract_benchmark_definitions(data_dir: str) -> List[Dict]:
    """Extract all benchmark definitions."""
    benchmarks_data = []
    benchmarks_dir = os.path.join(data_dir, "benchmarks")
    
    if not os.path.exists(benchmarks_dir):
        return benchmarks_data
    
    for benchmark_file in os.listdir(benchmarks_dir):
        if benchmark_file.endswith('.json'):
            file_path = os.path.join(benchmarks_dir, benchmark_file)
            benchmark_data = load_json_file(file_path)
            
            if benchmark_data:
                benchmark_data["data_type"] = "benchmark_definition"
                benchmarks_data.append(benchmark_data)
    
    return benchmarks_data

def extract_organizations_data(data_dir: str) -> List[Dict]:
    """Extract all organization data."""
    organizations_data = []
    organizations_dir = os.path.join(data_dir, "organizations")
    
    if not os.path.exists(organizations_dir):
        return organizations_data
    
    for org_dir in os.listdir(organizations_dir):
        org_path = os.path.join(organizations_dir, org_dir)
        if not os.path.isdir(org_path):
            continue
            
        org_file = os.path.join(org_path, "organization.json")
        org_data = load_json_file(org_file)
        
        if org_data:
            org_data["data_type"] = "organization"
            organizations_data.append(org_data)
    
    return organizations_data

def extract_providers_data(data_dir: str) -> List[Dict]:
    """Extract all provider data."""
    providers_data = []
    providers_dir = os.path.join(data_dir, "providers")
    
    if not os.path.exists(providers_dir):
        return providers_data
    
    for provider_dir in os.listdir(providers_dir):
        provider_path = os.path.join(providers_dir, provider_dir)
        if not os.path.isdir(provider_path):
            continue
            
        provider_file = os.path.join(provider_path, "provider.json")
        provider_data = load_json_file(provider_file)
        
        if provider_data:
            provider_data["data_type"] = "provider"
            providers_data.append(provider_data)
    
    return providers_data

def extract_licenses_data(data_dir: str) -> List[Dict]:
    """Extract all license data."""
    licenses_data = []
    licenses_dir = os.path.join(data_dir, "licenses")
    
    if not os.path.exists(licenses_dir):
        return licenses_data
    
    for license_file in os.listdir(licenses_dir):
        if license_file.endswith('.json'):
            file_path = os.path.join(licenses_dir, license_file)
            license_data = load_json_file(file_path)
            
            if license_data:
                # Extract license name from filename
                license_name = license_file.replace('.json', '')
                license_data["license_id"] = license_name
                license_data["data_type"] = "license"
                licenses_data.append(license_data)
    
    return licenses_data

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to string representation
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def get_all_columns(data_list: List[Dict]) -> set:
    """Get all unique column names from a list of dictionaries."""
    all_columns = set()
    for item in data_list:
        flattened = flatten_dict(item)
        all_columns.update(flattened.keys())
    return all_columns

def write_to_csv(data_list: List[Dict], filename: str, output_dir: str):
    """Write data to CSV file."""
    if not data_list:
        print(f"No data to write for {filename}")
        return
    
    # Get all unique columns
    all_columns = get_all_columns(data_list)
    all_columns = sorted(list(all_columns))
    
    # Flatten all data
    flattened_data = []
    for item in data_list:
        flattened_data.append(flatten_dict(item))
    
    # Write to CSV
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(flattened_data)
    
    print(f"Written {len(flattened_data)} records to {output_path}")

def main():
    """Main function to combine all LLM data."""
    data_dir = "data"
    output_dir = "combined_data"
    
    print("Starting LLM data combination process...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all data
    print("\nExtracting models data...")
    models_data = extract_models_data(data_dir)
    print(f"Found {len(models_data)} models")
    
    print("\nExtracting benchmark results...")
    benchmark_results = extract_benchmark_results(data_dir)
    print(f"Found {len(benchmark_results)} benchmark results")
    
    print("\nExtracting benchmark definitions...")
    benchmark_definitions = extract_benchmark_definitions(data_dir)
    print(f"Found {len(benchmark_definitions)} benchmark definitions")
    
    print("\nExtracting organizations data...")
    organizations_data = extract_organizations_data(data_dir)
    print(f"Found {len(organizations_data)} organizations")
    
    print("\nExtracting providers data...")
    providers_data = extract_providers_data(data_dir)
    print(f"Found {len(providers_data)} providers")
    
    print("\nExtracting licenses data...")
    licenses_data = extract_licenses_data(data_dir)
    print(f"Found {len(licenses_data)} licenses")
    
    # Write separate CSV files for each data type
    print("\nWriting CSV files...")
    write_to_csv(models_data, "models.csv", output_dir)
    write_to_csv(benchmark_results, "benchmark_results.csv", output_dir)
    write_to_csv(benchmark_definitions, "benchmark_definitions.csv", output_dir)
    write_to_csv(organizations_data, "organizations.csv", output_dir)
    write_to_csv(providers_data, "providers.csv", output_dir)
    write_to_csv(licenses_data, "licenses.csv", output_dir)
    
    # Create a combined CSV with all data
    print("\nCreating combined CSV...")
    all_data = []
    all_data.extend(models_data)
    all_data.extend(benchmark_results)
    all_data.extend(benchmark_definitions)
    all_data.extend(organizations_data)
    all_data.extend(providers_data)
    all_data.extend(licenses_data)
    
    write_to_csv(all_data, "all_llm_data.csv", output_dir)
    
    print(f"\nData combination complete!")
    print(f"Total records: {len(all_data)}")
    print(f"Output files saved in: {output_dir}/")
    
    # Print summary
    print("\nSummary:")
    print(f"- Models: {len(models_data)}")
    print(f"- Benchmark Results: {len(benchmark_results)}")
    print(f"- Benchmark Definitions: {len(benchmark_definitions)}")
    print(f"- Organizations: {len(organizations_data)}")
    print(f"- Providers: {len(providers_data)}")
    print(f"- Licenses: {len(licenses_data)}")

if __name__ == "__main__":
    main()
