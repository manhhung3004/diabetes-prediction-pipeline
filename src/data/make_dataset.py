import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to pdath
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import DATA_CONFIG

def download_dataset():
    """Download diabetes dataset (Pima Indians Diabetes Database)"""
    # For demo purposes, we'll create a sample dataset
    # In practice, you would download from UCI ML Repository or use sklearn.datasets
    
    print("Creating sample diabetes dataset...")
    
    # Create sample data (this would be replaced with actual data loading)
    np.random.seed(42)
    n_samples = 768
    
    data = {
        'Pregnancies': np.random.randint(0, 18, n_samples),
        'Glucose': np.random.normal(120, 30, n_samples).clip(0, 200),
        'BloodPressure': np.random.normal(70, 15, n_samples).clip(0, 122),
        'SkinThickness': np.random.normal(20, 10, n_samples).clip(0, 99),
        'Insulin': np.random.normal(80, 100, n_samples).clip(0, 846),
        'BMI': np.random.normal(32, 7, n_samples).clip(10, 70),
        'DiabetesPedigreeFunction': np.random.gamma(0.5, 1, n_samples).clip(0, 2.5),
        'Age': np.random.randint(21, 81, n_samples),
    }
    
    # Create target variable with some logic
    df = pd.DataFrame(data)
    
    # Simple logic for diabetes outcome
    risk_score = (
        (df['Glucose'] > 140) * 2 +
        (df['BMI'] > 30) * 1 +
        (df['Age'] > 50) * 1 +
        (df['Pregnancies'] > 5) * 0.5
    )
    
    # Add some randomness
    random_factor = np.random.normal(0, 1, n_samples)
    final_score = risk_score + random_factor
    
    df['Outcome'] = (final_score > 2).astype(int)
    
    return df

def clean_data(df):
    """Clean and preprocess the dataset"""
    print("Cleaning dataset...")
    
    # Check for missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics
    print("\nDataset statistics:")
    print(df.describe())
    
    # Check target distribution
    print(f"\nTarget distribution:")
    print(df['Outcome'].value_counts(normalize=True))
    
    return df

def save_dataset(df, filepath):
    """Save dataset to specified path"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to: {filepath}")

def main():
    """Main data preparation pipeline"""
    print("Starting data preparation...")
    
    # Create raw data directory
    os.makedirs(os.path.dirname(DATA_CONFIG["raw_data_path"]), exist_ok=True)
    
    # Download/create dataset
    df = download_dataset()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save raw data
    save_dataset(df_clean, DATA_CONFIG["raw_data_path"])
    
    # Save processed data (same as raw for now)
    save_dataset(df_clean, DATA_CONFIG["processed_data_path"])
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
