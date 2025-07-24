import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import DATA_CONFIG

def create_features(df):
    """Create new features from existing ones"""
    df_features = df.copy()
    
    # BMI categories
    df_features['BMI_Category'] = pd.cut(
        df_features['BMI'], 
        bins=[0, 18.5, 25, 30, float('inf')], 
        labels=['Underweight', 'Normal', 'Overweight', 'Obese']
    )
    
    # Age groups
    df_features['Age_Group'] = pd.cut(
        df_features['Age'],
        bins=[0, 30, 50, float('inf')],
        labels=['Young', 'Middle', 'Senior']
    )
    
    # Glucose level categories
    df_features['Glucose_Level'] = pd.cut(
        df_features['Glucose'],
        bins=[0, 100, 126, float('inf')],
        labels=['Normal', 'Prediabetic', 'Diabetic']
    )
    
    # Interaction features
    df_features['BMI_Age_Interaction'] = df_features['BMI'] * df_features['Age']
    df_features['Glucose_BMI_Ratio'] = df_features['Glucose'] / (df_features['BMI'] + 1)
    df_features['Insulin_Glucose_Ratio'] = df_features['Insulin'] / (df_features['Glucose'] + 1)
    
    return df_features

def encode_categorical_features(df):
    """Encode categorical features"""
    df_encoded = df.copy()
    
    # One-hot encode categorical features
    categorical_cols = df_encoded.select_dtypes(include=['category', 'object']).columns
    
    for col in categorical_cols:
        if col != 'Outcome':  # Don't encode target variable
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded

def select_features(X, y, k=10):
    """Select top k features using statistical tests"""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected features: {selected_features}")
    
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

def scale_features(df, method='standard'):
    """Scale numerical features"""
    df_scaled = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'Outcome']
    
    # Choose scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    # Scale numerical features
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    return df_scaled, scaler

def main():
    """Main feature engineering pipeline"""
    print("Starting feature engineering...")
    
    # Load processed data
    df = pd.read_csv(DATA_CONFIG["processed_data_path"])
    print(f"Loaded data shape: {df.shape}")
    
    # Create new features
    df_features = create_features(df)
    print(f"After feature creation: {df_features.shape}")
    
    # Encode categorical features
    df_encoded = encode_categorical_features(df_features)
    print(f"After encoding: {df_encoded.shape}")
    
    # Separate features and target
    X = df_encoded.drop('Outcome', axis=1)
    y = df_encoded['Outcome']
    
    # Feature selection (optional)
    # X_selected = select_features(X, y, k=15)
    
    # Scale features
    X_scaled, scaler = scale_features(pd.concat([X, y], axis=1))
    
    # Save processed features
    output_path = DATA_CONFIG["processed_data_path"].replace('.csv', '_features.csv')
    X_scaled.to_csv(output_path, index=False)
    
    print(f"Feature engineering completed!")
    print(f"Final dataset shape: {X_scaled.shape}")
    print(f"Features saved to: {output_path}")

if __name__ == "__main__":
    main()
