import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import MODEL_CONFIG, TRAINING_CONFIG, DATA_CONFIG

def load_data():
    """Load the diabetes dataset"""
    try:
        data = pd.read_csv(DATA_CONFIG["raw_data_path"])
        print(f"Data loaded successfully: {data.shape}")
        return data
    except FileNotFoundError:
        print("Dataset not found. Please ensure diabetes.csv is in data/raw/")
        return None

def preprocess_data(data):
    """Preprocess the data"""
    # Handle missing values (zeros in some columns indicate missing values)
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in columns_to_replace:
        if column in data.columns:
            data[column] = data[column].replace(0, data[column].median())
    
    # Separate features and target
    X = data[MODEL_CONFIG["features"]]
    y = data[MODEL_CONFIG["target"]]
    
    return X, y

def train_model(X, y):
    """Train the diabetes prediction model"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG["test_size"], 
        random_state=MODEL_CONFIG["random_state"],
        stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(**TRAINING_CONFIG)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, accuracy

def save_model(model, scaler):
    """Save the trained model and scaler"""
    os.makedirs(os.path.dirname(DATA_CONFIG["model_save_path"]), exist_ok=True)
    
    joblib.dump(model, DATA_CONFIG["model_save_path"])
    joblib.dump(scaler, DATA_CONFIG["scaler_save_path"])
    
    print(f"Model saved to: {DATA_CONFIG['model_save_path']}")
    print(f"Scaler saved to: {DATA_CONFIG['scaler_save_path']}")

def main():
    """Main training pipeline"""
    print("Starting diabetes prediction model training...")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train model
    model, scaler, accuracy = train_model(X, y)
    
    # Save model
    save_model(model, scaler)
    
    print(f"\nTraining completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
