# Model configuration
MODEL_CONFIG = {
    "name": "DiabetesPredictionModel",
    "version": "1.0.0",
    "algorithm": "RandomForest",
    "features": [
        "Pregnancies",
        "Glucose",
        "BloodPressure", 
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ],
    "target": "Outcome",
    "test_size": 0.2,
    "random_state": 42
}

# Training configuration
TRAINING_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

# API configuration
API_CONFIG = {
    "title": "Diabetes Prediction API",
    "description": "ML API for diabetes prediction based on health metrics",
    "version": "1.0.0",
    "host": "0.0.0.0",
    "port": 8000
}

# Data configuration
DATA_CONFIG = {
    "raw_data_path": "data/raw/diabetes.csv",
    "processed_data_path": "data/processed/diabetes_processed.csv",
    "model_save_path": "models/saved/diabetes_model.joblib",
    "scaler_save_path": "models/saved/scaler.joblib"
}
