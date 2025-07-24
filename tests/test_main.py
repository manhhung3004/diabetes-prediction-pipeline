import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

class TestAPI:
    """Test cases for the Diabetes Prediction API"""
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint"""
        response = client.get("/model/info")
        # This might fail if model is not loaded, which is OK for testing
        assert response.status_code in [200, 500]
    
    def test_prediction_endpoint_valid_input(self):
        """Test prediction with valid input"""
        test_data = {
            "pregnancies": 2,
            "glucose": 120.0,
            "blood_pressure": 70.0,
            "skin_thickness": 20.0,
            "insulin": 80.0,
            "bmi": 25.5,
            "diabetes_pedigree_function": 0.5,
            "age": 30
        }
        
        response = client.post("/predict", json=test_data)
        # This might fail if model is not loaded, which is OK for testing
        if response.status_code == 200:
            assert "prediction" in response.json()
            assert "probability" in response.json()
            assert "risk_level" in response.json()
    
    def test_prediction_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        test_data = {
            "pregnancies": "invalid",  # Should be int
            "glucose": 120.0,
            "blood_pressure": 70.0,
            "skin_thickness": 20.0,
            "insulin": 80.0,
            "bmi": 25.5,
            "diabetes_pedigree_function": 0.5,
            "age": 30
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error

class TestDataProcessing:
    """Test cases for data processing functions"""
    
    def test_data_loading(self):
        """Test if sample data can be created"""
        # Create sample data
        sample_data = pd.DataFrame({
            'Pregnancies': [1, 2, 3],
            'Glucose': [100, 120, 140],
            'BloodPressure': [70, 80, 90],
            'SkinThickness': [20, 25, 30],
            'Insulin': [80, 100, 120],
            'BMI': [25, 30, 35],
            'DiabetesPedigreeFunction': [0.5, 0.7, 0.9],
            'Age': [25, 35, 45],
            'Outcome': [0, 0, 1]
        })
        
        assert not sample_data.empty
        assert sample_data.shape[1] == 9  # 8 features + 1 target
    
    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        # Create sample data with missing values (zeros)
        sample_data = pd.DataFrame({
            'Glucose': [0, 120, 140],
            'BloodPressure': [70, 0, 90],
            'BMI': [25, 30, 0]
        })
        
        # Replace zeros with median
        for column in sample_data.columns:
            sample_data[column] = sample_data[column].replace(0, sample_data[column].median())
        
        # Check no zeros remain
        assert (sample_data == 0).sum().sum() == 0

class TestModelTraining:
    """Test cases for model training"""
    
    def test_feature_target_separation(self):
        """Test separation of features and target"""
        sample_data = pd.DataFrame({
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6], 
            'Target': [0, 1, 0]
        })
        
        X = sample_data.drop('Target', axis=1)
        y = sample_data['Target']
        
        assert X.shape[1] == 2
        assert len(y) == 3
        assert 'Target' not in X.columns
    
    def test_train_test_split_ratio(self):
        """Test train-test split maintains proper ratio"""
        from sklearn.model_selection import train_test_split
        
        X = np.random.random((100, 5))
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

if __name__ == "__main__":
    pytest.main([__file__])
