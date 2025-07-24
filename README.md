# Diabetes Prediction Model - MLOps Project

This project demonstrates building and deploying an ML model for diabetes prediction using health metrics, following MLOps best practices.

## Project Overview

This project helps you learn Building and Deploying an ML Model using a simple and real-world use case: predicting whether a person is diabetic based on health metrics.

## Project Pipeline

- **Model Training** - Train diabetes prediction model
- **Building the Model locally** - Local development and testing
- **API Deployment with FastAPI** - REST API for model serving
- **Dockerization** - Containerize the application
- **Kubernetes Deployment** - Deploy to Kubernetes cluster

## Project Structure

```
Diabetes-Prediction-Model/
├── data/
│   ├── raw/                    # Raw, unprocessed data
│   └── processed/              # Cleaned and processed data
├── notebooks/                  # Jupyter notebooks for EDA and experiments
├── src/
│   ├── data/                   # Data processing scripts
│   ├── features/               # Feature engineering scripts
│   └── models/                 # Model training and evaluation scripts
├── api/                        # FastAPI application
├── tests/                      # Unit and integration tests
├── deployment/
│   ├── docker/                 # Docker configuration
│   └── kubernetes/             # Kubernetes manifests
├── models/
│   └── saved/                  # Trained model artifacts
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── monitoring/                 # Model monitoring and logging
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
└── README.md                   # Project documentation
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download and process data
python src/data/make_dataset.py
```

### 3. Model Training

```bash
# Train the model
python src/models/train_model.py
```

### 4. Run API

```bash
# Start FastAPI server
uvicorn api.main:app --reload
```

### 5. Docker Deployment

```bash
# Build Docker image
docker build -t diabetes-prediction .

# Run container
docker run -p 8000:8000 diabetes-prediction
```

### 6. Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/
```

## Dataset

The dataset contains health metrics including:
- Glucose level
- Blood pressure
- BMI
- Age
- Insulin level
- Pregnancies
- Skin thickness
- Diabetes pedigree function

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Make prediction
- `GET /model/info` - Model information

## Tech Stack

- **ML Framework**: Scikit-learn, Pandas, NumPy
- **API**: FastAPI
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Monitoring**: Prometheus, Grafana
- **Testing**: Pytest
- **CI/CD**: GitHub Actions

## Monitoring

Model performance is monitored using:
- Prediction accuracy metrics
- Data drift detection
- API response times
- System resource usage

