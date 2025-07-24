#!/bin/bash

# Setup script for Diabetes Prediction Model project

echo "Setting up Diabetes Prediction Model project..."

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw data/processed models/saved logs

# Download/create dataset
echo "Preparing dataset..."
python src/data/make_dataset.py

# Train initial model
echo "Training initial model..."
python src/models/train_model.py

# Run tests
echo "Running tests..."
pytest tests/ -v

echo "Setup completed successfully!"
echo "To start the API server, run: uvicorn api.main:app --reload"
