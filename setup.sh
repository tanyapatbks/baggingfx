#!/bin/bash

# Setup script for Multi-Currency CNN-LSTM Forex Prediction System
# This script creates virtual environment and installs dependencies

echo "=============================================="
echo "Multi-Currency CNN-LSTM Forex Prediction"
echo "Environment Setup Script"
echo "=============================================="

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected"
else
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "forex_env" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf forex_env
fi

python -m venv forex_env

# Activate virtual environment
echo "Activating virtual environment..."
source forex_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing required packages..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ All packages installed successfully"
else
    echo "Error: requirements.txt not found"
    exit 1
fi

# Check if data directory exists
echo "Checking data directory..."
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
    echo "Please copy your CSV files to the data/ directory:"
    echo "  - EURUSD_1H.csv"
    echo "  - GBPUSD_1H.csv"
    echo "  - USDJPY_1H.csv"
fi

# Create other directories
echo "Creating project directories..."
mkdir -p models results logs

echo "=============================================="
echo "Setup completed successfully!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source forex_env/bin/activate"
echo ""
echo "To run the system:"
echo "  python main.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Copy your CSV data files to data/ directory"
echo "2. Activate the environment: source forex_env/bin/activate"
echo "3. Run the analysis: python main.py"
echo "=============================================="