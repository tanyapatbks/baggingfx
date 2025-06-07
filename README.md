Multi-Currency CNN-LSTM Forex Prediction System
A comprehensive deep learning system for forex trend prediction using multi-currency CNN-LSTM architecture. This system implements a sophisticated approach to forex market analysis by leveraging cross-currency correlations and temporal patterns.
🎯 Project Overview
This project implements a Master's thesis research on multi-currency forex prediction using:

CNN layers for spatial-temporal feature extraction
LSTM layers for sequential pattern learning
Multi-currency integration for cross-correlation analysis
Comprehensive baseline comparisons with traditional strategies

📋 Features
Core Functionality

✅ Multi-currency data integration (EURUSD, GBPUSD, USDJPY)
✅ Advanced CNN-LSTM neural network architecture
✅ Sophisticated data preprocessing and normalization
✅ Threshold-based trading strategy implementation
✅ Comprehensive baseline strategy comparison
✅ Market regime analysis across different time periods
✅ Statistical significance testing
✅ Interactive visualizations and reports

Analysis Components

Data Loading & Exploration: Comprehensive data quality analysis
Preprocessing: Missing value handling, normalization, feature engineering
Multi-Currency Integration: Cross-correlation analysis and unified datasets
Sequence Preparation: Sliding window creation for time series modeling
Model Architecture: CNN-LSTM with optimized hyperparameters
Training & Optimization: Advanced callbacks and logging system
Trading Strategies: Threshold-based decisions with risk management
Results Analysis: Publication-ready visualizations and reports

🚀 Quick Start
Prerequisites
bash# Python 3.8+ required
pip install -r requirements.txt
Data Preparation

Create the data directory structure:

forex_prediction/
├── data/
│   ├── EURUSD_1H.csv
│   ├── GBPUSD_1H.csv
│   └── USDJPY_1H.csv

Ensure your CSV files have the following columns:

Local time (datetime index)
Open (float)
High (float)
Low (float)
Close (float)
Volume (float)



Running the Analysis
bash# Execute the complete pipeline
python main.py
The system will automatically:

Load and validate your data
Perform comprehensive preprocessing
Train the CNN-LSTM model
Test multiple trading strategies
Generate detailed reports and visualizations

📊 Expected Results
Generated Files

results/research_report_[timestamp].html - Comprehensive analysis report
results/forex_analysis_dashboard.html - Interactive dashboard
results/results_[timestamp].tex - LaTeX tables for academic papers
results/model_architecture.png - Neural network architecture diagram
results/training_history_[timestamp].png - Training progress visualization
results/strategy_performance_comparison.png - Strategy comparison charts

Performance Metrics
The system evaluates strategies using:

Total Return: Cumulative profit/loss percentage
Sharpe Ratio: Risk-adjusted return measure
Win Rate: Percentage of profitable trades
Maximum Drawdown: Largest peak-to-trough decline
Statistical Significance: P-values for strategy comparisons

🏗️ System Architecture
Data Flow
Raw CSV Data → Preprocessing → Multi-Currency Integration → 
Sequence Creation → CNN-LSTM Training → Strategy Testing → 
Performance Analysis → Report Generation
Model Architecture
Input (60×15) → CNN Layers (64→128 filters) → MaxPooling → 
LSTM Layers (128→64 units) → Dense Layer (32 units) → 
Output (1 unit, sigmoid)
📈 Trading Strategy Framework
Threshold-Based Decisions

Conservative: Buy ≥ 0.7, Sell ≤ 0.3
Moderate: Buy ≥ 0.6, Sell ≤ 0.4
Aggressive: Buy ≥ 0.55, Sell ≤ 0.45

Risk Management

Minimum holding period: 4 hours
Stop loss: 2%
Take profit: 4%

Baseline Comparisons

Buy & Hold strategy
RSI-based trading (14-period, 30/70 thresholds)
MACD-based trading (12/26/9 parameters)
Single-currency CNN-LSTM models

🔧 Configuration
Key parameters can be modified in config/config.py:
python# Model Architecture
WINDOW_SIZE = 60          # Hours of historical data
CNN_FILTERS_1 = 64        # First CNN layer filters
CNN_FILTERS_2 = 128       # Second CNN layer filters
LSTM_UNITS_1 = 128        # First LSTM layer units
LSTM_UNITS_2 = 64         # Second LSTM layer units

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Data Splits
TRAIN_START = '2018-01-01'
TRAIN_END = '2020-12-31'
VAL_START = '2021-01-01'
VAL_END = '2021-12-31'
TEST_START = '2022-01-01'
TEST_END = '2022-12-31'
📝 Research Applications
This system is designed for academic research and supports:
Thesis Requirements

Comprehensive literature review implementation
Rigorous methodology documentation
Statistical significance testing
Publication-ready visualizations
LaTeX table exports for academic papers

Experimental Design

Temporal data splitting (avoiding look-ahead bias)
Cross-validation with time series constraints
Ablation studies with model variants
Market regime analysis for robustness testing

🔍 Troubleshooting
Common Issues

Missing Data Files
Error: Required data files not found
Solution: Ensure CSV files are in /data directory with correct names

Memory Issues
Error: Out of memory during training
Solution: Reduce BATCH_SIZE in config.py

Poor Model Performance
Issue: Low accuracy or poor trading returns
Solution: Check data quality, adjust WINDOW_SIZE, or modify architecture


Performance Optimization

Use GPU acceleration if available
Adjust batch size based on available memory
Monitor training logs for early stopping
Consider data augmentation for small datasets

📚 Dependencies
Core Libraries

tensorflow>=2.10.0 - Deep learning framework
pandas>=1.5.0 - Data manipulation
numpy>=1.21.0 - Numerical computing
scikit-learn>=1.1.0 - Machine learning utilities

Visualization

matplotlib>=3.5.0 - Static plotting
seaborn>=0.11.0 - Statistical visualizations
plotly>=5.0.0 - Interactive charts

Technical Analysis

ta>=0.10.0 - Technical indicators for baseline strategies

Utilities

psutil - System monitoring
tqdm - Progress bars
joblib - Model serialization