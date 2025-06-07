"""
Configuration file for Multi-Currency CNN-LSTM Forex Prediction
Contains all hyperparameters, file paths, and model settings
"""

import os
from datetime import datetime

class Config:
    """Configuration class containing all project settings"""
    
    # ============== PROJECT PATHS ==============
    # Base directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Data file paths
    CURRENCY_FILES = {
        'EURUSD': os.path.join(DATA_DIR, 'EURUSD_1H.csv'),
        'GBPUSD': os.path.join(DATA_DIR, 'GBPUSD_1H.csv'),
        'USDJPY': os.path.join(DATA_DIR, 'USDJPY_1H.csv')
    }
    
    # ============== DATA PARAMETERS ==============
    # Currency pair configuration
    CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
    FEATURES_PER_PAIR = 5  # OHLCV
    TOTAL_FEATURES = len(CURRENCY_PAIRS) * FEATURES_PER_PAIR  # 15 features
    
    # Time series configuration
    WINDOW_SIZE = 60  # 60 hours lookback window
    PREDICTION_HORIZON = 1  # Predict next hour
    
    # Data split configuration (temporal split)
    TRAIN_START = '2018-01-01'
    TRAIN_END = '2020-12-31'
    VAL_START = '2021-01-01'
    VAL_END = '2021-12-31'
    TEST_START = '2022-01-01'
    TEST_END = '2022-12-31'
    
    # ============== MODEL ARCHITECTURE ==============
    # CNN layer configuration
    CNN_FILTERS_1 = 64      # First CNN layer filters
    CNN_FILTERS_2 = 128     # Second CNN layer filters
    CNN_KERNEL_SIZE = 3     # Kernel size for both CNN layers
    CNN_ACTIVATION = 'relu'
    CNN_PADDING = 'same'
    
    # LSTM layer configuration
    LSTM_UNITS_1 = 128      # First LSTM layer units
    LSTM_UNITS_2 = 64       # Second LSTM layer units
    LSTM_DROPOUT = 0.2      # LSTM dropout rate
    LSTM_RECURRENT_DROPOUT = 0.2  # LSTM recurrent dropout
    
    # Dense layer configuration
    DENSE_UNITS = 32        # Dense layer before output
    DENSE_DROPOUT = 0.3     # Dense layer dropout
    DENSE_ACTIVATION = 'relu'
    
    # Output layer configuration
    OUTPUT_ACTIVATION = 'sigmoid'  # For binary classification
    
    # ============== TRAINING PARAMETERS ==============
    # Basic training configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Optimizer configuration
    OPTIMIZER = 'adam'
    LOSS_FUNCTION = 'binary_crossentropy'
    METRICS = ['accuracy']
    
    # Callback configuration
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 8
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_MIN_LR = 1e-7
    
    # ============== EXECUTION MODE CONFIGURATION ==============
    # Controls whether to use test set (only for final evaluation before thesis defense)
    DEVELOPMENT_MODE = True  # Set to False only for final thesis evaluation
    
    # ============== TRADING STRATEGY PARAMETERS ==============
    # Threshold configurations for different strategies
    TRADING_THRESHOLDS = {
        'conservative': {'buy': 0.7, 'sell': 0.3},
        'moderate': {'buy': 0.6, 'sell': 0.4},
        'aggressive': {'buy': 0.55, 'sell': 0.45}
    }
    
    # New Time-Based Trading Parameters
    MAX_HOLDING_HOURS = 3      # Maximum holding period: 3 hours
    MIN_HOLDING_HOURS = 1      # Minimum holding period: 1 hour  
    STOP_LOSS_PCT = 0.02       # 2% stop loss (still applies)
    
    # Weekend Trading Rules
    AVOID_WEEKEND_POSITIONS = True  # Close positions before weekend
    FRIDAY_CLOSE_HOUR = 20     # Close positions by Friday 8 PM
    MONDAY_OPEN_HOUR = 8       # No new positions until Monday 8 AM
    
    # Baseline strategy parameters
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # ============== PREPROCESSING PARAMETERS ==============
    # Normalization configuration
    NORMALIZATION_METHOD = 'mixed'  # 'standard', 'minmax', or 'mixed'
    
    # Missing value handling
    FILL_METHOD = 'forward'  # 'forward', 'backward', 'interpolate'
    MAX_FILL_HOURS = 6      # Maximum hours to forward fill
    INTERPOLATION_METHOD = 'linear'
    
    # ============== LOGGING CONFIGURATION ==============
    # Experiment tracking
    EXPERIMENT_NAME = f"forex_cnn_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    LOG_LEVEL = 'INFO'
    
    # Model checkpointing
    SAVE_BEST_MODEL = True
    SAVE_MODEL_FREQUENCY = 10  # Save every N epochs
    
    # ============== EVALUATION PARAMETERS ==============
    # Performance metrics to calculate
    EVALUATION_METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'sharpe_ratio', 'max_drawdown', 'win_rate',
        'avg_profit_per_trade', 'total_return'
    ]
    
    # Statistical significance testing
    CONFIDENCE_LEVEL = 0.95
    
    # ============== VISUALIZATION PARAMETERS ==============
    # Plot configuration
    FIGURE_SIZE = (12, 8)
    DPI = 300
    SAVE_PLOTS = True
    PLOT_FORMAT = 'png'
    
    # Color schemes for different strategies
    STRATEGY_COLORS = {
        'multi_currency': '#1f77b4',
        'single_currency': '#ff7f0e', 
        'buy_hold': '#2ca02c',
        'rsi': '#d62728',
        'macd': '#9467bd'
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get full path for saving/loading models"""
        return os.path.join(cls.MODELS_DIR, f"{model_name}.h5")
    
    @classmethod
    def get_results_path(cls, filename):
        """Get full path for saving results"""
        return os.path.join(cls.RESULTS_DIR, filename)
    
    @classmethod
    def get_log_path(cls, log_name):
        """Get full path for log files"""
        return os.path.join(cls.LOGS_DIR, f"{log_name}.log")