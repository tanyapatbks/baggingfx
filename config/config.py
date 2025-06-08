"""
Complete Configuration for Multi-Currency CNN-LSTM Forex Prediction System
This file contains all necessary parameters and configurations
"""

import os
from datetime import datetime

class Config:
    """
    Comprehensive configuration for Multi-Currency CNN-LSTM Forex Prediction System
    Contains all parameters needed for the entire pipeline
    """
    
    def __init__(self):
        # ========================================================================================
        # EXPERIMENT CONFIGURATION
        # ========================================================================================
        self.EXPERIMENT_NAME = f"forex_cnn_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.DEVELOPMENT_MODE = True  # Set to False for final evaluation on test set
        
        # ========================================================================================
        # DATA CONFIGURATION
        # ========================================================================================
        self.CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        # Data file paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(base_dir, 'data')
        
        self.CURRENCY_FILES = {
            'EURUSD': os.path.join(self.DATA_DIR, 'EURUSD_1H.csv'),
            'GBPUSD': os.path.join(self.DATA_DIR, 'GBPUSD_1H.csv'),
            'USDJPY': os.path.join(self.DATA_DIR, 'USDJPY_1H.csv')
        }
        
        # ========================================================================================
        # PREPROCESSING PARAMETERS
        # ========================================================================================
        self.MAX_FILL_HOURS = 4
        self.INTERPOLATION_METHOD = 'linear'
        
        # ========================================================================================
        # SEQUENCE PREPARATION
        # ========================================================================================
        self.WINDOW_SIZE = 60
        self.PREDICTION_HORIZON = 1
        self.TOTAL_FEATURES = 15  # 5 features × 3 currency pairs
        
        # ========================================================================================
        # TEMPORAL DATA SPLITS
        # ========================================================================================
        self.TRAIN_START = '2018-01-01'
        self.TRAIN_END = '2020-12-31'
        self.VAL_START = '2021-01-01'
        self.VAL_END = '2021-12-31'
        self.TEST_START = '2022-01-01'
        self.TEST_END = '2022-12-31'
        
        # ========================================================================================
        # MODEL ARCHITECTURE
        # ========================================================================================
        # CNN parameters
        self.CNN_FILTERS_1 = 64
        self.CNN_FILTERS_2 = 128
        self.CNN_KERNEL_SIZE = 3
        self.CNN_ACTIVATION = 'relu'
        self.CNN_PADDING = 'same'
        
        # LSTM parameters
        self.LSTM_UNITS_1 = 128
        self.LSTM_UNITS_2 = 64
        self.LSTM_DROPOUT = 0.2
        self.LSTM_RECURRENT_DROPOUT = 0.2
        
        # Dense layer parameters
        self.DENSE_UNITS = 32
        self.DENSE_DROPOUT = 0.3
        self.DENSE_ACTIVATION = 'relu'
        self.OUTPUT_ACTIVATION = 'sigmoid'
        
        # ========================================================================================
        # TRAINING PARAMETERS
        # ========================================================================================
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.OPTIMIZER = 'adam'
        self.LOSS_FUNCTION = 'binary_crossentropy'
        self.METRICS = ['accuracy']
        
        # Training callbacks
        self.EARLY_STOPPING_PATIENCE = 15
        self.REDUCE_LR_PATIENCE = 8
        self.REDUCE_LR_FACTOR = 0.5
        self.REDUCE_LR_MIN_LR = 1e-7
        self.SAVE_MODEL_FREQUENCY = 10
        
        # ========================================================================================
        # TRADING STRATEGY PARAMETERS
        # ========================================================================================
        # Threshold strategies
        self.TRADING_THRESHOLDS = {
            'conservative': {'buy': 0.7, 'sell': 0.3},
            'moderate': {'buy': 0.6, 'sell': 0.4},
            'aggressive': {'buy': 0.55, 'sell': 0.45}
        }
        
        # Risk management parameters
        self.MINIMUM_HOLDING_PERIOD = 4  # hours - แก้ไข error หลัก
        self.STOP_LOSS_PCT = 0.02       # 2% stop loss
        self.TAKE_PROFIT_PCT = 0.04     # 4% take profit
        
        # Time-based trading rules
        self.MIN_HOLDING_HOURS = 1      # Minimum holding time
        self.MAX_HOLDING_HOURS = 3      # Maximum holding time
        self.AVOID_WEEKEND_POSITIONS = True
        self.FRIDAY_CLOSE_HOUR = 21     # Close positions before weekend
        self.MONDAY_OPEN_HOUR = 1       # Avoid trading early Monday
        
        # RSI strategy parameters
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        
        # MACD strategy parameters
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # ========================================================================================
        # VISUALIZATION SETTINGS
        # ========================================================================================
        self.FIGURE_SIZE = (12, 8)
        self.DPI = 300
        self.SAVE_PLOTS = True
        self.PLOT_FORMAT = 'png'
        
        self.STRATEGY_COLORS = {
            'multi_currency_conservative': '#1f77b4',
            'multi_currency_moderate': '#ff7f0e',
            'multi_currency_aggressive': '#2ca02c',
            'buy_and_hold': '#d62728',
            'rsi': '#9467bd',
            'macd': '#8c564b'
        }
        
        # ========================================================================================
        # SYSTEM SETTINGS
        # ========================================================================================
        self.LOG_LEVEL = 'INFO'
        
        # Directory structure
        self.BASE_DIR = base_dir
        self.RESULTS_DIR = os.path.join(base_dir, 'results')
        self.MODELS_DIR = os.path.join(base_dir, 'models')
        self.LOGS_DIR = os.path.join(base_dir, 'logs')
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [self.DATA_DIR, self.RESULTS_DIR, self.MODELS_DIR, self.LOGS_DIR]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {directory}: {str(e)}")
    
    def get_results_path(self, filename: str) -> str:
        """Get full path for results file"""
        return os.path.join(self.RESULTS_DIR, filename)
    
    def get_models_path(self, filename: str) -> str:
        """Get full path for model file"""
        return os.path.join(self.MODELS_DIR, filename)
    
    def get_log_path(self, filename: str) -> str:
        """Get full path for log file"""
        return os.path.join(self.LOGS_DIR, f"{filename}.log")
    
    def update_experiment_name(self, new_name: str):
        """Update experiment name for resume functionality"""
        self.EXPERIMENT_NAME = new_name
    
    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Check data files exist
        for pair, path in self.CURRENCY_FILES.items():
            if not os.path.exists(path):
                errors.append(f"Missing data file for {pair}: {path}")
        
        # Check positive parameters
        if self.WINDOW_SIZE <= 0:
            errors.append("WINDOW_SIZE must be positive")
        
        if self.BATCH_SIZE <= 0:
            errors.append("BATCH_SIZE must be positive")
        
        if self.EPOCHS <= 0:
            errors.append("EPOCHS must be positive")
        
        if self.LEARNING_RATE <= 0:
            errors.append("LEARNING_RATE must be positive")
        
        # Check threshold values
        for strategy, thresholds in self.TRADING_THRESHOLDS.items():
            if thresholds['buy'] <= thresholds['sell']:
                errors.append(f"Buy threshold must be higher than sell threshold for {strategy}")
            
            if not (0 <= thresholds['sell'] <= thresholds['buy'] <= 1):
                errors.append(f"Thresholds must be between 0 and 1 for {strategy}")
        
        # Check time parameters
        if self.MIN_HOLDING_HOURS >= self.MAX_HOLDING_HOURS:
            errors.append("MIN_HOLDING_HOURS must be less than MAX_HOLDING_HOURS")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True
    
    def to_dict(self):
        """Convert config to dictionary for saving"""
        config_dict = {}
        
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # Convert non-serializable types
                if isinstance(value, (list, tuple, dict, str, int, float, bool)):
                    config_dict[key] = value
                else:
                    config_dict[key] = str(value)
        
        return config_dict
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        
        config_dict = self.to_dict()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            print(f"Configuration saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save configuration: {str(e)}")
    
    def print_config_summary(self):
        """Print a summary of key configuration parameters"""
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Experiment Name: {self.EXPERIMENT_NAME}")
        print(f"Development Mode: {self.DEVELOPMENT_MODE}")
        print(f"Currency Pairs: {', '.join(self.CURRENCY_PAIRS)}")
        print(f"Data Window: {self.WINDOW_SIZE} hours")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Model Architecture: CNN({self.CNN_FILTERS_1},{self.CNN_FILTERS_2}) → LSTM({self.LSTM_UNITS_1},{self.LSTM_UNITS_2}) → Dense({self.DENSE_UNITS})")
        print(f"Trading Strategies: {len(self.TRADING_THRESHOLDS)} threshold variants")
        print(f"Risk Management: Stop Loss {self.STOP_LOSS_PCT*100}%, Take Profit {self.TAKE_PROFIT_PCT*100}%")
        print(f"Results Directory: {self.RESULTS_DIR}")
        print("=" * 60)

# Create a default instance for easy importing
default_config = Config()

# Validation function that can be called independently
def validate_system_requirements():
    """
    Validate system requirements and dependencies
    """
    print("Checking system requirements...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]}")
    
    # Check required packages
    required_packages = {
        'tensorflow': '2.10.0',
        'pandas': '1.5.0',
        'numpy': '1.21.0',
        'scikit-learn': '1.1.0',
        'matplotlib': '3.5.0',
        'seaborn': '0.11.0'
    }
    
    missing_packages = []
    
    for package, min_version in required_packages.items():
        try:
            imported = __import__(package)
            version = getattr(imported, '__version__', 'unknown')
            print(f"✅ {package} {version}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All system requirements met")
    return True

if __name__ == "__main__":
    # Test configuration when run directly
    print("Testing configuration...")
    
    # Test system requirements
    if not validate_system_requirements():
        exit(1)
    
    # Test config creation
    try:
        config = Config()
        config.create_directories()
        config.validate_config()
        config.print_config_summary()
        print("✅ Configuration test passed")
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        exit(1)