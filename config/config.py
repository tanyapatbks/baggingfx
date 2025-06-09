"""
Enhanced Configuration File - config/config.py
เพิ่มการตั้งค่าสำหรับ Enhanced Visualization และ Analysis
"""

import os
from datetime import datetime
from typing import Dict, List

class Config:
    """Enhanced configuration class for Multi-Currency CNN-LSTM Forex Prediction System"""
    
    def __init__(self):
        # Base directories
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'models')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'results')
        self.LOGS_DIR = os.path.join(self.BASE_DIR, 'logs')
        
        # Currency pairs and data files
        self.CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY']
        self.CURRENCY_FILES = {
            'EURUSD': os.path.join(self.DATA_DIR, 'EURUSD_1H.csv'),
            'GBPUSD': os.path.join(self.DATA_DIR, 'GBPUSD_1H.csv'),
            'USDJPY': os.path.join(self.DATA_DIR, 'USDJPY_1H.csv')
        }
        
        # ==================== ENHANCED VISUALIZATION SETTINGS ====================
        # Plot appearance settings
        self.SAVE_PLOTS = True
        self.PLOT_FORMAT = 'png'  # Options: 'png', 'jpg', 'pdf', 'svg'
        self.DPI = 300  # High resolution for publication
        self.FIGURE_SIZE = (12, 8)  # Default figure size
        
        # Color schemes for different strategies
        self.STRATEGY_COLORS = {
            'multi_currency_conservative': '#1f77b4',  # Blue
            'multi_currency_moderate': '#ff7f0e',      # Orange
            'multi_currency_aggressive': '#2ca02c',    # Green
            'buy_and_hold': '#d62728',                 # Red
            'rsi': '#9467bd',                          # Purple
            'macd': '#8c564b',                         # Brown
            'training': '#1f77b4',                     # Blue
            'validation': '#ff7f0e'                    # Orange
        }
        
        # Analysis settings
        self.CREATE_TRAINING_PLOTS = True
        self.CREATE_CUMULATIVE_PLOTS = True
        self.CREATE_RISK_ANALYSIS = True
        self.CREATE_ARCHITECTURE_DIAGRAM = True
        self.CREATE_PREDICTION_ANALYSIS = True
        self.CREATE_INTERACTIVE_DASHBOARD = True
        
        # Enhanced analysis parameters
        self.ANALYSIS_YEARS = [2018, 2019, 2020, 2021]  # Years for yearly analysis
        self.MARKET_REGIMES = {
            '2018': {'type': 'trade_war_tensions', 'description': 'Trade war uncertainties'},
            '2019': {'type': 'stable_accommodative', 'description': 'Stable monetary policies'},
            '2020': {'type': 'covid_crisis', 'description': 'COVID-19 pandemic crisis'},
            '2021': {'type': 'recovery_inflation', 'description': 'Recovery and inflation concerns'}
        }
        
        # ==================== MODEL ARCHITECTURE SETTINGS ====================
        # Input parameters
        self.WINDOW_SIZE = 60  # Hours of historical data
        self.PREDICTION_HORIZON = 1  # Hours ahead to predict
        self.TOTAL_FEATURES = 15  # 5 features × 3 currency pairs
        
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
        
        # ==================== TRAINING SETTINGS ====================
        # Training parameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.OPTIMIZER = 'adam'
        self.LOSS_FUNCTION = 'binary_crossentropy'
        self.METRICS = ['accuracy']
        
        # Callback parameters
        self.EARLY_STOPPING_PATIENCE = 15
        self.REDUCE_LR_PATIENCE = 8
        self.REDUCE_LR_FACTOR = 0.5
        self.REDUCE_LR_MIN_LR = 1e-7
        self.SAVE_MODEL_FREQUENCY = 5  # Save model every N epochs
        
        # ==================== DATA PREPROCESSING SETTINGS ====================
        # Missing value handling
        self.MAX_FILL_HOURS = 3  # Maximum hours to forward fill
        self.INTERPOLATION_METHOD = 'linear'
        
        # Normalization parameters
        self.NORMALIZATION_METHOD = 'mixed'  # 'standard', 'minmax', 'mixed'
        self.FEATURE_SCALING = 'per_currency'  # 'global', 'per_currency', 'per_feature'
        
        # ==================== TEMPORAL DATA SPLITS ====================
        # Training period
        self.TRAIN_START = '2018-01-01'
        self.TRAIN_END = '2020-12-31'
        
        # Validation period
        self.VAL_START = '2021-01-01'
        self.VAL_END = '2021-12-31'
        
        # Test period
        self.TEST_START = '2022-01-01'
        self.TEST_END = '2022-12-31'
        
        # ==================== ENHANCED TRADING STRATEGY SETTINGS ====================
        # Threshold-based strategy parameters
        self.TRADING_THRESHOLDS = {
            'conservative': {'buy': 0.7, 'sell': 0.3},
            'moderate': {'buy': 0.6, 'sell': 0.4},
            'aggressive': {'buy': 0.55, 'sell': 0.45}
        }
        
        # Enhanced risk management parameters
        self.MINIMUM_HOLDING_PERIOD = 1  # Hours
        self.MIN_HOLDING_HOURS = 1  # Minimum hours to hold position
        self.MAX_HOLDING_HOURS = 3  # Maximum hours to hold position
        self.STOP_LOSS_PCT = 0.02  # 2% stop loss
        self.TAKE_PROFIT_PCT = 0.04  # 4% take profit
        
        # Weekend trading avoidance
        self.AVOID_WEEKEND_POSITIONS = True
        self.FRIDAY_CLOSE_HOUR = 21  # Close positions on Friday after this hour
        self.MONDAY_OPEN_HOUR = 1   # Don't open positions on Monday before this hour
        
        # ==================== BASELINE STRATEGY SETTINGS ====================
        # RSI strategy parameters
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        
        # MACD strategy parameters
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # ==================== EXPERIMENT SETTINGS ====================
        # Experiment identification
        self.EXPERIMENT_NAME = f"forex_cnn_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.DEVELOPMENT_MODE = True  # True = use validation set, False = use test set
        
        # Logging settings
        self.LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        self.DETAILED_LOGGING = True
        self.LOG_SYSTEM_STATS = True
        
        # Performance monitoring
        self.MONITOR_MEMORY = True
        self.MONITOR_GPU = True
        self.LOG_FREQUENCY = 5  # Log every N epochs
        
        # ==================== ENHANCED ANALYSIS SETTINGS ====================
        # Statistical analysis
        self.CONFIDENCE_LEVEL = 0.95  # For confidence intervals
        self.SIGNIFICANCE_LEVEL = 0.05  # For hypothesis testing
        self.BOOTSTRAP_SAMPLES = 1000  # For bootstrap analysis
        
        # Performance metrics
        self.RISK_FREE_RATE = 0.02  # Annual risk-free rate for Sharpe ratio
        self.TRADING_COST = 0.0001  # Transaction cost per trade (1 pip for major pairs)
        
        # Visualization preferences
        self.PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
        self.COLOR_PALETTE = 'husl'  # Seaborn color palette
        self.FONT_SIZE = 12
        self.TITLE_SIZE = 16
        
        # Report generation
        self.GENERATE_HTML_REPORT = True
        self.GENERATE_LATEX_TABLES = True
        self.GENERATE_INTERACTIVE_PLOTS = True
        self.INCLUDE_STATISTICAL_TESTS = True
        
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.MODELS_DIR, 
            self.RESULTS_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_results_path(self, filename: str) -> str:
        """Get full path for results file"""
        return os.path.join(self.RESULTS_DIR, filename)
    
    def get_model_path(self, filename: str) -> str:
        """Get full path for model file"""
        return os.path.join(self.MODELS_DIR, filename)
    
    def get_log_path(self, filename: str) -> str:
        """Get full path for log file"""
        return os.path.join(self.LOGS_DIR, f"{filename}.log")
    
    def update_experiment_name(self, experiment_id: str):
        """Update experiment name for resuming existing experiments"""
        self.EXPERIMENT_NAME = experiment_id
    
    def get_experiment_config(self) -> Dict:
        """Get experiment configuration summary"""
        return {
            'experiment_name': self.EXPERIMENT_NAME,
            'currency_pairs': self.CURRENCY_PAIRS,
            'model_params': {
                'window_size': self.WINDOW_SIZE,
                'cnn_filters': [self.CNN_FILTERS_1, self.CNN_FILTERS_2],
                'lstm_units': [self.LSTM_UNITS_1, self.LSTM_UNITS_2],
                'dense_units': self.DENSE_UNITS
            },
            'training_params': {
                'batch_size': self.BATCH_SIZE,
                'epochs': self.EPOCHS,
                'learning_rate': self.LEARNING_RATE
            },
            'data_splits': {
                'train': f"{self.TRAIN_START} to {self.TRAIN_END}",
                'validation': f"{self.VAL_START} to {self.VAL_END}",
                'test': f"{self.TEST_START} to {self.TEST_END}"
            },
            'enhanced_features': {
                'visualization_enabled': True,
                'training_plots': self.CREATE_TRAINING_PLOTS,
                'cumulative_analysis': self.CREATE_CUMULATIVE_PLOTS,
                'risk_analysis': self.CREATE_RISK_ANALYSIS,
                'architecture_diagram': self.CREATE_ARCHITECTURE_DIAGRAM,
                'prediction_analysis': self.CREATE_PREDICTION_ANALYSIS,
                'interactive_dashboard': self.CREATE_INTERACTIVE_DASHBOARD
            }
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check data files exist
        for pair, file_path in self.CURRENCY_FILES.items():
            if not os.path.exists(file_path):
                issues.append(f"Data file not found for {pair}: {file_path}")
        
        # Check date ranges
        from datetime import datetime
        try:
            train_start = datetime.strptime(self.TRAIN_START, '%Y-%m-%d')
            train_end = datetime.strptime(self.TRAIN_END, '%Y-%m-%d')
            val_start = datetime.strptime(self.VAL_START, '%Y-%m-%d')
            val_end = datetime.strptime(self.VAL_END, '%Y-%m-%d')
            test_start = datetime.strptime(self.TEST_START, '%Y-%m-%d')
            test_end = datetime.strptime(self.TEST_END, '%Y-%m-%d')
            
            if train_start >= train_end:
                issues.append("Training start date must be before end date")
            if val_start >= val_end:
                issues.append("Validation start date must be before end date")
            if test_start >= test_end:
                issues.append("Test start date must be before end date")
            if train_end >= val_start:
                issues.append("Training period should not overlap with validation period")
            if val_end >= test_start:
                issues.append("Validation period should not overlap with test period")
                
        except ValueError as e:
            issues.append(f"Invalid date format: {str(e)}")
        
        # Check model parameters
        if self.WINDOW_SIZE <= 0:
            issues.append("Window size must be positive")
        if self.PREDICTION_HORIZON <= 0:
            issues.append("Prediction horizon must be positive")
        if self.BATCH_SIZE <= 0:
            issues.append("Batch size must be positive")
        if self.EPOCHS <= 0:
            issues.append("Number of epochs must be positive")
        if self.LEARNING_RATE <= 0:
            issues.append("Learning rate must be positive")
        
        # Check trading parameters
        for threshold_type, thresholds in self.TRADING_THRESHOLDS.items():
            if thresholds['buy'] <= thresholds['sell']:
                issues.append(f"Buy threshold must be higher than sell threshold for {threshold_type}")
            if not (0 <= thresholds['sell'] <= thresholds['buy'] <= 1):
                issues.append(f"Thresholds must be between 0 and 1 for {threshold_type}")
        
        return issues
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration"""
        print("=" * 60)
        print("ENHANCED CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Experiment: {self.EXPERIMENT_NAME}")
        print(f"Mode: {'Development (Validation)' if self.DEVELOPMENT_MODE else 'Final (Test)'}")
        print(f"Currency Pairs: {', '.join(self.CURRENCY_PAIRS)}")
        print(f"Model: CNN-LSTM ({self.CNN_FILTERS_1}→{self.CNN_FILTERS_2} CNN, {self.LSTM_UNITS_1}→{self.LSTM_UNITS_2} LSTM)")
        print(f"Training: {self.EPOCHS} epochs, batch size {self.BATCH_SIZE}")
        print(f"Data Window: {self.WINDOW_SIZE}h → {self.PREDICTION_HORIZON}h ahead")
        print(f"Enhanced Features: ✅ ALL ENABLED")
        print(f"Visualization: {self.PLOT_FORMAT.upper()} @ {self.DPI} DPI")
        print("=" * 60)
        
        # Validation check
        issues = self.validate_configuration()
        if issues:
            print("⚠️  CONFIGURATION ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Configuration validation passed")
        print("=" * 60)