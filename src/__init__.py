"""
Multi-Currency CNN-LSTM Forex Prediction System
A comprehensive system for forex trend prediction using deep learning
"""

__version__ = "1.0.0"
__author__ = "Forex Prediction Research Team"
__description__ = "Multi-Currency CNN-LSTM Forex Prediction System for Master Thesis Research"

# Import main modules for easy access
from .data_loader import CurrencyDataLoader
from .preprocessing import ForexDataPreprocessor
from .multi_currency_integration import MultiCurrencyIntegrator
from .sequence_preparation import SequenceDataPreparator
from .model_architecture import CNNLSTMArchitecture
from .training import ModelTrainer
from .trading_strategy import TradingStrategyManager
from .visualization import ResultsAnalyzer
from .utils import (
    setup_logging, 
    ensure_directory_exists,
    calculate_execution_time,
    ProgressTracker
)

__all__ = [
    'CurrencyDataLoader',
    'ForexDataPreprocessor', 
    'MultiCurrencyIntegrator',
    'SequenceDataPreparator',
    'CNNLSTMArchitecture',
    'ModelTrainer',
    'TradingStrategyManager',
    'ResultsAnalyzer',
    'setup_logging',
    'ensure_directory_exists',
    'calculate_execution_time',
    'ProgressTracker'
]