"""
Utility Functions Module
Common helper functions used across the forex prediction system
Provides logging setup, file operations, and general utilities
"""

import os
import logging
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging configuration for the entire system
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure that a directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Absolute path to directory
    """
    abs_path = os.path.abspath(directory_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def save_json(data: Dict, file_path: str, indent: int = 2) -> bool:
    """
    Save dictionary to JSON file with error handling
    
    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
        
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save JSON to {file_path}: {str(e)}")
        return False

def load_json(file_path: str) -> Optional[Dict]:
    """
    Load dictionary from JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load JSON from {file_path}: {str(e)}")
        return None

def save_pickle(data: Any, file_path: str) -> bool:
    """
    Save object to pickle file with error handling
    
    Args:
        data: Object to save
        file_path: Path to output pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        
        with open(file_path, 'wb') as f:
            joblib.dump(data, f)
        
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save pickle to {file_path}: {str(e)}")
        return False

def load_pickle(file_path: str) -> Optional[Any]:
    """
    Load object from pickle file with error handling
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded object or None if failed
    """
    try:
        with open(file_path, 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load pickle from {file_path}: {str(e)}")
        return None

def calculate_memory_usage() -> Dict[str, float]:
    """
    Calculate current memory usage statistics
    
    Returns:
        Dictionary containing memory usage information
    """
    try:
        import psutil
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_percent': system_memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
    except ImportError:
        logging.getLogger(__name__).warning("psutil not available for memory monitoring")
        return {}
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating memory usage: {str(e)}")
        return {}

def format_large_number(number: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with appropriate suffixes
    
    Args:
        number: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if abs(number) >= 1e9:
        return f"{number / 1e9:.{precision}f}B"
    elif abs(number) >= 1e6:
        return f"{number / 1e6:.{precision}f}M"
    elif abs(number) >= 1e3:
        return f"{number / 1e3:.{precision}f}K"
    else:
        return f"{number:.{precision}f}"

def calculate_execution_time(func):
    """
    Decorator to calculate and log function execution time
    
    Args:
        func: Function to time
        
    Returns:
        Decorated function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper

def validate_data_frame(df: pd.DataFrame, required_columns: List[str] = None,
                       min_rows: int = 1) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame structure and content
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check if DataFrame is empty
    if df is None:
        return False, ["DataFrame is None"]
    
    if len(df) < min_rows:
        issues.append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for all NaN columns
    all_nan_columns = [col for col in df.columns if df[col].isna().all()]
    if all_nan_columns:
        issues.append(f"Columns with all NaN values: {all_nan_columns}")
    
    # Check data types
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        # Check if object columns should be numeric
        potentially_numeric = []
        for col in object_columns:
            try:
                pd.to_numeric(df[col].dropna().head(10))
                potentially_numeric.append(col)
            except:
                pass
        
        if potentially_numeric:
            issues.append(f"Object columns that might be numeric: {potentially_numeric}")
    
    return len(issues) == 0, issues

def create_experiment_summary(config, start_time: datetime, end_time: datetime,
                            results: Dict) -> Dict:
    """
    Create comprehensive experiment summary
    
    Args:
        config: Configuration object
        start_time: Experiment start time
        end_time: Experiment end time
        results: Experiment results
        
    Returns:
        Dictionary containing experiment summary
    """
    duration = end_time - start_time
    
    summary = {
        'experiment_metadata': {
            'experiment_id': config.EXPERIMENT_NAME,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration)
        },
        'configuration': {
            'currency_pairs': config.CURRENCY_PAIRS,
            'window_size': config.WINDOW_SIZE,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE
        },
        'system_info': calculate_memory_usage(),
        'results_summary': results
    }
    
    return summary

def compare_model_performance(results1: Dict, results2: Dict, 
                            metrics: List[str] = None) -> Dict:
    """
    Compare performance between two models
    
    Args:
        results1: First model results
        results2: Second model results
        metrics: List of metrics to compare
        
    Returns:
        Dictionary containing comparison results
    """
    if metrics is None:
        metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']
    
    comparison = {
        'model1_metrics': {},
        'model2_metrics': {},
        'improvements': {},
        'winner_by_metric': {}
    }
    
    for metric in metrics:
        # Extract values (handle nested dictionaries)
        val1 = extract_nested_value(results1, metric)
        val2 = extract_nested_value(results2, metric)
        
        if val1 is not None and val2 is not None:
            comparison['model1_metrics'][metric] = val1
            comparison['model2_metrics'][metric] = val2
            
            # Calculate improvement (model2 vs model1)
            if val1 != 0:
                improvement = (val2 - val1) / abs(val1) * 100
            else:
                improvement = 0
            
            comparison['improvements'][metric] = improvement
            
            # Determine winner (lower is better for drawdown)
            if metric == 'max_drawdown':
                comparison['winner_by_metric'][metric] = 'model1' if val1 < val2 else 'model2'
            else:
                comparison['winner_by_metric'][metric] = 'model1' if val1 > val2 else 'model2'
    
    return comparison

def extract_nested_value(data: Dict, key: str, default=None):
    """
    Extract value from nested dictionary using dot notation
    
    Args:
        data: Dictionary to search
        key: Key to find (supports dot notation like 'performance.total_return')
        default: Default value if key not found
        
    Returns:
        Found value or default
    """
    try:
        keys = key.split('.')
        value = data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    except:
        return default

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform division with safety check for zero denominator
    
    Args:
        numerator: Numerator value
        denominator: Denominator value  
        default: Default value when denominator is zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        return numerator / denominator
    except:
        return default

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dictionaries(*dicts) -> Dict:
    """
    Merge multiple dictionaries with nested key handling
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    
    for d in dicts:
        if not isinstance(d, dict):
            continue
            
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value)
            else:
                result[key] = value
    
    return result

def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Truncate if too long
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized

def convert_to_serializable(obj) -> Any:
    """
    Convert object to JSON serializable format
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def print_system_info():
    """Print comprehensive system information for debugging"""
    import platform
    import sys
    
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # Memory info
    memory_info = calculate_memory_usage()
    if memory_info:
        logger.info(f"Total Memory: {memory_info.get('system_total_gb', 0):.2f} GB")
        logger.info(f"Available Memory: {memory_info.get('system_available_gb', 0):.2f} GB")
    
    # Package versions
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow Version: {tf.__version__}")
    except ImportError:
        logger.warning("TensorFlow not available")
    
    try:
        import pandas as pd
        logger.info(f"Pandas Version: {pd.__version__}")
    except ImportError:
        logger.warning("Pandas not available")
    
    try:
        import numpy as np
        logger.info(f"NumPy Version: {np.__version__}")
    except ImportError:
        logger.warning("NumPy not available")
    
    logger.info("="*60)

class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        
    def update(self, step: int = None, description: str = None):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if description:
            self.description = description
            
        percentage = (self.current_step / self.total_steps) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current_step > 0:
            estimated_total = elapsed * (self.total_steps / self.current_step)
            remaining = estimated_total - elapsed
            
            self.logger.info(f"{self.description}: {self.current_step}/{self.total_steps} "
                           f"({percentage:.1f}%) - "
                           f"Elapsed: {str(elapsed).split('.')[0]}, "
                           f"Remaining: {str(remaining).split('.')[0]}")
        else:
            self.logger.info(f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)")
    
    def finish(self):
        """Mark as completed"""
        total_time = datetime.now() - self.start_time
        self.logger.info(f"{self.description} completed in {str(total_time).split('.')[0]}")

def debug_csv_datetime_format(file_path: str, column_name: str = 'Local time', 
                              max_samples: int = 10) -> Dict:
    """
    Debug CSV datetime format to understand parsing issues
    
    Args:
        file_path: Path to CSV file
        column_name: Name of datetime column
        max_samples: Number of sample values to examine
        
    Returns:
        Dictionary containing datetime format analysis
    """
    import pandas as pd
    
    try:
        # Read raw CSV without parsing
        df_raw = pd.read_csv(file_path)
        
        if column_name not in df_raw.columns:
            return {
                'error': f"Column '{column_name}' not found",
                'available_columns': df_raw.columns.tolist()
            }
        
        # Get sample values
        sample_values = df_raw[column_name].head(max_samples).tolist()
        
        # Try to identify the format
        datetime_info = {
            'sample_values': sample_values,
            'total_records': len(df_raw),
            'unique_formats': [],
            'parsing_results': {}
        }
        
        # Test different format patterns
        test_formats = [
            '%d.%m.%Y %H:%M:%S.%f GMT%z',
            '%d.%m.%Y %H:%M:%S GMT%z', 
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
        ]
        
        for fmt in test_formats:
            try:
                test_result = pd.to_datetime(sample_values[:3], format=fmt)
                datetime_info['parsing_results'][fmt] = 'SUCCESS'
            except:
                datetime_info['parsing_results'][fmt] = 'FAILED'
        
        # Try pandas infer
        try:
            inferred = pd.to_datetime(sample_values[:3], infer_datetime_format=True)
            datetime_info['pandas_infer'] = 'SUCCESS'
        except Exception as e:
            datetime_info['pandas_infer'] = f'FAILED: {str(e)}'
        
        return datetime_info
        
    except Exception as e:
        return {'error': f"Failed to debug CSV: {str(e)}"}

def safe_datetime_conversion(series: pd.Series, column_name: str = 'datetime') -> pd.Series:
    """
    Safely convert series to datetime with multiple fallback strategies
    
    Args:
        series: Pandas series to convert
        column_name: Name of column for logging
        
    Returns:
        Converted datetime series
    """
    logger = logging.getLogger(__name__)
    
    # Strategy 1: Direct conversion
    try:
        return pd.to_datetime(series, infer_datetime_format=True)
    except Exception as e1:
        logger.warning(f"Direct datetime conversion failed for {column_name}: {str(e1)}")
    
    # Strategy 2: Clean and retry
    try:
        # Remove common problematic parts
        cleaned = series.astype(str).str.replace(r' GMT[+-]\d{4}', '', regex=True)
        cleaned = cleaned.str.replace(r'\.000', '', regex=True)
        return pd.to_datetime(cleaned, infer_datetime_format=True)
    except Exception as e2:
        logger.warning(f"Cleaned datetime conversion failed for {column_name}: {str(e2)}")
    
    # Strategy 3: Try specific formats
    formats_to_try = [
        '%d.%m.%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%d.%m.%Y'
    ]
    
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(series, format=fmt)
        except:
            continue
    
    # Strategy 4: Manual parsing for complex formats
    try:
        def parse_complex_datetime(date_str):
            import re
            # Handle formats like "01.01.2018 00:00:00.000 GMT+0700"
            # Remove timezone and milliseconds
            cleaned = re.sub(r'\.000 GMT[+-]\d{4}', '', str(date_str))
            return pd.to_datetime(cleaned, format='%d.%m.%Y %H:%M:%S')
        
        return series.apply(parse_complex_datetime)
    except Exception as e4:
        logger.error(f"All datetime conversion strategies failed for {column_name}: {str(e4)}")
    
    raise ValueError(f"Unable to convert {column_name} to datetime. Sample values: {series.head().tolist()}")
    """
    Validate that all required files exist
    
    Args:
        file_paths: Dictionary mapping names to file paths
        
    Returns:
        Tuple of (all_valid, list_of_missing_files)
    """
    missing_files = []
    
    for name, path in file_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    return len(missing_files) == 0, missing_files

def validate_file_paths(file_paths: Dict[str, str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required files exist
    
    Args:
        file_paths: Dictionary mapping names to file paths
        
    Returns:
        Tuple of (all_valid, list_of_missing_files)
    """
    missing_files = []
    
    for name, path in file_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    return len(missing_files) == 0, missing_files

def debug_csv_datetime_format(file_path: str, column_name: str = 'Local time', 
                              max_samples: int = 10) -> Dict:
    """
    Debug CSV datetime format to understand parsing issues
    
    Args:
        file_path: Path to CSV file
        column_name: Name of datetime column
        max_samples: Number of sample values to examine
        
    Returns:
        Dictionary containing datetime format analysis
    """
    try:
        # Read raw CSV without parsing
        df_raw = pd.read_csv(file_path)
        
        if column_name not in df_raw.columns:
            return {
                'error': f"Column '{column_name}' not found",
                'available_columns': df_raw.columns.tolist()
            }
        
        # Get sample values
        sample_values = df_raw[column_name].head(max_samples).tolist()
        
        # Try to identify the format
        datetime_info = {
            'sample_values': sample_values,
            'total_records': len(df_raw),
            'parsing_results': {}
        }
        
        # Test different format patterns
        test_formats = [
            '%d.%m.%Y %H:%M:%S.%f GMT%z',
            '%d.%m.%Y %H:%M:%S GMT%z', 
            '%Y-%m-%d %H:%M:%S',
            '%d.%m.%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
        ]
        
        for fmt in test_formats:
            try:
                test_result = pd.to_datetime(sample_values[:3], format=fmt)
                datetime_info['parsing_results'][fmt] = 'SUCCESS'
            except:
                datetime_info['parsing_results'][fmt] = 'FAILED'
        
        # Try pandas infer
        try:
            inferred = pd.to_datetime(sample_values[:3], infer_datetime_format=True)
            datetime_info['pandas_infer'] = 'SUCCESS'
        except Exception as e:
            datetime_info['pandas_infer'] = f'FAILED: {str(e)}'
        
        return datetime_info
        
    except Exception as e:
        return {'error': f"Failed to debug CSV: {str(e)}"}

def safe_datetime_conversion(series: pd.Series, column_name: str = 'datetime') -> pd.Series:
    """
    Safely convert series to datetime with multiple fallback strategies
    
    Args:
        series: Pandas series to convert
        column_name: Name of column for logging
        
    Returns:
        Converted datetime series
    """
    logger = logging.getLogger(__name__)
    
    # Strategy 1: Direct conversion
    try:
        return pd.to_datetime(series, infer_datetime_format=True)
    except Exception as e1:
        logger.warning(f"Direct datetime conversion failed for {column_name}: {str(e1)}")
    
    # Strategy 2: Clean and retry
    try:
        # Remove common problematic parts
        cleaned = series.astype(str).str.replace(r' GMT[+-]\d{4}', '', regex=True)
        cleaned = cleaned.str.replace(r'\.000', '', regex=True)
        return pd.to_datetime(cleaned, infer_datetime_format=True)
    except Exception as e2:
        logger.warning(f"Cleaned datetime conversion failed for {column_name}: {str(e2)}")
    
    # Strategy 3: Try specific formats
    formats_to_try = [
        '%d.%m.%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%d.%m.%Y'
    ]
    
    for fmt in formats_to_try:
        try:
            return pd.to_datetime(series, format=fmt)
        except:
            continue
    
    # Strategy 4: Manual parsing for complex formats
    try:
        def parse_complex_datetime(date_str):
            import re
            # Handle formats like "01.01.2018 00:00:00.000 GMT+0700"
            # Remove timezone and milliseconds
            cleaned = re.sub(r'\.000 GMT[+-]\d{4}', '', str(date_str))
            return pd.to_datetime(cleaned, format='%d.%m.%Y %H:%M:%S')
        
        return series.apply(parse_complex_datetime)
    except Exception as e4:
        logger.error(f"All datetime conversion strategies failed for {column_name}: {str(e4)}")
    
    raise ValueError(f"Unable to convert {column_name} to datetime. Sample values: {series.head().tolist()}")