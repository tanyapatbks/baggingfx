"""
Sequence Data Preparation Module
Transforms aligned multi-currency data into sequences suitable for CNN-LSTM training
Handles sliding window creation, label generation, and temporal data splitting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Generator
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

class SequenceDataPreparator:
    """
    Advanced sequence preparation for multi-currency CNN-LSTM models
    Creates optimized sliding windows and manages temporal data splits
    """
    
    def __init__(self, config):
        """
        Initialize sequence preparator with configuration settings
        
        Args:
            config: Configuration object containing sequence parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Sequence parameters
        self.window_size = config.WINDOW_SIZE
        self.prediction_horizon = config.PREDICTION_HORIZON
        self.batch_size = config.BATCH_SIZE
        
        # Data split dates
        self.split_dates = {
            'train_start': pd.to_datetime(config.TRAIN_START),
            'train_end': pd.to_datetime(config.TRAIN_END),
            'val_start': pd.to_datetime(config.VAL_START),
            'val_end': pd.to_datetime(config.VAL_END),
            'test_start': pd.to_datetime(config.TEST_START),
            'test_end': pd.to_datetime(config.TEST_END)
        }
        
        # Sequence statistics
        self.sequence_stats = {}
        
    def create_sequences(self, data: pd.DataFrame, target_pair: str = 'EURUSD') -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Create sliding window sequences from unified multi-currency data
        
        Args:
            data: Unified DataFrame with all currency features
            target_pair: Currency pair to predict (default: EURUSD)
            
        Returns:
            Tuple of (X_sequences, y_labels, timestamps)
        """
        self.logger.info(f"Creating sequences with window size {self.window_size} for prediction of {target_pair}")
        
        # Validate input data
        if len(data) < self.window_size + self.prediction_horizon:
            raise ValueError(f"Insufficient data: need at least {self.window_size + self.prediction_horizon} records, "
                           f"got {len(data)}")
        
        # Extract features for sequence creation
        feature_columns = [col for col in data.columns if any(pair in col for pair in self.config.CURRENCY_PAIRS)]
        feature_matrix = data[feature_columns].values
        
        # Create target labels from the specified currency pair
        target_column = f'{target_pair}_Close_Return'
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        target_returns = data[target_column].values
        
        # Calculate the number of sequences we can create
        num_sequences = len(data) - self.window_size - self.prediction_horizon + 1
        num_features = feature_matrix.shape[1]
        
        self.logger.info(f"Creating {num_sequences:,} sequences with {num_features} features each")
        
        # Initialize arrays for sequences and labels
        X_sequences = np.zeros((num_sequences, self.window_size, num_features), dtype=np.float32)
        y_labels = np.zeros(num_sequences, dtype=np.float32)
        sequence_timestamps = []
        
        # Create sequences using sliding window approach
        for i in range(num_sequences):
            # Extract sequence window
            start_idx = i
            end_idx = i + self.window_size
            
            # Feature sequence (lookback window)
            X_sequences[i] = feature_matrix[start_idx:end_idx]
            
            # Target label (future return direction)
            future_idx = end_idx + self.prediction_horizon - 1
            future_return = target_returns[future_idx]
            
            # Convert return to binary classification (1 if positive, 0 if negative)
            y_labels[i] = 1.0 if future_return > 0 else 0.0
            
            # Store timestamp of the sequence end (prediction point)
            sequence_timestamps.append(data.index[end_idx - 1])
            
            # Log progress for large datasets
            if (i + 1) % 10000 == 0:
                self.logger.info(f"Created {i + 1:,}/{num_sequences:,} sequences ({(i + 1)/num_sequences*100:.1f}%)")
        
        # Convert timestamps to DatetimeIndex
        sequence_timestamps = pd.DatetimeIndex(sequence_timestamps)
        
        # Validate created sequences
        self._validate_sequences(X_sequences, y_labels, sequence_timestamps)
        
        # Store sequence statistics
        self.sequence_stats = {
            'total_sequences': num_sequences,
            'window_size': self.window_size,
            'num_features': num_features,
            'target_pair': target_pair,
            'positive_labels': int(y_labels.sum()),
            'negative_labels': int(len(y_labels) - y_labels.sum()),
            'class_balance': float(y_labels.mean()),
            'feature_names': feature_columns,
            'temporal_range': {
                'start': sequence_timestamps.min(),
                'end': sequence_timestamps.max()
            }
        }
        
        self.logger.info(f"Sequence creation complete: {num_sequences:,} sequences, "
                        f"class balance = {self.sequence_stats['class_balance']:.3f}")
        
        return X_sequences, y_labels, sequence_timestamps
    
    def _validate_sequences(self, X: np.ndarray, y: np.ndarray, timestamps: pd.DatetimeIndex):
        """
        Validate created sequences for quality and consistency
        
        Args:
            X: Feature sequences array
            y: Labels array
            timestamps: Timestamp index
        """
        self.logger.info("Validating created sequences")
        
        # Shape validation
        expected_shape = (len(y), self.window_size, self.config.TOTAL_FEATURES)
        if X.shape != expected_shape:
            self.logger.warning(f"Unexpected sequence shape: {X.shape}, expected: {expected_shape}")
        
        # Data quality checks
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values in sequences")
        
        if inf_count > 0:
            self.logger.warning(f"Found {inf_count} infinite values in sequences")
        
        # Label validation
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, [0., 1.]) and not np.array_equal(unique_labels, [0.]) and not np.array_equal(unique_labels, [1.]):
            self.logger.warning(f"Unexpected label values: {unique_labels}")
        
        # Temporal validation
        if len(timestamps) != len(y):
            self.logger.error(f"Timestamp count {len(timestamps)} doesn't match label count {len(y)}")
        
        # Class balance check
        class_balance = y.mean()
        if class_balance < 0.3 or class_balance > 0.7:
            self.logger.warning(f"Significant class imbalance detected: {class_balance:.3f}")
        
        self.logger.info("Sequence validation completed")
    
    def prepare_labels(self, data: pd.DataFrame, target_pair: str = 'EURUSD', 
                      label_type: str = 'direction') -> np.ndarray:
        """
        Prepare prediction labels with various strategies
        
        Args:
            data: DataFrame with currency data
            target_pair: Currency pair to create labels for
            label_type: Type of labels ('direction', 'magnitude', 'quantile')
            
        Returns:
            Array of labels
        """
        self.logger.info(f"Preparing {label_type} labels for {target_pair}")
        
        target_column = f'{target_pair}_Close_Return'
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found")
        
        returns = data[target_column].values
        
        if label_type == 'direction':
            # Binary directional labels (up/down)
            labels = (returns > 0).astype(np.float32)
            
        elif label_type == 'magnitude':
            # Magnitude-based labels (small/medium/large moves)
            abs_returns = np.abs(returns)
            q33, q67 = np.percentile(abs_returns, [33, 67])
            
            labels = np.zeros(len(returns), dtype=np.float32)
            labels[abs_returns <= q33] = 0  # Small moves
            labels[(abs_returns > q33) & (abs_returns <= q67)] = 1  # Medium moves
            labels[abs_returns > q67] = 2  # Large moves
            
        elif label_type == 'quantile':
            # Quantile-based labels
            q20, q80 = np.percentile(returns, [20, 80])
            
            labels = np.ones(len(returns), dtype=np.float32)  # Neutral (middle 60%)
            labels[returns <= q20] = 0  # Strong down
            labels[returns >= q80] = 2  # Strong up
            
        else:
            raise ValueError(f"Unknown label type: {label_type}")
        
        # Remove first window_size + prediction_horizon - 1 labels to align with sequences
        aligned_labels = labels[self.window_size + self.prediction_horizon - 1:]
        
        self.logger.info(f"Created {len(aligned_labels)} {label_type} labels")
        return aligned_labels
    
    def split_temporal_data(self, X: np.ndarray, y: np.ndarray, 
                           timestamps: pd.DatetimeIndex) -> Dict[str, Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """
        Split data temporally into train/validation/test sets
        
        Args:
            X: Feature sequences
            y: Labels
            timestamps: Sequence timestamps
            
        Returns:
            Dictionary containing train/val/test splits
        """
        self.logger.info("Performing temporal data split")
        
        # Create boolean masks for each split
        train_mask = (timestamps >= self.split_dates['train_start']) & (timestamps <= self.split_dates['train_end'])
        val_mask = (timestamps >= self.split_dates['val_start']) & (timestamps <= self.split_dates['val_end'])
        test_mask = (timestamps >= self.split_dates['test_start']) & (timestamps <= self.split_dates['test_end'])
        
        # Apply masks to create splits
        splits = {
            'train': (X[train_mask], y[train_mask], timestamps[train_mask]),
            'val': (X[val_mask], y[val_mask], timestamps[val_mask]),
            'test': (X[test_mask], y[test_mask], timestamps[test_mask])
        }
        
        # Log split statistics
        for split_name, (X_split, y_split, ts_split) in splits.items():
            class_balance = y_split.mean() if len(y_split) > 0 else 0
            self.logger.info(f"{split_name.upper()} set: {len(y_split):,} samples, "
                           f"class balance = {class_balance:.3f}, "
                           f"date range = {ts_split.min()} to {ts_split.max()}")
        
        # Validate splits don't overlap and cover expected periods
        self._validate_temporal_splits(splits)
        
        return splits
    
    def _validate_temporal_splits(self, splits: Dict):
        """Validate temporal splits for correctness"""
        # Check for temporal overlap
        train_end = splits['train'][2].max()
        val_start = splits['val'][2].min()
        val_end = splits['val'][2].max()
        test_start = splits['test'][2].min()
        
        if val_start <= train_end:
            self.logger.warning("Validation set overlaps with training set")
        
        if test_start <= val_end:
            self.logger.warning("Test set overlaps with validation set")
        
        # Check for minimum sample requirements
        min_samples = self.batch_size * 2  # At least 2 batches
        for split_name, (X_split, _, _) in splits.items():
            if len(X_split) < min_samples:
                self.logger.warning(f"{split_name} set has only {len(X_split)} samples "
                                  f"(minimum recommended: {min_samples})")
    
    def create_data_generators(self, X: np.ndarray, y: np.ndarray, 
                              batch_size: Optional[int] = None,
                              shuffle: bool = True) -> Generator:
        """
        Create efficient data generators for training
        
        Args:
            X: Feature sequences
            y: Labels
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle data
            
        Yields:
            Batches of (X_batch, y_batch)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        while True:  # Infinite generator for training
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                yield X_batch, y_batch
    
    def create_balanced_generator(self, X: np.ndarray, y: np.ndarray,
                                 batch_size: Optional[int] = None) -> Generator:
        """
        Create balanced data generator that ensures equal class representation
        
        Args:
            X: Feature sequences
            y: Labels
            batch_size: Batch size
            
        Yields:
            Balanced batches of (X_batch, y_batch)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Separate indices by class
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        
        self.logger.info(f"Creating balanced generator: {len(class_0_indices)} class 0, "
                        f"{len(class_1_indices)} class 1 samples")
        
        while True:
            # Shuffle each class independently
            np.random.shuffle(class_0_indices)
            np.random.shuffle(class_1_indices)
            
            # Create balanced batches
            samples_per_class = batch_size // 2
            
            max_batches = min(len(class_0_indices), len(class_1_indices)) // samples_per_class
            
            for batch_idx in range(max_batches):
                # Select equal samples from each class
                start_idx = batch_idx * samples_per_class
                end_idx = start_idx + samples_per_class
                
                batch_indices_0 = class_0_indices[start_idx:end_idx]
                batch_indices_1 = class_1_indices[start_idx:end_idx]
                
                # Combine and shuffle batch indices
                batch_indices = np.concatenate([batch_indices_0, batch_indices_1])
                np.random.shuffle(batch_indices)
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                yield X_batch, y_batch
    
    def get_sequence_statistics(self) -> Dict:
        """
        Get comprehensive statistics about created sequences
        
        Returns:
            Dictionary containing detailed sequence statistics
        """
        if not hasattr(self, 'sequence_stats') or not self.sequence_stats:
            return {"error": "No sequences have been created yet"}
        
        return self.sequence_stats.copy()
    
    def analyze_temporal_patterns(self, y: np.ndarray, timestamps: pd.DatetimeIndex) -> Dict:
        """
        Analyze temporal patterns in the target variable
        
        Args:
            y: Labels array
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing temporal pattern analysis
        """
        self.logger.info("Analyzing temporal patterns in target variable")
        
        # Create DataFrame for analysis
        df = pd.DataFrame({'labels': y, 'timestamp': timestamps})
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        
        temporal_analysis = {
            'yearly_patterns': df.groupby('year')['labels'].agg(['mean', 'count']).to_dict(),
            'monthly_patterns': df.groupby('month')['labels'].agg(['mean', 'count']).to_dict(),
            'weekday_patterns': df.groupby('day_of_week')['labels'].agg(['mean', 'count']).to_dict(),
            'hourly_patterns': df.groupby('hour')['labels'].agg(['mean', 'count']).to_dict()
        }
        
        # Identify potential biases
        yearly_means = [temporal_analysis['yearly_patterns']['mean'][year] 
                       for year in temporal_analysis['yearly_patterns']['mean']]
        monthly_means = [temporal_analysis['monthly_patterns']['mean'][month] 
                        for month in temporal_analysis['monthly_patterns']['mean']]
        
        temporal_analysis['bias_analysis'] = {
            'yearly_bias_range': max(yearly_means) - min(yearly_means) if yearly_means else 0,
            'monthly_bias_range': max(monthly_means) - min(monthly_means) if monthly_means else 0,
            'potential_seasonality': max(monthly_means) - min(monthly_means) > 0.1 if monthly_means else False
        }
        
        self.logger.info("Temporal pattern analysis completed")
        return temporal_analysis
    
    def export_sequence_summary(self, output_path: Optional[str] = None) -> str:
        """
        Export comprehensive sequence preparation summary
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to exported summary file
        """
        from datetime import datetime
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"sequence_summary_{timestamp}.txt"
        
        # Generate comprehensive report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("SEQUENCE PREPARATION SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Sequence configuration
        report_lines.append("SEQUENCE CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Window Size: {self.window_size} hours")
        report_lines.append(f"Prediction Horizon: {self.prediction_horizon} hour(s)")
        report_lines.append(f"Batch Size: {self.batch_size}")
        report_lines.append("")
        
        # Data split configuration
        report_lines.append("TEMPORAL SPLIT CONFIGURATION")
        report_lines.append("-" * 40)
        for split_name, dates in [('Training', ['train_start', 'train_end']),
                                 ('Validation', ['val_start', 'val_end']),
                                 ('Testing', ['test_start', 'test_end'])]:
            start_date = self.split_dates[dates[0]].strftime('%Y-%m-%d')
            end_date = self.split_dates[dates[1]].strftime('%Y-%m-%d')
            report_lines.append(f"{split_name}: {start_date} to {end_date}")
        report_lines.append("")
        
        # Sequence statistics
        if hasattr(self, 'sequence_stats') and self.sequence_stats:
            stats = self.sequence_stats
            report_lines.append("SEQUENCE STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Sequences: {stats['total_sequences']:,}")
            report_lines.append(f"Features per Sequence: {stats['num_features']}")
            report_lines.append(f"Target Currency: {stats['target_pair']}")
            report_lines.append(f"Positive Labels: {stats['positive_labels']:,} ({stats['class_balance']:.1%})")
            report_lines.append(f"Negative Labels: {stats['negative_labels']:,} ({1-stats['class_balance']:.1%})")
            report_lines.append(f"Temporal Range: {stats['temporal_range']['start']} to {stats['temporal_range']['end']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Sequence summary exported to: {output_path}")
        return output_path