"""
Data Preprocessing and Cleaning Module
Handles all data transformation steps including missing value treatment,
normalization, and feature engineering for multi-currency forex data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ForexDataPreprocessor:
    """
    Comprehensive preprocessor for multi-currency forex OHLCV data
    Implements sophisticated cleaning and normalization strategies
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration settings
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers for different normalization approaches
        self.scalers = {}
        self.feature_stats = {}
        
        # Preprocessing flags and metadata
        self.is_fitted = False
        self.preprocessing_steps = []
        
    def handle_missing_values(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Intelligent missing value handling with forex-specific logic
        
        Args:
            data: Dictionary of currency pair DataFrames
            
        Returns:
            Dictionary of DataFrames with missing values handled
        """
        self.logger.info("Starting comprehensive missing value treatment")
        
        processed_data = {}
        missing_value_report = {}
        
        for pair, df in data.items():
            self.logger.info(f"Processing missing values for {pair}")
            df_processed = df.copy()
            
            # Track missing values before processing
            initial_missing = df_processed.isnull().sum()
            missing_value_report[pair] = {'initial': initial_missing.to_dict()}
            
            # Step 1: Handle short gaps with forward fill
            for column in ['Open', 'High', 'Low', 'Close']:
                # Forward fill for gaps up to max_fill_hours
                df_processed[column] = df_processed[column].fillna(method='ffill', limit=self.config.MAX_FILL_HOURS)
                
                # Backward fill for any remaining gaps at the beginning
                df_processed[column] = df_processed[column].fillna(method='bfill', limit=2)
            
            # Step 2: Handle Volume separately (forex volume can be zero legitimately)
            # Use forward fill for volume, but don't interpolate - use zero for gaps
            df_processed['Volume'] = df_processed['Volume'].fillna(method='ffill', limit=self.config.MAX_FILL_HOURS)
            df_processed['Volume'] = df_processed['Volume'].fillna(0)  # Fill remaining with zero
            
            # Step 3: Detect and handle long gaps with interpolation
            long_gaps_mask = self._detect_long_gaps(df_processed)
            
            if long_gaps_mask.any():
                self.logger.info(f"{pair}: Found {long_gaps_mask.sum()} records in long gaps, applying interpolation")
                
                # Apply interpolation for long gaps
                price_columns = ['Open', 'High', 'Low', 'Close']
                for column in price_columns:
                    # Use linear interpolation for price data in long gaps
                    df_processed.loc[long_gaps_mask, column] = df_processed[column].interpolate(
                        method=self.config.INTERPOLATION_METHOD,
                        limit_direction='both'
                    )[long_gaps_mask]
            
            # Step 4: Handle any remaining missing values with market-specific logic
            remaining_missing = df_processed.isnull().sum()
            
            if remaining_missing.sum() > 0:
                self.logger.warning(f"{pair}: {remaining_missing.sum()} missing values remain after processing")
                
                # For any remaining missing values, use the most conservative approach
                for column in ['Open', 'High', 'Low', 'Close']:
                    if remaining_missing[column] > 0:
                        # Use the last valid value or market-typical value
                        if not df_processed[column].dropna().empty:
                            median_value = df_processed[column].median()
                            df_processed[column] = df_processed[column].fillna(median_value)
                        else:
                            self.logger.error(f"{pair}: No valid values found for {column}")
            
            # Step 5: OHLCV consistency check and correction
            df_processed = self._ensure_ohlcv_consistency(df_processed)
            
            # Track final missing values
            final_missing = df_processed.isnull().sum()
            missing_value_report[pair]['final'] = final_missing.to_dict()
            missing_value_report[pair]['improvement'] = ((initial_missing - final_missing) / initial_missing * 100).fillna(0).to_dict()
            
            processed_data[pair] = df_processed
            
            self.logger.info(f"{pair}: Missing value treatment complete. "
                           f"Removed {(initial_missing - final_missing).sum()} missing values")
        
        # Log overall missing value treatment summary
        self._log_missing_value_summary(missing_value_report)
        self.preprocessing_steps.append('missing_values_handled')
        
        return processed_data
    
    def _detect_long_gaps(self, df: pd.DataFrame, threshold_hours: int = None) -> pd.Series:
        """
        Detect records that are part of long market gaps (weekends, holidays)
        
        Args:
            df: DataFrame to analyze
            threshold_hours: Hours threshold for considering a gap as 'long'
            
        Returns:
            Boolean series indicating records in long gaps
        """
        if threshold_hours is None:
            threshold_hours = self.config.MAX_FILL_HOURS * 2
        
        # Calculate time differences
        time_diffs = df.index.to_series().diff()
        
        # Identify start of long gaps
        long_gap_starts = time_diffs > pd.Timedelta(hours=threshold_hours)
        
        # For simplicity, mark records with any missing values as potential gap records
        has_missing = df.isnull().any(axis=1)
        
        return has_missing & long_gap_starts
    
    def _ensure_ohlcv_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure OHLCV data consistency and fix logical violations
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with consistent OHLCV data
        """
        df_clean = df.copy()
        
        # Rule 1: High should be >= max(Open, Close) and >= Low
        df_clean['High'] = np.maximum(
            df_clean['High'],
            np.maximum(df_clean['Open'], np.maximum(df_clean['Close'], df_clean['Low']))
        )
        
        # Rule 2: Low should be <= min(Open, Close) and <= High
        df_clean['Low'] = np.minimum(
            df_clean['Low'],
            np.minimum(df_clean['Open'], np.minimum(df_clean['Close'], df_clean['High']))
        )
        
        # Rule 3: Volume should be non-negative
        df_clean['Volume'] = np.maximum(df_clean['Volume'], 0)
        
        return df_clean
    
    def calculate_percentage_returns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate percentage returns for OHLC data to create stationary series
        
        Args:
            data: Dictionary of currency pair DataFrames
            
        Returns:
            Dictionary of DataFrames with percentage returns for OHLC
        """
        self.logger.info("Calculating percentage returns for OHLC features")
        
        processed_data = {}
        
        for pair, df in data.items():
            df_returns = df.copy()
            
            # Calculate percentage returns for OHLC
            for column in ['Open', 'High', 'Low', 'Close']:
                # Calculate percentage change
                returns = df[column].pct_change()
                
                # Handle infinite values and first NaN
                returns = returns.replace([np.inf, -np.inf], np.nan)
                returns = returns.fillna(0)  # First observation gets 0 return
                
                # Additional safety: remove extreme outliers
                # Clip returns to reasonable range (-50% to +50% per hour)
                returns = returns.clip(-0.5, 0.5)
                
                # Store returns with clear naming
                df_returns[f'{column}_Return'] = returns
                
                # Keep original prices for reference
                df_returns[f'{column}_Price'] = df[column]
            
            # Volume remains as-is (will be normalized separately)
            # But calculate volume change rate for additional information
            volume_change = df['Volume'].pct_change().fillna(0)
            df_returns['Volume_Change'] = volume_change
            df_returns['Volume_Original'] = df['Volume']
            
            processed_data[pair] = df_returns
            
            # Log some statistics about the returns
            returns_stats = {}
            for column in ['Open', 'High', 'Low', 'Close']:
                returns_col = f'{column}_Return'
                returns_stats[returns_col] = {
                    'mean': df_returns[returns_col].mean(),
                    'std': df_returns[returns_col].std(),
                    'min': df_returns[returns_col].min(),
                    'max': df_returns[returns_col].max()
                }
            
            self.logger.info(f"{pair}: Percentage returns calculated. "
                           f"Close return volatility: {returns_stats['Close_Return']['std']:.6f}")
        
        self.preprocessing_steps.append('percentage_returns_calculated')
        return processed_data
    
    def apply_mixed_normalization(self, data: Dict[str, pd.DataFrame], 
                                 fit_data: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Apply mixed normalization strategy: StandardScaler + MinMaxScaler
        
        Args:
            data: Dictionary of currency pair DataFrames
            fit_data: Whether to fit scalers on this data (True for training)
            
        Returns:
            Dictionary of normalized DataFrames
        """
        self.logger.info("Applying mixed normalization strategy")
        
        if fit_data:
            self.scalers = {}
            self.feature_stats = {}
        
        normalized_data = {}
        
        for pair, df in data.items():
            df_normalized = df.copy()
            
            if fit_data:
                self.scalers[pair] = {}
                self.feature_stats[pair] = {}
            
            # Normalize return features (OHLC returns)
            return_columns = ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return']
            
            for column in return_columns:
                if column in df.columns:
                    # Check for problematic values before normalization
                    data_to_normalize = df[[column]].copy()
                    
                    # Remove infinite values
                    data_to_normalize = data_to_normalize.replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values with 0
                    data_to_normalize = data_to_normalize.fillna(0)
                    
                    # Additional safety check for extreme values
                    # Remove values beyond reasonable range for forex returns
                    Q1 = data_to_normalize[column].quantile(0.01)
                    Q99 = data_to_normalize[column].quantile(0.99)
                    data_to_normalize[column] = data_to_normalize[column].clip(Q1, Q99)
                    
                    if fit_data:
                        # Fit StandardScaler for returns
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(data_to_normalize)
                        self.scalers[pair][column] = scaler
                        
                        # Store statistics
                        self.feature_stats[pair][column] = {
                            'mean': scaler.mean_[0],
                            'std': scaler.scale_[0],
                            'type': 'standard'
                        }
                    else:
                        # Transform using fitted scaler
                        if pair in self.scalers and column in self.scalers[pair]:
                            scaled_data = self.scalers[pair][column].transform(data_to_normalize)
                        else:
                            self.logger.warning(f"No fitted scaler found for {pair}:{column}")
                            scaled_data = data_to_normalize.values
                    
                    df_normalized[column] = scaled_data.flatten()
            
            # Normalize volume separately with MinMaxScaler
            volume_columns = ['Volume_Original', 'Volume_Change']
            
            for column in volume_columns:
                if column in df.columns:
                    # Check for problematic values before normalization
                    data_to_normalize = df[[column]].copy()
                    
                    # Remove infinite values
                    data_to_normalize = data_to_normalize.replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values with 0 for volume data
                    data_to_normalize = data_to_normalize.fillna(0)
                    
                    # Ensure all values are non-negative for volume
                    data_to_normalize[column] = data_to_normalize[column].clip(lower=0)
                    
                    if fit_data:
                        # Fit MinMaxScaler for volume
                        scaler = MinMaxScaler(feature_range=(0, 1))
                        scaled_data = scaler.fit_transform(data_to_normalize)
                        self.scalers[pair][column] = scaler
                        
                        # Store statistics
                        self.feature_stats[pair][column] = {
                            'min': scaler.data_min_[0],
                            'max': scaler.data_max_[0],
                            'type': 'minmax'
                        }
                    else:
                        # Transform using fitted scaler
                        if pair in self.scalers and column in self.scalers[pair]:
                            scaled_data = self.scalers[pair][column].transform(data_to_normalize)
                        else:
                            self.logger.warning(f"No fitted scaler found for {pair}:{column}")
                            scaled_data = data_to_normalize.values
                    
                    df_normalized[column] = scaled_data.flatten()
            
            normalized_data[pair] = df_normalized
            
            if fit_data:
                self.logger.info(f"{pair}: Fitted scalers for {len(self.scalers[pair])} features")
            else:
                self.logger.info(f"{pair}: Applied normalization transformations")
        
        if fit_data:
            self.is_fitted = True
            self.preprocessing_steps.append('normalization_fitted')
        else:
            self.preprocessing_steps.append('normalization_applied')
        
        return normalized_data
    
    def detect_market_hours_gaps(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Detect and analyze market closure periods (weekends, holidays)
        
        Args:
            data: Dictionary of currency pair DataFrames
            
        Returns:
            Dictionary containing gap analysis for each currency pair
        """
        self.logger.info("Detecting market hours gaps and closure periods")
        
        gap_analysis = {}
        
        for pair, df in data.items():
            # Calculate time differences between consecutive records
            time_diffs = df.index.to_series().diff().dropna()
            
            # Standard hourly forex data should have 1-hour gaps
            normal_gap = pd.Timedelta(hours=1)
            
            # Identify gaps larger than normal
            large_gaps = time_diffs[time_diffs > normal_gap]
            
            # Classify gaps
            weekend_gaps = []
            holiday_gaps = []
            unusual_gaps = []
            
            for timestamp, gap_duration in large_gaps.items():
                gap_hours = gap_duration.total_seconds() / 3600
                
                # Check if gap occurs over weekend (Friday to Monday)
                gap_start = timestamp - gap_duration
                if gap_start.weekday() == 4 and timestamp.weekday() == 0:  # Friday to Monday
                    weekend_gaps.append((gap_start, timestamp, gap_hours))
                elif gap_hours > 24:  # Likely holiday
                    holiday_gaps.append((gap_start, timestamp, gap_hours))
                else:  # Unusual gap
                    unusual_gaps.append((gap_start, timestamp, gap_hours))
            
            gap_analysis[pair] = {
                'total_gaps': len(large_gaps),
                'weekend_gaps': len(weekend_gaps),
                'holiday_gaps': len(holiday_gaps),
                'unusual_gaps': len(unusual_gaps),
                'largest_gap_hours': large_gaps.max().total_seconds() / 3600 if len(large_gaps) > 0 else 0,
                'average_gap_hours': large_gaps.mean().total_seconds() / 3600 if len(large_gaps) > 0 else 0,
                'weekend_gap_details': weekend_gaps[:5],  # First 5 for logging
                'holiday_gap_details': holiday_gaps[:3]   # First 3 for logging
            }
            
            self.logger.info(f"{pair}: Found {len(large_gaps)} market gaps - "
                           f"{len(weekend_gaps)} weekends, {len(holiday_gaps)} holidays, "
                           f"{len(unusual_gaps)} unusual")
        
        return gap_analysis
    
    def create_feature_matrix(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create unified feature matrix for multi-currency modeling
        
        Args:
            data: Dictionary of normalized currency pair DataFrames
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        self.logger.info("Creating unified multi-currency feature matrix")
        
        # Define the features we want to use for modeling
        model_features = ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return', 'Volume_Original']
        
        feature_matrices = []
        feature_names = []
        
        # Get common time index across all pairs
        common_index = None
        for pair, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        self.logger.info(f"Common time index has {len(common_index)} timestamps")
        
        # Create feature matrix for each currency pair
        for pair in self.config.CURRENCY_PAIRS:  # Maintain consistent order
            if pair in data:
                df = data[pair].loc[common_index]  # Align to common index
                
                # Extract model features
                pair_features = df[model_features].copy()
                
                # Rename columns to include currency pair
                pair_features.columns = [f'{pair}_{col}' for col in pair_features.columns]
                
                feature_matrices.append(pair_features)
                feature_names.extend(pair_features.columns.tolist())
        
        # Concatenate all currency features horizontally
        if feature_matrices:
            unified_matrix = pd.concat(feature_matrices, axis=1)
            unified_matrix = unified_matrix.dropna()  # Remove any remaining NaN rows
            
            self.logger.info(f"Created unified feature matrix: {unified_matrix.shape[0]} samples × "
                           f"{unified_matrix.shape[1]} features")
            
            # Verify we have expected number of features (5 features × 3 currencies = 15)
            expected_features = len(self.config.CURRENCY_PAIRS) * len(model_features)
            if unified_matrix.shape[1] != expected_features:
                self.logger.warning(f"Expected {expected_features} features, got {unified_matrix.shape[1]}")
        else:
            self.logger.error("No feature matrices created - check input data")
            unified_matrix = pd.DataFrame()
            feature_names = []
        
        return unified_matrix, feature_names
    
    def _log_missing_value_summary(self, missing_report: Dict):
        """Log comprehensive missing value treatment summary"""
        self.logger.info("Missing Value Treatment Summary:")
        self.logger.info("-" * 50)
        
        for pair, report in missing_report.items():
            initial_total = sum(report['initial'].values())
            final_total = sum(report['final'].values())
            improvement_total = initial_total - final_total
            
            self.logger.info(f"{pair}:")
            self.logger.info(f"  Initial missing: {initial_total:,}")
            self.logger.info(f"  Final missing: {final_total:,}")
            self.logger.info(f"  Improvement: {improvement_total:,} ({improvement_total/max(initial_total, 1)*100:.1f}%)")
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get comprehensive summary of preprocessing steps and transformations
        
        Returns:
            Dictionary containing preprocessing summary
        """
        summary = {
            'steps_completed': self.preprocessing_steps,
            'is_fitted': self.is_fitted,
            'scalers_fitted': len(self.scalers) if hasattr(self, 'scalers') else 0,
            'currency_pairs_processed': list(self.scalers.keys()) if hasattr(self, 'scalers') else [],
            'feature_statistics': self.feature_stats if hasattr(self, 'feature_stats') else {}
        }
        
        return summary
    
    def save_preprocessing_artifacts(self, output_dir: str):
        """
        Save preprocessing artifacts (scalers, statistics) for later use
        
        Args:
            output_dir: Directory to save artifacts
        """
        import joblib
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.is_fitted:
            # Save scalers
            scaler_path = os.path.join(output_dir, 'preprocessing_scalers.joblib')
            joblib.dump(self.scalers, scaler_path)
            
            # Save feature statistics
            stats_path = os.path.join(output_dir, 'feature_statistics.joblib')
            joblib.dump(self.feature_stats, stats_path)
            
            # Save preprocessing summary
            summary_path = os.path.join(output_dir, 'preprocessing_summary.joblib')
            joblib.dump(self.get_preprocessing_summary(), summary_path)
            
            self.logger.info(f"Preprocessing artifacts saved to {output_dir}")
        else:
            self.logger.warning("Preprocessor not fitted - no artifacts to save")
    
    def load_preprocessing_artifacts(self, input_dir: str):
        """
        Load preprocessing artifacts from saved files
        
        Args:
            input_dir: Directory containing saved artifacts
        """
        import joblib
        import os
        
        try:
            # Load scalers
            scaler_path = os.path.join(input_dir, 'preprocessing_scalers.joblib')
            if os.path.exists(scaler_path):
                self.scalers = joblib.load(scaler_path)
            
            # Load feature statistics
            stats_path = os.path.join(input_dir, 'feature_statistics.joblib')
            if os.path.exists(stats_path):
                self.feature_stats = joblib.load(stats_path)
            
            self.is_fitted = True
            self.logger.info(f"Preprocessing artifacts loaded from {input_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessing artifacts: {str(e)}")
            raise