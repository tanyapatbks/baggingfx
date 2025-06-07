"""
Data Loading and Initial Exploration Module
Handles loading of multi-currency OHLCV data and performs initial data quality analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

class CurrencyDataLoader:
    """
    Comprehensive data loader for multi-currency forex data
    Provides data loading, validation, and initial exploration capabilities
    """
    
    def __init__(self, config):
        """
        Initialize the data loader with configuration settings
        
        Args:
            config: Configuration object containing file paths and parameters
        """
        self.config = config
        self.currency_pairs = config.CURRENCY_PAIRS
        self.file_paths = config.CURRENCY_FILES
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage for loaded data
        self.raw_data = {}
        self.data_info = {}
        
    def load_currency_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all currency pairs
        
        Returns:
            Dictionary containing DataFrames for each currency pair
        """
        self.logger.info("Starting to load currency data for all pairs")
        
        for pair in self.currency_pairs:
            try:
                file_path = self.file_paths[pair]
                
                # Check if file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data file for {pair} not found at {file_path}")
                
                # Load CSV data with proper parsing
                self.logger.info(f"Loading data for {pair} from {file_path}")
                
                df = pd.read_csv(file_path)
                
                # Validate required columns exist
                required_columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns for {pair}: {missing_columns}")
                
                # Debug and convert Local time to datetime
                self.logger.info(f"Converting Local time to datetime for {pair}...")
                self.logger.info(f"Sample datetime values: {df['Local time'].head(3).tolist()}")
                
                try:
                    # Strategy 1: Direct conversion
                    df['Local time'] = pd.to_datetime(df['Local time'], infer_datetime_format=True)
                    self.logger.info(f"Direct datetime conversion successful for {pair}")
                except Exception as e1:
                    self.logger.warning(f"Direct conversion failed for {pair}: {str(e1)}")
                    
                    try:
                        # Strategy 2: Clean and retry (remove timezone info)
                        df['Local time'] = df['Local time'].astype(str).str.replace(r' GMT[+-]\d{4}', '', regex=True)
                        df['Local time'] = df['Local time'].str.replace(r'\.000', '', regex=True)
                        df['Local time'] = pd.to_datetime(df['Local time'], infer_datetime_format=True)
                        self.logger.info(f"Cleaned datetime conversion successful for {pair}")
                    except Exception as e2:
                        self.logger.warning(f"Cleaned conversion failed for {pair}: {str(e2)}")
                        
                        try:
                            # Strategy 3: Specific format
                            df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S')
                            self.logger.info(f"Format-specific conversion successful for {pair}")
                        except Exception as e3:
                            self.logger.error(f"All datetime conversion strategies failed for {pair}")
                            raise ValueError(f"Unable to parse datetime in {pair}. Sample: {df['Local time'].head().tolist()}")
                
                # Set as index and sort
                df.set_index('Local time', inplace=True)
                df.sort_index(inplace=True)
                
                # Verify datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    raise ValueError(f"Failed to create DatetimeIndex for {pair}")
                
                # Log successful conversion
                self.logger.info(f"Successfully created DatetimeIndex for {pair}: "
                               f"{df.index.min()} to {df.index.max()} ({len(df)} records)")
                
                # Store the loaded data
                self.raw_data[pair] = df
                self.logger.info(f"Successfully loaded and processed {len(df)} rows for {pair}")
                
            except Exception as e:
                self.logger.error(f"Error loading data for {pair}: {str(e)}")
                raise
        
        self.logger.info(f"Successfully loaded data for {len(self.raw_data)} currency pairs")
        return self.raw_data
    
    def explore_data_structure(self) -> Dict[str, Dict]:
        """
        Perform comprehensive data structure analysis
        
        Returns:
            Dictionary containing structural information for each currency pair
        """
        self.logger.info("Analyzing data structure for all currency pairs")
        
        structure_info = {}
        
        for pair, df in self.raw_data.items():
            # Verify that index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error(f"Index for {pair} is not DatetimeIndex: {type(df.index)}")
                continue
                
            try:
                info = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'index_type': type(df.index).__name__,
                    'date_range': {
                        'start': df.index.min(),
                        'end': df.index.max(),
                        'duration_days': (df.index.max() - df.index.min()).days
                    }
                }
                structure_info[pair] = info
                
                self.logger.info(f"{pair}: {info['shape'][0]} rows × {info['shape'][1]} columns, "
                               f"from {info['date_range']['start']} to {info['date_range']['end']}")
                               
            except Exception as e:
                self.logger.error(f"Error analyzing structure for {pair}: {str(e)}")
                # Provide basic info even if date calculation fails
                structure_info[pair] = {
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'dtypes': df.dtypes.to_dict(),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'index_type': type(df.index).__name__,
                    'error': str(e)
                }
        
        self.data_info['structure'] = structure_info
        return structure_info
    
    def check_data_quality(self) -> Dict[str, Dict]:
        """
        Comprehensive data quality assessment
        
        Returns:
            Dictionary containing quality metrics for each currency pair
        """
        self.logger.info("Performing data quality assessment")
        
        quality_info = {}
        
        for pair, df in self.raw_data.items():
            # Calculate missing values
            missing_values = df.isnull().sum()
            missing_percentages = (missing_values / len(df) * 100).round(2)
            
            # Check for duplicated timestamps
            duplicated_timestamps = df.index.duplicated().sum()
            
            # Calculate zero/negative values for price data
            price_columns = ['Open', 'High', 'Low', 'Close']
            zero_prices = {col: (df[col] <= 0).sum() for col in price_columns}
            
            # Check for OHLC consistency
            ohlc_issues = {
                'high_below_low': (df['High'] < df['Low']).sum(),
                'high_below_open': (df['High'] < df['Open']).sum(),
                'high_below_close': (df['High'] < df['Close']).sum(),
                'low_above_open': (df['Low'] > df['Open']).sum(),
                'low_above_close': (df['Low'] > df['Close']).sum()
            }
            
            # Calculate volume statistics
            volume_stats = {
                'zero_volume': (df['Volume'] == 0).sum(),
                'negative_volume': (df['Volume'] < 0).sum(),
                'zero_volume_percentage': ((df['Volume'] == 0).sum() / len(df) * 100).round(2)
            }
            
            quality_info[pair] = {
                'missing_values': missing_values.to_dict(),
                'missing_percentages': missing_percentages.to_dict(),
                'duplicated_timestamps': duplicated_timestamps,
                'zero_prices': zero_prices,
                'ohlc_consistency_issues': ohlc_issues,
                'volume_statistics': volume_stats,
                'data_completeness': (1 - missing_values.sum() / (len(df) * len(df.columns))) * 100
            }
            
            self.logger.info(f"{pair} data quality: {quality_info[pair]['data_completeness']:.2f}% complete")
            
            # Log any significant issues
            if duplicated_timestamps > 0:
                self.logger.warning(f"{pair}: Found {duplicated_timestamps} duplicated timestamps")
            
            if sum(ohlc_issues.values()) > 0:
                self.logger.warning(f"{pair}: Found OHLC consistency issues: {ohlc_issues}")
        
        self.data_info['quality'] = quality_info
        return quality_info
    
    def analyze_date_ranges(self) -> Dict[str, Dict]:
        """
        Analyze temporal coverage and identify gaps
        
        Returns:
            Dictionary containing date range analysis for each currency pair
        """
        self.logger.info("Analyzing date ranges and temporal gaps")
        
        date_analysis = {}
        
        for pair, df in self.raw_data.items():
            # Calculate expected hourly frequency
            date_range = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq='1H'
            )
            
            # Find missing timestamps
            missing_timestamps = date_range.difference(df.index)
            
            # Identify gaps larger than expected
            if len(df) > 1:
                time_diffs = df.index.to_series().diff()
                large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=1)]
            else:
                large_gaps = pd.Series([], dtype='timedelta64[ns]')
            
            # Weekend/holiday coverage analysis
            weekday_coverage = df.groupby(df.index.dayofweek).size()
            
            date_analysis[pair] = {
                'expected_records': len(date_range),
                'actual_records': len(df),
                'coverage_percentage': round((len(df) / len(date_range) * 100), 2),
                'missing_timestamps_count': len(missing_timestamps),
                'large_gaps_count': len(large_gaps),
                'largest_gap_hours': large_gaps.max().total_seconds() / 3600 if len(large_gaps) > 0 else 0,
                'weekday_distribution': weekday_coverage.to_dict(),
                'first_record': df.index.min(),
                'last_record': df.index.max()
            }
            
            self.logger.info(f"{pair}: {date_analysis[pair]['coverage_percentage']:.2f}% temporal coverage, "
                           f"{len(missing_timestamps)} missing hours")
        
        self.data_info['dates'] = date_analysis
        return date_analysis
    
    def calculate_basic_statistics(self) -> Dict[str, Dict]:
        """
        Calculate comprehensive statistical summaries
        
        Returns:
            Dictionary containing statistical information for each currency pair
        """
        self.logger.info("Calculating basic statistics for all currency pairs")
        
        statistics = {}
        
        for pair, df in self.raw_data.items():
            # Basic descriptive statistics
            desc_stats = df.describe()
            
            # Price-specific statistics
            price_stats = {
                'daily_return_volatility': df['Close'].pct_change().std() * np.sqrt(24),  # Hourly to daily
                'price_range': {
                    'min_price': df[['Open', 'High', 'Low', 'Close']].min().min(),
                    'max_price': df[['Open', 'High', 'Low', 'Close']].max().max(),
                    'price_spread': df[['Open', 'High', 'Low', 'Close']].max().max() - 
                                   df[['Open', 'High', 'Low', 'Close']].min().min()
                },
                'typical_spreads': {
                    'high_low_spread_mean': (df['High'] - df['Low']).mean(),
                    'high_low_spread_std': (df['High'] - df['Low']).std()
                }
            }
            
            # Volume analysis
            volume_stats = {
                'volume_statistics': df['Volume'].describe().to_dict(),
                'zero_volume_ratio': (df['Volume'] == 0).mean(),
                'volume_distribution': {
                    'q25': df['Volume'].quantile(0.25),
                    'q50': df['Volume'].quantile(0.50),
                    'q75': df['Volume'].quantile(0.75),
                    'q95': df['Volume'].quantile(0.95)
                }
            }
            
            # Trend analysis
            returns = df['Close'].pct_change().dropna()
            trend_stats = {
                'returns_statistics': {
                    'mean_return': returns.mean(),
                    'return_volatility': returns.std(),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'positive_return_ratio': (returns > 0).mean()
                }
            }
            
            statistics[pair] = {
                'descriptive_statistics': desc_stats.to_dict(),
                'price_analysis': price_stats,
                'volume_analysis': volume_stats,
                'trend_analysis': trend_stats
            }
            
            self.logger.info(f"{pair}: Mean price = {price_stats['price_range']['min_price']:.5f} - "
                           f"{price_stats['price_range']['max_price']:.5f}, "
                           f"Daily volatility = {price_stats['daily_return_volatility']:.4f}")
        
        self.data_info['statistics'] = statistics
        return statistics
    
    def validate_temporal_alignment(self) -> Dict[str, any]:
        """
        Check if all currency pairs have aligned timestamps for multi-currency analysis
        
        Returns:
            Dictionary containing alignment validation results
        """
        self.logger.info("Validating temporal alignment across currency pairs")
        
        if len(self.raw_data) < 2:
            return {'status': 'insufficient_data', 'message': 'Need at least 2 currency pairs'}
        
        # Get common time index
        common_index = None
        for pair, df in self.raw_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        alignment_info = {
            'total_common_timestamps': len(common_index),
            'alignment_percentage': {},
            'missing_in_pairs': {},
            'alignment_status': 'perfect' if len(common_index) > 0 else 'misaligned'
        }
        
        for pair, df in self.raw_data.items():
            alignment_percentage = (len(common_index) / len(df) * 100) if len(df) > 0 else 0
            missing_count = len(df.index.difference(common_index))
            
            alignment_info['alignment_percentage'][pair] = alignment_percentage
            alignment_info['missing_in_pairs'][pair] = missing_count
            
            self.logger.info(f"{pair}: {alignment_percentage:.2f}% aligned with common timestamps")
        
        if len(common_index) == 0:
            self.logger.error("No common timestamps found across currency pairs!")
            alignment_info['alignment_status'] = 'critical_misalignment'
        elif min(alignment_info['alignment_percentage'].values()) < 90:
            self.logger.warning("Poor temporal alignment detected between currency pairs")
            alignment_info['alignment_status'] = 'poor_alignment'
        
        return alignment_info
    
    def get_summary_report(self) -> str:
        """
        Generate comprehensive summary report of loaded data
        
        Returns:
            Formatted string containing complete data summary
        """
        if not self.raw_data:
            return "No data loaded. Please run load_currency_data() first."
        
        report = []
        report.append("="*80)
        report.append("MULTI-CURRENCY FOREX DATA LOADING SUMMARY REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overview section
        report.append("OVERVIEW")
        report.append("-" * 40)
        report.append(f"Currency Pairs Loaded: {len(self.raw_data)}")
        report.append(f"Pairs: {', '.join(self.raw_data.keys())}")
        report.append("")
        
        # Individual pair details
        for pair, df in self.raw_data.items():
            report.append(f"{pair} DETAILS")
            report.append("-" * 40)
            report.append(f"Records: {len(df):,}")
            report.append(f"Date Range: {df.index.min()} to {df.index.max()}")
            report.append(f"Columns: {', '.join(df.columns)}")
            
            if 'quality' in self.data_info:
                quality = self.data_info['quality'][pair]
                report.append(f"Data Completeness: {quality['data_completeness']:.2f}%")
                report.append(f"Zero Volume Records: {quality['volume_statistics']['zero_volume']:,}")
            
            if 'statistics' in self.data_info:
                stats = self.data_info['statistics'][pair]
                price_range = stats['price_analysis']['price_range']
                report.append(f"Price Range: {price_range['min_price']:.5f} - {price_range['max_price']:.5f}")
            
            report.append("")
        
        # Temporal alignment summary
        if hasattr(self, 'alignment_info'):
            report.append("TEMPORAL ALIGNMENT")
            report.append("-" * 40)
            report.append(f"Common Timestamps: {self.alignment_info['total_common_timestamps']:,}")
            report.append(f"Alignment Status: {self.alignment_info['alignment_status']}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)
    
    def export_data_summary(self, output_path: Optional[str] = None) -> str:
        """
        Export comprehensive data summary to file
        
        Args:
            output_path: Optional path for output file
            
        Returns:
            Path to exported summary file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"data_summary_{timestamp}.txt"
        
        summary_report = self.get_summary_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        self.logger.info(f"Data summary exported to: {output_path}")
        return output_path