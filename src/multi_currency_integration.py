"""
Multi-Currency Data Integration Module
Handles merging and integration of multiple currency pairs into unified datasets
while preserving cross-currency correlations and temporal relationships
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class MultiCurrencyIntegrator:
    """
    Sophisticated integrator for multi-currency forex data
    Combines multiple currency pairs while preserving temporal and correlation structures
    """
    
    def __init__(self, config):
        """
        Initialize the multi-currency integrator
        
        Args:
            config: Configuration object containing integration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Integration metadata
        self.integration_stats = {}
        self.correlation_matrix = {}
        self.alignment_info = {}
        
        # Expected feature order for consistency
        self.feature_order = ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return', 'Volume_Original']
        
    def merge_currency_pairs(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple currency pairs into unified dataset with temporal alignment
        
        Args:
            data: Dictionary containing preprocessed DataFrames for each currency pair
            
        Returns:
            Unified DataFrame with all currency features aligned temporally
        """
        self.logger.info("Starting multi-currency data integration process")
        
        # Validate input data
        if len(data) != len(self.config.CURRENCY_PAIRS):
            raise ValueError(f"Expected {len(self.config.CURRENCY_PAIRS)} currency pairs, "
                           f"got {len(data)}")
        
        # Find common temporal index across all currency pairs
        common_index = self._find_common_temporal_index(data)
        self.logger.info(f"Found {len(common_index)} common timestamps across all currency pairs")
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps found across currency pairs")
        
        # Prepare individual currency DataFrames aligned to common index
        aligned_dataframes = []
        feature_columns = []
        
        for pair in self.config.CURRENCY_PAIRS:  # Maintain consistent order
            if pair not in data:
                raise ValueError(f"Missing data for currency pair: {pair}")
            
            # Extract and align data to common index
            pair_df = data[pair].loc[common_index].copy()
            
            # Select and reorder features consistently
            selected_features = []
            for feature in self.feature_order:
                if feature in pair_df.columns:
                    selected_features.append(feature)
                else:
                    self.logger.warning(f"Feature {feature} not found in {pair} data")
            
            # Create feature matrix for this currency pair
            pair_features = pair_df[selected_features].copy()
            
            # Rename columns to include currency pair prefix
            pair_features.columns = [f'{pair}_{col}' for col in pair_features.columns]
            
            aligned_dataframes.append(pair_features)
            feature_columns.extend(pair_features.columns.tolist())
            
            self.logger.info(f"{pair}: Aligned {len(pair_features)} records with "
                           f"{len(selected_features)} features")
        
        # Concatenate all currency features horizontally
        unified_data = pd.concat(aligned_dataframes, axis=1)
        
        # Handle any remaining missing values after alignment
        initial_missing = unified_data.isnull().sum().sum()
        if initial_missing > 0:
            self.logger.warning(f"Found {initial_missing} missing values after alignment")
            unified_data = unified_data.dropna()
            self.logger.info(f"Removed {initial_missing} rows, {len(unified_data)} rows remaining")
        
        # Store integration statistics
        self.integration_stats = {
            'total_records': len(unified_data),
            'total_features': len(feature_columns),
            'feature_names': feature_columns,
            'currency_pairs': list(self.config.CURRENCY_PAIRS),
            'features_per_pair': len(self.feature_order),
            'temporal_range': {
                'start': unified_data.index.min(),
                'end': unified_data.index.max(),
                'duration_days': (unified_data.index.max() - unified_data.index.min()).days
            }
        }
        
        self.logger.info(f"Successfully created unified dataset: {unified_data.shape[0]} samples × "
                        f"{unified_data.shape[1]} features")
        
        # Validate final feature count
        expected_features = len(self.config.CURRENCY_PAIRS) * len(self.feature_order)
        if unified_data.shape[1] != expected_features:
            self.logger.warning(f"Expected {expected_features} features, got {unified_data.shape[1]}")
        
        return unified_data
    
    def _find_common_temporal_index(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """
        Find common timestamps across all currency pairs
        
        Args:
            data: Dictionary of currency pair DataFrames
            
        Returns:
            DatetimeIndex containing common timestamps
        """
        common_index = None
        index_stats = {}
        
        for pair, df in data.items():
            pair_index = df.index
            index_stats[pair] = {
                'total_records': len(pair_index),
                'start_date': pair_index.min(),
                'end_date': pair_index.max(),
                'unique_timestamps': len(pair_index.unique())
            }
            
            if common_index is None:
                common_index = pair_index
            else:
                # Find intersection of timestamps
                common_index = common_index.intersection(pair_index)
        
        # Store alignment information
        self.alignment_info = {
            'individual_stats': index_stats,
            'common_timestamps': len(common_index),
            'alignment_percentage': {}
        }
        
        # Calculate alignment percentages
        for pair, stats in index_stats.items():
            alignment_pct = (len(common_index) / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
            self.alignment_info['alignment_percentage'][pair] = alignment_pct
            
            self.logger.info(f"{pair}: {alignment_pct:.2f}% of records align with common index")
        
        return common_index
    
    def create_concatenated_features(self, unified_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create concatenated feature array suitable for CNN-LSTM input
        
        Args:
            unified_data: Unified DataFrame with all currency features
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        self.logger.info("Creating concatenated feature array for CNN-LSTM input")
        
        # Ensure features are in correct order
        ordered_features = []
        feature_names = []
        
        for pair in self.config.CURRENCY_PAIRS:
            for feature in self.feature_order:
                column_name = f'{pair}_{feature}'
                if column_name in unified_data.columns:
                    ordered_features.append(column_name)
                    feature_names.append(column_name)
                else:
                    self.logger.error(f"Expected feature {column_name} not found in unified data")
        
        # Create feature array
        if ordered_features:
            feature_array = unified_data[ordered_features].values
            
            # Validate array shape
            expected_shape = (len(unified_data), len(self.config.CURRENCY_PAIRS) * len(self.feature_order))
            actual_shape = feature_array.shape
            
            if actual_shape != expected_shape:
                self.logger.warning(f"Feature array shape {actual_shape} differs from expected {expected_shape}")
            
            self.logger.info(f"Created feature array: {feature_array.shape[0]} samples × "
                           f"{feature_array.shape[1]} features")
            
            # Check for any remaining issues
            if np.isnan(feature_array).any():
                nan_count = np.isnan(feature_array).sum()
                self.logger.warning(f"Found {nan_count} NaN values in feature array")
            
            if np.isinf(feature_array).any():
                inf_count = np.isinf(feature_array).sum()
                self.logger.warning(f"Found {inf_count} infinite values in feature array")
        
        else:
            self.logger.error("No valid features found for concatenation")
            feature_array = np.array([])
            feature_names = []
        
        return feature_array, feature_names
    
    def validate_temporal_alignment(self, unified_data: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of temporal alignment in unified dataset
        
        Args:
            unified_data: Unified DataFrame to validate
            
        Returns:
            Dictionary containing detailed alignment validation results
        """
        self.logger.info("Performing comprehensive temporal alignment validation")
        
        validation_results = {
            'timestamp_analysis': {},
            'gap_analysis': {},
            'frequency_analysis': {},
            'alignment_quality': {}
        }
        
        # Timestamp analysis
        timestamps = unified_data.index
        validation_results['timestamp_analysis'] = {
            'total_timestamps': len(timestamps),
            'unique_timestamps': len(timestamps.unique()),
            'duplicated_timestamps': (len(timestamps) - len(timestamps.unique())),
            'date_range': {
                'start': timestamps.min(),
                'end': timestamps.max(),
                'duration_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600
            }
        }
        
        # Gap analysis
        if len(timestamps) > 1:
            time_diffs = timestamps.to_series().diff().dropna()
            expected_gap = pd.Timedelta(hours=1)
            
            validation_results['gap_analysis'] = {
                'normal_gaps': (time_diffs == expected_gap).sum(),
                'large_gaps': (time_diffs > expected_gap).sum(),
                'gap_statistics': {
                    'min_gap_hours': time_diffs.min().total_seconds() / 3600,
                    'max_gap_hours': time_diffs.max().total_seconds() / 3600,
                    'median_gap_hours': time_diffs.median().total_seconds() / 3600,
                    'mean_gap_hours': time_diffs.mean().total_seconds() / 3600
                }
            }
        
        # Frequency analysis
        if len(timestamps) > 1:
            inferred_freq = pd.infer_freq(timestamps[:100])  # Sample for speed
            validation_results['frequency_analysis'] = {
                'inferred_frequency': inferred_freq,
                'expected_frequency': '1H',
                'frequency_consistent': inferred_freq == '1H' if inferred_freq else False
            }
        
        # Overall alignment quality assessment
        quality_score = 100  # Start with perfect score
        
        # Deduct points for issues
        if validation_results['timestamp_analysis']['duplicated_timestamps'] > 0:
            quality_score -= 20
            
        if 'gap_analysis' in validation_results:
            large_gaps_ratio = (validation_results['gap_analysis']['large_gaps'] / 
                              max(len(timestamps) - 1, 1))
            quality_score -= min(large_gaps_ratio * 50, 30)  # Max 30 point deduction
        
        if not validation_results['frequency_analysis'].get('frequency_consistent', True):
            quality_score -= 10
        
        validation_results['alignment_quality'] = {
            'overall_score': max(quality_score, 0),
            'quality_rating': self._get_quality_rating(quality_score),
            'recommendations': self._get_quality_recommendations(quality_score, validation_results)
        }
        
        self.logger.info(f"Temporal alignment quality: {quality_score:.1f}/100 "
                        f"({validation_results['alignment_quality']['quality_rating']})")
        
        return validation_results
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def _get_quality_recommendations(self, score: float, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if validation_results['timestamp_analysis']['duplicated_timestamps'] > 0:
            recommendations.append("Remove or aggregate duplicated timestamps")
        
        if 'gap_analysis' in validation_results:
            if validation_results['gap_analysis']['large_gaps'] > 0:
                recommendations.append("Consider handling large temporal gaps with interpolation")
        
        if not validation_results['frequency_analysis'].get('frequency_consistent', True):
            recommendations.append("Regularize timestamp frequency to consistent intervals")
        
        if score < 60:
            recommendations.append("Consider additional data cleaning before model training")
        
        return recommendations
    
    def calculate_cross_correlation(self, unified_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive cross-correlation analysis between currency pairs
        
        Args:
            unified_data: Unified DataFrame with all currency features
            
        Returns:
            Dictionary containing detailed correlation analysis
        """
        self.logger.info("Calculating cross-currency correlation analysis")
        
        correlation_results = {
            'pair_correlations': {},
            'feature_correlations': {},
            'correlation_matrix': {},
            'correlation_summary': {}
        }
        
        # Extract close price returns for each currency pair
        close_returns = {}
        for pair in self.config.CURRENCY_PAIRS:
            close_col = f'{pair}_Close_Return'
            if close_col in unified_data.columns:
                close_returns[pair] = unified_data[close_col]
        
        # Calculate pairwise correlations between currency pairs
        pair_names = list(close_returns.keys())
        correlation_matrix = np.zeros((len(pair_names), len(pair_names)))
        
        for i, pair1 in enumerate(pair_names):
            for j, pair2 in enumerate(pair_names):
                if i <= j:  # Calculate upper triangle only
                    corr_coef, p_value = pearsonr(close_returns[pair1], close_returns[pair2])
                    correlation_matrix[i, j] = corr_coef
                    correlation_matrix[j, i] = corr_coef  # Mirror to lower triangle
                    
                    if i != j:  # Don't store self-correlation
                        correlation_results['pair_correlations'][f'{pair1}_{pair2}'] = {
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'significance': 'significant' if p_value < 0.05 else 'not_significant'
                        }
        
        # Store correlation matrix
        correlation_results['correlation_matrix'] = pd.DataFrame(
            correlation_matrix,
            index=pair_names,
            columns=pair_names
        )
        
        # Feature-level correlation analysis
        for feature in self.feature_order:
            feature_correlations = {}
            feature_data = {}
            
            for pair in self.config.CURRENCY_PAIRS:
                feature_col = f'{pair}_{feature}'
                if feature_col in unified_data.columns:
                    feature_data[pair] = unified_data[feature_col]
            
            # Calculate correlations between same feature across pairs
            if len(feature_data) > 1:
                feature_pairs = list(feature_data.keys())
                for i, pair1 in enumerate(feature_pairs):
                    for j, pair2 in enumerate(feature_pairs):
                        if i < j:
                            corr_coef, p_value = pearsonr(feature_data[pair1], feature_data[pair2])
                            feature_correlations[f'{pair1}_{pair2}'] = {
                                'correlation': corr_coef,
                                'p_value': p_value
                            }
                
                correlation_results['feature_correlations'][feature] = feature_correlations
        
        # Summary statistics
        all_correlations = [result['correlation'] for result in correlation_results['pair_correlations'].values()]
        if all_correlations:
            correlation_results['correlation_summary'] = {
                'mean_correlation': np.mean(all_correlations),
                'std_correlation': np.std(all_correlations),
                'min_correlation': np.min(all_correlations),
                'max_correlation': np.max(all_correlations),
                'high_correlation_pairs': [
                    pair for pair, result in correlation_results['pair_correlations'].items()
                    if abs(result['correlation']) > 0.7
                ]
            }
            
            self.logger.info(f"Cross-currency correlation summary: "
                           f"Mean = {correlation_results['correlation_summary']['mean_correlation']:.3f}, "
                           f"Range = [{correlation_results['correlation_summary']['min_correlation']:.3f}, "
                           f"{correlation_results['correlation_summary']['max_correlation']:.3f}]")
        
        self.correlation_matrix = correlation_results
        return correlation_results
    
    def get_integration_summary(self) -> Dict:
        """
        Get comprehensive summary of integration process and results
        
        Returns:
            Dictionary containing complete integration summary
        """
        summary = {
            'integration_statistics': self.integration_stats,
            'temporal_alignment': self.alignment_info,
            'correlation_analysis': self.correlation_matrix.get('correlation_summary', {}),
            'data_quality_metrics': {}
        }
        
        # Add data quality metrics if available
        if hasattr(self, 'validation_results'):
            summary['data_quality_metrics'] = self.validation_results.get('alignment_quality', {})
        
        return summary
    
    def export_integration_report(self, unified_data: pd.DataFrame, output_path: Optional[str] = None) -> str:
        """
        Export comprehensive integration analysis report
        
        Args:
            unified_data: Unified DataFrame to analyze
            output_path: Optional path for output file
            
        Returns:
            Path to exported report file
        """
        from datetime import datetime
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"integration_report_{timestamp}.txt"
        
        # Generate comprehensive report
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MULTI-CURRENCY INTEGRATION ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Integration overview
        report_lines.append("INTEGRATION OVERVIEW")
        report_lines.append("-" * 40)
        if self.integration_stats:
            stats = self.integration_stats
            report_lines.append(f"Currency Pairs: {', '.join(stats['currency_pairs'])}")
            report_lines.append(f"Total Records: {stats['total_records']:,}")
            report_lines.append(f"Total Features: {stats['total_features']}")
            report_lines.append(f"Features per Pair: {stats['features_per_pair']}")
            report_lines.append(f"Temporal Range: {stats['temporal_range']['start']} to {stats['temporal_range']['end']}")
            report_lines.append(f"Duration: {stats['temporal_range']['duration_days']} days")
        report_lines.append("")
        
        # Alignment analysis
        report_lines.append("TEMPORAL ALIGNMENT ANALYSIS")
        report_lines.append("-" * 40)
        if self.alignment_info:
            for pair, pct in self.alignment_info['alignment_percentage'].items():
                report_lines.append(f"{pair}: {pct:.2f}% aligned")
            report_lines.append(f"Common Timestamps: {self.alignment_info['common_timestamps']:,}")
        report_lines.append("")
        
        # Correlation analysis
        report_lines.append("CROSS-CURRENCY CORRELATION ANALYSIS")
        report_lines.append("-" * 40)
        if self.correlation_matrix and 'correlation_summary' in self.correlation_matrix:
            corr_summary = self.correlation_matrix['correlation_summary']
            report_lines.append(f"Mean Correlation: {corr_summary.get('mean_correlation', 0):.3f}")
            report_lines.append(f"Correlation Range: [{corr_summary.get('min_correlation', 0):.3f}, {corr_summary.get('max_correlation', 0):.3f}]")
            
            if 'high_correlation_pairs' in corr_summary and corr_summary['high_correlation_pairs']:
                report_lines.append(f"High Correlation Pairs: {', '.join(corr_summary['high_correlation_pairs'])}")
        report_lines.append("")
        
        # Data quality summary
        report_lines.append("DATA QUALITY SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Final Dataset Shape: {unified_data.shape}")
        report_lines.append(f"Missing Values: {unified_data.isnull().sum().sum()}")
        report_lines.append(f"Data Types: {len(unified_data.dtypes.unique())} unique types")
        report_lines.append("")
        
        report_lines.append("="*80)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Integration report exported to: {output_path}")
        return output_path