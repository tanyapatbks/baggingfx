"""
Trading Strategy and Model Comparison Module
Implements trading decision logic, baseline strategies, and comprehensive performance comparison
Handles threshold-based trading, risk management, and market regime analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import ta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TradingStrategyManager:
    """
    Comprehensive trading strategy implementation and comparison framework
    Handles multiple trading approaches and performance analysis
    """
    
    def __init__(self, config):
        """
        Initialize trading strategy manager
        
        Args:
            config: Configuration object containing trading parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Trading strategy configurations
        self.thresholds = config.TRADING_THRESHOLDS
        self.risk_params = {
            'min_holding_period': config.MINIMUM_HOLDING_PERIOD,
            'stop_loss': config.STOP_LOSS_PCT,
            'take_profit': config.TAKE_PROFIT_PCT
        }
        
        # Performance tracking
        self.strategy_results = {}
        self.comparison_metrics = {}
        
    def apply_threshold_strategy(self, predictions: np.ndarray, prices: pd.Series,
                                timestamps: pd.DatetimeIndex, threshold_type: str = 'moderate') -> Dict:
        """
        Apply time-based threshold trading strategy to model predictions
        
        New Strategy Rules:
        - Open position based on prediction thresholds
        - Hold for minimum 1 hour, maximum 3 hours
        - Close at t+1 if profitable, otherwise wait
        - Force close at t+3 regardless of profit/loss
        - Stop loss still applies throughout holding period
        
        Args:
            predictions: Model prediction probabilities (0-1)
            prices: Price series for the target currency pair
            timestamps: Corresponding timestamps
            threshold_type: Type of threshold strategy ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Dictionary containing trading signals and performance metrics
        """
        self.logger.info(f"Applying {threshold_type} time-based threshold trading strategy")
        
        if threshold_type not in self.thresholds:
            raise ValueError(f"Unknown threshold type: {threshold_type}")
        
        thresholds = self.thresholds[threshold_type]
        buy_threshold = thresholds['buy']
        sell_threshold = thresholds['sell']
        
        # Initialize trading signals and tracking arrays
        signals = np.zeros(len(predictions))  # 0 = hold, 1 = buy, -1 = sell
        positions = np.zeros(len(predictions))  # Current position
        trades = []
        
        # Position tracking variables
        current_position = 0  # 0 = no position, 1 = long, -1 = short
        entry_time = None
        entry_price = None
        entry_index = None
        
        for i, (pred, price, timestamp) in enumerate(zip(predictions, prices, timestamps)):
            # Check if we should avoid weekend trading
            if self._is_weekend_period(timestamp):
                if current_position != 0:
                    # Force close position before weekend
                    exit_price = price
                    pnl = self._calculate_pnl(current_position, entry_price, exit_price)
                    trades.append({
                        'type': f'close_{"long" if current_position == 1 else "short"}_weekend',
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'holding_hours': (timestamp - entry_time).total_seconds() / 3600,
                        'reason': 'weekend_close'
                    })
                    current_position = 0
                    entry_price = None
                    entry_time = None
                    entry_index = None
                # Skip trading during weekend
                positions[i] = current_position
                continue
            
            # Handle existing position
            if current_position != 0:
                hours_held = (timestamp - entry_time).total_seconds() / 3600
                current_pnl = self._calculate_pnl(current_position, entry_price, price)
                
                # Check for stop loss (immediate close)
                if current_pnl <= -self.config.STOP_LOSS_PCT:
                    trades.append({
                        'type': f'close_{"long" if current_position == 1 else "short"}_stop_loss',
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl_pct': current_pnl,
                        'holding_hours': hours_held,
                        'reason': 'stop_loss'
                    })
                    current_position = 0
                    entry_price = None
                    entry_time = None
                    entry_index = None
                    
                # Check time-based closing rules
                elif hours_held >= self.config.MIN_HOLDING_HOURS:
                    should_close = False
                    close_reason = ''
                    
                    if hours_held >= self.config.MAX_HOLDING_HOURS:
                        # Force close after 3 hours
                        should_close = True
                        close_reason = 'time_limit'
                    elif current_pnl > 0:
                        # Close if profitable after minimum holding period
                        should_close = True
                        close_reason = 'profit_early'
                    
                    if should_close:
                        trades.append({
                            'type': f'close_{"long" if current_position == 1 else "short"}_{close_reason}',
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl_pct': current_pnl,
                            'holding_hours': hours_held,
                            'reason': close_reason
                        })
                        current_position = 0
                        entry_price = None
                        entry_time = None
                        entry_index = None
                
                positions[i] = current_position
                continue
            
            # Generate new trading signals (only when no position)
            if current_position == 0:
                if pred >= buy_threshold:
                    # Open long position
                    signals[i] = 1
                    current_position = 1
                    entry_price = price
                    entry_time = timestamp
                    entry_index = i
                    
                elif pred <= sell_threshold:
                    # Open short position
                    signals[i] = -1
                    current_position = -1
                    entry_price = price
                    entry_time = timestamp
                    entry_index = i
            
            positions[i] = current_position
        
        # Handle any remaining open position at the end
        if current_position != 0:
            final_price = prices.iloc[-1]
            final_timestamp = timestamps[-1]
            final_pnl = self._calculate_pnl(current_position, entry_price, final_price)
            hours_held = (final_timestamp - entry_time).total_seconds() / 3600
            
            trades.append({
                'type': f'close_{"long" if current_position == 1 else "short"}_end',
                'entry_time': entry_time,
                'exit_time': final_timestamp,
                'entry_price': entry_price,
                'exit_price': final_price,
                'pnl_pct': final_pnl,
                'holding_hours': hours_held,
                'reason': 'end_of_data'
            })
        
        # Calculate strategy performance metrics
        performance = self._calculate_strategy_performance(trades, threshold_type)
        
        strategy_result = {
            'threshold_type': threshold_type,
            'signals': signals,
            'positions': positions,
            'trades': trades,
            'performance': performance,
            'thresholds_used': thresholds,
            'strategy_type': 'time_based'
        }
        
        self.logger.info(f"{threshold_type} time-based strategy: {len(trades)} trades, "
                        f"total return: {performance['total_return']:.4f}")
        
        return strategy_result
    
    def _is_weekend_period(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if timestamp falls in weekend trading avoidance period
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if should avoid trading (weekend period)
        """
        if not self.config.AVOID_WEEKEND_POSITIONS:
            return False
        
        weekday = timestamp.weekday()  # 0 = Monday, 6 = Sunday
        hour = timestamp.hour
        
        # Friday after closing hour
        if weekday == 4 and hour >= self.config.FRIDAY_CLOSE_HOUR:
            return True
        
        # Saturday and Sunday (full day)
        if weekday in [5, 6]:
            return True
        
        # Monday before opening hour
        if weekday == 0 and hour < self.config.MONDAY_OPEN_HOUR:
            return True
        
        return False
    
    def _calculate_pnl(self, position_type: int, entry_price: float, exit_price: float) -> float:
        """
        Calculate profit/loss percentage for a position
        
        Args:
            position_type: 1 for long, -1 for short
            entry_price: Price when position was opened
            exit_price: Price when position was closed
            
        Returns:
            PnL as percentage (decimal form)
        """
        if position_type == 1:  # Long position
            return (exit_price - entry_price) / entry_price
        elif position_type == -1:  # Short position
            return (entry_price - exit_price) / entry_price
        else:
            return 0.0
    
    def _calculate_strategy_performance(self, trades: List[Dict], strategy_name: str) -> Dict:
        """Calculate comprehensive performance metrics for a trading strategy"""
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_holding_hours': 0.0
            }
        
        # Extract trade results
        trade_returns = [trade['pnl_pct'] for trade in trades]
        holding_periods = [trade['holding_hours'] for trade in trades]
        
        # Basic performance metrics
        total_return = sum(trade_returns)
        win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)
        avg_profit_per_trade = np.mean(trade_returns)
        
        # Risk-adjusted metrics
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe_ratio = avg_profit_per_trade / np.std(trade_returns) * np.sqrt(252 * 24)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Additional metrics
        profitable_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        performance = {
            'total_trades': len(trades),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_holding_hours': np.mean(holding_periods),
            'best_trade': max(trade_returns) if trade_returns else 0,
            'worst_trade': min(trade_returns) if trade_returns else 0,
            'avg_winning_trade': np.mean(profitable_trades) if profitable_trades else 0,
            'avg_losing_trade': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(profitable_trades) / sum(losing_trades)) if losing_trades else float('inf'),
            'consecutive_wins': self._calculate_max_consecutive(trade_returns, positive=True),
            'consecutive_losses': self._calculate_max_consecutive(trade_returns, positive=False)
        }
        
        return performance
    
    def _calculate_max_consecutive(self, returns: List[float], positive: bool = True) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not returns:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if (positive and ret > 0) or (not positive and ret < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def implement_buy_hold_strategy(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> Dict:
        """
        Implement simple buy-and-hold strategy as baseline
        
        Args:
            prices: Price series
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing buy-hold performance
        """
        self.logger.info("Implementing buy-and-hold baseline strategy")
        
        if len(prices) < 2:
            return {'error': 'Insufficient data for buy-hold strategy'}
        
        # Simple buy-and-hold: buy at start, sell at end
        entry_price = prices.iloc[0]
        exit_price = prices.iloc[-1]
        total_return = (exit_price - entry_price) / entry_price
        
        # Calculate intermediate drawdowns
        normalized_prices = prices / entry_price
        running_max = normalized_prices.expanding().max()
        drawdowns = (running_max - normalized_prices) / running_max
        max_drawdown = drawdowns.max()
        
        # Calculate volatility and Sharpe ratio
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 * 24)  # Annualized
        sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
        
        buy_hold_performance = {
            'strategy_name': 'buy_and_hold',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'holding_period_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            'total_trades': 1
        }
        
        self.logger.info(f"Buy-hold strategy: {total_return:.4f} total return, "
                        f"max drawdown: {max_drawdown:.4f}")
        
        return buy_hold_performance
    
    def implement_rsi_strategy(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> Dict:
        """
        Implement RSI-based trading strategy
        
        Args:
            prices: Price series
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing RSI strategy performance
        """
        self.logger.info("Implementing RSI baseline strategy")
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(close=prices, window=self.config.RSI_PERIOD).rsi()
        
        # Generate trading signals
        signals = np.zeros(len(prices))
        positions = np.zeros(len(prices))
        trades = []
        
        current_position = 0
        entry_price = None
        entry_time = None
        
        for i in range(1, len(rsi)):
            if pd.isna(rsi.iloc[i]):
                continue
                
            if rsi.iloc[i] <= self.config.RSI_OVERSOLD and current_position != 1:
                # Buy signal (oversold)
                if current_position == -1:
                    # Close short first
                    exit_price = prices.iloc[i]
                    pnl = (entry_price - exit_price) / entry_price
                    trades.append({
                        'type': 'close_short',
                        'entry_time': entry_time,
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'signal': 'rsi_oversold'
                    })
                
                signals[i] = 1
                current_position = 1
                entry_price = prices.iloc[i]
                entry_time = timestamps[i]
                
            elif rsi.iloc[i] >= self.config.RSI_OVERBOUGHT and current_position != -1:
                # Sell signal (overbought)
                if current_position == 1:
                    # Close long first
                    exit_price = prices.iloc[i]
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        'type': 'close_long',
                        'entry_time': entry_time,
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'signal': 'rsi_overbought'
                    })
                
                signals[i] = -1
                current_position = -1
                entry_price = prices.iloc[i]
                entry_time = timestamps[i]
            
            positions[i] = current_position
        
        performance = self._calculate_strategy_performance(trades, 'rsi')
        performance['rsi_parameters'] = {
            'period': self.config.RSI_PERIOD,
            'oversold_threshold': self.config.RSI_OVERSOLD,
            'overbought_threshold': self.config.RSI_OVERBOUGHT
        }
        
        rsi_result = {
            'strategy_name': 'rsi',
            'signals': signals,
            'positions': positions,
            'trades': trades,
            'performance': performance,
            'rsi_values': rsi.values
        }
        
        self.logger.info(f"RSI strategy: {len(trades)} trades, "
                        f"total return: {performance['total_return']:.4f}")
        
        return rsi_result
    
    def implement_macd_strategy(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> Dict:
        """
        Implement MACD-based trading strategy
        
        Args:
            prices: Price series
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing MACD strategy performance
        """
        self.logger.info("Implementing MACD baseline strategy")
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            close=prices,
            window_fast=self.config.MACD_FAST,
            window_slow=self.config.MACD_SLOW,
            window_sign=self.config.MACD_SIGNAL
        )
        
        macd_line = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_histogram = macd_indicator.macd_diff()
        
        # Generate trading signals based on MACD crossovers
        signals = np.zeros(len(prices))
        positions = np.zeros(len(prices))
        trades = []
        
        current_position = 0
        entry_price = None
        entry_time = None
        
        for i in range(1, len(macd_line)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                continue
            
            # MACD bullish crossover (MACD line crosses above signal line)
            if (macd_line.iloc[i] > macd_signal.iloc[i] and 
                macd_line.iloc[i-1] <= macd_signal.iloc[i-1] and 
                current_position != 1):
                
                if current_position == -1:
                    # Close short first
                    exit_price = prices.iloc[i]
                    pnl = (entry_price - exit_price) / entry_price
                    trades.append({
                        'type': 'close_short',
                        'entry_time': entry_time,
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'signal': 'macd_bullish_crossover'
                    })
                
                signals[i] = 1
                current_position = 1
                entry_price = prices.iloc[i]
                entry_time = timestamps[i]
                
            # MACD bearish crossover (MACD line crosses below signal line)
            elif (macd_line.iloc[i] < macd_signal.iloc[i] and 
                  macd_line.iloc[i-1] >= macd_signal.iloc[i-1] and 
                  current_position != -1):
                
                if current_position == 1:
                    # Close long first
                    exit_price = prices.iloc[i]
                    pnl = (exit_price - entry_price) / entry_price
                    trades.append({
                        'type': 'close_long',
                        'entry_time': entry_time,
                        'exit_time': timestamps[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'signal': 'macd_bearish_crossover'
                    })
                
                signals[i] = -1
                current_position = -1
                entry_price = prices.iloc[i]
                entry_time = timestamps[i]
            
            positions[i] = current_position
        
        performance = self._calculate_strategy_performance(trades, 'macd')
        performance['macd_parameters'] = {
            'fast_period': self.config.MACD_FAST,
            'slow_period': self.config.MACD_SLOW,
            'signal_period': self.config.MACD_SIGNAL
        }
        
        macd_result = {
            'strategy_name': 'macd',
            'signals': signals,
            'positions': positions,
            'trades': trades,
            'performance': performance,
            'macd_line': macd_line.values,
            'macd_signal': macd_signal.values,
            'macd_histogram': macd_histogram.values
        }
        
        self.logger.info(f"MACD strategy: {len(trades)} trades, "
                        f"total return: {performance['total_return']:.4f}")
        
        return macd_result
    
    def create_single_currency_models(self, data_splits: Dict, target_pairs: List[str]) -> Dict:
        """
        Create and evaluate single-currency models for comparison
        
        Args:
            data_splits: Dictionary containing train/val/test data splits
            target_pairs: List of currency pairs to create models for
            
        Returns:
            Dictionary containing single-currency model results
        """
        self.logger.info("Creating single-currency models for comparison")
        
        single_currency_results = {}
        
        for pair in target_pairs:
            self.logger.info(f"Training single-currency model for {pair}")
            
            # This would typically involve:
            # 1. Extracting single-pair features from multi-currency data
            # 2. Training a model on just that pair's data
            # 3. Evaluating performance
            # For now, we'll create a placeholder structure
            
            single_currency_results[pair] = {
                'model_type': 'single_currency_cnn_lstm',
                'target_pair': pair,
                'performance': {
                    'accuracy': 0.0,  # Would be calculated from actual training
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                },
                'trading_performance': {
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0
                }
            }
        
        return single_currency_results
    
    def test_bagging_approach_by_pair(self, multi_currency_model, single_currency_results: Dict,
                                     test_data: Dict, price_data: Dict) -> Dict:
        """
        Test bagging approach performance on individual currency pairs
        
        Args:
            multi_currency_model: Trained multi-currency model
            single_currency_results: Results from single-currency models
            test_data: Test dataset
            price_data: Price data for each currency pair
            
        Returns:
            Dictionary containing comparison results by currency pair
        """
        self.logger.info("Testing bagging approach performance by currency pair")
        
        pair_comparison = {}
        
        for pair in self.config.CURRENCY_PAIRS:
            self.logger.info(f"Analyzing bagging performance for {pair}")
            
            # Extract test data for this specific pair
            # In a real implementation, this would involve:
            # 1. Using multi-currency model to predict this pair
            # 2. Comparing with single-currency model predictions
            # 3. Evaluating trading performance for this specific pair
            
            pair_comparison[pair] = {
                'multi_currency_performance': {
                    'prediction_accuracy': 0.0,
                    'directional_accuracy': 0.0,
                    'trading_metrics': {
                        'total_return': 0.0,
                        'win_rate': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    }
                },
                'single_currency_performance': single_currency_results.get(pair, {}),
                'improvement_metrics': {
                    'accuracy_improvement': 0.0,
                    'return_improvement': 0.0,
                    'risk_improvement': 0.0
                },
                'statistical_significance': {
                    'p_value': 1.0,
                    'significant': False
                }
            }
        
        return pair_comparison
    
    def analyze_market_regime_performance(self, strategy_results: Dict, 
                                        timestamps: pd.DatetimeIndex) -> Dict:
        """
        Analyze strategy performance across different market regimes
        
        Args:
            strategy_results: Dictionary containing strategy results
            timestamps: Time index for regime analysis
            
        Returns:
            Dictionary containing regime-specific performance analysis
        """
        self.logger.info("Analyzing performance across market regimes")
        
        # Define market regimes based on years
        regimes = {
            '2018': {'start': '2018-01-01', 'end': '2018-12-31', 'type': 'trade_war_tensions'},
            '2019': {'start': '2019-01-01', 'end': '2019-12-31', 'type': 'stable_accommodative'},
            '2020': {'start': '2020-01-01', 'end': '2020-12-31', 'type': 'covid_crisis'},
            '2021': {'start': '2021-01-01', 'end': '2021-12-31', 'type': 'recovery_inflation'},
            '2022': {'start': '2022-01-01', 'end': '2022-12-31', 'type': 'rate_hikes_war'}
        }
        
        regime_analysis = {}
        
        for year, regime_info in regimes.items():
            start_date = pd.to_datetime(regime_info['start'])
            end_date = pd.to_datetime(regime_info['end'])
            
            # Filter data for this regime
            regime_mask = (timestamps >= start_date) & (timestamps <= end_date)
            
            if not regime_mask.any():
                continue
            
            regime_performance = {}
            
            # Analyze each strategy's performance in this regime
            for strategy_name, results in strategy_results.items():
                if 'trades' in results:
                    # Filter trades to this regime
                    regime_trades = [
                        trade for trade in results['trades']
                        if (pd.to_datetime(trade['entry_time']) >= start_date and 
                            pd.to_datetime(trade['exit_time']) <= end_date)
                    ]
                    
                    # Calculate regime-specific performance
                    regime_perf = self._calculate_strategy_performance(regime_trades, f"{strategy_name}_{year}")
                    regime_performance[strategy_name] = regime_perf
            
            regime_analysis[year] = {
                'regime_type': regime_info['type'],
                'date_range': {'start': start_date, 'end': end_date},
                'strategy_performance': regime_performance,
                'market_characteristics': self._analyze_market_characteristics(timestamps[regime_mask])
            }
        
        # Identify best performing strategies by regime
        regime_summary = self._summarize_regime_performance(regime_analysis)
        
        self.logger.info("Market regime analysis completed")
        return {
            'regime_details': regime_analysis,
            'regime_summary': regime_summary
        }
    
    def _analyze_market_characteristics(self, regime_timestamps: pd.DatetimeIndex) -> Dict:
        """Analyze general market characteristics for a regime period"""
        return {
            'trading_days': len(regime_timestamps),
            'regime_duration_days': (regime_timestamps.max() - regime_timestamps.min()).days,
            'data_coverage': len(regime_timestamps) / (24 * (regime_timestamps.max() - regime_timestamps.min()).days) if len(regime_timestamps) > 0 else 0
        }
    
    def _summarize_regime_performance(self, regime_analysis: Dict) -> Dict:
        """Summarize performance across all regimes"""
        summary = {
            'best_strategy_by_regime': {},
            'most_consistent_strategy': None,
            'regime_specific_insights': {}
        }
        
        # Find best strategy in each regime
        for year, regime_data in regime_analysis.items():
            best_strategy = None
            best_return = float('-inf')
            
            for strategy, performance in regime_data['strategy_performance'].items():
                if performance['total_return'] > best_return:
                    best_return = performance['total_return']
                    best_strategy = strategy
            
            summary['best_strategy_by_regime'][year] = {
                'strategy': best_strategy,
                'return': best_return,
                'regime_type': regime_data['regime_type']
            }
        
        return summary
    
    def compare_all_strategies(self, multi_currency_results: Dict, baseline_results: Dict,
                              single_currency_results: Dict) -> Dict:
        """
        Comprehensive comparison of all trading strategies
        
        Args:
            multi_currency_results: Results from multi-currency CNN-LSTM
            baseline_results: Results from baseline strategies
            single_currency_results: Results from single-currency models
            
        Returns:
            Dictionary containing comprehensive strategy comparison
        """
        self.logger.info("Performing comprehensive strategy comparison")
        
        # Compile all results
        all_results = {
            'multi_currency_cnn_lstm': multi_currency_results,
            **baseline_results,
            **single_currency_results
        }
        
        # Create comparison metrics
        comparison_metrics = {}
        strategy_rankings = {}
        
        # Define metrics for comparison
        metrics_to_compare = [
            'total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown',
            'avg_profit_per_trade', 'total_trades'
        ]
        
        for metric in metrics_to_compare:
            metric_values = {}
            
            for strategy_name, results in all_results.items():
                if 'performance' in results and metric in results['performance']:
                    metric_values[strategy_name] = results['performance'][metric]
                elif metric in results:
                    metric_values[strategy_name] = results[metric]
            
            # Rank strategies by this metric
            if metric == 'max_drawdown':  # Lower is better
                ranked = sorted(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better
                ranked = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            comparison_metrics[metric] = {
                'values': metric_values,
                'ranking': ranked,
                'best_strategy': ranked[0][0] if ranked else None,
                'best_value': ranked[0][1] if ranked else None
            }
        
        # Calculate overall performance score
        overall_scores = self._calculate_overall_scores(all_results, comparison_metrics)
        
        # Statistical significance testing
        significance_tests = self._perform_significance_tests(all_results)
        
        comprehensive_comparison = {
            'strategy_results': all_results,
            'metric_comparisons': comparison_metrics,
            'overall_rankings': overall_scores,
            'statistical_significance': significance_tests,
            'summary_insights': self._generate_comparison_insights(comparison_metrics, overall_scores)
        }
        
        self.logger.info("Strategy comparison completed")
        return comprehensive_comparison
    
    def _calculate_overall_scores(self, all_results: Dict, comparison_metrics: Dict) -> Dict:
        """Calculate overall performance scores for strategy ranking"""
        strategy_scores = {}
        
        # Weight different metrics based on importance
        metric_weights = {
            'total_return': 0.3,
            'sharpe_ratio': 0.25,
            'win_rate': 0.2,
            'max_drawdown': 0.15,  # Inverted (lower is better)
            'avg_profit_per_trade': 0.1
        }
        
        for strategy in all_results.keys():
            total_score = 0
            
            for metric, weight in metric_weights.items():
                if metric in comparison_metrics and strategy in comparison_metrics[metric]['values']:
                    # Normalize score based on ranking
                    ranking = comparison_metrics[metric]['ranking']
                    strategy_rank = next((i for i, (name, _) in enumerate(ranking) if name == strategy), len(ranking))
                    
                    if metric == 'max_drawdown':
                        # Invert for drawdown (lower rank is better)
                        normalized_score = (len(ranking) - strategy_rank - 1) / max(len(ranking) - 1, 1)
                    else:
                        normalized_score = (len(ranking) - strategy_rank - 1) / max(len(ranking) - 1, 1)
                    
                    total_score += normalized_score * weight
            
            strategy_scores[strategy] = total_score
        
        # Sort by overall score
        ranked_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'scores': strategy_scores,
            'ranking': ranked_strategies,
            'best_overall_strategy': ranked_strategies[0][0] if ranked_strategies else None
        }
    
    def _perform_significance_tests(self, all_results: Dict) -> Dict:
        """Perform statistical significance tests between strategies"""
        significance_results = {}
        
        # Extract returns for each strategy that has trade data
        strategy_returns = {}
        
        for strategy_name, results in all_results.items():
            if 'trades' in results and results['trades']:
                returns = [trade['pnl_pct'] for trade in results['trades']]
                if len(returns) > 1:
                    strategy_returns[strategy_name] = returns
        
        # Perform pairwise t-tests
        strategy_names = list(strategy_returns.keys())
        
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names[i+1:], i+1):
                returns1 = strategy_returns[strategy1]
                returns2 = strategy_returns[strategy2]
                
                if len(returns1) > 1 and len(returns2) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(returns1, returns2)
                        
                        significance_results[f"{strategy1}_vs_{strategy2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant_at_5pct': p_value < 0.05,
                            'significant_at_1pct': p_value < 0.01,
                            'better_strategy': strategy1 if np.mean(returns1) > np.mean(returns2) else strategy2
                        }
                    except:
                        # Handle cases where test fails
                        significance_results[f"{strategy1}_vs_{strategy2}"] = {
                            'error': 'Unable to perform t-test'
                        }
        
        return significance_results
    
    def _generate_comparison_insights(self, comparison_metrics: Dict, overall_scores: Dict) -> List[str]:
        """Generate textual insights from strategy comparison"""
        insights = []
        
        # Best overall strategy
        if overall_scores['best_overall_strategy']:
            insights.append(f"Best overall strategy: {overall_scores['best_overall_strategy']}")
        
        # Best strategy by specific metrics
        for metric in ['total_return', 'sharpe_ratio', 'win_rate']:
            if metric in comparison_metrics and comparison_metrics[metric]['best_strategy']:
                best_strategy = comparison_metrics[metric]['best_strategy']
                best_value = comparison_metrics[metric]['best_value']
                insights.append(f"Best {metric}: {best_strategy} ({best_value:.4f})")
        
        # Risk analysis
        if 'max_drawdown' in comparison_metrics:
            safest_strategy = comparison_metrics['max_drawdown']['ranking'][0][0]
            safest_drawdown = comparison_metrics['max_drawdown']['ranking'][0][1]
            insights.append(f"Lowest risk (max drawdown): {safest_strategy} ({safest_drawdown:.4f})")
        
        return insights