"""
Main Execution Script for Multi-Currency CNN-LSTM Forex Prediction
Complete pipeline from data loading to results analysis and visualization
"""

import os
import sys
import traceback
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple, List, Any, Optional, Union
# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

# Import configuration and modules
from config import Config
from src.utils import setup_logging, print_system_info, ProgressTracker, validate_file_paths
from src.data_loader import CurrencyDataLoader
from src.preprocessing import ForexDataPreprocessor
from src.multi_currency_integration import MultiCurrencyIntegrator
from src.sequence_preparation import SequenceDataPreparator
from src.model_architecture import CNNLSTMArchitecture
from src.training import ModelTrainer
from src.trading_strategy import TradingStrategyManager
from src.visualization import ResultsAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

def run_final_evaluation(model_path: str, data_splits: Dict, unified_data: pd.DataFrame, 
                        correlation_analysis: Dict, config) -> Dict:
    """
    Run final evaluation on test set - USE ONLY ONCE BEFORE THESIS DEFENSE
    
    This function should be called only when ready for final thesis results.
    Do not modify any parameters after running this function.
    
    Args:
        model_path: Path to the best trained model
        data_splits: Dictionary containing train/val/test splits
        unified_data: Unified multi-currency dataset
        correlation_analysis: Cross-currency correlation results
        config: Configuration object
        
    Returns:
        Dictionary containing final evaluation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*80)
    logger.info("WARNING: This is the final evaluation that will be used for thesis defense.")
    logger.info("Do not run this function multiple times or modify parameters afterward.")
    logger.info("="*80)
    
    # Load the best model
    import tensorflow as tf
    best_model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from: {model_path}")
    
    # Get test data
    X_test, y_test, test_timestamps = data_splits['test']
    logger.info(f"Test set size: {len(X_test)} samples")
    
    # Generate final predictions
    logger.info("Generating final predictions on test set...")
    final_predictions = best_model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    final_predictions = final_predictions.flatten()
    
    # Extract test price data
    test_price_data = unified_data.loc[test_timestamps]['EURUSD_Close_Price']
    
    # Initialize strategy manager
    strategy_manager = TradingStrategyManager(config)
    
    # Test all trading strategies on test set
    logger.info("Testing final trading strategies on test set...")
    final_threshold_results = {}
    
    for threshold_type in ['conservative', 'moderate', 'aggressive']:
        logger.info(f"Final evaluation: {threshold_type} strategy...")
        threshold_result = strategy_manager.apply_threshold_strategy(
            final_predictions, test_price_data, test_timestamps, threshold_type
        )
        final_threshold_results[f'final_{threshold_type}'] = threshold_result
    
    # Test baseline strategies
    logger.info("Testing baseline strategies on test set...")
    final_buy_hold = strategy_manager.implement_buy_hold_strategy(test_price_data, test_timestamps)
    final_rsi = strategy_manager.implement_rsi_strategy(test_price_data, test_timestamps)
    final_macd = strategy_manager.implement_macd_strategy(test_price_data, test_timestamps)
    
    # Final comprehensive comparison
    final_comparison = strategy_manager.compare_all_strategies(
        final_threshold_results['final_moderate'],
        {'buy_and_hold': final_buy_hold, 'rsi': final_rsi, 'macd': final_macd},
        {}  # No single currency comparison in final evaluation
    )
    
    # Create final results visualization
    analyzer = ResultsAnalyzer(config)
    
    # Generate final performance plots
    final_performance_plot = analyzer.plot_strategy_performance(
        final_comparison, 'FINAL_strategy_performance_test_set'
    )
    
    # Final confusion matrix
    final_predictions_dict = {'Final_CNN-LSTM': (final_predictions > 0.5).astype(int)}
    final_confusion_plot = analyzer.plot_confusion_matrices(
        final_predictions_dict, y_test, 'FINAL_confusion_matrix_test_set'
    )
    
    # Export final results
    final_results = {
        'final_strategy_comparison': final_comparison,
        'final_threshold_results': final_threshold_results,
        'final_baseline_results': {
            'buy_and_hold': final_buy_hold,
            'rsi': final_rsi,
            'macd': final_macd
        },
        'test_predictions': final_predictions,
        'test_labels': y_test,
        'evaluation_timestamp': datetime.now().isoformat()
    }
    
    # Save final results
    final_report_path = analyzer.create_research_report(
        final_results, f'FINAL_EVALUATION_{config.EXPERIMENT_NAME}'
    )
    
    final_latex_path = analyzer.export_results_to_latex(
        final_results, f'FINAL_EVALUATION_{config.EXPERIMENT_NAME}'
    )
    
    logger.info("="*80)
    logger.info("FINAL EVALUATION COMPLETED")
    logger.info("="*80)
    logger.info(f"Final report: {final_report_path}")
    logger.info(f"Final LaTeX results: {final_latex_path}")
    logger.info("These results should be used for thesis defense.")
    logger.info("DO NOT re-run final evaluation or modify parameters.")
    logger.info("="*80)
    
    return final_results

def main():
    """
    Main execution function that orchestrates the entire forex prediction pipeline
    """
    # Initialize configuration and create directories
    config = Config()
    config.create_directories()
    
    # Setup logging
    log_file = config.get_log_path('main_execution')
    logger = setup_logging(config.LOG_LEVEL, log_file)
    
    # Print system information
    print_system_info()
    
    logger.info("="*80)
    logger.info("MULTI-CURRENCY CNN-LSTM FOREX PREDICTION SYSTEM")
    logger.info("="*80)
    logger.info(f"Experiment ID: {config.EXPERIMENT_NAME}")
    logger.info(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Record start time
    experiment_start_time = datetime.now()
    
    try:
        # Validate required files exist
        logger.info("Validating input files...")
        files_valid, missing_files = validate_file_paths(config.CURRENCY_FILES)
        if not files_valid:
            logger.error("Missing required data files:")
            for missing in missing_files:
                logger.error(f"  - {missing}")
            raise FileNotFoundError("Required data files not found. Please ensure CSV files are in /data directory.")
        
        logger.info("All required files found. Proceeding with analysis...")
        
        # ========================================================================================
        # STEP 1: DATA LOADING AND EXPLORATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA LOADING AND EXPLORATION")
        logger.info("="*60)
        
        data_loader = CurrencyDataLoader(config)
        
        # Load all currency data
        logger.info("Loading currency data for all pairs...")
        raw_data = data_loader.load_currency_data()
        
        # Perform comprehensive data exploration
        logger.info("Analyzing data structure...")
        structure_info = data_loader.explore_data_structure()
        
        logger.info("Checking data quality...")
        quality_info = data_loader.check_data_quality()
        
        logger.info("Analyzing date ranges...")
        date_info = data_loader.analyze_date_ranges()
        
        logger.info("Calculating basic statistics...")
        stats_info = data_loader.calculate_basic_statistics()
        
        logger.info("Validating temporal alignment...")
        alignment_info = data_loader.validate_temporal_alignment()
        
        # Export data summary
        summary_path = data_loader.export_data_summary(
            config.get_results_path('data_loading_summary.txt')
        )
        
        logger.info(f"Data loading completed. Summary exported to: {summary_path}")
        
        # ========================================================================================
        # STEP 2: DATA PREPROCESSING AND CLEANING
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 2: DATA PREPROCESSING AND CLEANING")
        logger.info("="*60)
        
        preprocessor = ForexDataPreprocessor(config)
        
        # Handle missing values
        logger.info("Handling missing values...")
        cleaned_data = preprocessor.handle_missing_values(raw_data)
        
        # Calculate percentage returns for OHLC
        logger.info("Calculating percentage returns...")
        returns_data = preprocessor.calculate_percentage_returns(cleaned_data)
        
        # Apply mixed normalization strategy
        logger.info("Applying normalization...")
        normalized_data = preprocessor.apply_mixed_normalization(returns_data, fit_data=True)
        
        # Detect market hours gaps
        logger.info("Analyzing market gaps...")
        gap_analysis = preprocessor.detect_market_hours_gaps(normalized_data)
        
        # Create unified feature matrix
        logger.info("Creating unified feature matrix...")
        feature_matrix, feature_names = preprocessor.create_feature_matrix(normalized_data)
        
        logger.info(f"Preprocessing completed. Feature matrix shape: {feature_matrix.shape}")
        
        # Save preprocessing artifacts
        preprocessor.save_preprocessing_artifacts(config.RESULTS_DIR)
        
        # ========================================================================================
        # STEP 3: MULTI-CURRENCY INTEGRATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 3: MULTI-CURRENCY INTEGRATION")
        logger.info("="*60)
        
        integrator = MultiCurrencyIntegrator(config)
        
        # Merge currency pairs into unified dataset
        logger.info("Merging currency pairs...")
        unified_data = integrator.merge_currency_pairs(normalized_data)
        
        # Create concatenated features for CNN-LSTM
        logger.info("Creating concatenated features...")
        features_array, feature_names_ordered = integrator.create_concatenated_features(unified_data)
        
        # Validate temporal alignment
        logger.info("Validating temporal alignment...")
        alignment_validation = integrator.validate_temporal_alignment(unified_data)
        
        # Calculate cross-currency correlations
        logger.info("Calculating cross-currency correlations...")
        correlation_analysis = integrator.calculate_cross_correlation(unified_data)
        
        # Export integration report
        integration_report_path = integrator.export_integration_report(
            unified_data, 
            config.get_results_path('integration_report.txt')
        )
        
        logger.info(f"Integration completed. Report saved to: {integration_report_path}")
        
        # ========================================================================================
        # STEP 4: SEQUENCE PREPARATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 4: SEQUENCE PREPARATION")
        logger.info("="*60)
        
        sequence_preparator = SequenceDataPreparator(config)
        
        # Create sequences for CNN-LSTM training
        logger.info(f"Creating sequences with window size {config.WINDOW_SIZE}...")
        X_sequences, y_labels, timestamps = sequence_preparator.create_sequences(
            unified_data, target_pair='EURUSD'
        )
        
        # Split data temporally (train/validation/test)
        logger.info("Splitting data temporally...")
        data_splits = sequence_preparator.split_temporal_data(X_sequences, y_labels, timestamps)
        
        # Analyze temporal patterns
        logger.info("Analyzing temporal patterns...")
        temporal_analysis = sequence_preparator.analyze_temporal_patterns(y_labels, timestamps)
        
        # Export sequence summary
        sequence_summary_path = sequence_preparator.export_sequence_summary(
            config.get_results_path('sequence_preparation_summary.txt')
        )
        
        logger.info(f"Sequence preparation completed. Summary saved to: {sequence_summary_path}")
        logger.info(f"Training samples: {len(data_splits['train'][0])}")
        logger.info(f"Validation samples: {len(data_splits['val'][0])}")
        logger.info(f"Test samples: {len(data_splits['test'][0])}")
        
        # ========================================================================================
        # STEP 5: MODEL ARCHITECTURE CREATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 5: MODEL ARCHITECTURE CREATION")
        logger.info("="*60)
        
        architecture = CNNLSTMArchitecture(config)
        
        # Build and compile the multi-currency CNN-LSTM model
        logger.info("Building CNN-LSTM architecture...")
        model = architecture.compile_multi_currency_model()
        
        # Display model architecture
        logger.info("Displaying model architecture...")
        architecture_plot_path = architecture.display_model_architecture(
            save_plot=True,
            output_path=config.get_results_path('model_architecture.png')
        )
        
        # Analyze model complexity
        logger.info("Analyzing model complexity...")
        complexity_analysis = architecture.analyze_model_complexity()
        
        # Create model variants for comparison
        logger.info("Creating model variants...")
        model_variants = architecture.create_model_variants()
        
        logger.info("Model architecture creation completed.")
        
        # ========================================================================================
        # STEP 6: MODEL TRAINING AND OPTIMIZATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 6: MODEL TRAINING AND OPTIMIZATION")
        logger.info("="*60)
        
        trainer = ModelTrainer(config)
        
        # Log model configuration
        model_summary = architecture.get_model_summary()
        trainer.log_model_configuration(model, model_summary)
        
        # Prepare data generators
        X_train, y_train, _ = data_splits['train']
        X_val, y_val, _ = data_splits['val']
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // config.BATCH_SIZE
        validation_steps = len(X_val) // config.BATCH_SIZE
        
        # Create data generators
        train_generator = sequence_preparator.create_data_generators(
            X_train, y_train, config.BATCH_SIZE, shuffle=True
        )
        val_generator = sequence_preparator.create_data_generators(
            X_val, y_val, config.BATCH_SIZE, shuffle=False
        )
        
        # Train the model
        logger.info("Starting model training...")
        training_results = trainer.train_model(
            model, train_generator, val_generator, 
            steps_per_epoch, validation_steps
        )
        
        # Load best model
        logger.info("Loading best trained model...")
        best_model = trainer.load_best_model()
        
        logger.info("Model training completed successfully.")
        
        # ========================================================================================
        # STEP 7: TRADING STRATEGY IMPLEMENTATION AND COMPARISON (VALIDATION SET)
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 7: TRADING STRATEGY IMPLEMENTATION AND COMPARISON (VALIDATION SET)")
        logger.info("="*60)
        
        if config.DEVELOPMENT_MODE:
            logger.info("DEVELOPMENT MODE: Using validation set for strategy testing and comparison")
            logger.info("Test set will be reserved for final evaluation before thesis defense")
        else:
            logger.info("FINAL EVALUATION MODE: Using test set for final performance assessment")
        
        strategy_manager = TradingStrategyManager(config)
        
        # Get validation data for strategy evaluation (or test data if in final mode)
        if config.DEVELOPMENT_MODE:
            X_eval, y_eval, eval_timestamps = data_splits['val']
            eval_set_name = "validation"
        else:
            X_eval, y_eval, eval_timestamps = data_splits['test']
            eval_set_name = "test"
        
        # Generate predictions from the trained model
        logger.info(f"Generating model predictions on {eval_set_name} set...")
        model_predictions = best_model.predict(X_eval, batch_size=config.BATCH_SIZE, verbose=1)
        model_predictions = model_predictions.flatten()
        
        # Extract price data for evaluation period (using EURUSD as target)
        eval_price_data = unified_data.loc[eval_timestamps]['EURUSD_Close_Price']
        
        # Apply different threshold strategies with new time-based approach
        logger.info("Testing time-based threshold trading strategies...")
        threshold_results = {}
        
        for threshold_type in ['conservative', 'moderate', 'aggressive']:
            logger.info(f"Testing {threshold_type} time-based threshold strategy...")
            threshold_result = strategy_manager.apply_threshold_strategy(
                model_predictions, eval_price_data, eval_timestamps, threshold_type
            )
            threshold_results[f'multi_currency_{threshold_type}'] = threshold_result
        
        # Implement baseline strategies on the same data
        logger.info("Implementing baseline strategies...")
        
        # Buy & Hold strategy
        buy_hold_result = strategy_manager.implement_buy_hold_strategy(
            eval_price_data, eval_timestamps
        )
        
        # RSI strategy
        rsi_result = strategy_manager.implement_rsi_strategy(
            eval_price_data, eval_timestamps
        )
        
        # MACD strategy
        macd_result = strategy_manager.implement_macd_strategy(
            eval_price_data, eval_timestamps
        )
        
        # Create single-currency models for comparison (using validation set)
        logger.info("Creating single-currency model comparison...")
        if config.DEVELOPMENT_MODE:
            comparison_splits = {'val': data_splits['val']}  # Only use validation for development
        else:
            comparison_splits = {'test': data_splits['test']}  # Use test for final evaluation
            
        single_currency_results = strategy_manager.create_single_currency_models(
            comparison_splits, config.CURRENCY_PAIRS
        )
        
        # Test bagging approach by currency pair
        logger.info("Testing bagging approach by currency pair...")
        pair_comparison = strategy_manager.test_bagging_approach_by_pair(
            best_model, single_currency_results, 
            (X_eval, y_eval, eval_timestamps),
            {pair: unified_data[f'{pair}_Close_Price'] for pair in config.CURRENCY_PAIRS}
        )
        
        # Analyze performance across market regimes
        logger.info("Analyzing market regime performance...")
        all_strategy_results = {
            **threshold_results,
            'buy_and_hold': buy_hold_result,
            'rsi': rsi_result,
            'macd': macd_result
        }
        
        regime_analysis = strategy_manager.analyze_market_regime_performance(
            all_strategy_results, eval_timestamps
        )
        
        # Comprehensive strategy comparison
        logger.info("Performing comprehensive strategy comparison...")
        strategy_comparison = strategy_manager.compare_all_strategies(
            threshold_results['multi_currency_moderate'],
            {'buy_and_hold': buy_hold_result, 'rsi': rsi_result, 'macd': macd_result},
            single_currency_results
        )
        
        logger.info(f"Trading strategy analysis completed on {eval_set_name} set.")
        
        # ========================================================================================
        # STEP 8: RESULTS ANALYSIS AND VISUALIZATION
        # ========================================================================================
        logger.info("\n" + "="*60)
        logger.info("STEP 8: RESULTS ANALYSIS AND VISUALIZATION")
        logger.info("="*60)
        
        analyzer = ResultsAnalyzer(config)
        
        # Plot training history
        logger.info("Creating training history visualization...")
        training_plot_path = analyzer.plot_training_history(
            training_results['history'], config.EXPERIMENT_NAME
        )
        
        # Visualize correlation matrix
        logger.info("Creating correlation analysis visualization...")
        correlation_plot_path = analyzer.visualize_correlation_matrix(
            correlation_analysis, 'cross_currency_correlations'
        )
        
        # Plot strategy performance comparison
        logger.info("Creating strategy performance comparison...")
        performance_plot_path = analyzer.plot_strategy_performance(
            strategy_comparison, f'strategy_performance_comparison_{eval_set_name}'
        )
        
        # Create confusion matrices (using evaluation set)
        logger.info("Creating confusion matrices...")
        model_predictions_dict = {
            'CNN-LSTM': (model_predictions > 0.5).astype(int)
        }
        confusion_plot_path = analyzer.plot_confusion_matrices(
            model_predictions_dict, y_eval, f'model_confusion_matrices_{eval_set_name}'
        )
        
        # Plot predictions vs actual (using evaluation set)
        logger.info("Creating prediction vs actual visualization...")
        prediction_plot_path = analyzer.plot_prediction_vs_actual(
            model_predictions, y_eval, eval_timestamps, f'predictions_vs_actual_{eval_set_name}'
        )
        
        # Generate statistical test results
        logger.info("Generating statistical test results...")
        statistical_results = analyzer.generate_statistical_tests(strategy_comparison)
        
        # Create interactive dashboard
        logger.info("Creating interactive dashboard...")
        comprehensive_results = {
            'strategy_comparison': strategy_comparison,
            'regime_analysis': regime_analysis,
            'correlation_analysis': correlation_analysis,
            'statistical_tests': statistical_results,
            'pair_comparison': pair_comparison,
            'evaluation_set': eval_set_name
        }
        
        dashboard_path = analyzer.create_interactive_dashboard(
            comprehensive_results, f'forex_analysis_dashboard_{eval_set_name}'
        )
        
        # Create comprehensive research report
        logger.info("Generating comprehensive research report...")
        report_path = analyzer.create_research_report(
            comprehensive_results, f'{config.EXPERIMENT_NAME}_{eval_set_name}'
        )
        
        # Export results to LaTeX
        logger.info("Exporting results to LaTeX format...")
        latex_path = analyzer.export_results_to_latex(
            comprehensive_results, f'{config.EXPERIMENT_NAME}_{eval_set_name}'
        )
        
        logger.info(f"Results analysis and visualization completed using {eval_set_name} set.")
        
        # ========================================================================================
        # FINAL SUMMARY AND CLEANUP
        # ========================================================================================
        experiment_end_time = datetime.now()
        total_duration = experiment_end_time - experiment_start_time
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total execution time: {str(total_duration).split('.')[0]}")
        logger.info(f"Experiment ID: {config.EXPERIMENT_NAME}")
        logger.info(f"Results saved to: {config.RESULTS_DIR}")
        logger.info(f"Evaluation performed on: {eval_set_name.upper()} SET")
        
        # Print key results summary
        logger.info("\nKEY RESULTS SUMMARY:")
        logger.info("-" * 40)
        
        if 'overall_rankings' in strategy_comparison:
            best_strategy = strategy_comparison['overall_rankings'].get('best_overall_strategy', 'Unknown')
            logger.info(f"Best Overall Strategy: {best_strategy}")
        
        # Print performance metrics for multi-currency approach
        multi_currency_perf = threshold_results['multi_currency_moderate']['performance']
        logger.info(f"Multi-Currency CNN-LSTM Performance:")
        logger.info(f"  - Total Return: {multi_currency_perf['total_return']:.4f}")
        logger.info(f"  - Sharpe Ratio: {multi_currency_perf['sharpe_ratio']:.4f}")
        logger.info(f"  - Win Rate: {multi_currency_perf['win_rate']:.4f}")
        logger.info(f"  - Max Drawdown: {multi_currency_perf['max_drawdown']:.4f}")
        logger.info(f"  - Average Holding Hours: {multi_currency_perf.get('avg_holding_hours', 0):.2f}")
        
        # Print file locations
        logger.info("\nGENERATED FILES:")
        logger.info("-" * 40)
        logger.info(f"Research Report: {report_path}")
        logger.info(f"Interactive Dashboard: {dashboard_path}")
        logger.info(f"LaTeX Results: {latex_path}")
        logger.info(f"Model Architecture: {architecture_plot_path}")
        logger.info(f"Training History: {training_plot_path}")
        logger.info(f"Performance Comparison: {performance_plot_path}")
        
        # Important warning about development mode
        if config.DEVELOPMENT_MODE:
            logger.info("\n" + "!"*80)
            logger.info("IMPORTANT: DEVELOPMENT MODE NOTICE")
            logger.info("!"*80)
            logger.info("This analysis was performed using the VALIDATION SET.")
            logger.info("The TEST SET has been preserved for final evaluation before thesis defense.")
            logger.info("")
            logger.info("To run final evaluation on test set:")
            logger.info("1. Set DEVELOPMENT_MODE = False in config/config.py")
            logger.info("2. Run the system one final time before thesis defense")
            logger.info("3. Do NOT modify any parameters after final test evaluation")
            logger.info("!"*80)
        else:
            logger.info("\n" + "!"*80)
            logger.info("FINAL EVALUATION COMPLETED")
            logger.info("!"*80)
            logger.info("This analysis was performed using the TEST SET.")
            logger.info("Results are final and should be used for thesis defense.")
            logger.info("DO NOT modify parameters or re-run evaluation after this point.")
            logger.info("!"*80)
        
        logger.info("\n" + "="*80)
        logger.info("Thank you for using the Multi-Currency CNN-LSTM Forex Prediction System!")
        logger.info("="*80)
        
        return {
            'success': True,
            'experiment_id': config.EXPERIMENT_NAME,
            'duration': total_duration,
            'results': comprehensive_results,
            'evaluation_set': eval_set_name,
            'development_mode': config.DEVELOPMENT_MODE,
            'file_locations': {
                'report': report_path,
                'dashboard': dashboard_path,
                'latex': latex_path
            }
        }
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

if __name__ == "__main__":
    """
    Main execution entry point
    """
    print("Multi-Currency CNN-LSTM Forex Prediction System")
    print("=" * 60)
    print("Starting comprehensive forex prediction analysis...")
    print("This may take several hours to complete.")
    print("=" * 60)
    
    # Execute main pipeline
    results = main()
    
    # Print final status
    if results['success']:
        print("\n🎉 Analysis completed successfully!")
        print(f"📊 Check the results in the 'results' directory")
        print(f"📈 Interactive dashboard: {results['file_locations']['dashboard']}")
        print(f"📄 Research report: {results['file_locations']['report']}")
    else:
        print("\n❌ Analysis failed!")
        print(f"Error: {results['error']}")
        print("Check the log files for detailed error information.")
    
    print("\n" + "=" * 60)
    print("Execution completed.")
    print("=" * 60)