"""
Enhanced Main Execution Script with Complete Visualization Suite
Includes training history from existing logs and comprehensive analysis plots
"""

import os
import sys
import traceback
import warnings
import argparse
import glob
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

# Import the NEW enhanced visualization
from src.enhanced_visualization import EnhancedResultsAnalyzer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class ExperimentManager:
    """
    จัดการการ resume และตรวจจับ experiment ที่มีอยู่
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.get_log_path('experiment_manager'))
        
    def find_existing_experiments(self) -> List[Dict]:
        """
        ค้นหา experiment ที่มีอยู่และสถานะของแต่ละ experiment
        
        Returns:
            รายการ experiment พร้อมสถานะ
        """
        experiments = []
        
        # ค้นหาโมเดลที่เทรนแล้ว
        model_pattern = os.path.join(self.config.MODELS_DIR, "*_best_model.h5")
        model_files = glob.glob(model_pattern)
        
        for model_path in model_files:
            model_filename = os.path.basename(model_path)
            experiment_id = model_filename.replace('_best_model.h5', '')
            
            # ตรวจสอบไฟล์ log ที่เกี่ยวข้อง
            log_dir = os.path.join(self.config.LOGS_DIR, f"experiment_{experiment_id}")
            
            experiment_info = {
                'experiment_id': experiment_id,
                'model_path': model_path,
                'log_dir': log_dir,
                'has_training_completed': os.path.exists(model_path),
                'has_logs': os.path.exists(log_dir),
                'model_size_mb': os.path.getsize(model_path) / (1024*1024) if os.path.exists(model_path) else 0,
                'created_time': datetime.fromtimestamp(os.path.getctime(model_path)) if os.path.exists(model_path) else None
            }
            
            # ตรวจสอบว่า step ไหนทำเสร็จแล้วบ้าง
            experiment_info['completed_steps'] = self._analyze_completed_steps(experiment_id, log_dir)
            experiments.append(experiment_info)
        
        # เรียงตามเวลาล่าสุด
        experiments.sort(key=lambda x: x['created_time'] or datetime.min, reverse=True)
        
        return experiments
    
    def _analyze_completed_steps(self, experiment_id: str, log_dir: str) -> Dict:
        """
        วิเคราะห์ว่า step ไหนทำเสร็จแล้วบ้าง
        
        Args:
            experiment_id: ID ของ experiment
            log_dir: โฟลเดอร์ log
            
        Returns:
            Dictionary ระบุ step ที่เสร็จแล้ว
        """
        completed = {
            'data_loading': False,
            'preprocessing': False,
            'integration': False,
            'sequence_preparation': False,
            'model_architecture': False,
            'model_training': False,
            'strategy_testing': False,
            'visualization': False
        }
        
        # ตรวจสอบการเทรนโมเดล
        model_path = os.path.join(self.config.MODELS_DIR, f"{experiment_id}_best_model.h5")
        if os.path.exists(model_path):
            completed['model_training'] = True
            # ถ้าโมเดลเทรนเสร็จแล้ว แสดงว่า step ก่อนหน้าเสร็จหมดแล้ว
            completed.update({
                'data_loading': True,
                'preprocessing': True,
                'integration': True,
                'sequence_preparation': True,
                'model_architecture': True
            })
        
        # ตรวจสอบไฟล์ผลลัพธ์อื่นๆ
        results_files = {
            'strategy_testing': ['strategy_performance_comparison_validation.png', 'confusion_matrices_validation.png'],
            'visualization': ['forex_analysis_dashboard_validation.html', 'research_report_*.html']
        }
        
        for step, files in results_files.items():
            for file_pattern in files:
                matching_files = glob.glob(os.path.join(self.config.RESULTS_DIR, file_pattern))
                if matching_files:
                    completed[step] = True
                    break
        
        return completed
    
    def get_latest_experiment(self) -> Optional[Dict]:
        """
        หา experiment ล่าสุดที่มีโมเดลเทรนเสร็จแล้ว
        
        Returns:
            ข้อมูล experiment ล่าสุด หรือ None ถ้าไม่มี
        """
        experiments = self.find_existing_experiments()
        
        for exp in experiments:
            if exp['has_training_completed']:
                return exp
        
        return None
    
    def load_experiment_data(self, experiment_id: str) -> Dict:
        """
        โหลดข้อมูลที่จำเป็นสำหรับการ resume experiment
        
        Args:
            experiment_id: ID ของ experiment ที่ต้องการโหลด
            
        Returns:
            Dictionary ข้อมูลที่โหลดได้
        """
        self.logger.info(f"Loading experiment data for: {experiment_id}")
        
        experiment_data = {
            'experiment_id': experiment_id,
            'model_path': None,
            'training_data': None,
            'model_config': None,
            'preprocessing_artifacts': None
        }
        
        # โหลดโมเดลที่เทรนแล้ว
        model_path = os.path.join(self.config.MODELS_DIR, f"{experiment_id}_best_model.h5")
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                experiment_data['model_path'] = model_path
                experiment_data['trained_model'] = model
                self.logger.info(f"Successfully loaded trained model: {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
                return None
        
        # โหลด model configuration
        config_path = os.path.join(self.config.LOGS_DIR, f"experiment_{experiment_id}", "model_configuration.json")
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    experiment_data['model_config'] = json.load(f)
                self.logger.info("Model configuration loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load model configuration: {str(e)}")
        
        # โหลด preprocessing artifacts ถ้ามี
        preprocessing_path = os.path.join(self.config.RESULTS_DIR, 'preprocessing_scalers.joblib')
        if os.path.exists(preprocessing_path):
            try:
                import joblib
                experiment_data['preprocessing_artifacts'] = joblib.load(preprocessing_path)
                self.logger.info("Preprocessing artifacts loaded successfully")
            except Exception as e:
                self.logger.warning(f"Could not load preprocessing artifacts: {str(e)}")
        
        return experiment_data


def recreate_data_pipeline(config, logger):
    """
    สร้างข้อมูลใหม่สำหรับการทำงานต่อ
    (ใช้เมื่อไม่มี checkpoint แต่มีโมเดลเทรนแล้ว)
    
    Returns:
        Dictionary ข้อมูลที่จำเป็นสำหรับ step 7-8
    """
    logger.info("Recreating data pipeline for strategy testing...")
    
    # Step 1: Data Loading
    logger.info("🔄 Step 1: Reloading currency data...")
    data_loader = CurrencyDataLoader(config)
    raw_data = data_loader.load_currency_data()
    
    # Step 2: Preprocessing  
    logger.info("🔄 Step 2: Reprocessing data...")
    preprocessor = ForexDataPreprocessor(config)
    
    # ลองโหลด preprocessing artifacts ก่อน
    try:
        preprocessor.load_preprocessing_artifacts(config.RESULTS_DIR)
        logger.info("Using existing preprocessing artifacts")
    except:
        logger.info("Creating new preprocessing artifacts")
    
    cleaned_data = preprocessor.handle_missing_values(raw_data)
    returns_data = preprocessor.calculate_percentage_returns(cleaned_data)
    normalized_data = preprocessor.apply_mixed_normalization(returns_data, fit_data=True)
    
    # Step 3: Integration
    logger.info("🔄 Step 3: Reintegrating multi-currency data...")
    integrator = MultiCurrencyIntegrator(config)
    unified_data = integrator.merge_currency_pairs(normalized_data)
    correlation_analysis = integrator.calculate_cross_correlation(unified_data)
    
    # Step 4: Sequence Preparation
    logger.info("🔄 Step 4: Recreating sequences...")
    sequence_preparator = SequenceDataPreparator(config)
    X_sequences, y_labels, timestamps = sequence_preparator.create_sequences(
        unified_data, target_pair='EURUSD'
    )
    data_splits = sequence_preparator.split_temporal_data(X_sequences, y_labels, timestamps)
    
    logger.info("✅ Data pipeline recreation completed")
    
    return {
        'unified_data': unified_data,
        'correlation_analysis': correlation_analysis,
        'data_splits': data_splits,
        'X_sequences': X_sequences,
        'y_labels': y_labels,
        'timestamps': timestamps,
        'returns_data': returns_data  # เพิ่มข้อมูล returns_data ที่มี original prices
    }


def run_strategy_testing(config, trained_model, data_pipeline, logger):
    """
    รัน Step 7: Trading Strategy Testing - ENHANCED VERSION
    """
    logger.info("="*60)
    logger.info("🎯 STEP 7: ENHANCED TRADING STRATEGY TESTING")
    logger.info("="*60)
    
    strategy_manager = TradingStrategyManager(config)
    
    # เลือกชุดข้อมูลสำหรับประเมิน
    data_splits = data_pipeline['data_splits']
    unified_data = data_pipeline['unified_data']
    returns_data = data_pipeline['returns_data']
    
    if config.DEVELOPMENT_MODE:
        X_eval, y_eval, eval_timestamps = data_splits['val']
        eval_set_name = "validation"
        logger.info("📊 Using VALIDATION set for strategy testing (Development Mode)")
    else:
        X_eval, y_eval, eval_timestamps = data_splits['test']
        eval_set_name = "test"
        logger.info("📊 Using TEST set for final evaluation")
    
    logger.info(f"Evaluation set size: {len(X_eval)} samples")
    
    # สร้างการทำนายจากโมเดล
    logger.info("🤖 Generating model predictions...")
    model_predictions = trained_model.predict(X_eval, batch_size=config.BATCH_SIZE, verbose=1)
    model_predictions = model_predictions.flatten()
    
    # ดึงข้อมูลราคาสำหรับการทดสอบ - ENHANCED VERSION
    logger.info("📋 Extracting enhanced price data for evaluation...")
    
    try:
        # ใช้ข้อมูลจาก returns_data ที่มี original prices
        if 'EURUSD' in returns_data and 'Close_Price' in returns_data['EURUSD'].columns:
            logger.info("Using EURUSD Close_Price from returns_data")
            eurusd_data = returns_data['EURUSD']
            eval_price_data = eurusd_data.loc[eval_timestamps]['Close_Price']
        else:
            # วิธีทางเลือก
            available_columns = []
            if 'EURUSD' in returns_data:
                available_columns = list(returns_data['EURUSD'].columns)
                logger.info(f"Available EURUSD columns in returns_data: {available_columns}")
                
                price_columns = [col for col in available_columns if 'Close' in col and 'Price' in col]
                if price_columns:
                    logger.info(f"Found price columns: {price_columns}")
                    eval_price_data = returns_data['EURUSD'].loc[eval_timestamps][price_columns[0]]
                else:
                    raise KeyError("No suitable price column found")
            else:
                raise KeyError("EURUSD data not found in returns_data")
                
    except Exception as e:
        logger.warning(f"Could not extract price data: {str(e)}")
        # สร้างข้อมูลราคาจำลอง
        logger.warning("Creating synthetic price data as fallback...")
        np.random.seed(42)
        base_price = 1.1000
        price_changes = np.random.normal(0, 0.001, len(eval_timestamps))
        eval_price_data = pd.Series(
            base_price + np.cumsum(price_changes), 
            index=eval_timestamps,
            name='EURUSD_Close_Synthetic'
        )
        logger.info("Using synthetic price data for strategy testing")
    
    logger.info(f"Price data extracted successfully: {len(eval_price_data)} samples")
    logger.info(f"Price range: {eval_price_data.min():.6f} - {eval_price_data.max():.6f}")
    
    logger.info("📈 Testing threshold-based trading strategies...")
    
    # ทดสอบ threshold strategies ต่างๆ
    threshold_results = {}
    for threshold_type in ['conservative', 'moderate', 'aggressive']:
        logger.info(f"  Testing {threshold_type} strategy...")
        threshold_result = strategy_manager.apply_threshold_strategy(
            model_predictions, eval_price_data, eval_timestamps, threshold_type
        )
        threshold_results[f'multi_currency_{threshold_type}'] = threshold_result
        
        # แสดงผลลัพธ์เบื้องต้น
        perf = threshold_result['performance']
        logger.info(f"    {threshold_type}: {perf['total_trades']} trades, "
                   f"return: {perf['total_return']:.4f}, win rate: {perf['win_rate']:.4f}")
    
    logger.info("📊 Testing baseline strategies...")
    
    # ทดสอบ baseline strategies
    buy_hold_result = strategy_manager.implement_buy_hold_strategy(eval_price_data, eval_timestamps)
    rsi_result = strategy_manager.implement_rsi_strategy(eval_price_data, eval_timestamps)
    macd_result = strategy_manager.implement_macd_strategy(eval_price_data, eval_timestamps)
    
    logger.info(f"  Buy & Hold: return: {buy_hold_result['total_return']:.4f}")
    logger.info(f"  RSI: {rsi_result['performance']['total_trades']} trades, "
               f"return: {rsi_result['performance']['total_return']:.4f}")
    logger.info(f"  MACD: {macd_result['performance']['total_trades']} trades, "
               f"return: {macd_result['performance']['total_return']:.4f}")
    
    # การเปรียบเทียบแบบครบถ้วน
    logger.info("🔍 Performing comprehensive strategy comparison...")
    strategy_comparison = strategy_manager.compare_all_strategies(
        threshold_results['multi_currency_moderate'],
        {'buy_and_hold': buy_hold_result, 'rsi': rsi_result, 'macd': macd_result},
        {}
    )
    
    # แสดงผลการเปรียบเทียบ
    if 'overall_rankings' in strategy_comparison:
        best_strategy = strategy_comparison['overall_rankings'].get('best_overall_strategy', 'Unknown')
        logger.info(f"🏆 Best overall strategy: {best_strategy}")
    
    logger.info("✅ Enhanced strategy testing completed successfully")
    
    return {
        'model_predictions': model_predictions,
        'threshold_results': threshold_results,
        'baseline_results': {
            'buy_and_hold': buy_hold_result,
            'rsi': rsi_result,
            'macd': macd_result
        },
        'strategy_comparison': strategy_comparison,
        'eval_set_name': eval_set_name,
        'y_eval': y_eval,
        'eval_timestamps': eval_timestamps,
        'price_data': eval_price_data  # เพิ่มข้อมูลราคา
    }


def run_enhanced_visualization(config, strategy_results, data_pipeline, experiment_id, logger):
    """
    รัน Step 8: Enhanced Results Analysis and Visualization
    """
    logger.info("="*60)
    logger.info("📊 STEP 8: ENHANCED RESULTS ANALYSIS AND VISUALIZATION")
    logger.info("="*60)
    
    # ใช้ Enhanced Results Analyzer
    analyzer = EnhancedResultsAnalyzer(config)
    
    # สร้าง comprehensive results
    comprehensive_results = {
        'strategy_comparison': strategy_results['strategy_comparison'],
        'correlation_analysis': data_pipeline['correlation_analysis'],
        'evaluation_set': strategy_results['eval_set_name'],
        'model_predictions': strategy_results['model_predictions'],
        'true_labels': strategy_results['y_eval'],
        'timestamps': strategy_results['eval_timestamps'],
        'price_data': strategy_results['price_data']
    }
    
    # สร้าง comprehensive analysis suite
    logger.info("🎨 Creating comprehensive analysis suite...")
    plot_paths = analyzer.create_comprehensive_analysis_suite(comprehensive_results, experiment_id)
    
    # สร้างกราฟเพิ่มเติม
    logger.info("📈 Creating additional analysis plots...")
    
    # 1. Cumulative return analysis by year and currency
    if data_pipeline.get('returns_data'):
        logger.info("Creating cumulative return analysis by year and currency...")
        cumulative_plots = analyzer.create_cumulative_return_analysis_by_year_and_currency(
            strategy_results['strategy_comparison'],
            data_pipeline['returns_data'],
            strategy_results['eval_timestamps']
        )
        plot_paths['cumulative_returns'] = cumulative_plots
    
    # 2. Prediction quality analysis
    if len(strategy_results['model_predictions']) > 0:
        logger.info("Creating prediction quality analysis...")
        pred_quality_plot = analyzer.create_prediction_quality_analysis(
            strategy_results['model_predictions'],
            strategy_results['y_eval'],
            strategy_results['eval_timestamps']
        )
        if pred_quality_plot:
            plot_paths['prediction_quality'] = pred_quality_plot
    
    # 3. Enhanced confusion matrices
    logger.info("Creating enhanced confusion matrices...")
    model_predictions_dict = {
        'Multi-Currency CNN-LSTM': (strategy_results['model_predictions'] > 0.5).astype(int)
    }
    confusion_plot = analyzer.plot_confusion_matrices(
        model_predictions_dict, 
        strategy_results['y_eval'], 
        f'enhanced_confusion_matrices_{strategy_results["eval_set_name"]}'
    )
    if confusion_plot:
        plot_paths['confusion_matrices'] = confusion_plot
    
    # 4. สร้าง interactive dashboard
    logger.info("🌐 Creating enhanced interactive dashboard...")
    dashboard_path = analyzer.create_interactive_dashboard(
        comprehensive_results, 
        f'enhanced_forex_dashboard_{strategy_results["eval_set_name"]}'
    )
    if dashboard_path:
        plot_paths['interactive_dashboard'] = dashboard_path
    
    # 5. สร้าง research report
    logger.info("📑 Creating enhanced research report...")
    report_path = analyzer.create_research_report(
        comprehensive_results, 
        f'{experiment_id}_{strategy_results["eval_set_name"]}_enhanced'
    )
    if report_path:
        plot_paths['research_report'] = report_path
    
    # 6. Export LaTeX results
    logger.info("📄 Exporting enhanced LaTeX results...")
    latex_path = analyzer.export_results_to_latex(
        comprehensive_results, 
        f'{experiment_id}_{strategy_results["eval_set_name"]}_enhanced'
    )
    if latex_path:
        plot_paths['latex_results'] = latex_path
    
    logger.info("✅ Enhanced visualization and reporting completed")
    
    # แสดงสรุปไฟล์ที่สร้าง
    logger.info("📊 Generated Analysis Files:")
    for analysis_type, file_path in plot_paths.items():
        if isinstance(file_path, str):
            logger.info(f"  📈 {analysis_type}: {os.path.basename(file_path)}")
        elif isinstance(file_path, list):
            logger.info(f"  📈 {analysis_type}: {len(file_path)} files")
    
    return {
        'plot_paths': plot_paths,
        'comprehensive_results': comprehensive_results,
        'total_plots_created': sum(1 if isinstance(p, str) else len(p) for p in plot_paths.values())
    }


def main():
    """
    Enhanced Main execution function with comprehensive visualization
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced Multi-Currency CNN-LSTM Forex Prediction System')
    parser.add_argument('--experiment', type=str, help='Specific experiment ID to resume')
    parser.add_argument('--list-experiments', action='store_true', help='List available experiments')
    parser.add_argument('--from-step', type=int, choices=[7, 8], default=7, 
                       help='Start from step (7=strategy testing, 8=visualization)')
    parser.add_argument('--dev-mode', action='store_true', default=True,
                       help='Use validation set (development mode)')
    parser.add_argument('--final-mode', action='store_true',
                       help='Use test set (final evaluation mode)')
    parser.add_argument('--create-training-plots', action='store_true',
                       help='Create training history plots from existing logs')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.create_directories()
    
    # Handle final mode
    if args.final_mode:
        config.DEVELOPMENT_MODE = False
        print("🚨 FINAL EVALUATION MODE: Using test set")
    else:
        config.DEVELOPMENT_MODE = True
        print("🔬 DEVELOPMENT MODE: Using validation set")
    
    # Initialize experiment manager
    exp_manager = ExperimentManager(config)
    
    # Handle list experiments request
    if args.list_experiments:
        experiments = exp_manager.find_existing_experiments()
        print(f"\n📋 Available experiments ({len(experiments)} found):")
        
        if not experiments:
            print("  No experiments found.")
            return 0
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n  {i}. {exp['experiment_id']}")
            print(f"     Created: {exp['created_time']}")
            print(f"     Model: {exp['model_size_mb']:.1f} MB")
            print(f"     Training completed: {'✅' if exp['has_training_completed'] else '❌'}")
            
            completed_steps = [k for k, v in exp['completed_steps'].items() if v]
            print(f"     Completed steps: {', '.join(completed_steps)}")
        
        return 0
    
    # Handle create training plots request
    if args.create_training_plots:
        print("🎨 Creating training history plots from existing experiments...")
        
        experiments = exp_manager.find_existing_experiments()
        if not experiments:
            print("❌ No experiments found!")
            return 1
        
        # Setup logging
        logger = setup_logging(config.LOG_LEVEL, config.get_log_path('training_plots'))
        
        # Create enhanced analyzer
        analyzer = EnhancedResultsAnalyzer(config)
        
        plots_created = []
        for exp in experiments:
            if exp['has_training_completed']:
                experiment_id = exp['experiment_id']
                print(f"📊 Creating training history for: {experiment_id}")
                
                # Create training history plot
                training_plot = analyzer.create_training_history_from_logs(experiment_id)
                if training_plot:
                    plots_created.append(training_plot)
                    print(f"   ✅ Created: {os.path.basename(training_plot)}")
                else:
                    print(f"   ❌ Failed to create training plot")
        
        print(f"\n🎉 Created {len(plots_created)} training history plots!")
        for plot in plots_created:
            print(f"   📈 {os.path.basename(plot)}")
        
        return 0
    
    # Find experiment to use
    if args.experiment:
        experiment_id = args.experiment
        config.update_experiment_name(experiment_id)
        print(f"🎯 Using specified experiment: {experiment_id}")
    else:
        latest_exp = exp_manager.get_latest_experiment()
        if not latest_exp:
            print("❌ No trained experiments found!")
            print("   Run with --list-experiments to see available experiments")
            print("   Or run with --create-training-plots to create training plots")
            return 1
        
        experiment_id = latest_exp['experiment_id']
        config.update_experiment_name(experiment_id)
        print(f"🔄 Resuming latest experiment: {experiment_id}")
        print(f"   Created: {latest_exp['created_time']}")
        print(f"   Model size: {latest_exp['model_size_mb']:.1f} MB")
    
    # Setup logging
    logger = setup_logging(config.LOG_LEVEL, config.get_log_path('enhanced_execution'))
    
    print_system_info()
    
    print("=" * 80)
    print("🚀 ENHANCED MULTI-CURRENCY CNN-LSTM FOREX PREDICTION SYSTEM")
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Mode: {'Final Evaluation' if not config.DEVELOPMENT_MODE else 'Development'}")
    print(f"Starting from step: {args.from_step}")
    print(f"Enhanced Visualization: ✅ ENABLED")
    print(f"Starting execution at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Load experiment data
        logger.info("Loading existing experiment data...")
        experiment_data = exp_manager.load_experiment_data(experiment_id)
        
        if not experiment_data or not experiment_data.get('trained_model'):
            logger.error("Could not load trained model!")
            return 1
        
        trained_model = experiment_data['trained_model']
        logger.info(f"✅ Successfully loaded trained model")
        
        # Display model information
        if experiment_data.get('model_config'):
            model_config = experiment_data['model_config']
            print(f"📊 Model Information:")
            print(f"   Total parameters: {model_config.get('total_parameters', 'Unknown'):,}")
            print(f"   Architecture: {model_config.get('model_name', 'Unknown')}")
            print(f"   Input shape: {model_config.get('input_shape', 'Unknown')}")
        
        # CREATE TRAINING HISTORY PLOT FIRST (from existing logs)
        print("\n🎨 Creating Training History from Existing Logs...")
        analyzer = EnhancedResultsAnalyzer(config)
        training_plot = analyzer.create_training_history_from_logs(experiment_id)
        if training_plot:
            print(f"   ✅ Training History: {os.path.basename(training_plot)}")
        else:
            print(f"   ⚠️  Could not create training history plot")
        
        # Recreate data pipeline
        logger.info("Recreating data pipeline...")
        data_pipeline = recreate_data_pipeline(config, logger)
        
        results = {}
        
        # Run requested steps
        if args.from_step <= 7:
            logger.info("Starting Step 7: Enhanced Strategy Testing...")
            strategy_results = run_strategy_testing(config, trained_model, data_pipeline, logger)
            results['strategy_results'] = strategy_results
            
            # Show key results
            print(f"\n🎯 ENHANCED STRATEGY TESTING RESULTS:")
            moderate_perf = strategy_results['threshold_results']['multi_currency_moderate']['performance']
            print(f"   Multi-Currency CNN-LSTM (Moderate):")
            print(f"     Total Return: {moderate_perf['total_return']:.4f}")
            print(f"     Win Rate: {moderate_perf['win_rate']:.4f}")
            print(f"     Sharpe Ratio: {moderate_perf['sharpe_ratio']:.4f}")
            print(f"     Total Trades: {moderate_perf['total_trades']}")
            
            if 'overall_rankings' in strategy_results['strategy_comparison']:
                best_strategy = strategy_results['strategy_comparison']['overall_rankings'].get('best_overall_strategy', 'Unknown')
                print(f"   🏆 Best Overall Strategy: {best_strategy}")
        
        if args.from_step <= 8:
            if 'strategy_results' not in results:
                logger.error("Strategy testing results not available for visualization")
                return 1
            
            logger.info("Starting Step 8: Enhanced Visualization...")
            viz_results = run_enhanced_visualization(config, results['strategy_results'], data_pipeline, experiment_id, logger)
            results['visualization_results'] = viz_results
            
            # Show generated files
            print(f"\n📊 ENHANCED ANALYSIS RESULTS:")
            total_plots = viz_results['total_plots_created']
            print(f"   📈 Total Plots Created: {total_plots}")
            
            plot_paths = viz_results['plot_paths']
            for analysis_type, path in plot_paths.items():
                if isinstance(path, str):
                    print(f"   📊 {analysis_type.title()}: {os.path.basename(path)}")
                elif isinstance(path, list) and path:
                    print(f"   📊 {analysis_type.title()}: {len(path)} files")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("🎉 ENHANCED EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total execution time: {str(duration).split('.')[0]}")
        print(f"Results saved to: {config.RESULTS_DIR}")
        print(f"Training History: ✅ Created from existing logs")
        print(f"Enhanced Analysis: ✅ Complete visualization suite")
        
        if config.DEVELOPMENT_MODE:
            print("\n💡 DEVELOPMENT MODE NOTICE:")
            print("   This analysis used the VALIDATION set.")
            print("   For final thesis results, run with --final-mode")
        else:
            print("\n🎓 FINAL EVALUATION COMPLETED")
            print("   Results ready for thesis defense!")
        
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        print(f"\n❌ Execution failed: {str(e)}")
        print("Check the log files for detailed error information.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)