"""
Training and Optimization Module with Comprehensive Logging
Handles model training, optimization, and detailed experiment tracking
Implements sophisticated callbacks and logging for research reproducibility
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import pandas as pd
import logging
import os
import time
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model trainer with advanced logging and monitoring capabilities
    Tracks all aspects of training for research reproducibility and analysis
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration and logging setup
        
        Args:
            config: Configuration object containing training parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Training parameters
        self.epochs = config.EPOCHS
        self.batch_size = config.BATCH_SIZE
        self.learning_rate = config.LEARNING_RATE
        
        # Experiment tracking
        self.experiment_id = config.EXPERIMENT_NAME
        self.experiment_start_time = None
        self.training_history = {}
        self.training_metrics = {}
        
        # Model and training state
        self.model = None
        self.callbacks = []
        self.training_logs = []
        
        # Setup comprehensive logging system
        self.setup_logging_system()
        
    def setup_logging_system(self):
        """
        Setup comprehensive logging system for experiment tracking
        """
        self.logger.info("Setting up comprehensive logging system")
        
        # Create experiment-specific log directory
        self.experiment_log_dir = os.path.join(
            self.config.LOGS_DIR, 
            f"experiment_{self.experiment_id}"
        )
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        
        # Setup different log files for different purposes
        self.log_files = {
            'training': os.path.join(self.experiment_log_dir, 'training_progress.log'),
            'metrics': os.path.join(self.experiment_log_dir, 'training_metrics.csv'),
            'system': os.path.join(self.experiment_log_dir, 'system_monitor.log'),
            'experiment': os.path.join(self.experiment_log_dir, 'experiment_metadata.json')
        }
        
        # Initialize system monitoring
        self.system_stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamps': []
        }
        
        self.logger.info(f"Logging system initialized for experiment: {self.experiment_id}")
        self.logger.info(f"Log directory: {self.experiment_log_dir}")
    
    def log_model_configuration(self, model, architecture_summary: Dict):
        """
        Log comprehensive model configuration and architecture details
        
        Args:
            model: Compiled Keras model
            architecture_summary: Dictionary containing architecture details
        """
        self.logger.info("Logging model configuration and architecture")
        
        # Store model reference
        self.model = model
        
        # Create detailed model configuration log
        model_config = {
            'model_name': model.name,
            'total_parameters': model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'model_architecture': architecture_summary,
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'optimizer_config': {
                'name': model.optimizer.__class__.__name__,
                'learning_rate': float(model.optimizer.learning_rate),
                'configuration': model.optimizer.get_config()
            },
            'loss_function': model.loss,
            'metrics': model.metrics_names,
            'layer_details': []
        }
        
        # Detailed layer information
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'parameters': layer.count_params(),
                'output_shape': str(layer.output_shape) if hasattr(layer, 'output_shape') else 'N/A',
                'trainable': layer.trainable
            }
            
            # Add layer-specific configuration
            try:
                layer_info['config'] = layer.get_config()
            except:
                layer_info['config'] = 'Unable to retrieve'
            
            model_config['layer_details'].append(layer_info)
        
        # Save model configuration
        config_path = os.path.join(self.experiment_log_dir, 'model_configuration.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)
        
        # Log summary statistics
        self.logger.info(f"Model: {model.name}")
        self.logger.info(f"Total parameters: {model_config['total_parameters']:,}")
        self.logger.info(f"Trainable parameters: {model_config['trainable_parameters']:,}")
        self.logger.info(f"Layers: {len(model_config['layer_details'])}")
        self.logger.info(f"Model configuration saved to: {config_path}")
    
    def log_training_progress(self, epoch: int, logs: Dict):
        """
        Log detailed training progress for each epoch
        
        Args:
            epoch: Current epoch number
            logs: Training logs from Keras
        """
        # Record system statistics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        gpu_info = self._get_gpu_info()
        
        # Create comprehensive training log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch + 1,
            'training_metrics': dict(logs),
            'system_stats': {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory_info.percent,
                'memory_available_gb': memory_info.available / (1024**3),
                'memory_used_gb': memory_info.used / (1024**3)
            }
        }
        
        # Add GPU information if available
        if gpu_info:
            log_entry['system_stats']['gpu_info'] = gpu_info
        
        # Store for later analysis
        self.training_logs.append(log_entry)
        
        # Update system monitoring arrays
        self.system_stats['cpu_usage'].append(cpu_percent)
        self.system_stats['memory_usage'].append(memory_info.percent)
        self.system_stats['timestamps'].append(datetime.now())
        
        # Log to file and console
        self.logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Loss: {logs.get('loss', 0):.6f}, "
                        f"Accuracy: {logs.get('accuracy', 0):.6f}, "
                        f"Val_Loss: {logs.get('val_loss', 0):.6f}, "
                        f"Val_Accuracy: {logs.get('val_accuracy', 0):.6f}")
        
        # Check for potential issues
        self._check_training_issues(logs, epoch)
    
    def _get_gpu_info(self) -> Optional[Dict]:
        """Get GPU information if available"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Try to get GPU memory info
                gpu_info = {
                    'gpu_count': len(gpus),
                    'gpu_names': [gpu.name for gpu in gpus]
                }
                
                # Get memory info if possible
                try:
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    gpu_info['memory_current_mb'] = memory_info['current'] / (1024**2)
                    gpu_info['memory_peak_mb'] = memory_info['peak'] / (1024**2)
                except:
                    pass
                
                return gpu_info
        except:
            pass
        return None
    
    def _check_training_issues(self, logs: Dict, epoch: int):
        """Check for potential training issues and log warnings"""
        # Check for loss explosion
        if 'loss' in logs and logs['loss'] > 10:
            self.logger.warning(f"Epoch {epoch + 1}: High training loss detected: {logs['loss']:.6f}")
        
        # Check for NaN values
        for metric, value in logs.items():
            if np.isnan(value) or np.isinf(value):
                self.logger.error(f"Epoch {epoch + 1}: {metric} is {value}")
        
        # Check for overfitting
        if 'loss' in logs and 'val_loss' in logs:
            if logs['val_loss'] > logs['loss'] * 1.5:
                self.logger.warning(f"Epoch {epoch + 1}: Potential overfitting detected - "
                                  f"val_loss ({logs['val_loss']:.6f}) >> train_loss ({logs['loss']:.6f})")
        
        # Check for learning stagnation
        if epoch > 10 and len(self.training_logs) > 5:
            recent_losses = [log['training_metrics'].get('val_loss', 0) for log in self.training_logs[-5:]]
            if len(set([round(loss, 4) for loss in recent_losses])) == 1:
                self.logger.warning(f"Epoch {epoch + 1}: Learning may have stagnated - "
                                  f"validation loss unchanged for 5 epochs")
    
    def log_memory_usage(self):
        """Log current memory usage statistics"""
        from src.utils import calculate_memory_usage
        
        memory_info = calculate_memory_usage()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        memory_log = {
            'timestamp': datetime.now().isoformat(),
            'system_memory': memory_info,
            'process_memory': {
                'rss_gb': process_memory.rss / (1024**3),
                'vms_gb': process_memory.vms / (1024**3)
            }
        }
        
        # Log GPU memory if available
        try:
            if tf.config.experimental.list_physical_devices('GPU'):
                memory_info_gpu = tf.config.experimental.get_memory_info('GPU:0')
                memory_log['gpu_memory'] = {
                    'current_mb': memory_info_gpu['current'] / (1024**2),
                    'peak_mb': memory_info_gpu['peak'] / (1024**2)
                }
        except:
            pass
        
        # Save to system log
        with open(self.log_files['system'], 'a') as f:
            f.write(json.dumps(memory_log) + '\n')
        
        # Log to console using the dict values
        if memory_info and 'system_percent' in memory_info:
            self.logger.info(f"Memory usage: {memory_info['system_percent']:.1f}% "
                           f"({memory_info.get('system_used_gb', 0):.2f}GB / {memory_info.get('system_total_gb', 0):.2f}GB)")
        else:
            self.logger.info("Memory usage information not available")
    
    def log_computation_time(self, start_time: float, end_time: float, operation: str):
        """
        Log computation time for specific operations
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp  
            operation: Description of the operation
        """
        duration = end_time - start_time
        
        time_log = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': duration,
            'duration_formatted': self._format_duration(duration)
        }
        
        self.logger.info(f"{operation} completed in {self._format_duration(duration)}")
        
        # Store in training metrics
        if 'computation_times' not in self.training_metrics:
            self.training_metrics['computation_times'] = []
        self.training_metrics['computation_times'].append(time_log)
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"
    
    def setup_callbacks(self) -> List:
        """
        Setup comprehensive callbacks for training monitoring and control
        
        Returns:
            List of configured callbacks
        """
        self.logger.info("Setting up training callbacks")
        
        callbacks = []
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        )
        callbacks.append(early_stopping)
        self.logger.info(f"Early Stopping: patience={self.config.EARLY_STOPPING_PATIENCE}")
        
        # Model Checkpoint
        checkpoint_path = os.path.join(
            self.config.MODELS_DIR,
            f"{self.experiment_id}_best_model.h5"
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='min'
        )
        callbacks.append(model_checkpoint)
        self.logger.info(f"Model Checkpoint: {checkpoint_path}")
        
        # Reduce Learning Rate
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.REDUCE_LR_FACTOR,
            patience=self.config.REDUCE_LR_PATIENCE,
            min_lr=self.config.REDUCE_LR_MIN_LR,
            verbose=1,
            mode='min'
        )
        callbacks.append(reduce_lr)
        self.logger.info(f"Reduce LR: factor={self.config.REDUCE_LR_FACTOR}, "
                        f"patience={self.config.REDUCE_LR_PATIENCE}")
        
        # CSV Logger for metrics
        csv_logger = CSVLogger(
            filename=self.log_files['metrics'],
            separator=',',
            append=False
        )
        callbacks.append(csv_logger)
        self.logger.info(f"CSV Logger: {self.log_files['metrics']}")
        
        # Custom logging callback
        custom_logger = CustomLoggingCallback(self)
        callbacks.append(custom_logger)
        
        # Memory monitoring callback
        memory_monitor = MemoryMonitorCallback(self)
        callbacks.append(memory_monitor)
        
        self.callbacks = callbacks
        return callbacks
    
    def create_experiment_metadata(self, additional_info: Dict = None):
        """
        Create comprehensive experiment metadata for reproducibility
        
        Args:
            additional_info: Additional information to include in metadata
        """
        metadata = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'start_time': datetime.now().isoformat(),
                'tensorflow_version': tf.__version__,
                'python_version': __import__('sys').version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            },
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'platform': __import__('platform').platform(),
                'gpu_available': len(tf.config.experimental.list_physical_devices('GPU')) > 0,
                'gpu_count': len(tf.config.experimental.list_physical_devices('GPU'))
            },
            'training_config': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'early_stopping_patience': self.config.EARLY_STOPPING_PATIENCE,
                'reduce_lr_patience': self.config.REDUCE_LR_PATIENCE,
                'reduce_lr_factor': self.config.REDUCE_LR_FACTOR
            },
            'data_config': {
                'window_size': self.config.WINDOW_SIZE,
                'prediction_horizon': self.config.PREDICTION_HORIZON,
                'currency_pairs': self.config.CURRENCY_PAIRS,
                'total_features': self.config.TOTAL_FEATURES
            }
        }
        
        # Add additional information if provided
        if additional_info:
            metadata['additional_info'] = additional_info
        
        # Save metadata
        with open(self.log_files['experiment'], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Experiment metadata saved: {self.log_files['experiment']}")
    
    def train_model(self, model, train_generator, val_generator, 
                   steps_per_epoch: int, validation_steps: int) -> Dict:
        """
        Train model with comprehensive logging and monitoring
        
        Args:
            model: Compiled Keras model
            train_generator: Training data generator
            val_generator: Validation data generator
            steps_per_epoch: Number of steps per epoch
            validation_steps: Number of validation steps
            
        Returns:
            Dictionary containing training history and metrics
        """
        self.logger.info("Starting model training with comprehensive monitoring")
        
        # Record experiment start time
        self.experiment_start_time = time.time()
        
        # Log initial memory state
        self.log_memory_usage()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Create experiment metadata
        self.create_experiment_metadata({
            'steps_per_epoch': steps_per_epoch,
            'validation_steps': validation_steps,
            'total_training_samples': steps_per_epoch * self.batch_size,
            'total_validation_samples': validation_steps * self.batch_size
        })
        
        try:
            # Start training with timing
            training_start = time.time()
            
            history = model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=val_generator,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                workers=1,
                use_multiprocessing=False
            )
            
            training_end = time.time()
            
            # Log training completion
            self.log_computation_time(training_start, training_end, "Model Training")
            
            # Store training history
            self.training_history = history.history
            
            # Calculate final training metrics
            final_metrics = self._calculate_final_metrics(history)
            
            # Save comprehensive training summary
            self._save_training_summary(final_metrics)
            
            self.logger.info("Model training completed successfully")
            return {
                'history': history.history,
                'final_metrics': final_metrics,
                'training_duration': training_end - training_start,
                'experiment_id': self.experiment_id
            }
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            
            # Save error information
            error_info = {
                'error_message': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat(),
                'training_logs': self.training_logs
            }
            
            error_path = os.path.join(self.experiment_log_dir, 'training_error.json')
            with open(error_path, 'w') as f:
                json.dump(error_info, f, indent=2, default=str)
            
            raise
    
    def _calculate_final_metrics(self, history) -> Dict:
        """Calculate comprehensive final training metrics"""
        final_metrics = {
            'best_epoch': np.argmin(history.history['val_loss']) + 1,
            'best_val_loss': float(np.min(history.history['val_loss'])),
            'best_val_accuracy': float(np.max(history.history.get('val_accuracy', [0]))),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_train_accuracy': float(history.history.get('accuracy', [0])[-1]),
            'total_epochs_trained': len(history.history['loss']),
            'training_stability': {
                'loss_variance': float(np.var(history.history['loss'])),
                'val_loss_variance': float(np.var(history.history['val_loss'])),
                'loss_trend': 'decreasing' if history.history['loss'][-1] < history.history['loss'][0] else 'increasing'
            }
        }
        
        # Calculate learning rate changes if available
        if 'lr' in history.history:
            final_metrics['learning_rate_changes'] = len(set(history.history['lr']))
            final_metrics['final_learning_rate'] = float(history.history['lr'][-1])
        
        return final_metrics
    
    def _save_training_summary(self, final_metrics: Dict):
        """Save comprehensive training summary"""
        training_summary = {
            'experiment_metadata': {
                'experiment_id': self.experiment_id,
                'completion_time': datetime.now().isoformat(),
                'total_experiment_duration': time.time() - self.experiment_start_time
            },
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'system_monitoring': {
                'average_cpu_usage': np.mean(self.system_stats['cpu_usage']) if self.system_stats['cpu_usage'] else 0,
                'max_memory_usage': np.max(self.system_stats['memory_usage']) if self.system_stats['memory_usage'] else 0,
                'training_stability_indicators': self._analyze_training_stability()
            },
            'computation_times': self.training_metrics.get('computation_times', [])
        }
        
        # Save summary
        summary_path = os.path.join(self.experiment_log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        # Save training logs
        logs_path = os.path.join(self.experiment_log_dir, 'detailed_training_logs.json')
        with open(logs_path, 'w') as f:
            json.dump(self.training_logs, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_path}")
    
    def _analyze_training_stability(self) -> Dict:
        """Analyze training stability indicators"""
        if not self.training_logs:
            return {}
        
        # Extract loss values
        losses = [log['training_metrics'].get('loss', 0) for log in self.training_logs]
        val_losses = [log['training_metrics'].get('val_loss', 0) for log in self.training_logs]
        
        stability = {
            'loss_oscillations': len([i for i in range(1, len(losses)) if losses[i] > losses[i-1]]),
            'val_loss_oscillations': len([i for i in range(1, len(val_losses)) if val_losses[i] > val_losses[i-1]]),
            'convergence_rate': 'fast' if len(losses) < self.epochs * 0.5 else 'slow',
            'overfitting_episodes': len([i for i, log in enumerate(self.training_logs) 
                                       if log['training_metrics'].get('val_loss', 0) > 
                                       log['training_metrics'].get('loss', 0) * 1.5])
        }
        
        return stability
    
    def save_training_checkpoints(self, model, epoch: int):
        """Save model checkpoints at regular intervals"""
        if epoch % self.config.SAVE_MODEL_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                self.config.MODELS_DIR,
                f"{self.experiment_id}_epoch_{epoch}.h5"
            )
            model.save(checkpoint_path)
            self.logger.info(f"Model checkpoint saved: {checkpoint_path}")
    
    def load_best_model(self) -> tf.keras.Model:
        """Load the best performing model from training"""
        best_model_path = os.path.join(
            self.config.MODELS_DIR,
            f"{self.experiment_id}_best_model.h5"
        )
        
        if os.path.exists(best_model_path):
            model = tf.keras.models.load_model(best_model_path)
            self.logger.info(f"Best model loaded from: {best_model_path}")
            return model
        else:
            raise FileNotFoundError(f"Best model not found at: {best_model_path}")


class CustomLoggingCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed epoch-by-epoch logging"""
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.trainer.log_training_progress(epoch, logs)


class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Callback for monitoring memory usage during training"""
    
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Log every 5 epochs
            self.trainer.log_memory_usage()