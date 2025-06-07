"""
CNN-LSTM Model Architecture Module
Implements the multi-currency CNN-LSTM architecture for forex prediction
Combines convolutional feature extraction with LSTM temporal modeling
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Activation, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class CNNLSTMArchitecture:
    """
    Advanced CNN-LSTM architecture for multi-currency forex prediction
    Implements sophisticated feature extraction and temporal modeling
    """
    
    def __init__(self, config):
        """
        Initialize model architecture with configuration settings
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model architecture parameters
        self.input_shape = (config.WINDOW_SIZE, config.TOTAL_FEATURES)
        self.model = None
        self.model_history = None
        
        # Architecture specifications
        self.cnn_specs = {
            'filters_1': config.CNN_FILTERS_1,
            'filters_2': config.CNN_FILTERS_2,
            'kernel_size': config.CNN_KERNEL_SIZE,
            'activation': config.CNN_ACTIVATION,
            'padding': config.CNN_PADDING
        }
        
        self.lstm_specs = {
            'units_1': config.LSTM_UNITS_1,
            'units_2': config.LSTM_UNITS_2,
            'dropout': config.LSTM_DROPOUT,
            'recurrent_dropout': config.LSTM_RECURRENT_DROPOUT
        }
        
        self.dense_specs = {
            'units': config.DENSE_UNITS,
            'dropout': config.DENSE_DROPOUT,
            'activation': config.DENSE_ACTIVATION
        }
        
        # Training specifications
        self.training_specs = {
            'optimizer': config.OPTIMIZER,
            'learning_rate': config.LEARNING_RATE,
            'loss': config.LOSS_FUNCTION,
            'metrics': config.METRICS
        }
        
    def build_cnn_layers(self, input_layer) -> tf.keras.layers.Layer:
        """
        Build CNN layers for spatial-temporal feature extraction
        
        Args:
            input_layer: Input layer from the model
            
        Returns:
            Output layer from CNN processing
        """
        self.logger.info("Building CNN layers for feature extraction")
        
        # First Convolutional Layer
        # Purpose: Extract basic local patterns from multi-currency data
        conv1 = Conv1D(
            filters=self.cnn_specs['filters_1'],
            kernel_size=self.cnn_specs['kernel_size'],
            padding=self.cnn_specs['padding'],
            name='conv1d_1'
        )(input_layer)
        
        # Batch normalization for training stability
        conv1_bn = BatchNormalization(name='bn_conv1')(conv1)
        
        # ReLU activation
        conv1_relu = Activation(self.cnn_specs['activation'], name='relu_conv1')(conv1_bn)
        
        self.logger.info(f"CNN Layer 1: {self.cnn_specs['filters_1']} filters, "
                        f"kernel size {self.cnn_specs['kernel_size']}")
        
        # Second Convolutional Layer
        # Purpose: Extract complex patterns and cross-currency relationships
        conv2 = Conv1D(
            filters=self.cnn_specs['filters_2'],
            kernel_size=self.cnn_specs['kernel_size'],
            padding=self.cnn_specs['padding'],
            name='conv1d_2'
        )(conv1_relu)
        
        # Batch normalization
        conv2_bn = BatchNormalization(name='bn_conv2')(conv2)
        
        # ReLU activation
        conv2_relu = Activation(self.cnn_specs['activation'], name='relu_conv2')(conv2_bn)
        
        self.logger.info(f"CNN Layer 2: {self.cnn_specs['filters_2']} filters, "
                        f"kernel size {self.cnn_specs['kernel_size']}")
        
        # MaxPooling for dimensionality reduction
        # Purpose: Reduce temporal dimension and focus on most salient features
        pooled = MaxPooling1D(
            pool_size=2,
            strides=2,
            name='maxpool_1'
        )(conv2_relu)
        
        self.logger.info("Applied MaxPooling: pool_size=2, stride=2")
        
        return pooled
    
    def build_lstm_layers(self, cnn_output) -> tf.keras.layers.Layer:
        """
        Build LSTM layers for temporal sequence modeling
        
        Args:
            cnn_output: Output from CNN layers
            
        Returns:
            Output layer from LSTM processing
        """
        self.logger.info("Building LSTM layers for temporal modeling")
        
        # First LSTM Layer
        # Purpose: Learn long-term temporal dependencies in forex patterns
        lstm1 = LSTM(
            units=self.lstm_specs['units_1'],
            return_sequences=True,  # Pass sequences to next LSTM layer
            dropout=self.lstm_specs['dropout'],
            recurrent_dropout=self.lstm_specs['recurrent_dropout'],
            name='lstm_1'
        )(cnn_output)
        
        # Batch normalization for LSTM output
        lstm1_bn = BatchNormalization(name='bn_lstm1')(lstm1)
        
        self.logger.info(f"LSTM Layer 1: {self.lstm_specs['units_1']} units, "
                        f"return_sequences=True")
        
        # Second LSTM Layer
        # Purpose: Further temporal processing and dimensionality reduction
        lstm2 = LSTM(
            units=self.lstm_specs['units_2'],
            return_sequences=False,  # Final sequence output only
            dropout=self.lstm_specs['dropout'],
            recurrent_dropout=self.lstm_specs['recurrent_dropout'],
            name='lstm_2'
        )(lstm1_bn)
        
        # Batch normalization
        lstm2_bn = BatchNormalization(name='bn_lstm2')(lstm2)
        
        self.logger.info(f"LSTM Layer 2: {self.lstm_specs['units_2']} units, "
                        f"return_sequences=False")
        
        return lstm2_bn
    
    def build_dense_layers(self, lstm_output) -> tf.keras.layers.Layer:
        """
        Build dense layers for final prediction processing
        
        Args:
            lstm_output: Output from LSTM layers
            
        Returns:
            Output layer ready for final prediction
        """
        self.logger.info("Building dense layers for prediction processing")
        
        # Dense layer for feature refinement
        dense = Dense(
            units=self.dense_specs['units'],
            activation=self.dense_specs['activation'],
            name='dense_1'
        )(lstm_output)
        
        # Dropout for regularization
        dense_dropout = Dropout(
            rate=self.dense_specs['dropout'],
            name='dropout_dense'
        )(dense)
        
        # Batch normalization
        dense_bn = BatchNormalization(name='bn_dense')(dense_dropout)
        
        self.logger.info(f"Dense Layer: {self.dense_specs['units']} units, "
                        f"dropout={self.dense_specs['dropout']}")
        
        # Output layer for binary classification
        output = Dense(
            units=1,
            activation=self.config.OUTPUT_ACTIVATION,
            name='output_prediction'
        )(dense_bn)
        
        self.logger.info(f"Output Layer: 1 unit, activation={self.config.OUTPUT_ACTIVATION}")
        
        return output
    
    def compile_multi_currency_model(self) -> Model:
        """
        Compile the complete multi-currency CNN-LSTM model
        
        Returns:
            Compiled Keras model ready for training
        """
        self.logger.info("Compiling complete multi-currency CNN-LSTM model")
        self.logger.info(f"Input shape: {self.input_shape}")
        
        # Define input layer
        input_layer = Input(
            shape=self.input_shape,
            name='multi_currency_input'
        )
        
        # Build CNN feature extraction layers
        cnn_output = self.build_cnn_layers(input_layer)
        
        # Build LSTM temporal modeling layers
        lstm_output = self.build_lstm_layers(cnn_output)
        
        # Build dense prediction layers
        final_output = self.build_dense_layers(lstm_output)
        
        # Create the complete model
        model = Model(
            inputs=input_layer,
            outputs=final_output,
            name='MultiCurrency_CNN_LSTM'
        )
        
        # Configure optimizer
        if self.training_specs['optimizer'].lower() == 'adam':
            optimizer = Adam(learning_rate=self.training_specs['learning_rate'])
        else:
            optimizer = self.training_specs['optimizer']
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=self.training_specs['loss'],
            metrics=self.training_specs['metrics']
        )
        
        self.model = model
        
        # Log model summary
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        self.logger.info(f"Model compiled successfully:")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Optimizer: {self.training_specs['optimizer']} "
                        f"(lr={self.training_specs['learning_rate']})")
        self.logger.info(f"Loss function: {self.training_specs['loss']}")
        
        return model
    
    def display_model_architecture(self, save_plot: bool = True, 
                                 output_path: Optional[str] = None) -> str:
        """
        Display and optionally save model architecture visualization
        
        Args:
            save_plot: Whether to save the architecture plot
            output_path: Optional path for saving the plot
            
        Returns:
            Path to saved plot or empty string if not saved
        """
        if self.model is None:
            raise ValueError("Model must be compiled before displaying architecture")
        
        self.logger.info("Generating model architecture summary")
        
        # Print detailed model summary
        self.model.summary()
        
        # Create architecture visualization
        if save_plot:
            try:
                if output_path is None:
                    output_path = self.config.get_results_path('model_architecture.png')
                
                tf.keras.utils.plot_model(
                    self.model,
                    to_file=output_path,
                    show_shapes=True,
                    show_layer_names=True,
                    rankdir='TB',
                    expand_nested=True,
                    dpi=self.config.DPI
                )
                
                self.logger.info(f"Model architecture plot saved to: {output_path}")
                return output_path
                
            except Exception as e:
                self.logger.warning(f"Could not save model plot: {str(e)}")
                return ""
        
        return ""
    
    def get_layer_output_shapes(self) -> Dict[str, Tuple]:
        """
        Get output shapes for all layers in the model
        
        Returns:
            Dictionary mapping layer names to output shapes
        """
        if self.model is None:
            raise ValueError("Model must be compiled first")
        
        layer_shapes = {}
        for layer in self.model.layers:
            try:
                if hasattr(layer, 'output_shape'):
                    layer_shapes[layer.name] = layer.output_shape
                elif hasattr(layer, 'output') and hasattr(layer.output, 'shape'):
                    layer_shapes[layer.name] = layer.output.shape.as_list()
            except:
                layer_shapes[layer.name] = "Unable to determine"
        
        return layer_shapes
    
    def analyze_model_complexity(self) -> Dict:
        """
        Analyze model complexity and computational requirements
        
        Returns:
            Dictionary containing complexity analysis
        """
        if self.model is None:
            raise ValueError("Model must be compiled first")
        
        # Count parameters by layer type
        param_count = {
            'conv1d': 0,
            'lstm': 0,
            'dense': 0,
            'other': 0
        }
        
        layer_info = []
        
        for layer in self.model.layers:
            layer_params = layer.count_params()
            layer_type = layer.__class__.__name__.lower()
            
            layer_info.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'parameters': layer_params,
                'output_shape': getattr(layer, 'output_shape', None)
            })
            
            if 'conv' in layer_type:
                param_count['conv1d'] += layer_params
            elif 'lstm' in layer_type:
                param_count['lstm'] += layer_params
            elif 'dense' in layer_type:
                param_count['dense'] += layer_params
            else:
                param_count['other'] += layer_params
        
        total_params = sum(param_count.values())
        
        complexity_analysis = {
            'total_parameters': total_params,
            'parameter_distribution': param_count,
            'parameter_percentages': {
                k: (v / total_params * 100) if total_params > 0 else 0 
                for k, v in param_count.items()
            },
            'layer_details': layer_info,
            'memory_estimate_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'dominant_component': max(param_count, key=param_count.get)
        }
        
        self.logger.info(f"Model complexity analysis:")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"CNN parameters: {param_count['conv1d']:,} ({complexity_analysis['parameter_percentages']['conv1d']:.1f}%)")
        self.logger.info(f"LSTM parameters: {param_count['lstm']:,} ({complexity_analysis['parameter_percentages']['lstm']:.1f}%)")
        self.logger.info(f"Dense parameters: {param_count['dense']:,} ({complexity_analysis['parameter_percentages']['dense']:.1f}%)")
        self.logger.info(f"Estimated memory: {complexity_analysis['memory_estimate_mb']:.2f} MB")
        
        return complexity_analysis
    
    def create_model_variants(self) -> Dict[str, Model]:
        """
        Create model variants for comparison and ablation studies
        
        Returns:
            Dictionary containing different model variants
        """
        self.logger.info("Creating model variants for comparison")
        
        variants = {}
        
        # 1. CNN-only model (without LSTM)
        input_layer = Input(shape=self.input_shape, name='input_cnn_only')
        cnn_out = self.build_cnn_layers(input_layer)
        
        # Global pooling instead of LSTM
        global_pooled = GlobalMaxPooling1D(name='global_max_pool')(cnn_out)
        dense_out = self.build_dense_layers(global_pooled)
        
        cnn_only_model = Model(inputs=input_layer, outputs=dense_out, name='CNN_Only')
        cnn_only_model.compile(
            optimizer=Adam(learning_rate=self.training_specs['learning_rate']),
            loss=self.training_specs['loss'],
            metrics=self.training_specs['metrics']
        )
        variants['cnn_only'] = cnn_only_model
        
        # 2. LSTM-only model (without CNN)
        input_layer = Input(shape=self.input_shape, name='input_lstm_only')
        
        # Direct LSTM processing without CNN
        lstm1 = LSTM(
            units=self.lstm_specs['units_1'],
            return_sequences=True,
            dropout=self.lstm_specs['dropout'],
            recurrent_dropout=self.lstm_specs['recurrent_dropout'],
            name='lstm_only_1'
        )(input_layer)
        
        lstm2 = LSTM(
            units=self.lstm_specs['units_2'],
            return_sequences=False,
            dropout=self.lstm_specs['dropout'],
            recurrent_dropout=self.lstm_specs['recurrent_dropout'],
            name='lstm_only_2'
        )(lstm1)
        
        dense_out = self.build_dense_layers(lstm2)
        
        lstm_only_model = Model(inputs=input_layer, outputs=dense_out, name='LSTM_Only')
        lstm_only_model.compile(
            optimizer=Adam(learning_rate=self.training_specs['learning_rate']),
            loss=self.training_specs['loss'],
            metrics=self.training_specs['metrics']
        )
        variants['lstm_only'] = lstm_only_model
        
        # 3. Simplified CNN-LSTM (fewer parameters)
        input_layer = Input(shape=self.input_shape, name='input_simple')
        
        # Simplified CNN
        conv_simple = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(input_layer)
        pooled_simple = MaxPooling1D(pool_size=2)(conv_simple)
        
        # Simplified LSTM
        lstm_simple = LSTM(units=32, return_sequences=False, dropout=0.2)(pooled_simple)
        
        # Simplified dense
        dense_simple = Dense(units=16, activation='relu')(lstm_simple)
        output_simple = Dense(units=1, activation='sigmoid')(dense_simple)
        
        simple_model = Model(inputs=input_layer, outputs=output_simple, name='Simple_CNN_LSTM')
        simple_model.compile(
            optimizer=Adam(learning_rate=self.training_specs['learning_rate']),
            loss=self.training_specs['loss'],
            metrics=self.training_specs['metrics']
        )
        variants['simple'] = simple_model
        
        self.logger.info(f"Created {len(variants)} model variants for comparison")
        
        for name, model in variants.items():
            params = model.count_params()
            self.logger.info(f"{name} variant: {params:,} parameters")
        
        return variants
    
    def get_model_summary(self) -> Dict:
        """
        Get comprehensive model summary information
        
        Returns:
            Dictionary containing complete model information
        """
        if self.model is None:
            return {"error": "Model not compiled yet"}
        
        summary = {
            'architecture_specs': {
                'input_shape': self.input_shape,
                'cnn_specs': self.cnn_specs,
                'lstm_specs': self.lstm_specs,
                'dense_specs': self.dense_specs
            },
            'training_specs': self.training_specs,
            'model_stats': {
                'total_parameters': self.model.count_params(),
                'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'total_layers': len(self.model.layers),
                'model_name': self.model.name
            },
            'layer_shapes': self.get_layer_output_shapes()
        }
        
        return summary