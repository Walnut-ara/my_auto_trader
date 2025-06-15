"""
Machine Learning based trading strategy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib
from typing import Dict, Tuple
import joblib
from pathlib import Path

from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """Machine learning strategy using RandomForest"""
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = Path('models/ml_strategy_model.pkl')
        self.scaler_path = Path('models/ml_strategy_scaler.pkl')
        
    def get_default_params(self) -> Dict:
        """Default parameters for ML strategy"""
        return {
            'lookback_period': 20,
            'prediction_horizon': 5,
            'min_confidence': 0.65,
            'n_estimators': 100,
            'max_depth': 10,
            'position_size_pct': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'retrain_frequency': 1000,  # Retrain every N bars
            'feature_set': 'full'  # 'basic' or 'full'
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML-based trading signals"""
        
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # Generate features
        df = self._generate_features(df)
        
        # Load or train model
        if self.model is None:
            if self.model_path.exists():
                self._load_model()
            else:
                logger.info("No trained model found. Training new model...")
                self._train_model(df)
        
        # Check if retraining is needed
        if len(df) % self.params['retrain_frequency'] == 0:
            logger.info("Retraining model...")
            self._train_model(df)
        
        # Make predictions
        df['signal'] = 0
        df['prediction_confidence'] = 0
        
        if self.model is not None and len(self.feature_columns) > 0:
            # Prepare features for prediction
            X = df[self.feature_columns].dropna()
            
            if len(X) > 0:
                # Scale features
                X_scaled = self.scaler.transform(X)
                
                # Get predictions and probabilities
                predictions = self.model.predict(X_scaled)
                probabilities = self.model.predict_proba(X_scaled)
                
                # Map predictions to signals
                df.loc[X.index, 'signal'] = predictions
                
                # Get confidence (max probability)
                df.loc[X.index, 'prediction_confidence'] = probabilities.max(axis=1)
                
                # Filter by minimum confidence
                low_confidence = df['prediction_confidence'] < self.params['min_confidence']
                df.loc[low_confidence, 'signal'] = 0
        
        # Add additional filters
        df = self._apply_risk_filters(df)
        
        return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for ML model"""
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volatility features
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        
        if self.params['feature_set'] == 'full':
            # Additional advanced features
            
            # Pattern recognition
            df['candle_range'] = (df['high'] - df['low']) / df['open']
            df['body_range'] = abs(df['close'] - df['open']) / df['open']
            df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open']
            df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open']
            
            # Microstructure
            df['bid_ask_proxy'] = df['high'] - df['low']
            df['close_location'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Momentum
            df['roc_5'] = talib.ROC(df['close'], timeperiod=5)
            df['roc_10'] = talib.ROC(df['close'], timeperiod=10)
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            
            # Market regime
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            df['trend_strength'] = df['adx'] / 100
        
        # Create lagged features
        for col in ['returns', 'rsi', 'volume_ratio']:
            for lag in range(1, 6):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in 
                              ['open', 'high', 'low', 'close', 'volume', 
                               'signal', 'prediction_confidence']]
        
        return df
    
    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """Create labels for training"""
        
        # Calculate future returns
        horizon = self.params['prediction_horizon']
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        # Create labels: -1 (down), 0 (neutral), 1 (up)
        labels = pd.Series(0, index=df.index)
        
        # Define thresholds
        up_threshold = 0.01  # 1% up
        down_threshold = -0.01  # 1% down
        
        labels[future_returns > up_threshold] = 1
        labels[future_returns < down_threshold] = -1
        
        return labels
    
    def _train_model(self, df: pd.DataFrame):
        """Train the ML model"""
        
        # Create labels
        labels = self._create_labels(df)
        
        # Prepare training data
        X = df[self.feature_columns].copy()
        y = labels
        
        # Remove NaN values
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            logger.warning("Insufficient data for training")
            return
        
        # Split data (use last 20% for validation)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.params['n_estimators'],
            max_depth=self.params['max_depth'],
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        logger.info(f"Model trained. Train accuracy: {train_score:.3f}, Val accuracy: {val_score:.3f}")
        
        # Save model
        self._save_model()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 features: {feature_importance.head()['feature'].tolist()}")
    
    def _apply_risk_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional risk filters to signals"""
        
        # Don't trade during high volatility
        high_volatility = df['atr_pct'] > 0.05
        df.loc[high_volatility, 'signal'] = 0
        
        # Don't trade when volume is too low
        low_volume = df['volume_ratio'] < 0.5
        df.loc[low_volume, 'signal'] = 0
        
        # Reduce signals during unclear market conditions
        unclear_market = (df['adx'] < 20) if 'adx' in df.columns else False
        if isinstance(unclear_market, pd.Series):
            df.loc[unclear_market, 'signal'] = 0
        
        return df
    
    def _save_model(self):
        """Save trained model and scaler"""
        
        if self.model is not None:
            self.model_path.parent.mkdir(exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def _load_model(self):
        """Load saved model and scaler"""
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None