"""
Deep learning model for price prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out


class PricePredictor:
    """Price prediction system using LSTM"""
    
    def __init__(self, lookback_period: int = 60, prediction_horizon: int = 5):
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        
        # Create features
        features = self._create_features(df)
        
        # Scale features
        scaled_features = self.feature_scaler.fit_transform(features)
        
        # Scale target (price)
        prices = df['close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.lookback_period, len(scaled_features) - self.prediction_horizon):
            X.append(scaled_features[i - self.lookback_period:i])
            y.append(scaled_prices[i + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for price prediction"""
        
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['close'] = df['close']
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        features['volume'] = df['volume']
        features['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'])
        features['ma_7'] = df['close'].rolling(7).mean()
        features['ma_21'] = df['close'].rolling(21).mean()
        features['ma_ratio'] = features['ma_7'] / features['ma_21']
        
        # Price changes
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Remove NaN values
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        if len(X) == 0:
            logger.error("Insufficient data for training")
            return
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMPredictor(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                          f"Val Loss = {val_loss:.4f}")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """Make price predictions"""
        
        if self.model is None:
            logger.error("Model not trained")
            return {}
        
        # Prepare recent data
        features = self._create_features(df)
        scaled_features = self.feature_scaler.transform(features)
        
        # Get last sequence
        if len(scaled_features) < self.lookback_period:
            logger.error("Insufficient data for prediction")
            return {}
        
        last_sequence = scaled_features[-self.lookback_period:]
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X).cpu().numpy()
        
        # Inverse transform
        predicted_price = self.scaler.inverse_transform(prediction)[0, 0]
        current_price = df['close'].iloc[-1]
        
        # Calculate metrics
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # Confidence based on recent accuracy (simplified)
        confidence = self._calculate_confidence(df)
        
        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'prediction_horizon': self.prediction_horizon,
            'confidence': confidence,
            'signal': 1 if price_change_pct > 1 else -1 if price_change_pct < -1 else 0
        }
    
    def _calculate_confidence(self, df: pd.DataFrame) -> float:
        """Calculate prediction confidence"""
        
        # Simplified confidence based on recent volatility
        recent_volatility = df['close'].pct_change().tail(20).std()
        
        # Lower confidence in high volatility
        if recent_volatility < 0.01:
            confidence = 0.8
        elif recent_volatility < 0.02:
            confidence = 0.6
        elif recent_volatility < 0.03:
            confidence = 0.4
        else:
            confidence = 0.2
        
        return confidence
    
    def _save_model(self):
        """Save model state"""
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'lookback_period': self.lookback_period,
                'prediction_horizon': self.prediction_horizon,
            }, 'models/price_predictor.pth')
    
    def load_model(self, path: str):
        """Load saved model"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Initialize model with saved parameters
        # Note: Need to know input size, which should be saved too
        self.model = LSTMPredictor(input_size=12).to(self.device)  # Adjust input size
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.lookback_period = checkpoint['lookback_period']
        self.prediction_horizon = checkpoint['prediction_horizon']