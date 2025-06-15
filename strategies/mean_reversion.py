"""
Mean reversion strategy for ranging markets
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict
from typing import Tuple

from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using RSI, Bollinger Bands, and Z-score"""
    
    def get_default_params(self) -> Dict:
        """Default parameters for mean reversion"""
        return {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'bb_period': 20,
            'bb_std': 2.0,
            'zscore_period': 20,
            'zscore_threshold': 2.0,
            'volume_confirm': True,
            'atr_period': 14,
            'position_size_pct': 0.08,
            'stop_loss': 0.015,
            'take_profit': 0.02,
            'dynamic_thresholds': True
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals"""
        
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # Calculate indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.params['rsi_period'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'],
            timeperiod=self.params['bb_period'],
            nbdevup=self.params['bb_std'],
            nbdevdn=self.params['bb_std']
        )
        
        # Z-score calculation
        df['zscore'] = self._calculate_zscore(df['close'], self.params['zscore_period'])
        
        # ATR for volatility filtering
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 
                             timeperiod=self.params['atr_period'])
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Stochastic RSI for additional confirmation
        df['stoch_rsi'], df['stoch_rsi_d'] = talib.STOCHRSI(
            df['close'], timeperiod=14, fastk_period=5, fastd_period=3
        )
        
        # Dynamic thresholds based on market conditions
        if self.params['dynamic_thresholds']:
            oversold, overbought = self._calculate_dynamic_thresholds(df)
        else:
            oversold = self.params['rsi_oversold']
            overbought = self.params['rsi_overbought']
        
        # Generate signals
        df['signal'] = 0
        
        # Buy conditions (oversold)
        buy_conditions = (
            (df['rsi'] < oversold) &
            (df['close'] < df['bb_lower']) &
            (df['zscore'] < -self.params['zscore_threshold']) &
            (df['stoch_rsi'] < 20) &
            (df['atr_pct'] < 0.05)  # Not too volatile
        )
        
        # Add volume confirmation if enabled
        if self.params['volume_confirm']:
            buy_conditions &= (df['volume_ratio'] > 1.2)
        
        # Sell conditions (overbought)
        sell_conditions = (
            (df['rsi'] > overbought) &
            (df['close'] > df['bb_upper']) &
            (df['zscore'] > self.params['zscore_threshold']) &
            (df['stoch_rsi'] > 80) &
            (df['atr_pct'] < 0.05)  # Not too volatile
        )
        
        if self.params['volume_confirm']:
            sell_conditions &= (df['volume_ratio'] > 1.2)
        
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Apply signal filtering
        df['signal'] = self._filter_signals(df)
        
        # Add mean reversion exit signals
        df['exit_signal'] = self._generate_exit_signals(df)
        
        return df
    
    def _calculate_zscore(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Z-score"""
        
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        zscore = (prices - rolling_mean) / rolling_std
        
        return zscore
    
    def _calculate_dynamic_thresholds(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate dynamic RSI thresholds based on recent distribution"""
        
        # Use recent RSI values to determine thresholds
        recent_rsi = df['rsi'].tail(100)
        
        # Calculate percentiles
        oversold = recent_rsi.quantile(0.2)  # 20th percentile
        overbought = recent_rsi.quantile(0.8)  # 80th percentile
        
        # Apply bounds
        oversold = max(20, min(35, oversold))
        overbought = min(80, max(65, overbought))
        
        return oversold, overbought
    
    def _filter_signals(self, df: pd.DataFrame) -> pd.Series:
        """Filter signals to avoid false reversals"""
        
        signals = df['signal'].copy()
        
        # Check market regime - avoid mean reversion in strong trends
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        # Calculate trend strength
        trend_strength = abs(sma_20 - sma_50) / sma_50
        
        # Remove signals in strong trends
        strong_trend = trend_strength > 0.03  # 3% difference
        signals[strong_trend] = 0
        
        # Remove signals that occur too frequently
        signal_changes = signals.diff().abs()
        recent_changes = signal_changes.rolling(10).sum()
        
        # If too many signal changes recently, skip new signals
        signals[recent_changes > 4] = 0
        
        return signals
    
    def _generate_exit_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate exit signals when price returns to mean"""
        
        exit_signals = pd.Series(0, index=df.index)
        
        # Exit long when price returns above middle band
        exit_long = (
            (df['signal'].shift(1) == 1) &  # Was in long position
            (
                (df['close'] > df['bb_middle']) |  # Price above middle band
                (df['rsi'] > 50) |  # RSI normalized
                (df['zscore'] > 0)  # Z-score positive
            )
        )
        
        # Exit short when price returns below middle band
        exit_short = (
            (df['signal'].shift(1) == -1) &  # Was in short position
            (
                (df['close'] < df['bb_middle']) |  # Price below middle band
                (df['rsi'] < 50) |  # RSI normalized
                (df['zscore'] < 0)  # Z-score negative
            )
        )
        
        exit_signals[exit_long | exit_short] = 1
        
        return exit_signals
    
    def calculate_adaptive_position_size(self, df: pd.DataFrame) -> float:
        """Calculate position size based on market conditions"""
        
        base_size = self.params['position_size_pct']
        
        # Current market conditions
        current_rsi = df['rsi'].iloc[-1]
        current_zscore = abs(df['zscore'].iloc[-1])
        current_volatility = df['atr_pct'].iloc[-1]
        
        # Increase size for extreme oversold/overbought
        if current_rsi < 20 or current_rsi > 80:
            extremity_multiplier = 1.5
        elif current_rsi < 25 or current_rsi > 75:
            extremity_multiplier = 1.25
        else:
            extremity_multiplier = 1.0
        
        # Adjust for Z-score
        zscore_multiplier = min(1.5, max(0.5, current_zscore / 2))
        
        # Reduce size in high volatility
        volatility_multiplier = max(0.5, min(1.0, 0.02 / current_volatility))
        
        # Final position size
        position_size = base_size * extremity_multiplier * zscore_multiplier * volatility_multiplier
        
        # Apply limits
        return np.clip(position_size, 0.02, 0.15)