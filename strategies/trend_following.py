"""
Trend following strategy using multiple indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple
import talib
from typing import Dict

from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):
    """Multi-timeframe trend following strategy"""
    
    def get_default_params(self) -> Dict:
        """Default parameters for trend following"""
        return {
            'ema_fast': 12,
            'ema_slow': 26,
            'ema_trend': 50,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'adx_period': 14,
            'adx_threshold': 25,
            'volume_ma': 20,
            'position_size_pct': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.04
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend following signals"""
        
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # Calculate indicators
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.params['ema_fast'])
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.params['ema_slow'])
        df['ema_trend'] = talib.EMA(df['close'], timeperiod=self.params['ema_trend'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.params['ema_fast'],
            slowperiod=self.params['ema_slow'],
            signalperiod=9
        )
        
        # ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 
                             timeperiod=self.params['adx_period'])
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 
                             timeperiod=self.params['atr_period'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.params['volume_ma']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Supertrend calculation
        hl_avg = (df['high'] + df['low']) / 2
        df['supertrend_up'] = hl_avg - self.params['atr_multiplier'] * df['atr']
        df['supertrend_down'] = hl_avg + self.params['atr_multiplier'] * df['atr']
        
        # Generate signals
        df['signal'] = 0
        
        # Strong uptrend conditions
        long_conditions = (
            (df['ema_fast'] > df['ema_slow']) &
            (df['ema_slow'] > df['ema_trend']) &
            (df['macd'] > df['macd_signal']) &
            (df['adx'] > self.params['adx_threshold']) &
            (df['close'] > df['supertrend_up']) &
            (df['volume_ratio'] > 1.0)
        )
        
        # Strong downtrend conditions (for short or exit)
        short_conditions = (
            (df['ema_fast'] < df['ema_slow']) &
            (df['ema_slow'] < df['ema_trend']) &
            (df['macd'] < df['macd_signal']) &
            (df['adx'] > self.params['adx_threshold']) &
            (df['close'] < df['supertrend_down']) &
            (df['volume_ratio'] > 1.0)
        )
        
        df.loc[long_conditions, 'signal'] = 1
        df.loc[short_conditions, 'signal'] = -1
        
        # Filter signals to reduce whipsaws
        df['signal'] = self._filter_signals(df['signal'])
        
        # Add exit conditions
        df['exit_signal'] = self._generate_exit_signals(df)
        
        return df
    
    def _filter_signals(self, signals: pd.Series) -> pd.Series:
        """Filter signals to reduce false signals"""
        
        filtered = signals.copy()
        
        # Remove signals that reverse too quickly
        for i in range(2, len(filtered)):
            if filtered.iloc[i] == -filtered.iloc[i-1]:
                # Check if reversal is too quick (within 2 bars)
                filtered.iloc[i] = 0
        
        return filtered
    
    def _generate_exit_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate exit signals based on trend weakness"""
        
        exit_signals = pd.Series(0, index=df.index)
        
        # Exit long when trend weakens
        exit_long = (
            (df['signal'].shift(1) == 1) &  # Was in long position
            (
                (df['ema_fast'] < df['ema_slow']) |  # Fast crosses below slow
                (df['macd'] < df['macd_signal']) |   # MACD turns bearish
                (df['adx'] < 20)  # Trend strength weakens
            )
        )
        
        # Exit short when trend weakens
        exit_short = (
            (df['signal'].shift(1) == -1) &  # Was in short position
            (
                (df['ema_fast'] > df['ema_slow']) |  # Fast crosses above slow
                (df['macd'] > df['macd_signal']) |   # MACD turns bullish
                (df['adx'] < 20)  # Trend strength weakens
            )
        )
        
        exit_signals[exit_long | exit_short] = 1
        
        return exit_signals
    
    def calculate_dynamic_stops(self, df: pd.DataFrame, position: int) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit based on ATR"""
        
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Dynamic multipliers based on trend strength
        adx = df['adx'].iloc[-1]
        
        if adx > 40:  # Very strong trend
            stop_multiplier = 2.5
            profit_multiplier = 4.0
        elif adx > 25:  # Strong trend
            stop_multiplier = 2.0
            profit_multiplier = 3.0
        else:  # Weak trend
            stop_multiplier = 1.5
            profit_multiplier = 2.0
        
        if position > 0:  # Long position
            stop_loss = current_price - (atr * stop_multiplier)
            take_profit = current_price + (atr * profit_multiplier)
        else:  # Short position
            stop_loss = current_price + (atr * stop_multiplier)
            take_profit = current_price - (atr * profit_multiplier)
        
        return stop_loss, take_profit