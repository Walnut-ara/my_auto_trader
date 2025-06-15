"""
Enhanced multi-strategy trading engine with market regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import talib
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class StrategyEngine:
    """Advanced strategy engine with multiple strategies"""
    
    def __init__(self):
        self.strategies = {
            'rsi_mean_reversion': self._rsi_mean_reversion_signal,
            'trend_following': self._trend_following_signal,
            'breakout': self._breakout_signal,
            'volume_momentum': self._volume_momentum_signal,
            'pattern_recognition': self._pattern_recognition_signal,
            'market_making': self._market_making_signal
        }
        
    def generate_signals(
        self,
        data: pd.DataFrame,
        strategy_weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """Generate combined signals from multiple strategies"""
        
        if strategy_weights is None:
            # Default equal weights
            strategy_weights = {s: 1.0/len(self.strategies) for s in self.strategies}
        
        # Detect market regime
        regime = self._detect_market_regime(data)
        
        # Adjust weights based on regime
        adjusted_weights = self._adjust_weights_by_regime(strategy_weights, regime)
        
        # Generate signals from each strategy
        all_signals = pd.DataFrame(index=data.index)
        
        for strategy_name, strategy_func in self.strategies.items():
            if adjusted_weights.get(strategy_name, 0) > 0:
                signals = strategy_func(data)
                all_signals[strategy_name] = signals
        
        # Combine signals
        data['signal'] = self._combine_signals(all_signals, adjusted_weights)
        
        # Add position sizing
        data['position_size'] = self._calculate_position_size(data, regime)
        
        # Add stop loss and take profit
        data['stop_loss'], data['take_profit'] = self._calculate_exits(data, regime)
        
        return data
    
    def _detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        
        # Calculate indicators
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        # ADX for trend strength
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        
        current_price = data['close'].iloc[-1]
        current_vol = volatility.iloc[-1]
        current_adx = adx.iloc[-1]
        
        # Determine regime
        if current_adx > 25:
            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                return MarketRegime.TRENDING_UP
            elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
                return MarketRegime.TRENDING_DOWN
        
        if current_vol > volatility.quantile(0.8).iloc[-1]:
            return MarketRegime.VOLATILE
        elif current_vol < volatility.quantile(0.2).iloc[-1]:
            return MarketRegime.QUIET
        else:
            return MarketRegime.RANGING
    
    def _adjust_weights_by_regime(
        self,
        base_weights: Dict[str, float],
        regime: MarketRegime
    ) -> Dict[str, float]:
        """Adjust strategy weights based on market regime"""
        
        regime_adjustments = {
            MarketRegime.TRENDING_UP: {
                'trend_following': 2.0,
                'breakout': 1.5,
                'volume_momentum': 1.2,
                'rsi_mean_reversion': 0.5,
                'pattern_recognition': 1.0,
                'market_making': 0.3
            },
            MarketRegime.TRENDING_DOWN: {
                'trend_following': 2.0,
                'breakout': 1.5,
                'rsi_mean_reversion': 0.5,
                'pattern_recognition': 1.0,
                'volume_momentum': 0.8,
                'market_making': 0.3
            },
            MarketRegime.RANGING: {
                'rsi_mean_reversion': 2.0,
                'market_making': 1.8,
                'pattern_recognition': 1.2,
                'trend_following': 0.5,
                'breakout': 0.8,
                'volume_momentum': 1.0
            },
            MarketRegime.VOLATILE: {
                'breakout': 1.8,
                'volume_momentum': 1.5,
                'trend_following': 1.2,
                'pattern_recognition': 1.0,
                'rsi_mean_reversion': 0.8,
                'market_making': 0.5
            },
            MarketRegime.QUIET: {
                'market_making': 2.0,
                'rsi_mean_reversion': 1.5,
                'pattern_recognition': 1.2,
                'trend_following': 0.5,
                'breakout': 0.5,
                'volume_momentum': 0.8
            }
        }
        
        adjustments = regime_adjustments[regime]
        adjusted_weights = {}
        
        for strategy, base_weight in base_weights.items():
            adjusted_weights[strategy] = base_weight * adjustments.get(strategy, 1.0)
        
        # Normalize weights
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _rsi_mean_reversion_signal(self, data: pd.DataFrame) -> pd.Series:
        """RSI-based mean reversion strategy"""
        
        rsi = talib.RSI(data['close'], timeperiod=14)
        
        # Dynamic thresholds based on recent RSI distribution
        rsi_mean = rsi.rolling(50).mean()
        rsi_std = rsi.rolling(50).std()
        
        oversold = rsi_mean - 1.5 * rsi_std
        overbought = rsi_mean + 1.5 * rsi_std
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[rsi < oversold] = 1  # Buy
        signals[rsi > overbought] = -1  # Sell
        
        return signals
    
    def _trend_following_signal(self, data: pd.DataFrame) -> pd.Series:
        """Trend following with multiple timeframes"""
        
        # Multiple EMAs
        ema_fast = talib.EMA(data['close'], timeperiod=12)
        ema_slow = talib.EMA(data['close'], timeperiod=26)
        ema_trend = talib.EMA(data['close'], timeperiod=50)
        
        # MACD
        macd, signal, hist = talib.MACD(data['close'])
        
        # Supertrend
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=10)
        hl_avg = (data['high'] + data['low']) / 2
        supertrend_up = hl_avg - 2 * atr
        supertrend_down = hl_avg + 2 * atr
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Bullish conditions
        bullish = (
            (ema_fast > ema_slow) & 
            (ema_slow > ema_trend) & 
            (macd > signal) &
            (data['close'] > supertrend_up)
        )
        
        # Bearish conditions
        bearish = (
            (ema_fast < ema_slow) & 
            (ema_slow < ema_trend) & 
            (macd < signal) &
            (data['close'] < supertrend_down)
        )
        
        signals[bullish] = 1
        signals[bearish] = -1
        
        return signals
    
    def _breakout_signal(self, data: pd.DataFrame) -> pd.Series:
        """Breakout strategy with volume confirmation"""
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            data['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Keltner Channels
        ema = talib.EMA(data['close'], timeperiod=20)
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=20)
        kc_upper = ema + 2 * atr
        kc_lower = ema - 2 * atr
        
        # Volume analysis
        volume_sma = data['volume'].rolling(20).mean()
        volume_spike = data['volume'] > 1.5 * volume_sma
        
        # Donchian Channels
        dc_upper = data['high'].rolling(20).max()
        dc_lower = data['low'].rolling(20).min()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Breakout conditions
        upper_breakout = (
            (data['close'] > bb_upper) & 
            (data['close'] > kc_upper) & 
            volume_spike
        )
        
        lower_breakout = (
            (data['close'] < bb_lower) & 
            (data['close'] < kc_lower) & 
            volume_spike
        )
        
        signals[upper_breakout] = 1
        signals[lower_breakout] = -1
        
        return signals
    
    def _volume_momentum_signal(self, data: pd.DataFrame) -> pd.Series:
        """Volume-based momentum strategy"""
        
        # On Balance Volume
        obv = talib.OBV(data['close'], data['volume'])
        obv_sma = talib.SMA(obv, timeperiod=20)
        
        # Money Flow Index
        mfi = talib.MFI(data['high'], data['low'], data['close'], 
                       data['volume'], timeperiod=14)
        
        # Volume Weighted Average Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        # Accumulation/Distribution
        ad = talib.AD(data['high'], data['low'], data['close'], data['volume'])
        ad_sma = talib.SMA(ad, timeperiod=20)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Bullish volume momentum
        bullish = (
            (obv > obv_sma) & 
            (mfi < 80) & 
            (data['close'] > vwap) &
            (ad > ad_sma)
        )
        
        # Bearish volume momentum
        bearish = (
            (obv < obv_sma) & 
            (mfi > 20) & 
            (data['close'] < vwap) &
            (ad < ad_sma)
        )
        
        signals[bullish] = 1
        signals[bearish] = -1
        
        return signals
    
    def _pattern_recognition_signal(self, data: pd.DataFrame) -> pd.Series:
        """Candlestick pattern recognition"""
        
        # Detect various candlestick patterns
        patterns = {
            'hammer': talib.CDLHAMMER(data['open'], data['high'], 
                                     data['low'], data['close']),
            'doji': talib.CDLDOJI(data['open'], data['high'], 
                                 data['low'], data['close']),
            'engulfing': talib.CDLENGULFING(data['open'], data['high'], 
                                           data['low'], data['close']),
            'morning_star': talib.CDLMORNINGSTAR(data['open'], data['high'], 
                                                data['low'], data['close']),
            'evening_star': talib.CDLEVENINGSTAR(data['open'], data['high'], 
                                                data['low'], data['close']),
            'three_white': talib.CDL3WHITESOLDIERS(data['open'], data['high'], 
                                                   data['low'], data['close']),
            'three_black': talib.CDL3BLACKCROWS(data['open'], data['high'], 
                                               data['low'], data['close'])
        }
        
        # Combine pattern signals
        signals = pd.Series(0, index=data.index)
        
        # Bullish patterns
        for pattern in ['hammer', 'engulfing', 'morning_star', 'three_white']:
            if pattern in patterns:
                signals += (patterns[pattern] > 0).astype(int)
        
        # Bearish patterns
        for pattern in ['evening_star', 'three_black']:
            if pattern in patterns:
                signals -= (patterns[pattern] > 0).astype(int)
        
        # Normalize to -1, 0, 1
        signals = np.sign(signals)
        
        return signals
    
    def _market_making_signal(self, data: pd.DataFrame) -> pd.Series:
        """Market making strategy for ranging markets"""
        
        # Calculate price levels
        close = data['close']
        sma = talib.SMA(close, timeperiod=20)
        std = close.rolling(20).std()
        
        # Mean reversion bands
        upper_band = sma + std
        lower_band = sma - std
        
        # Microstructure analysis
        spread = data['high'] - data['low']
        avg_spread = spread.rolling(20).mean()
        
        # Order flow imbalance (simplified)
        buy_pressure = (close - data['low']) / (data['high'] - data['low'])
        sell_pressure = (data['high'] - close) / (data['high'] - data['low'])
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Buy at lower band with positive order flow
        buy_condition = (
            (close < lower_band) & 
            (buy_pressure > 0.6) &
            (spread < 1.5 * avg_spread)
        )
        
        # Sell at upper band with negative order flow
        sell_condition = (
            (close > upper_band) & 
            (sell_pressure > 0.6) &
            (spread < 1.5 * avg_spread)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _combine_signals(
        self,
        all_signals: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """Combine signals from multiple strategies"""
        
        combined = pd.Series(0, index=all_signals.index)
        
        for strategy, weight in weights.items():
            if strategy in all_signals.columns:
                combined += all_signals[strategy] * weight
        
        # Convert to discrete signals
        threshold = 0.3
        final_signals = pd.Series(0, index=combined.index)
        final_signals[combined > threshold] = 1
        final_signals[combined < -threshold] = -1
        
        return final_signals
    
    def _calculate_position_size(
        self,
        data: pd.DataFrame,
        regime: MarketRegime
    ) -> pd.Series:
        """Dynamic position sizing based on market conditions"""
        
        # Base position size
        base_size = 0.1  # 10% of capital
        
        # Volatility adjustment
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        vol_scalar = 0.02 / volatility  # Target 2% volatility
        vol_scalar = vol_scalar.clip(0.5, 2.0)  # Limit adjustment
        
        # Regime adjustment
        regime_multipliers = {
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.RANGING: 1.0,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.QUIET: 1.3
        }
        
        regime_mult = regime_multipliers[regime]
        
        # Calculate final position size
        position_size = base_size * vol_scalar * regime_mult
        
        # Apply Kelly Criterion (simplified)
        if 'win_rate' in data.columns and 'avg_win_loss_ratio' in data.columns:
            win_rate = data['win_rate']
            win_loss_ratio = data['avg_win_loss_ratio']
            kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly = kelly.clip(0, 0.25)  # Max 25% Kelly
            position_size *= kelly
        
        return position_size.clip(0.02, 0.20)  # 2% to 20% limits
    
    def _calculate_exits(
        self,
        data: pd.DataFrame,
        regime: MarketRegime
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate dynamic stop loss and take profit levels"""
        
        # ATR-based stops
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Regime-based multipliers
        stop_multipliers = {
            MarketRegime.TRENDING_UP: 2.0,
            MarketRegime.TRENDING_DOWN: 1.5,
            MarketRegime.RANGING: 1.0,
            MarketRegime.VOLATILE: 2.5,
            MarketRegime.QUIET: 0.8
        }
        
        profit_multipliers = {
            MarketRegime.TRENDING_UP: 3.0,
            MarketRegime.TRENDING_DOWN: 2.0,
            MarketRegime.RANGING: 1.5,
            MarketRegime.VOLATILE: 3.5,
            MarketRegime.QUIET: 1.2
        }
        
        stop_mult = stop_multipliers[regime]
        profit_mult = profit_multipliers[regime]
        
        # Calculate levels
        stop_loss = atr * stop_mult / data['close']
        take_profit = atr * profit_mult / data['close']
        
        # Apply limits
        stop_loss = stop_loss.clip(0.005, 0.05)  # 0.5% to 5%
        take_profit = take_profit.clip(0.01, 0.10)  # 1% to 10%
        
        return stop_loss, take_profit
    
    def get_strategy_performance(
        self,
        data: pd.DataFrame,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """Evaluate individual strategy performance"""
        
        performance = {}
        
        for strategy_name, strategy_func in self.strategies.items():
            # Generate signals
            signals = strategy_func(data[-lookback_days:])
            
            # Simple performance metric
            returns = data['close'].pct_change()
            strategy_returns = returns * signals.shift(1)
            
            performance[strategy_name] = {
                'total_return': (1 + strategy_returns).prod() - 1,
                'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252),
                'trades': (signals.diff() != 0).sum()
            }
        
        return performance