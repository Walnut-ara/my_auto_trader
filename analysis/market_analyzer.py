"""
Market analysis module for symbol evaluation and scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib
from datetime import datetime, timedelta

from core.data_manager import DataManager
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketAnalyzer:
    """Comprehensive market analysis"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    def analyze_symbol(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        """Complete symbol analysis"""
        
        if data is None:
            data = self.data_manager.get_historical_data(
                symbol, '15',
                datetime.now() - timedelta(days=180),
                datetime.now()
            )
        
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            return {}
        
        # Add indicators
        data = self.data_manager.calculate_indicators(data)
        
        analysis = {
            'symbol': symbol,
            'period': {
                'start': str(data.index[0]),
                'end': str(data.index[-1]),
                'days': (data.index[-1] - data.index[0]).days
            },
            'volatility': self._analyze_volatility(data),
            'trend': self._analyze_trend(data),
            'volume': self._analyze_volume(data),
            'patterns': self._analyze_patterns(data),
            'momentum': self._analyze_momentum(data),
            'support_resistance': self._find_support_resistance(data),
            'market_regime': self._detect_market_regime(data)
        }
        
        return analysis
    
    def score_symbol(self, symbol: str) -> Dict:
        """Score symbol for trading suitability"""
        
        analysis = self.analyze_symbol(symbol)
        
        if not analysis:
            return None
        
        score = 0
        max_score = 100
        
        # Volatility score (20 points)
        vol_score = self._score_volatility(analysis['volatility'])
        score += vol_score * 20
        
        # Trend score (25 points)
        trend_score = self._score_trend(analysis['trend'])
        score += trend_score * 25
        
        # Volume score (20 points)
        volume_score = self._score_volume(analysis['volume'])
        score += volume_score * 20
        
        # Pattern score (15 points)
        pattern_score = self._score_patterns(analysis['patterns'])
        score += pattern_score * 15
        
        # Momentum score (20 points)
        momentum_score = self._score_momentum(analysis['momentum'])
        score += momentum_score * 20
        
        # Determine grade
        if score >= 85:
            grade = 'A'
        elif score >= 70:
            grade = 'B'
        elif score >= 55:
            grade = 'C'
        elif score >= 40:
            grade = 'D'
        else:
            grade = 'F'
        
        # Expected return based on analysis
        expected_return = self._estimate_expected_return(analysis, score)
        
        return {
            'symbol': symbol,
            'total_score': round(score, 1),
            'grade': grade,
            'volatility': analysis['volatility']['daily_volatility'],
            'trend_strength': analysis['trend']['strength'],
            'volume_score': volume_score,
            'expected_return': expected_return,
            'analysis': analysis
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze price volatility"""
        
        returns = data['close'].pct_change().dropna()
        
        # Basic volatility metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # ATR-based volatility
        if 'ATR' in data.columns:
            atr = data['ATR'].iloc[-1]
            atr_pct = atr / data['close'].iloc[-1]
        else:
            atr = talib.ATR(data['high'], data['low'], data['close'])
            atr_pct = atr.iloc[-1] / data['close'].iloc[-1]
        
        # Volatility regime
        vol_percentile = (returns.rolling(20).std().rank(pct=True).iloc[-1])
        
        if vol_percentile > 0.8:
            vol_regime = 'High'
        elif vol_percentile > 0.6:
            vol_regime = 'Above Average'
        elif vol_percentile > 0.4:
            vol_regime = 'Average'
        elif vol_percentile > 0.2:
            vol_regime = 'Below Average'
        else:
            vol_regime = 'Low'
        
        # Volatility clustering
        garch_like = returns.rolling(5).std().autocorr()
        
        return {
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'atr_percentage': atr_pct,
            'volatility_regime': vol_regime,
            'volatility_percentile': vol_percentile,
            'volatility_clustering': garch_like,
            'recent_spike': returns.iloc[-5:].std() > returns.std() * 1.5
        }
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze price trend"""
        
        close = data['close']
        
        # Moving averages
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        
        current_price = close.iloc[-1]
        
        # Trend direction
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_direction = 'Strong Uptrend'
        elif current_price > sma_20.iloc[-1]:
            trend_direction = 'Uptrend'
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend_direction = 'Strong Downtrend'
        elif current_price < sma_20.iloc[-1]:
            trend_direction = 'Downtrend'
        else:
            trend_direction = 'Sideways'
        
        # ADX for trend strength
        adx = talib.ADX(data['high'], data['low'], close)
        trend_strength = 'Strong' if adx.iloc[-1] > 25 else 'Weak'
        
        # Linear regression slope
        x = np.arange(len(close))
        slope, _ = np.polyfit(x[-20:], close.iloc[-20:].values, 1)
        slope_angle = np.degrees(np.arctan(slope / close.iloc[-1]))
        
        return {
            'current': trend_direction,
            'strength': trend_strength,
            'adx': adx.iloc[-1],
            'slope_angle': slope_angle,
            'price_vs_sma20': (current_price / sma_20.iloc[-1] - 1) * 100,
            'price_vs_sma50': (current_price / sma_50.iloc[-1] - 1) * 100 if not pd.isna(sma_50.iloc[-1]) else 0,
            'sma_alignment': current_price > sma_20.iloc[-1] > sma_50.iloc[-1]
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze trading volume"""
        
        volume = data['volume']
        close = data['close']
        
        # Volume moving averages
        vol_sma_20 = volume.rolling(20).mean()
        
        # Volume trend
        vol_slope, _ = np.polyfit(range(20), volume.iloc[-20:].values, 1)
        vol_trend = 'Increasing' if vol_slope > 0 else 'Decreasing'
        
        # Price-volume correlation
        pv_corr = close.pct_change().corr(volume.pct_change())
        
        # Volume spikes
        vol_zscore = (volume - vol_sma_20) / volume.rolling(20).std()
        recent_spikes = (vol_zscore.iloc[-20:] > 2).sum()
        
        # On-Balance Volume trend
        obv = talib.OBV(close, volume)
        obv_slope, _ = np.polyfit(range(20), obv.iloc[-20:].values, 1)
        
        return {
            'average_volume': vol_sma_20.iloc[-1],
            'current_vs_average': volume.iloc[-1] / vol_sma_20.iloc[-1],
            'trend': vol_trend,
            'price_volume_correlation': pv_corr,
            'recent_spikes': recent_spikes,
            'obv_trend': 'Positive' if obv_slope > 0 else 'Negative',
            'liquidity_score': min(10, vol_sma_20.iloc[-1] / 1e6)  # Score based on millions
        }
    
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze price patterns"""
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Recent highs and lows
        recent_high = high.iloc[-20:].max()
        recent_low = low.iloc[-20:].min()
        current_position = (close.iloc[-1] - recent_low) / (recent_high - recent_low)
        
        # Channel detection
        upper_channel = high.rolling(20).max()
        lower_channel = low.rolling(20).min()
        channel_width = (upper_channel - lower_channel) / close
        
        # Volatility contraction
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
        bb_width = (bb_upper - bb_lower) / bb_middle
        volatility_contraction = bb_width.iloc[-1] < bb_width.rolling(50).quantile(0.2).iloc[-1]
        
        # Pattern recognition
        patterns_detected = []
        
        # Simple pattern detection
        if close.iloc[-3:].is_monotonic_increasing:
            patterns_detected.append('Ascending')
        elif close.iloc[-3:].is_monotonic_decreasing:
            patterns_detected.append('Descending')
        
        if volatility_contraction:
            patterns_detected.append('Volatility Squeeze')
        
        return {
            'current_position': current_position,
            'channel_width': channel_width.iloc[-1],
            'volatility_contraction': volatility_contraction,
            'patterns_detected': patterns_detected,
            'near_resistance': current_position > 0.8,
            'near_support': current_position < 0.2
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze price momentum"""
        
        close = data['close']
        
        # RSI
        rsi = talib.RSI(close)
        current_rsi = rsi.iloc[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(close)
        macd_bullish = macd.iloc[-1] > signal.iloc[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(data['high'], data['low'], close)
        
        # Rate of Change
        roc = talib.ROC(close, timeperiod=10)
        
        # Money Flow Index
        mfi = talib.MFI(data['high'], data['low'], close, data['volume'])
        
        return {
            'rsi': current_rsi,
            'rsi_condition': 'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral',
            'macd_signal': 'Bullish' if macd_bullish else 'Bearish',
            'macd_histogram': hist.iloc[-1],
            'stochastic_k': slowk.iloc[-1],
            'rate_of_change': roc.iloc[-1],
            'mfi': mfi.iloc[-1],
            'momentum_score': self._calculate_momentum_score(current_rsi, macd_bullish, roc.iloc[-1])
        }
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Find support and resistance levels"""
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Recent pivots
        pivot_highs = high.rolling(5).max() == high
        pivot_lows = low.rolling(5).min() == low
        
        # Get significant levels
        resistance_levels = high[pivot_highs].iloc[-10:].unique()
        support_levels = low[pivot_lows].iloc[-10:].unique()
        
        # Current price position
        current_price = close.iloc[-1]
        
        # Find nearest levels
        if len(resistance_levels) > 0:
            nearest_resistance = resistance_levels[resistance_levels > current_price].min() if any(resistance_levels > current_price) else None
        else:
            nearest_resistance = None
            
        if len(support_levels) > 0:
            nearest_support = support_levels[support_levels < current_price].max() if any(support_levels < current_price) else None
        else:
            nearest_support = None
        
        return {
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_
            }