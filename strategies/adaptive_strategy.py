"""
Adaptive strategy that switches between different strategies based on market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from typing import Tuple
import talib

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
import logging

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    """Adaptive strategy that combines multiple strategies based on market regime"""
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        
        # Initialize sub-strategies
        self.trend_strategy = TrendFollowingStrategy()
        self.mean_reversion_strategy = MeanReversionStrategy()
        
        # Market regime history
        self.regime_history = []
        
    def get_default_params(self) -> Dict:
        """Default parameters for adaptive strategy"""
        return {
            'regime_lookback': 50,
            'regime_threshold': 0.6,
            'volatility_window': 20,
            'trend_strength_window': 20,
            'position_size_pct': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.03,
            'use_ensemble': True,
            'min_agreement': 0.5
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive signals based on market regime"""
        
        if not self.validate_data(data):
            return pd.DataFrame()
        
        df = data.copy()
        
        # Detect market regime
        df['market_regime'] = self._detect_market_regime(df)
        
        # Get signals from each strategy
        trend_signals = self.trend_strategy.generate_signals(df.copy())
        mean_rev_signals = self.mean_reversion_strategy.generate_signals(df.copy())
        
        # Initialize signals
        df['signal'] = 0
        df['strategy_used'] = ''
        
        if self.params['use_ensemble']:
            # Ensemble approach - combine signals
            df = self._ensemble_signals(df, trend_signals, mean_rev_signals)
        else:
            # Switch between strategies based on regime
            df = self._switch_strategies(df, trend_signals, mean_rev_signals)
        
        # Apply adaptive position sizing
        df['position_size'] = self._calculate_adaptive_position_size(df)
        
        # Dynamic exit levels
        df['stop_loss'], df['take_profit'] = self._calculate_adaptive_exits(df)
        
        return df
    
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect current market regime"""
        
        regime = pd.Series('unknown', index=df.index)
        
        # Calculate indicators for regime detection
        
        # 1. Trend strength using ADX
        adx = talib.ADX(df['high'], df['low'], df['close'], 
                       timeperiod=self.params['trend_strength_window'])
        
        # 2. Volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(self.params['volatility_window']).std()
        volatility_percentile = volatility.rolling(100).rank(pct=True)
        
        # 3. Price position relative to moving averages
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        price_vs_sma20 = (df['close'] - sma_20) / sma_20
        
        # 4. Market efficiency ratio
        efficiency_ratio = self._calculate_efficiency_ratio(df['close'])
        
        # 5. Volume patterns
        volume_sma = df['volume'].rolling(20).mean()
        volume_trend = (df['volume'] - volume_sma) / volume_sma
        
        # Classify regime
        for i in range(len(df)):
            if i < 50:  # Need minimum data
                regime.iloc[i] = 'unknown'
                continue
            
            # Strong trend conditions
            if adx.iloc[i] > 30 and efficiency_ratio.iloc[i] > 0.7:
                if price_vs_sma20.iloc[i] > 0.02:
                    regime.iloc[i] = 'strong_uptrend'
                elif price_vs_sma20.iloc[i] < -0.02:
                    regime.iloc[i] = 'strong_downtrend'
                else:
                    regime.iloc[i] = 'trend'
            
            # Ranging market
            elif adx.iloc[i] < 20 and volatility_percentile.iloc[i] < 0.5:
                regime.iloc[i] = 'ranging'
            
            # Volatile market
            elif volatility_percentile.iloc[i] > 0.8:
                regime.iloc[i] = 'volatile'
            
            # Breakout conditions
            elif volume_trend.iloc[i] > 0.5 and efficiency_ratio.iloc[i] > 0.6:
                regime.iloc[i] = 'breakout'
            
            # Default to choppy
            else:
                regime.iloc[i] = 'choppy'
        
        # Smooth regime transitions
        regime = self._smooth_regime_transitions(regime)
        
        # Update regime history
        current_regime = regime.iloc[-1]
        self.regime_history.append({
            'timestamp': df.index[-1],
            'regime': current_regime
        })
        
        return regime
    
    def _calculate_efficiency_ratio(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""
        
        change = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(period).sum()
        
        efficiency_ratio = change / volatility
        efficiency_ratio = efficiency_ratio.fillna(0)
        
        return efficiency_ratio
    
    def _smooth_regime_transitions(self, regime: pd.Series) -> pd.Series:
        """Smooth regime transitions to avoid whipsaws"""
        
        smoothed = regime.copy()
        
        # Don't change regime unless it persists for at least 3 bars
        for i in range(3, len(smoothed)):
            if smoothed.iloc[i] != smoothed.iloc[i-1]:
                # Check if new regime persists
                if (i + 2 < len(smoothed) and 
                    smoothed.iloc[i] == smoothed.iloc[i+1] == smoothed.iloc[i+2]):
                    # Keep new regime
                    pass
                else:
                    # Revert to previous regime
                    smoothed.iloc[i] = smoothed.iloc[i-1]
        
        return smoothed
    
    def _ensemble_signals(
        self, 
        df: pd.DataFrame,
        trend_signals: pd.DataFrame,
        mean_rev_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine signals from multiple strategies"""
        
        # Get regime-based weights
        weights = self._get_strategy_weights(df['market_regime'])
        
        # Combine signals
        for i in range(len(df)):
            regime = df['market_regime'].iloc[i]
            
            # Get individual signals
            trend_sig = trend_signals['signal'].iloc[i] if i < len(trend_signals) else 0
            mean_rev_sig = mean_rev_signals['signal'].iloc[i] if i < len(mean_rev_signals) else 0
            
            # Weight signals
            trend_weight = weights[regime]['trend']
            mean_rev_weight = weights[regime]['mean_reversion']
            
            # Calculate weighted signal
            weighted_signal = (trend_sig * trend_weight + 
                             mean_rev_sig * mean_rev_weight)
            
            # Determine final signal
            if abs(weighted_signal) >= self.params['min_agreement']:
                df.loc[df.index[i], 'signal'] = np.sign(weighted_signal)
                
                # Record which strategy contributed more
                if abs(trend_sig * trend_weight) > abs(mean_rev_sig * mean_rev_weight):
                    df.loc[df.index[i], 'strategy_used'] = 'trend'
                else:
                    df.loc[df.index[i], 'strategy_used'] = 'mean_reversion'
            else:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'strategy_used'] = 'none'
        
        return df
    
    def _switch_strategies(
        self,
        df: pd.DataFrame,
        trend_signals: pd.DataFrame,
        mean_rev_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Switch between strategies based on regime"""
        
        regime_strategy_map = {
            'strong_uptrend': 'trend',
            'strong_downtrend': 'trend',
            'trend': 'trend',
            'ranging': 'mean_reversion',
            'choppy': 'mean_reversion',
            'volatile': 'none',  # Don't trade in very volatile conditions
            'breakout': 'trend',
            'unknown': 'none'
        }
        
        for i in range(len(df)):
            regime = df['market_regime'].iloc[i]
            strategy = regime_strategy_map.get(regime, 'none')
            
            if strategy == 'trend' and i < len(trend_signals):
                df.loc[df.index[i], 'signal'] = trend_signals['signal'].iloc[i]
                df.loc[df.index[i], 'strategy_used'] = 'trend'
            elif strategy == 'mean_reversion' and i < len(mean_rev_signals):
                df.loc[df.index[i], 'signal'] = mean_rev_signals['signal'].iloc[i]
                df.loc[df.index[i], 'strategy_used'] = 'mean_reversion'
            else:
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'strategy_used'] = 'none'
        
        return df
    
    def _get_strategy_weights(self, regime: pd.Series) -> Dict[str, Dict[str, float]]:
        """Get strategy weights for each market regime"""
        
        weights = {
            'strong_uptrend': {'trend': 0.9, 'mean_reversion': 0.1},
            'strong_downtrend': {'trend': 0.9, 'mean_reversion': 0.1},
            'trend': {'trend': 0.7, 'mean_reversion': 0.3},
            'ranging': {'trend': 0.2, 'mean_reversion': 0.8},
            'choppy': {'trend': 0.3, 'mean_reversion': 0.7},
            'volatile': {'trend': 0.5, 'mean_reversion': 0.5},
            'breakout': {'trend': 0.8, 'mean_reversion': 0.2},
            'unknown': {'trend': 0.5, 'mean_reversion': 0.5}
        }
        
        return weights
    
    def _calculate_adaptive_position_size(self, df: pd.DataFrame) -> pd.Series:
        """Calculate position size based on market conditions"""
        
        position_sizes = pd.Series(self.params['position_size_pct'], index=df.index)
        
        # Adjust based on regime
        regime_multipliers = {
            'strong_uptrend': 1.2,
            'strong_downtrend': 0.8,  # More conservative on shorts
            'trend': 1.0,
            'ranging': 0.8,
            'choppy': 0.6,
            'volatile': 0.4,  # Very small in volatile markets
            'breakout': 1.1,
            'unknown': 0.5
        }
        
        for regime, multiplier in regime_multipliers.items():
            mask = df['market_regime'] == regime
            position_sizes[mask] *= multiplier
        
        # Further adjust based on volatility
        if 'atr_pct' in df.columns:
            volatility_adjustment = 0.02 / df['atr_pct'].clip(lower=0.01)
            volatility_adjustment = volatility_adjustment.clip(0.5, 1.5)
            position_sizes *= volatility_adjustment
        
        # Apply limits
        return position_sizes.clip(0.02, 0.20)
    
    def _calculate_adaptive_exits(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate adaptive stop loss and take profit levels"""
        
        stop_losses = pd.Series(self.params['stop_loss'], index=df.index)
        take_profits = pd.Series(self.params['take_profit'], index=df.index)
        
        # Adjust based on regime
        for i in range(len(df)):
            regime = df['market_regime'].iloc[i]
            
            if regime in ['strong_uptrend', 'strong_downtrend', 'trend']:
                # Wider stops in trending markets
                stop_losses.iloc[i] = 0.025
                take_profits.iloc[i] = 0.05
            elif regime in ['ranging', 'choppy']:
                # Tighter stops in ranging markets
                stop_losses.iloc[i] = 0.015
                take_profits.iloc[i] = 0.02
            elif regime == 'volatile':
                # Very wide stops in volatile markets
                stop_losses.iloc[i] = 0.04
                take_profits.iloc[i] = 0.06
            elif regime == 'breakout':
                # Asymmetric stops for breakouts
                stop_losses.iloc[i] = 0.015
                take_profits.iloc[i] = 0.08
        
        return stop_losses, take_profits