"""
Strategy configurations
"""

class StrategyConfig:
    """Strategy-specific configurations"""
    
    # Available strategies
    STRATEGIES = {
        'rsi_mean_reversion': {
            'name': 'RSI Mean Reversion',
            'description': 'Mean reversion based on RSI oversold/overbought',
            'default_params': {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'dynamic_thresholds': True
            }
        },
        'trend_following': {
            'name': 'Trend Following',
            'description': 'Multi-timeframe trend following with MACD',
            'default_params': {
                'ema_fast': 12,
                'ema_slow': 26,
                'ema_trend': 50,
                'atr_multiplier': 2.0
            }
        },
        'breakout': {
            'name': 'Breakout Strategy',
            'description': 'Volatility breakout with volume confirmation',
            'default_params': {
                'bb_period': 20,
                'bb_std': 2.0,
                'volume_threshold': 1.5,
                'atr_period': 20
            }
        },
        'volume_momentum': {
            'name': 'Volume Momentum',
            'description': 'Momentum strategy based on volume indicators',
            'default_params': {
                'obv_period': 20,
                'mfi_period': 14,
                'mfi_oversold': 20,
                'mfi_overbought': 80
            }
        },
        'pattern_recognition': {
            'name': 'Pattern Recognition',
            'description': 'Candlestick pattern-based signals',
            'default_params': {
                'min_pattern_strength': 100,
                'confirmation_bars': 2
            }
        },
        'market_making': {
            'name': 'Market Making',
            'description': 'Mean reversion for ranging markets',
            'default_params': {
                'band_period': 20,
                'band_std': 1.0,
                'min_spread': 0.001,
                'order_flow_threshold': 0.6
            }
        }
    }
    
    # Strategy weights by market condition
    DEFAULT_WEIGHTS = {
        'equal': {
            'rsi_mean_reversion': 0.167,
            'trend_following': 0.167,
            'breakout': 0.167,
            'volume_momentum': 0.167,
            'pattern_recognition': 0.167,
            'market_making': 0.167
        },
        'trend_focused': {
            'trend_following': 0.3,
            'breakout': 0.25,
            'volume_momentum': 0.2,
            'rsi_mean_reversion': 0.1,
            'pattern_recognition': 0.1,
            'market_making': 0.05
        },
        'range_focused': {
            'market_making': 0.3,
            'rsi_mean_reversion': 0.25,
            'pattern_recognition': 0.2,
            'volume_momentum': 0.15,
            'trend_following': 0.05,
            'breakout': 0.05
        }
    }