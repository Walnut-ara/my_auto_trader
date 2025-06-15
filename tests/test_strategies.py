"""
Unit tests for trading strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.ml_strategy import MLStrategy


class TestTrendFollowingStrategy:
    @pytest.fixture
    def strategy(self):
        return TrendFollowingStrategy()
    
    @pytest.fixture
    def sample_data(self):
        # Generate sample OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def test_generate_signals(self, strategy, sample_data):
        signals = strategy.generate_signals(sample_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert 'position' in signals.columns
        assert len(signals) == len(sample_data)
    
    def test_signal_values(self, strategy, sample_data):
        signals = strategy.generate_signals(sample_data)
        
        # Check signal values are valid
        assert signals['signal'].isin([1, -1, 0]).all()
        assert signals['position'].isin([1, -1, 0]).all()


class TestMeanReversionStrategy:
    @pytest.fixture
    def strategy(self):
        return MeanReversionStrategy()
    
    def test_z_score_calculation(self, strategy):
        data = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
        z_scores = strategy._calculate_z_score(data, window=5)
        
        assert len(z_scores) == len(data)
        assert z_scores.isna().sum() == 4  # First 4 values should be NaN


class TestMLStrategy:
    @pytest.fixture
    def strategy(self):
        return MLStrategy()
    
    def test_feature_generation(self, strategy):
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5T')
        data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        features = strategy._generate_features(data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert features.shape[1] > 5  # Should have multiple features