"""
Base strategy class for all trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, params: Dict = None):
        """Initialize strategy with parameters"""
        self.params = params or self.get_default_params()
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals (-1, 0, 1)
        """
        pass
    
    @abstractmethod
    def get_default_params(self) -> Dict:
        """Get default parameters for the strategy"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns in data")
            return False
            
        if len(data) < 50:
            logger.warning(f"Insufficient data: {len(data)} rows")
            return False
            
        return True
    
    def calculate_position_size(
        self, 
        signal: int,
        capital: float,
        current_price: float,
        volatility: float = 0.02
    ) -> float:
        """Calculate position size based on signal and risk"""
        
        if signal == 0:
            return 0
            
        # Base position size from parameters
        base_size = self.params.get('position_size_pct', 0.1)
        
        # Adjust for volatility
        vol_adjusted = base_size * (0.02 / volatility)
        vol_adjusted = np.clip(vol_adjusted, 0.05, 0.20)
        
        # Calculate position
        position_value = capital * vol_adjusted
        position_size = position_value / current_price
        
        return position_size * signal
    
    def get_exit_levels(
        self,
        entry_price: float,
        signal: int,
        atr: float = None
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        # Get parameters
        stop_loss_pct = self.params.get('stop_loss', 0.02)
        take_profit_pct = self.params.get('take_profit', 0.03)
        
        if signal > 0:  # Long position
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        else:  # Short position
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
            
        return stop_loss, take_profit
    
    def update_params(self, new_params: Dict):
        """Update strategy parameters"""
        self.params.update(new_params)
        logger.info(f"{self.name} parameters updated: {new_params}")