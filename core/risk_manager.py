"""
Risk management module for position sizing and exposure control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from config.settings import TradingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self):
        self.max_daily_loss = TradingConfig.MAX_DAILY_LOSS_PCT
        self.max_portfolio_risk = TradingConfig.MAX_PORTFOLIO_RISK_PCT
        self.max_positions = TradingConfig.MAX_POSITIONS
        
        # Track daily P&L
        self.daily_pnl = {}
        self.current_date = None
        
    def calculate_position_size(
        self,
        capital: float,
        symbol: str,
        entry_price: float,
        stop_loss_price: float,
        volatility: float,
        existing_positions: Dict[str, Dict]
    ) -> float:
        """Calculate optimal position size using multiple methods"""
        
        # Method 1: Fixed percentage
        fixed_size = capital * TradingConfig.DEFAULT_POSITION_SIZE_PCT
        
        # Method 2: Volatility-based sizing
        target_risk = capital * 0.01  # 1% risk per trade
        price_risk = abs(entry_price - stop_loss_price) / entry_price
        volatility_size = target_risk / price_risk if price_risk > 0 else fixed_size
        
        # Method 3: Kelly Criterion (simplified)
        # Requires win rate and profit/loss ratio
        kelly_size = fixed_size  # Placeholder
        
        # Method 4: Portfolio heat check
        current_exposure = self._calculate_portfolio_exposure(existing_positions)
        remaining_capacity = (self.max_portfolio_risk - current_exposure) * capital
        heat_adjusted_size = min(fixed_size, remaining_capacity)
        
        # Take the minimum for safety
        position_size = min(fixed_size, volatility_size, heat_adjusted_size)
        
        # Apply position limits
        position_size = self._apply_position_limits(position_size, entry_price)
        
        logger.debug(f"{symbol} position size: ${position_size:.2f}")
        return position_size
    
    def check_trade_allowed(
        self,
        symbol: str,
        capital: float,
        existing_positions: Dict[str, Dict],
        current_pnl: float = 0
    ) -> Tuple[bool, str]:
        """Check if new trade is allowed based on risk rules"""
        
        # Check daily loss limit
        if self._check_daily_loss_limit(current_pnl):
            return False, "Daily loss limit reached"
        
        # Check maximum positions
        if len(existing_positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check portfolio heat
        current_exposure = self._calculate_portfolio_exposure(existing_positions)
        if current_exposure >= self.max_portfolio_risk:
            return False, f"Portfolio risk limit ({self.max_portfolio_risk:.1%}) reached"
        
        # Check correlation limits
        if self._check_correlation_limits(symbol, existing_positions):
            return False, "Correlation limits exceeded"
        
        return True, "Trade allowed"
    
    def update_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        current_stop: float,
        strategy: str = 'trailing'
    ) -> float:
        """Dynamic stop loss adjustment"""
        
        if strategy == 'trailing':
            # Trailing stop loss
            trail_pct = 0.02  # 2% trailing
            new_stop = current_price * (1 - trail_pct)
            
            # Only move stop up (for long positions)
            if new_stop > current_stop:
                logger.info(f"{symbol}: Moving stop loss from {current_stop:.2f} to {new_stop:.2f}")
                return new_stop
                
        elif strategy == 'breakeven':
            # Move to breakeven after certain profit
            profit_pct = (current_price - entry_price) / entry_price
            
            if profit_pct > 0.01 and current_stop < entry_price:  # 1% profit
                logger.info(f"{symbol}: Moving stop to breakeven at {entry_price:.2f}")
                return entry_price
                
        elif strategy == 'atr':
            # ATR-based dynamic stop
            # Would need ATR data passed in
            pass
        
        return current_stop
    
    def calculate_portfolio_var(
        self,
        positions: Dict[str, Dict],
        confidence_level: float = 0.95,
        lookback_days: int = 30
    ) -> Dict[str, float]:
        """Calculate portfolio Value at Risk"""
        
        if not positions:
            return {'var_dollar': 0, 'var_percent': 0}
        
        # Get historical returns for all positions
        returns_data = {}
        total_value = 0
        
        for symbol, position in positions.items():
            # Would need historical data here
            # Placeholder for now
            position_value = position.get('size', 0) * position.get('current_price', 0)
            total_value += position_value
        
        # Calculate portfolio VaR
        # Simplified calculation
        portfolio_volatility = 0.02  # 2% daily volatility placeholder
        z_score = 1.645 if confidence_level == 0.95 else 2.326  # 95% or 99%
        
        var_percent = portfolio_volatility * z_score
        var_dollar = total_value * var_percent
        
        return {
            'var_dollar': var_dollar,
            'var_percent': var_percent,
            'confidence_level': confidence_level
        }
    
    def get_risk_metrics(self, positions: Dict[str, Dict]) -> Dict:
        """Get current risk metrics"""
        
        total_exposure = sum(
            pos.get('size', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        )
        
        portfolio_heat = self._calculate_portfolio_exposure(positions)
        
        # Calculate current day P&L
        today = datetime.now().date()
        daily_pnl = self.daily_pnl.get(today, 0)
        
        # Position concentration
        position_values = [
            pos.get('size', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        ]
        
        if position_values and total_exposure > 0:
            max_position = max(position_values)
            concentration = max_position / total_exposure
        else:
            concentration = 0
        
        return {
            'total_exposure': total_exposure,
            'portfolio_heat': portfolio_heat,
            'daily_pnl': daily_pnl,
            'position_count': len(positions),
            'max_concentration': concentration,
            'var_metrics': self.calculate_portfolio_var(positions)
        }
    
    def _calculate_portfolio_exposure(self, positions: Dict[str, Dict]) -> float:
        """Calculate total portfolio risk exposure"""
        
        total_risk = 0
        
        for symbol, position in positions.items():
            # Risk = position size * distance to stop loss
            position_value = position.get('size', 0) * position.get('entry_price', 0)
            stop_distance = abs(
                position.get('stop_loss', 0) - position.get('entry_price', 0)
            ) / position.get('entry_price', 1)
            
            position_risk = position_value * stop_distance
            total_risk += position_risk
        
        # Return as percentage of capital
        # Would need total capital passed in
        return total_risk / 10000  # Placeholder
    
    def _check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit has been reached"""
        
        today = datetime.now().date()
        
        # Reset daily P&L if new day
        if self.current_date != today:
            self.current_date = today
            self.daily_pnl[today] = 0
        
        # Update daily P&L
        self.daily_pnl[today] = current_pnl
        
        # Check limit (would need capital to calculate percentage)
        return current_pnl < -500  # Placeholder: $500 daily loss limit
    
    def _check_correlation_limits(
        self,
        symbol: str,
        existing_positions: Dict[str, Dict]
    ) -> bool:
        """Check correlation limits between positions"""
        
        # Simplified check - would need correlation matrix
        # For now, just limit same base currency
        base_currency = symbol[:3]
        
        same_base_count = sum(
            1 for s in existing_positions.keys()
            if s.startswith(base_currency)
        )
        
        return same_base_count >= 3  # Max 3 positions with same base
    
    def _apply_position_limits(self, position_size: float, price: float) -> float:
        """Apply minimum and maximum position limits"""
        
        min_size_usd = 10  # Minimum $10
        max_size_usd = 10000  # Maximum $10,000
        
        position_size = max(min_size_usd, position_size)
        position_size = min(max_size_usd, position_size)
        
        return position_size