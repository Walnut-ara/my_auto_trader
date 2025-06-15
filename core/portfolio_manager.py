"""
Portfolio management and optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config.settings import TradingConfig
from core.risk_manager import RiskManager
from utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """Portfolio construction and management"""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.positions = {}
        self.cash = TradingConfig.BACKTEST_INITIAL_CAPITAL
        self.initial_capital = self.cash
        
    def construct_portfolio(
        self,
        symbol_scores: List[Dict],
        capital: float,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Construct optimal portfolio from scored symbols"""
        
        if not symbol_scores:
            return {'allocations': {}, 'total_symbols': 0}
        
        # Default constraints
        if constraints is None:
            constraints = {
                'max_position_size': 0.20,  # 20% max per position
                'min_position_size': 0.02,  # 2% min per position
                'max_positions': TradingConfig.MAX_POSITIONS,
                'min_score': 50  # Minimum score to include
            }
        
        # Filter symbols by minimum score
        qualified_symbols = [
            s for s in symbol_scores
            if s.get('total_score', 0) >= constraints['min_score']
        ]
        
        if not qualified_symbols:
            logger.warning("No symbols meet minimum score requirement")
            return {'allocations': {}, 'total_symbols': 0}
        
        # Sort by score
        qualified_symbols.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Limit to max positions
        selected_symbols = qualified_symbols[:constraints['max_positions']]
        
        # Calculate allocations
        allocations = self._calculate_allocations(
            selected_symbols,
            capital,
            constraints
        )
        
        # Risk parity adjustment
        allocations = self._apply_risk_parity(allocations, selected_symbols)
        
        return {
            'allocations': allocations,
            'total_symbols': len(allocations),
            'total_allocated': sum(allocations.values()),
            'symbols': list(allocations.keys())
        }
    
    def rebalance_portfolio(
        self,
        current_positions: Dict[str, Dict],
        target_allocations: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Dict[str, Dict]:
        """Calculate rebalancing trades"""
        
        rebalance_trades = {}
        
        # Calculate current values
        total_value = self.cash
        current_allocations = {}
        
        for symbol, position in current_positions.items():
            position_value = position['size'] * current_prices.get(symbol, 0)
            total_value += position_value
            current_allocations[symbol] = position_value
        
        # Calculate required trades
        for symbol, target_pct in target_allocations.items():
            target_value = total_value * target_pct
            current_value = current_allocations.get(symbol, 0)
            
            difference = target_value - current_value
            
            if abs(difference) > total_value * 0.01:  # 1% threshold
                rebalance_trades[symbol] = {
                    'action': 'buy' if difference > 0 else 'sell',
                    'amount': abs(difference),
                    'shares': abs(difference) / current_prices.get(symbol, 1),
                    'current_allocation': current_value / total_value,
                    'target_allocation': target_pct
                }
        
        return rebalance_trades
    
    def update_positions(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        timestamp: datetime
    ):
        """Update portfolio positions"""
        
        if action == 'buy':
            if symbol in self.positions:
                # Average up/down
                current = self.positions[symbol]
                new_size = current['size'] + size
                new_cost = current['cost'] + (size * price)
                
                self.positions[symbol] = {
                    'size': new_size,
                    'cost': new_cost,
                    'avg_price': new_cost / new_size,
                    'last_update': timestamp
                }
            else:
                # New position
                self.positions[symbol] = {
                    'size': size,
                    'cost': size * price,
                    'avg_price': price,
                    'entry_time': timestamp,
                    'last_update': timestamp
                }
            
            self.cash -= size * price
            
        elif action == 'sell':
            if symbol in self.positions:
                current = self.positions[symbol]
                
                if size >= current['size']:
                    # Close position
                    self.cash += current['size'] * price
                    del self.positions[symbol]
                else:
                    # Partial close
                    current['size'] -= size
                    current['cost'] -= size * current['avg_price']
                    self.cash += size * price
    
    def get_portfolio_metrics(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate portfolio performance metrics"""
        
        # Current values
        position_values = {}
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['avg_price'])
            position_value = position['size'] * current_price
            position_values[symbol] = position_value
            total_value += position_value
        
        # Returns
        total_return = (total_value / self.initial_capital - 1) * 100
        
        # Position metrics
        if self.positions:
            position_returns = {}
            for symbol, position in self.positions.items():
                current_price = current_prices.get(symbol, position['avg_price'])
                position_returns[symbol] = (
                    (current_price / position['avg_price'] - 1) * 100
                )
            
            best_performer = max(position_returns.items(), key=lambda x: x[1])
            worst_performer = min(position_returns.items(), key=lambda x: x[1])
        else:
            position_returns = {}
            best_performer = (None, 0)
            worst_performer = (None, 0)
        
        # Concentration
        if position_values and total_value > 0:
            concentrations = {
                symbol: value / total_value * 100
                for symbol, value in position_values.items()
            }
            max_concentration = max(concentrations.values())
        else:
            concentrations = {}
            max_concentration = 0
        
        return {
            'total_value': total_value,
            'cash': self.cash,
            'total_return': total_return,
            'position_count': len(self.positions),
            'position_values': position_values,
            'position_returns': position_returns,
            'concentrations': concentrations,
            'max_concentration': max_concentration,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'cash_percentage': (self.cash / total_value * 100) if total_value > 0 else 100
        }
    
    def _calculate_allocations(
        self,
        symbols: List[Dict],
        capital: float,
        constraints: Dict
    ) -> Dict[str, float]:
        """Calculate position allocations"""
        
        allocations = {}
        
        # Score-weighted allocation
        total_score = sum(s['total_score'] for s in symbols)
        
        for symbol_data in symbols:
            symbol = symbol_data['symbol']
            score = symbol_data['total_score']
            
            # Base allocation proportional to score
            base_allocation = (score / total_score) if total_score > 0 else 0
            
            # Apply constraints
            allocation = max(
                constraints['min_position_size'],
                min(base_allocation, constraints['max_position_size'])
            )
            
            allocations[symbol] = allocation
        
        # Normalize allocations to sum to 1
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {
                symbol: alloc / total_allocation
                for symbol, alloc in allocations.items()
            }
        
        return allocations
    
    def _apply_risk_parity(
        self,
        allocations: Dict[str, float],
        symbol_data: List[Dict]
    ) -> Dict[str, float]:
        """Apply risk parity adjustments"""
        
        # Get volatilities
        volatilities = {}
        for data in symbol_data:
            symbol = data['symbol']
            volatility = data.get('volatility', 0.02)  # Default 2%
            volatilities[symbol] = volatility
        
        # Inverse volatility weighting
        risk_adjusted = {}
        total_inv_vol = sum(1/v for v in volatilities.values() if v > 0)
        
        for symbol, allocation in allocations.items():
            if symbol in volatilities and volatilities[symbol] > 0:
                # Higher allocation to lower volatility
                inv_vol_weight = (1 / volatilities[symbol]) / total_inv_vol
                
                # Blend original and risk-parity allocation
                risk_adjusted[symbol] = 0.7 * allocation + 0.3 * inv_vol_weight
            else:
                risk_adjusted[symbol] = allocation
        
        # Normalize
        total = sum(risk_adjusted.values())
        if total > 0:
            risk_adjusted = {
                symbol: alloc / total
                for symbol, alloc in risk_adjusted.items()
            }
        
        return risk_adjusted