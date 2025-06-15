"""
Paper trading implementation for strategy testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json

from config.settings import TradingConfig, PaperTradingConfig
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from utils.logger import get_logger
from utils.database import DatabaseManager

logger = get_logger(__name__)


class PaperTrader:
    """Simulated trading with real-time data"""
    
    def __init__(
        self,
        data_manager,
        strategy_engine,
        initial_capital: float = PaperTradingConfig.INITIAL_CAPITAL
    ):
        self.data_manager = data_manager
        self.strategy_engine = strategy_engine
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager()
        self.db = DatabaseManager()
        
        # Trading state
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.pending_orders = {}
        
        # Performance tracking
        self.equity_curve = []
        self.daily_stats = {}
        
        # Control
        self.running = False
        self.start_time = None
        
        # Paper trading specific
        self.simulated_fills = PaperTradingConfig.SIMULATE_SLIPPAGE
        self.latency_ms = PaperTradingConfig.LATENCY_MS
        
    async def start(
        self,
        symbols: List[str],
        timeframes: List[str] = ['5']
    ):
        """Start paper trading"""
        
        logger.info(f"Starting paper trading with {len(symbols)} symbols")
        
        self.running = True
        self.start_time = datetime.now()
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Initialize positions
        for symbol in symbols:
            self.positions[symbol] = {
                'size': 0,
                'entry_price': 0,
                'current_price': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0
            }
        
        # Main trading loop
        try:
            while self.running:
                await self._trading_iteration()
                await asyncio.sleep(PaperTradingConfig.UPDATE_INTERVAL)
                
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            self.stop()
    
    async def _trading_iteration(self):
        """Single trading iteration"""
        
        # Get latest data
        market_data = await self._get_market_data()
        
        if not market_data:
            return
        
        # Update positions with current prices
        self._update_positions(market_data)
        
        # Check risk limits
        risk_check = self._check_risk_limits()
        if not risk_check['allowed']:
            logger.warning(f"Risk limit reached: {risk_check['reason']}")
            return
        
        # Generate signals
        signals = self._generate_signals(market_data)
        
        # Execute trades
        for signal in signals:
            await self._process_signal(signal, market_data)
        
        # Update pending orders
        await self._update_pending_orders(market_data)
        
        # Record performance
        self._record_performance()
    
    async def _get_market_data(self) -> Dict:
        """Get latest market data"""
        
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get real-time data
                data = await self.data_manager.get_realtime_data(
                    symbol,
                    self.timeframes
                )
                
                if data:
                    market_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error getting data for {symbol}: {e}")
        
        return market_data
    
    def _generate_signals(self, market_data: Dict) -> List[Dict]:
        """Generate trading signals"""
        
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # Get current position
                position = self.positions.get(symbol, {})
                
                # Generate signal
                signal = self.strategy_engine.generate_signal(
                    symbol=symbol,
                    data=data,
                    position_size=position.get('size', 0)
                )
                
                if signal and signal['action'] != 'hold':
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _process_signal(self, signal: Dict, market_data: Dict):
        """Process trading signal"""
        
        symbol = signal['symbol']
        action = signal['action']
        
        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        # Get current price
        current_data = market_data.get(symbol, {})
        if not current_data:
            return
        
        current_price = current_data.get('close', 0)
        
        # Apply slippage
        if self.simulated_fills:
            slippage = self._calculate_slippage(
                symbol,
                action,
                current_price
            )
            execution_price = current_price + slippage
        else:
            execution_price = current_price
        
        # Check if trade is allowed
        trade_allowed, reason = self.risk_manager.check_trade_allowed(
            symbol=symbol,
            capital=self.capital,
            existing_positions=self.positions,
            current_pnl=self._calculate_total_pnl()
        )
        
        if not trade_allowed:
            logger.warning(f"Trade not allowed for {symbol}: {reason}")
            return
        
        # Calculate position size
        if action == 'buy':
            position_size = self.risk_manager.calculate_position_size(
                capital=self.capital,
                symbol=symbol,
                entry_price=execution_price,
                stop_loss_price=signal.get('stop_loss', execution_price * 0.98),
                volatility=signal.get('volatility', 0.02),
                existing_positions=self.positions
            )
            
            # Execute buy
            self._execute_trade(
                symbol=symbol,
                action='buy',
                size=position_size,
                price=execution_price,
                signal=signal
            )
            
        elif action == 'sell' and self.positions[symbol]['size'] > 0:
            # Execute sell
            self._execute_trade(
                symbol=symbol,
                action='sell',
                size=self.positions[symbol]['size'],
                price=execution_price,
                signal=signal
            )
    
    def _execute_trade(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        signal: Dict
    ):
        """Execute paper trade"""
        
        timestamp = datetime.now()
        position = self.positions[symbol]
        
        if action == 'buy':
            # Calculate trade value
            trade_value = size * price
            
            if trade_value > self.capital:
                logger.warning(f"Insufficient capital for {symbol}")
                return
            
            # Update position
            if position['size'] == 0:
                # New position
                position['size'] = size
                position['entry_price'] = price
                position['entry_time'] = timestamp
            else:
                # Add to position
                total_cost = position['size'] * position['entry_price'] + trade_value
                position['size'] += size
                position['entry_price'] = total_cost / position['size']
            
            # Update capital
            self.capital -= trade_value
            
            # Set stop loss and take profit
            if 'stop_loss' in signal:
                self._create_pending_order(
                    symbol=symbol,
                    order_type='stop_loss',
                    price=signal['stop_loss'],
                    size=size
                )
            
            if 'take_profit' in signal:
                self._create_pending_order(
                    symbol=symbol,
                    order_type='take_profit',
                    price=signal['take_profit'],
                    size=size
                )
            
        elif action == 'sell':
            if position['size'] == 0:
                logger.warning(f"No position to sell for {symbol}")
                return
            
            # Calculate P&L
            sell_value = size * price
            cost_basis = size * position['entry_price']
            pnl = sell_value - cost_basis
            pnl_pct = (price / position['entry_price'] - 1) * 100
            
            # Update position
            position['size'] -= size
            position['realized_pnl'] += pnl
            
            if position['size'] == 0:
                position['entry_price'] = 0
                position['unrealized_pnl'] = 0
            
            # Update capital
            self.capital += sell_value
            
            # Record trade
            trade = {
                'symbol': symbol,
                'entry_time': position.get('entry_time', timestamp),
                'exit_time': timestamp,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'size': size,
                'side': 'long',
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'exit_reason': signal.get('reason', 'signal'),
                'strategy': signal.get('strategy', 'unknown'),
                'commission': size * price * PaperTradingConfig.COMMISSION_RATE
            }
            
            self.trades.append(trade)
            
            # Save to database
            self.db.save_trade(trade)
            
            # Cancel pending orders
            self._cancel_pending_orders(symbol)
        
        # Log trade
        logger.info(
            f"Paper trade executed: {action} {size:.2f} {symbol} @ {price:.4f}"
        )
    
    def _update_positions(self, market_data: Dict):
        """Update position values"""
        
        for symbol, position in self.positions.items():
            if position['size'] > 0 and symbol in market_data:
                current_price = market_data[symbol].get('close', position['entry_price'])
                position['current_price'] = current_price
                
                # Calculate unrealized P&L
                position['unrealized_pnl'] = (
                    (current_price - position['entry_price']) * position['size']
                )
                position['unrealized_pnl_pct'] = (
                    (current_price / position['entry_price'] - 1) * 100
                    if position['entry_price'] > 0 else 0
                )
    
    async def _update_pending_orders(self, market_data: Dict):
        """Check and execute pending orders"""
        
        for order_id, order in list(self.pending_orders.items()):
            symbol = order['symbol']
            
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol].get('close', 0)
            
            # Check if order should be executed
            should_execute = False
            
            if order['type'] == 'stop_loss' and current_price <= order['price']:
                should_execute = True
                reason = 'stop_loss'
                
            elif order['type'] == 'take_profit' and current_price >= order['price']:
                should_execute = True
                reason = 'take_profit'
            
            if should_execute:
                # Create sell signal
                signal = {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': reason,
                    'price': current_price
                }
                
                # Process signal
                await self._process_signal(signal, market_data)
                
                # Remove executed order
                del self.pending_orders[order_id]
    
    def _create_pending_order(
        self,
        symbol: str,
        order_type: str,
        price: float,
        size: float
    ):
        """Create pending order"""
        
        order_id = f"{symbol}_{order_type}_{datetime.now().timestamp()}"
        
        self.pending_orders[order_id] = {
            'symbol': symbol,
            'type': order_type,
            'price': price,
            'size': size,
            'created_at': datetime.now()
        }
    
    def _cancel_pending_orders(self, symbol: str):
        """Cancel all pending orders for symbol"""
        
        orders_to_cancel = [
            order_id for order_id, order in self.pending_orders.items()
            if order['symbol'] == symbol
        ]
        
        for order_id in orders_to_cancel:
            del self.pending_orders[order_id]
    
    def _calculate_slippage(
        self,
        symbol: str,
        action: str,
        price: float
    ) -> float:
        """Calculate simulated slippage"""
        
        # Base slippage
        base_slippage = price * PaperTradingConfig.SLIPPAGE_PCT
        
        # Random component
        random_factor = np.random.uniform(0.5, 1.5)
        
        # Direction based on action
        if action == 'buy':
            return base_slippage * random_factor  # Pay more
        else:
            return -base_slippage * random_factor  # Receive less
    
    def _check_risk_limits(self) -> Dict:
        """Check if risk limits are exceeded"""
        
        # Calculate current exposure
        total_exposure = sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
            if pos['size'] > 0
        )
        
        # Check daily loss
        daily_pnl = self._calculate_daily_pnl()
        if daily_pnl < -self.capital * TradingConfig.MAX_DAILY_LOSS_PCT:
            return {'allowed': False, 'reason': 'Daily loss limit exceeded'}
        
        # Check exposure limit
        if total_exposure > self.capital * TradingConfig.MAX_PORTFOLIO_RISK_PCT:
            return {'allowed': False, 'reason': 'Portfolio exposure limit exceeded'}
        
        return {'allowed': True, 'reason': None}
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total P&L"""
        
        realized_pnl = sum(
            pos.get('realized_pnl', 0)
            for pos in self.positions.values()
        )
        
        unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0)
            for pos in self.positions.values()
        )
        
        return realized_pnl + unrealized_pnl
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        
        today = datetime.now().date()
        
        daily_trades = [
            trade for trade in self.trades
            if trade['exit_time'].date() == today
        ]
        
        daily_realized = sum(trade['pnl'] for trade in daily_trades)
        daily_unrealized = sum(
            pos.get('unrealized_pnl', 0)
            for pos in self.positions.values()
        )
        
        return daily_realized + daily_unrealized
    
    def _record_performance(self):
        """Record performance snapshot"""
        
        # Calculate metrics
        total_value = self.capital + sum(
            pos['size'] * pos.get('current_price', pos['entry_price'])
            for pos in self.positions.values()
            if pos['size'] > 0
        )
        
        snapshot = {
            'timestamp': datetime.now(),
            'equity': total_value,
            'cash': self.capital,
            'positions_value': total_value - self.capital,
            'daily_pnl': self._calculate_daily_pnl(),
            'total_pnl': total_value - self.initial_capital,
            'return_pct': (total_value / self.initial_capital - 1) * 100
        }
        
        self.equity_curve.append(snapshot)
        
        # Save to database periodically
        if len(self.equity_curve) % 10 == 0:
            self.db.save_performance_snapshot(snapshot)
    
    def get_results(self) -> Dict:
        """Get paper trading results"""
        
        # Create DataFrames
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        
        # Current positions
        open_positions = {
            symbol: pos for symbol, pos in self.positions.items()
            if pos['size'] > 0
        }
        
        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'positions': open_positions,
            'final_equity': self.capital + sum(
                pos['size'] * pos.get('current_price', pos['entry_price'])
                for pos in self.positions.values()
                if pos['size'] > 0
            ),
            'total_return': (
                (self.capital + sum(
                    pos['size'] * pos.get('current_price', pos['entry_price'])
                    for pos in self.positions.values()
                    if pos['size'] > 0
                )) / self.initial_capital - 1
            ) * 100,
            'trade_count': len(self.trades),
            'win_rate': (
                len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades) * 100
                if self.trades else 0
            ),
            'runtime': datetime.now() - self.start_time if self.start_time else timedelta(0)
        }
    
    def stop(self):
        """Stop paper trading"""
        
        self.running = False
        
        # Close all positions
        for symbol, position in self.positions.items():
            if position['size'] > 0:
                signal = {
                    'symbol': symbol,
                    'action': 'sell',
                    'reason': 'session_end'
                }
                
                self._execute_trade(
                    symbol=symbol,
                    action='sell',
                    size=position['size'],
                    price=position.get('current_price', position['entry_price']),
                    signal=signal
                )
        
        logger.info("Paper trading stopped")