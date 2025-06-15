"""
Live trading implementation with broker integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
from abc import ABC, abstractmethod

from config.settings import TradingConfig, BrokerConfig
from core.portfolio_manager import PortfolioManager
from core.risk_manager import RiskManager
from utils.logger import get_logger
from utils.database import DatabaseManager

logger = get_logger(__name__)


class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    async def connect(self):
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Dict) -> str:
        """Place order and return order ID"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        pass


class AlpacaBroker(BrokerInterface):
    """Alpaca broker implementation"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url or BrokerConfig.ALPACA_BASE_URL
        self.api = None
    
    async def connect(self):
        """Connect to Alpaca"""
        try:
            import alpaca_trade_api as tradeapi
            
            self.api = tradeapi.REST(
                self.api_key,
                self.secret_key,
                self.base_url
            )
            
            # Test connection
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account: {account.account_number}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        self.api = None
        logger.info("Disconnected from Alpaca")
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        account = self.api.get_account()
        
        return {
            'account_id': account.account_number,
            'buying_power': float(account.buying_power),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'day_trade_count': int(account.daytrade_count),
            'pattern_day_trader': account.pattern_day_trader
        }
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        positions = self.api.list_positions()
        
        return [
            {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'market_value': float(pos.market_value),
                'cost_basis': float(pos.cost_basis),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc),
                'current_price': float(pos.current_price)
            }
            for pos in positions
        ]
    
    async def place_order(self, order: Dict) -> str:
        """Place order"""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        try:
            alpaca_order = self.api.submit_order(
                symbol=order['symbol'],
                qty=order['quantity'],
                side=order['side'],
                type=order.get('order_type', 'market'),
                time_in_force=order.get('time_in_force', 'day'),
                limit_price=order.get('limit_price'),
                stop_price=order.get('stop_price'),
                client_order_id=order.get('client_order_id')
            )
            
            logger.info(f"Order placed: {alpaca_order.id}")
            return alpaca_order.id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        if not self.api:
            raise ConnectionError("Not connected to broker")
        
        try:
            order = self.api.get_order(order_id)
            
            return {
                'order_id': order.id,
                'status': order.status,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'created_at': order.created_at,
                'updated_at': order.updated_at
            }
            
        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            return {}


class LiveTrader:
    """Live trading implementation"""
    
    def __init__(
        self,
        broker: BrokerInterface,
        data_manager,
        strategy_engine,
        use_stop_loss: bool = True,
        use_take_profit: bool = True
    ):
        self.broker = broker
        self.data_manager = data_manager
        self.strategy_engine = strategy_engine
        self.portfolio_manager = PortfolioManager()
        self.risk_manager = RiskManager()
        self.db = DatabaseManager()
        
        # Trading parameters
        self.use_stop_loss = use_stop_loss
        self.use_take_profit = use_take_profit
        
        # State
        self.running = False
        self.positions = {}
        self.orders = {}
        self.trades = []
        
        # Performance tracking
        self.start_time = None
        self.equity_curve = []
    
    async def start(self, symbols: List[str], timeframes: List[str] = ['5']):
        """Start live trading"""
        
        logger.info(f"Starting live trading with {len(symbols)} symbols")
        
        # Connect to broker
        await self.broker.connect()
        
        # Get initial account info
        account_info = await self.broker.get_account_info()
        logger.info(f"Account info: {account_info}")
        
        self.running = True
        self.start_time = datetime.now()
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Main trading loop
        try:
            while self.running:
                await self._trading_iteration()
                await asyncio.sleep(TradingConfig.LIVE_UPDATE_INTERVAL)
                
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            
        finally:
            await self.stop()
    
    async def _trading_iteration(self):
        """Single trading iteration"""
        
        try:
            # Update account and positions
            await self._update_account_state()
            
            # Get market data
            market_data = await self._get_market_data()
            
            if not market_data:
                return
            
            # Check risk limits
            risk_check = await self._check_risk_limits()
            if not risk_check['allowed']:
                logger.warning(f"Risk limit reached: {risk_check['reason']}")
                return
            
            # Generate signals
            signals = self._generate_signals(market_data)
            
            # Process signals
            for signal in signals:
                await self._process_signal(signal)
            
            # Update pending orders
            await self._update_orders()
            
            # Record performance
            self._record_performance()
            
        except Exception as e:
            logger.error(f"Trading iteration error: {e}")
    
    async def _update_account_state(self):
        """Update account and position information"""
        
        # Get account info
        self.account_info = await self.broker.get_account_info()
        
        # Get positions
        broker_positions = await self.broker.get_positions()
        
        # Update internal positions
        self.positions = {}
        for pos in broker_positions:
            self.positions[pos['symbol']] = pos
    
    async def _get_market_data(self) -> Dict:
        """Get latest market data"""
        
        market_data = {}
        
        for symbol in self.symbols:
            try:
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
                position_size = position.get('qty', 0)
                
                # Generate signal
                signal = self.strategy_engine.generate_signal(
                    symbol=symbol,
                    data=data,
                    position_size=position_size
                )
                
                if signal and signal['action'] != 'hold':
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    async def _process_signal(self, signal: Dict):
        """Process trading signal"""
        
        symbol = signal['symbol']
        action = signal['action']
        
        # Check if trade is allowed
        trade_allowed, reason = self.risk_manager.check_trade_allowed(
            symbol=symbol,
            capital=self.account_info['buying_power'],
            existing_positions=self.positions,
            current_pnl=self._calculate_current_pnl()
        )
        
        if not trade_allowed:
            logger.warning(f"Trade not allowed for {symbol}: {reason}")
            return
        
        # Prepare order
        if action == 'buy':
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                capital=self.account_info['buying_power'],
                symbol=symbol,
                entry_price=signal.get('price'),
                stop_loss_price=signal.get('stop_loss'),
                volatility=signal.get('volatility', 0.02),
                existing_positions=self.positions
            )
            
            if position_size > 0:
                order = {
                    'symbol': symbol,
                    'quantity': int(position_size),
                    'side': 'buy',
                    'order_type': 'market',
                    'time_in_force': 'day'
                }
                
                # Place order
                order_id = await self.broker.place_order(order)
                
                if order_id:
                    self.orders[order_id] = {
                        'symbol': symbol,
                        'signal': signal,
                        'status': 'pending',
                        'created_at': datetime.now()
                    }
                    
                    # Set stop loss and take profit orders
                    if self.use_stop_loss and 'stop_loss' in signal:
                        await self._place_stop_loss(
                            symbol,
                            position_size,
                            signal['stop_loss']
                        )
                    
                    if self.use_take_profit and 'take_profit' in signal:
                        await self._place_take_profit(
                            symbol,
                            position_size,
                            signal['take_profit']
                        )
        
        elif action == 'sell':
            position = self.positions.get(symbol, {})
            
            if position and position['qty'] > 0:
                order = {
                    'symbol': symbol,
                    'quantity': int(position['qty']),
                    'side': 'sell',
                    'order_type': 'market',
                    'time_in_force': 'day'
                }
                
                # Place order
                order_id = await self.broker.place_order(order)
                
                if order_id:
                    self.orders[order_id] = {
                        'symbol': symbol,
                        'signal': signal,
                        'status': 'pending',
                        'created_at': datetime.now()
                    }
    
    async def _place_stop_loss(
        self,
        symbol: str,
        quantity: int,
        stop_price: float
    ):
        """Place stop loss order"""
        
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'side': 'sell',
            'order_type': 'stop',
            'stop_price': stop_price,
            'time_in_force': 'gtc'
        }
        
        order_id = await self.broker.place_order(order)
        
        if order_id:
            self.orders[order_id] = {
                'symbol': symbol,
                'type': 'stop_loss',
                'status': 'pending',
                'created_at': datetime.now()
            }
    
    async def _place_take_profit(
        self,
        symbol: str,
        quantity: int,
        limit_price: float
    ):
        """Place take profit order"""
        
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'side': 'sell',
            'order_type': 'limit',
            'limit_price': limit_price,
            'time_in_force': 'gtc'
        }
        
        order_id = await self.broker.place_order(order)
        
        if order_id:
            self.orders[order_id] = {
                'symbol': symbol,
                'type': 'take_profit',
                'status': 'pending',
                'created_at': datetime.now()
            }
    
    async def _update_orders(self):
        """Update order statuses"""
        
        for order_id, order_info in list(self.orders.items()):
            if order_info['status'] == 'pending':
                status = await self.broker.get_order_status(order_id)
                
                if status and status['status'] == 'filled':
                    order_info['status'] = 'filled'
                    order_info['filled_at'] = datetime.now()
                    order_info['filled_price'] = status['filled_avg_price']
                    order_info['filled_qty'] = status['filled_qty']
                    
                    # Record trade if it's a closing order
                    if 'signal' in order_info and order_info['signal']['action'] == 'sell':
                        self._record_trade(order_info)
                
                elif status and status['status'] in ['cancelled', 'expired', 'rejected']:
                    order_info['status'] = status['status']
                    del self.orders[order_id]
    
    def _record_trade(self, order_info: Dict):
        """Record completed trade"""
        
        symbol = order_info['symbol']
        position = self.positions.get(symbol, {})
        
        if position:
            trade = {
                'symbol': symbol,
                'entry_time': position.get('created_at', datetime.now()),
                'exit_time': order_info['filled_at'],
                'entry_price': position['avg_entry_price'],
                'exit_price': order_info['filled_price'],
                'size': order_info['filled_qty'],
                'side': 'long',
                'pnl': position.get('unrealized_pl', 0),
                'pnl_pct': position.get('unrealized_plpc', 0) * 100,
                'exit_reason': order_info['signal'].get('reason', 'signal'),
                'strategy': order_info['signal'].get('strategy', 'unknown')
            }
            
            self.trades.append(trade)
            
            # Save to database
            self.db.save_trade(trade)
    
    async def _check_risk_limits(self) -> Dict:
        """Check if risk limits are exceeded"""
        
        # Check daily loss
        daily_pnl = self._calculate_daily_pnl()
        
        if daily_pnl < -self.account_info['portfolio_value'] * TradingConfig.MAX_DAILY_LOSS_PCT:
            return {'allowed': False, 'reason': 'Daily loss limit exceeded'}
        
        # Check position concentration
        for symbol, position in self.positions.items():
            position_value = position['market_value']
            concentration = position_value / self.account_info['portfolio_value']
            
            if concentration > TradingConfig.MAX_POSITION_SIZE_PCT:
                return {'allowed': False, 'reason': f'Position concentration too high for {symbol}'}
        
        return {'allowed': True, 'reason': None}
    
    def _calculate_current_pnl(self) -> float:
        """Calculate current P&L"""
        
        return sum(
            pos.get('unrealized_pl', 0)
            for pos in self.positions.values()
        )
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        
        today = datetime.now().date()
        
        # Daily trades P&L
        daily_trades = [
            trade for trade in self.trades
            if trade['exit_time'].date() == today
        ]
        
        daily_realized = sum(trade['pnl'] for trade in daily_trades)
        
        # Unrealized P&L
        daily_unrealized = self._calculate_current_pnl()
        
        return daily_realized + daily_unrealized
    
    def _record_performance(self):
        """Record performance snapshot"""
        
        snapshot = {
            'timestamp': datetime.now(),
            'equity': self.account_info['portfolio_value'],
            'cash': self.account_info['cash'],
            'positions_value': self.account_info['portfolio_value'] - self.account_info['cash'],
            'daily_pnl': self._calculate_daily_pnl(),
            'total_pnl': self._calculate_current_pnl()
        }
        
        self.equity_curve.append(snapshot)
        
        # Save to database periodically
        if len(self.equity_curve) % 10 == 0:
            self.db.save_performance_snapshot(snapshot)
    
    async def stop(self):
        """Stop live trading"""
        
        self.running = False
        
        # Cancel all pending orders
        for order_id, order_info in self.orders.items():
            if order_info['status'] == 'pending':
                await self.broker.cancel_order(order_id)
        
        # Disconnect from broker
        if self.broker:
            await self.broker.disconnect()
        
        logger.info("Live trading stopped")
    
    def get_results(self) -> Dict:
        """Get trading results"""
        
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        
        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'positions': self.positions,
            'final_equity': self.account_info.get('portfolio_value', 0),
            'trade_count': len(self.trades),
            'runtime': datetime.now() - self.start_time if self.start_time else timedelta(0)
        }