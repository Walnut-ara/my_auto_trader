"""
Comprehensive backtesting engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

from core.risk_manager import RiskManager
from utils.validators import Validators, DataValidator

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Advanced backtesting engine with realistic simulation"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.0006):
        """
        Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission (0.06% for Bybit)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.risk_manager = RiskManager()
        
        # State tracking
        self.reset()
        
    def reset(self):
        """Reset backtesting state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.orders = {}
        self.current_time = None
        
    def run(self, data: pd.DataFrame, strategy: Any, params: Dict = None) -> Dict:
        """
        Run backtest with given strategy
        
        Args:
            data: OHLCV data
            strategy: Strategy instance or function
            params: Strategy parameters
            
        Returns:
            Backtest results
        """
        # Validate data
        Validators.validate_ohlcv_data(data)
        
        # Reset state
        self.reset()
        
        logger.info(f"Starting backtest with {len(data)} bars")
        
        # Generate signals
        if hasattr(strategy, 'generate_signals'):
            signals_df = strategy.generate_signals(data, params)
        else:
            signals_df = strategy(data, params)
        
        # Ensure we have required columns
        if 'signal' not in signals_df.columns:
            logger.error("No signals generated")
            return self._compile_results()
        
        # Process each bar
        for i in range(len(signals_df)):
            self.current_time = signals_df.index[i]
            bar = signals_df.iloc[i]
            
            # Update equity
            self._update_equity(bar)
            
            # Check for exits
            self._check_exits(bar)
            
            # Process new signals
            if bar['signal'] != 0 and not np.isnan(bar['signal']):
                self._process_signal(bar, signals_df.iloc[max(0, i-20):i+1])
            
            # Record equity
            self.equity_curve.append({
                'timestamp': self.current_time,
                'equity': self._calculate_equity(bar),
                'cash': self.capital,
                'positions_value': self._calculate_positions_value(bar)
            })
        
        # Close any remaining positions
        if len(signals_df) > 0:
            self._close_all_positions(signals_df.iloc[-1])
        
        return self._compile_results()
    
    def _process_signal(self, bar: pd.Series, recent_data: pd.DataFrame):
        """Process trading signal"""
        
        signal = int(bar['signal'])
        symbol = bar.get('symbol', 'UNKNOWN')
        
        # Check if we already have a position
        if symbol in self.positions:
            current_position = self.positions[symbol]
            
            # Check if signal is opposite to current position
            if np.sign(current_position['side']) != np.sign(signal):
                # Close current position first
                self._close_position(symbol, bar)
                # Then open new position
                self._open_position(symbol, signal, bar, recent_data)
            # If same direction, consider adding to position
            elif abs(len(self.positions)) < 5:  # Max 5 concurrent positions
                pass  # Could implement position scaling here
        else:
            # Open new position
            allowed, reason = self.risk_manager.check_trade_allowed(
                symbol, self.capital, self.positions
            )
            
            if allowed:
                self._open_position(symbol, signal, bar, recent_data)
            else:
                logger.debug(f"Trade not allowed: {reason}")
    
    def _open_position(self, symbol: str, signal: int, bar: pd.Series, recent_data: pd.DataFrame):
        """Open a new position"""
        
        entry_price = bar['close']
        
        # Calculate position size
        stop_loss_pct = bar.get('stop_loss', 0.02)
        stop_loss_price = entry_price * (1 - stop_loss_pct) if signal > 0 else entry_price * (1 + stop_loss_pct)
        
        # Calculate volatility
        volatility = recent_data['close'].pct_change().std() if len(recent_data) > 1 else 0.02
        
        position_size = self.risk_manager.calculate_position_size(
            self.capital,
            symbol,
            entry_price,
            stop_loss_price,
            volatility,
            self.positions
        )
        
        # Apply commission
        commission_cost = position_size * self.commission
        
        if position_size > self.capital * 0.95:  # Don't use more than 95% of capital
            position_size = self.capital * 0.95
        
        # Create position
        self.positions[symbol] = {
            'entry_time': self.current_time,
            'entry_price': entry_price,
            'side': signal,
            'size': position_size / entry_price,  # Convert to quantity
            'value': position_size,
            'stop_loss': stop_loss_price,
            'take_profit': entry_price * (1 + bar.get('take_profit', 0.03)) if signal > 0 else entry_price * (1 - bar.get('take_profit', 0.03)),
            'commission_paid': commission_cost
        }
        
        # Update capital
        self.capital -= (position_size + commission_cost)
        
        logger.debug(f"Opened {['short', '', 'long'][signal+1]} position in {symbol} at {entry_price:.2f}")
    
    def _close_position(self, symbol: str, bar: pd.Series, reason: str = 'signal'):
        """Close an existing position"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        exit_price = bar['close']
        
        # Calculate P&L
        if position['side'] > 0:  # Long position
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:  # Short position
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
        
        gross_pnl = position['value'] * pnl_pct
        
        # Apply commission
        exit_commission = position['value'] * self.commission
        net_pnl = gross_pnl - exit_commission
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'entry_time': position['entry_time'],
            'exit_time': self.current_time,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['size'],
            'gross_pnl': gross_pnl,
            'commission': position['commission_paid'] + exit_commission,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'exit_reason': reason,
            'duration': (self.current_time - position['entry_time']).total_seconds() / 60  # minutes
        })
        
        # Update capital
        self.capital += position['value'] + net_pnl
        
        # Remove position
        del self.positions[symbol]
        
        logger.debug(f"Closed {symbol} position at {exit_price:.2f}, P&L: {net_pnl:.2f}")
    
    def _check_exits(self, bar: pd.Series):
        """Check for stop loss and take profit exits"""
        
        symbols_to_close = []
        
        for symbol, position in self.positions.items():
            should_exit = False
            exit_reason = ''
            
            # Check stop loss
            if position['side'] > 0:  # Long position
                if bar['low'] <= position['stop_loss']:
                    should_exit = True
                    exit_reason = 'stop_loss'
                    # Adjust exit price for slippage
                    bar = bar.copy()
                    bar['close'] = position['stop_loss'] * 0.9995
                elif bar['high'] >= position['take_profit']:
                    should_exit = True
                    exit_reason = 'take_profit'
                    bar = bar.copy()
                    bar['close'] = position['take_profit'] * 0.9998
            else:  # Short position
                if bar['high'] >= position['stop_loss']:
                    should_exit = True
                    exit_reason = 'stop_loss'
                    bar = bar.copy()
                    bar['close'] = position['stop_loss'] * 1.0005
                elif bar['low'] <= position['take_profit']:
                    should_exit = True
                    exit_reason = 'take_profit'
                    bar = bar.copy()
                    bar['close'] = position['take_profit'] * 1.0002
            
            # Check for exit signal
            if bar.get('exit_signal', 0) == 1:
                should_exit = True
                exit_reason = 'exit_signal'
            
            if should_exit:
                symbols_to_close.append((symbol, bar, exit_reason))
        
        # Close positions
        for symbol, exit_bar, reason in symbols_to_close:
            self._close_position(symbol, exit_bar, reason)
    
    def _update_equity(self, bar: pd.Series):
        """Update position values with current prices"""
        
        for symbol, position in self.positions.items():
            current_price = bar['close']
            
            if position['side'] > 0:  # Long
                position['current_value'] = position['size'] * current_price
                position['unrealized_pnl'] = position['current_value'] - position['value']
            else:  # Short
                price_change = position['entry_price'] - current_price
                position['unrealized_pnl'] = position['size'] * price_change
                position['current_value'] = position['value'] + position['unrealized_pnl']
    
    def _calculate_equity(self, bar: pd.Series) -> float:
        """Calculate total equity"""
        
        total_equity = self.capital
        
        for position in self.positions.values():
            total_equity += position.get('current_value', position['value'])
        
        return total_equity
    
    def _calculate_positions_value(self, bar: pd.Series) -> float:
        """Calculate total positions value"""
        
        total_value = 0
        
        for position in self.positions.values():
            total_value += position.get('current_value', position['value'])
        
        return total_value
    
    def _close_all_positions(self, last_bar: pd.Series):
        """Close all remaining positions at end of backtest"""
        
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            self._close_position(symbol, last_bar, 'end_of_backtest')
    
    def _compile_results(self) -> Dict:
        """Compile backtest results"""
        
        # Convert lists to DataFrames
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        if not trades_df.empty:
            trades_df = trades_df.set_index('exit_time')
        
        if not equity_df.empty:
            equity_df = equity_df.set_index('timestamp')
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades_df, equity_df)
        
        return {
            'trades': trades_df,
            'equity_curve': equity_df,
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital
        }
    
    def _calculate_metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        
        metrics = {}
        
        if not trades_df.empty:
            # Basic metrics
            metrics['total_trades'] = len(trades_df)
            metrics['winning_trades'] = len(trades_df[trades_df['net_pnl'] > 0])
            metrics['losing_trades'] = len(trades_df[trades_df['net_pnl'] < 0])
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
            
            # P&L metrics
            metrics['total_pnl'] = trades_df['net_pnl'].sum()
            metrics['average_pnl'] = trades_df['net_pnl'].mean()
            metrics['average_win'] = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if metrics['winning_trades'] > 0 else 0
            metrics['average_loss'] = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if metrics['losing_trades'] > 0 else 0
            
            # Risk metrics
            if not equity_df.empty:
                returns = equity_df['equity'].pct_change().dropna()
                metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                metrics['max_drawdown'] = self._calculate_max_drawdown(equity_df['equity'])
            
            # Additional metrics
            metrics['profit_factor'] = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum()) if metrics['losing_trades'] > 0 else np.inf
            metrics['average_duration'] = trades_df['duration'].mean() if 'duration' in trades_df.columns else 0
            
        return metrics
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        cumulative = (1 + equity_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min() * 100 if len(drawdown) > 0 else 0