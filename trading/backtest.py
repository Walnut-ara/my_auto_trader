"""
Backtesting engine for strategy evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from config.settings import TradingConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """High-performance backtesting engine"""
    
    def __init__(self):
        self.commission = TradingConfig.BACKTEST_COMMISSION
        self.slippage = TradingConfig.BACKTEST_SLIPPAGE
        
    def run(
        self,
        data: pd.DataFrame,
        parameters: Dict,
        initial_capital: float = TradingConfig.BACKTEST_INITIAL_CAPITAL,
        position_size_pct: float = None,
        strategy_type: str = 'rsi_mean_reversion'
    ) -> Dict:
        """Run backtest with given parameters"""
        
        if data.empty:
            logger.warning("Empty data provided to backtest")
            return self._empty_results()
        
        logger.debug(f"Running backtest with {len(data)} bars")
        
        # Generate signals
        signals = self._generate_signals(data, parameters, strategy_type)
        
        # Simulate trading
        trades, equity_curve = self._simulate_trading(
            data, signals, parameters,
            initial_capital, position_size_pct
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve, initial_capital)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'signals': signals
        }
    
    def _generate_signals(
        self,
        data: pd.DataFrame,
        parameters: Dict,
        strategy_type: str
    ) -> pd.DataFrame:
        """Generate trading signals based on strategy"""
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['position'] = 0
        
        if strategy_type == 'rsi_mean_reversion':
            signals = self._rsi_signals(data, parameters, signals)
        elif strategy_type == 'trend_following':
            signals = self._trend_signals(data, parameters, signals)
        elif strategy_type == 'multi_strategy':
            # Import strategy engine for multi-strategy
            from core.strategy_engine import StrategyEngine
            engine = StrategyEngine()
            data_with_signals = engine.generate_signals(data)
            signals['signal'] = data_with_signals['signal']
        else:
            logger.warning(f"Unknown strategy type: {strategy_type}")
        
        # Generate positions from signals
        signals['position'] = signals['signal'].replace(to_replace=0, method='ffill')
        
        return signals
    
    def _rsi_signals(
        self,
        data: pd.DataFrame,
        parameters: Dict,
        signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate RSI-based signals"""
        
        # Calculate RSI
        rsi_period = parameters.get('rsi_period', 14)
        rsi = self._calculate_rsi(data['close'], rsi_period)
        
        # Get thresholds
        oversold = parameters.get('rsi_oversold', 30)
        overbought = parameters.get('rsi_overbought', 70)
        
        # Generate signals
        signals.loc[rsi < oversold, 'signal'] = 1  # Buy
        signals.loc[rsi > overbought, 'signal'] = -1  # Sell
        
        return signals
    
    def _trend_signals(
        self,
        data: pd.DataFrame,
        parameters: Dict,
        signals: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate trend-following signals"""
        
        # Calculate moving averages
        fast_period = parameters.get('ma_fast', 12)
        slow_period = parameters.get('ma_slow', 26)
        
        ma_fast = data['close'].rolling(fast_period).mean()
        ma_slow = data['close'].rolling(slow_period).mean()
        
        # Generate signals
        signals.loc[ma_fast > ma_slow, 'signal'] = 1
        signals.loc[ma_fast < ma_slow, 'signal'] = -1
        
        return signals
    
    def _simulate_trading(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        parameters: Dict,
        initial_capital: float,
        position_size_pct: Optional[float]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate trading with signals"""
        
        # Initialize tracking
        trades = []
        equity = initial_capital
        position = 0
        entry_price = 0
        entry_time = None
        
        # Position sizing
        if position_size_pct is None:
            position_size_pct = parameters.get('position_size_pct', 0.1)
        
        # Stop loss and take profit
        stop_loss_pct = parameters.get('stop_loss', 0.02)
        take_profit_pct = parameters.get('take_profit', 0.03)
        
        # Equity curve
        equity_curve = []
        
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_signal = signals['signal'].iloc[i]
            
            # Check for exit conditions if in position
            if position != 0:
                # Calculate P&L
                if position > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Check stop loss or take profit
                    if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                        # Exit position
                        exit_price = current_price * (1 - self.slippage * np.sign(position))
                        pnl = position * (exit_price - entry_price) - abs(position) * self.commission
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'pnl': pnl,
                            'pnl_pct': pnl / (position * entry_price),
                            'duration': (current_time - entry_time).total_seconds() / 60,
                            'exit_reason': 'stop_loss' if pnl_pct <= -stop_loss_pct else 'take_profit'
                        })
                        
                        equity += pnl
                        position = 0
                
                # Check for signal-based exit
                elif current_signal != 0 and np.sign(current_signal) != np.sign(position):
                    # Exit on opposite signal
                    exit_price = current_price * (1 - self.slippage * np.sign(position))
                    pnl = position * (exit_price - entry_price) - abs(position) * self.commission
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': pnl,
                        'pnl_pct': pnl / (abs(position) * entry_price),
                        'duration': (current_time - entry_time).total_seconds() / 60,
                        'exit_reason': 'signal'
                    })
                    
                    equity += pnl
                    position = 0
            
            # Check for entry signal
            if position == 0 and current_signal != 0:
                # Calculate position size
                position_value = equity * position_size_pct
                position = position_value / current_price * current_signal
                
                # Entry with slippage
                entry_price = current_price * (1 + self.slippage * current_signal)
                entry_time = current_time
                
                # Apply commission
                equity -= abs(position * entry_price) * self.commission
            
            # Record equity
            equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'position': position,
                'price': current_price
            })
        
        # Close any remaining position
        if position != 0:
            exit_price = data['close'].iloc[-1]
            pnl = position * (exit_price - entry_price) - abs(position) * self.commission
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': pnl,
                'pnl_pct': pnl / (abs(position) * entry_price),
                'duration': (data.index[-1] - entry_time).total_seconds() / 60,
                'exit_reason': 'end_of_data'
            })
            
            equity += pnl
        
        # Convert to DataFrames
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.set_index('entry_time', inplace=True)
        
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        return trades_df, equity_df
    
    def _calculate_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """Calculate performance metrics"""
        
        if trades.empty or equity_curve.empty:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate daily returns for risk metrics
        daily_equity = equity_curve['equity'].resample('D').last().dropna()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 0:
            avg_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            sharpe_ratio = np.sqrt(252) * avg_daily_return / std_daily_return if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win/loss streaks
        win_loss_series = (trades['pnl'] > 0).astype(int)
        streaks = self._calculate_streaks(win_loss_series)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_consecutive_wins': streaks['max_wins'],
            'max_consecutive_losses': streaks['max_losses'],
            'avg_trade_duration': trades['duration'].mean() if 'duration' in trades.columns else 0,
            'expectancy': (win_rate * avg_win + (1 - win_rate) * avg_loss)
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_streaks(self, win_loss_series: pd.Series) -> Dict:
        """Calculate consecutive win/loss streaks"""
        
        if win_loss_series.empty:
            return {'max_wins': 0, 'max_losses': 0}
        
        # Group consecutive values
        groups = (win_loss_series != win_loss_series.shift()).cumsum()
        grouped = win_loss_series.groupby(groups)
        
        # Calculate streak lengths
        streak_lengths = grouped.size()
        win_streaks = streak_lengths[grouped.first() == 1]
        loss_streaks = streak_lengths[grouped.first() == 0]
        
        return {
            'max_wins': win_streaks.max() if len(win_streaks) > 0 else 0,
            'max_losses': loss_streaks.max() if len(loss_streaks) > 0 else 0
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        
        return {
            'trades': pd.DataFrame(),
            'equity_curve': pd.DataFrame(),
            'metrics': self._empty_metrics(),
            'signals': pd.DataFrame()
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_trade_duration': 0,
            'expectancy': 0
        }
    
    def save_results(self, results: Dict, filepath: str):
        """Save backtest results"""
        
        # Create directory if needed
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save trades
        if not results['trades'].empty:
            trades_file = filepath + '_trades.csv'
            results['trades'].to_csv(trades_file)
        
        # Save equity curve
        if not results['equity_curve'].empty:
            equity_file = filepath + '_equity.csv'
            results['equity_curve'].to_csv(equity_file)
        
        # Save metrics
        metrics_file = filepath + '_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2)
        
        logger.info(f"Backtest results saved to {filepath}")