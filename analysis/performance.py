"""
Detailed performance analysis with monthly/weekly breakdowns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def analyze_performance(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_capital: float = 10000
    ) -> Dict:
        """Complete performance analysis with detailed breakdowns"""
        
        if trades.empty or equity_curve.empty:
            return self._empty_metrics()
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(trades, equity_curve, initial_capital)
        
        # Time-based analysis
        monthly_analysis = self._analyze_by_period(trades, equity_curve, 'M')
        weekly_analysis = self._analyze_by_period(trades, equity_curve, 'W')
        daily_analysis = self._analyze_by_period(trades, equity_curve, 'D')
        
        # Trade analysis
        trade_analysis = self._analyze_trades(trades)
        
        # Risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curve, initial_capital)
        
        # Pattern analysis
        pattern_analysis = self._analyze_patterns(trades, equity_curve)
        
        return {
            'summary': basic_metrics,
            'monthly': monthly_analysis,
            'weekly': weekly_analysis,
            'daily': daily_analysis,
            'trades': trade_analysis,
            'risk': risk_metrics,
            'patterns': pattern_analysis,
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_basic_metrics(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """Calculate basic performance metrics"""
        
        # Total return
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Period calculations
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        months = days / 30.44
        years = days / 365.25
        
        # Returns
        monthly_return = (((final_equity / initial_capital) ** (1 / months)) - 1) * 100 if months > 0 else 0
        annual_return = (((final_equity / initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Win rate
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        total_trades = len(trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Average trade
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        avg_trade = trades['pnl'].mean() if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)
        
        return {
            'total_return': round(total_return, 2),
            'monthly_return': round(monthly_return, 2),
            'annual_return': round(annual_return, 2),
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_trade': round(avg_trade, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'net_profit': round(gross_profit - gross_loss, 2),
            'trading_days': days,
            'trades_per_month': round(total_trades / months, 1) if months > 0 else 0
        }
    
    def _analyze_by_period(
        self,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        period: str
    ) -> List[Dict]:
        """Analyze performance by time period (M=Monthly, W=Weekly, D=Daily)"""
        
        # Group equity curve by period
        equity_grouped = equity_curve.resample(period).last()
        
        # Calculate period returns
        equity_grouped['return'] = equity_grouped['equity'].pct_change() * 100
        
        # Group trades by period
        trades_copy = trades.copy()
        if 'exit_time' in trades_copy.columns:
            trades_copy['period'] = pd.to_datetime(trades_copy['exit_time']).dt.to_period(period)
        else:
            trades_copy['period'] = pd.to_datetime(trades_copy.index).to_period(period)
        
        period_analysis = []
        
        for idx, row in equity_grouped.iterrows():
            period_str = idx.strftime('%Y-%m-%d')
            
            # Get trades for this period
            if not trades_copy.empty and 'period' in trades_copy.columns:
                period_trades = trades_copy[trades_copy['period'] == idx.to_period(period)]
            else:
                period_trades = pd.DataFrame()
            
            # Calculate metrics
            num_trades = len(period_trades)
            
            if num_trades > 0:
                wins = len(period_trades[period_trades['pnl'] > 0])
                losses = len(period_trades[period_trades['pnl'] < 0])
                win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
                total_pnl = period_trades['pnl'].sum()
            else:
                wins = losses = 0
                win_rate = 0
                total_pnl = 0
            
            period_data = {
                'period': period_str,
                'return': round(row['return'], 2) if pd.notna(row['return']) else 0,
                'equity': round(row['equity'], 2),
                'trades': num_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': round(win_rate, 2),
                'pnl': round(total_pnl, 2)
            }
            
            period_analysis.append(period_data)
        
        return period_analysis
    
    def _analyze_trades(self, trades: pd.DataFrame) -> Dict:
        """Detailed trade analysis"""
        
        if trades.empty:
            return {}
        
        # Duration analysis
        if 'duration' in trades.columns:
            avg_duration = trades['duration'].mean()
            max_duration = trades['duration'].max()
            min_duration = trades['duration'].min()
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Consecutive wins/losses
        trades_sorted = trades.sort_index()
        is_win = trades_sorted['pnl'] > 0
        
        # Calculate consecutive streaks
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        last_was_win = None
        
        for win in is_win:
            if last_was_win is None:
                current_streak = 1
                last_was_win = win
            elif win == last_was_win:
                current_streak += 1
            else:
                if last_was_win:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_streak = 1
                last_was_win = win
        
        # Add final streak
        if last_was_win is not None:
            if last_was_win:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
        
        # Best and worst trades
        best_trade = trades.loc[trades['pnl'].idxmax()] if not trades.empty else None
        worst_trade = trades.loc[trades['pnl'].idxmin()] if not trades.empty else None
        
        # Trade distribution
        pnl_distribution = {
            'p90': trades['pnl'].quantile(0.9),
            'p75': trades['pnl'].quantile(0.75),
            'p50': trades['pnl'].quantile(0.5),
            'p25': trades['pnl'].quantile(0.25),
            'p10': trades['pnl'].quantile(0.1)
        }
        
        return {
            'avg_duration_minutes': round(avg_duration, 1) if avg_duration else 0,
            'max_duration_minutes': round(max_duration, 1) if max_duration else 0,
            'min_duration_minutes': round(min_duration, 1) if min_duration else 0,
            'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
            'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
            'avg_consecutive_wins': round(np.mean(win_streaks), 1) if win_streaks else 0,
            'avg_consecutive_losses': round(np.mean(loss_streaks), 1) if loss_streaks else 0,
            'best_trade': {
                'pnl': round(best_trade['pnl'], 2),
                'date': str(best_trade.name) if best_trade is not None else None
            } if best_trade is not None else None,
            'worst_trade': {
                'pnl': round(worst_trade['pnl'], 2),
                'date': str(worst_trade.name) if worst_trade is not None else None
            } if worst_trade is not None else None,
            'pnl_distribution': {k: round(v, 2) for k, v in pnl_distribution.items()}
        }
    
    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame, initial_capital: float) -> Dict:
        """Calculate risk-related metrics"""
        
        # Daily returns
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # Sharpe ratio
        if len(daily_returns) > 0:
            avg_daily_return = daily_returns.mean()
            daily_std = daily_returns.std()
            sharpe_ratio = np.sqrt(252) * avg_daily_return / daily_std if daily_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * avg_daily_return / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Drawdown analysis
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for idx, dd in drawdown.items():
            if dd < -1 and not in_drawdown:  # Start of drawdown (> 1%)
                in_drawdown = True
                start_idx = idx
            elif dd >= -0.1 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if start_idx:
                    duration = (idx - start_idx).days
                    max_dd_period = drawdown[start_idx:idx].min()
                    drawdown_periods.append({
                        'start': start_idx,
                        'end': idx,
                        'duration_days': duration,
                        'max_drawdown': round(max_dd_period, 2)
                    })
        
        # Calmar ratio
        annual_return = ((equity_curve['equity'].iloc[-1] / initial_capital) ** 
                        (365 / len(equity_curve)) - 1) * 100
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = daily_returns.quantile(0.05) * 100
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = daily_returns[daily_returns <= daily_returns.quantile(0.05)].mean() * 100
        
        return {
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'var_95': round(var_95, 2),
            'cvar_95': round(cvar_95, 2),
            'volatility_annual': round(daily_std * np.sqrt(252) * 100, 2) if daily_std else 0,
            'downside_volatility': round(downside_std * np.sqrt(252) * 100, 2) if downside_std else 0,
            'drawdown_periods': drawdown_periods[-5:] if drawdown_periods else []  # Last 5 drawdowns
        }
    
    def _analyze_patterns(self, trades: pd.DataFrame, equity_curve: pd.DataFrame) -> Dict:
        """Analyze trading patterns"""
        
        if trades.empty:
            return {}
        
        # Time of day analysis (if timestamps available)
        if trades.index.dtype == 'datetime64[ns]':
            trades['hour'] = trades.index.hour
            hourly_performance = trades.groupby('hour').agg({
                'pnl': ['count', 'sum', 'mean'],
            })
            
            best_hours = hourly_performance['pnl']['mean'].nlargest(3).index.tolist()
            worst_hours = hourly_performance['pnl']['mean'].nsmallest(3).index.tolist()
        else:
            best_hours = worst_hours = []
        
        # Day of week analysis
        if trades.index.dtype == 'datetime64[ns]':
            trades['dayofweek'] = trades.index.dayofweek
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_performance = trades.groupby('dayofweek').agg({
                'pnl': ['count', 'sum', 'mean']
            })
            
            best_days = [day_names[i] for i in daily_performance['pnl']['mean'].nlargest(2).index]
            worst_days = [day_names[i] for i in daily_performance['pnl']['mean'].nsmallest(2).index]
        else:
            best_days = worst_days = []
        
        # Win/loss patterns
        if 'symbol' in trades.columns:
            symbol_performance = trades.groupby('symbol').agg({
                'pnl': ['count', 'sum', 'mean'],
            }).round(2)
            
            best_symbols = symbol_performance['pnl']['mean'].nlargest(5).to_dict()
            worst_symbols = symbol_performance['pnl']['mean'].nsmallest(5).to_dict()
        else:
            best_symbols = worst_symbols = {}
        
        return {
            'best_trading_hours': best_hours,
            'worst_trading_hours': worst_hours,
            'best_trading_days': best_days,
            'worst_trading_days': worst_days,
            'best_symbols': best_symbols,
            'worst_symbols': worst_symbols
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        
        return {
            'summary': {
                'total_return': 0,
                'monthly_return': 0,
                'annual_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0
            },
            'monthly': [],
            'weekly': [],
            'daily': [],
            'trades': {},
            'risk': {},
            'patterns': {}
        }
    
    def generate_performance_summary(self, analysis: Dict) -> str:
        """Generate a text summary of performance"""
        
        summary = analysis.get('summary', {})
        risk = analysis.get('risk', {})
        
        text = f"""
Performance Summary
==================

Returns:
- Total Return: {summary.get('total_return', 0):.2f}%
- Monthly Return: {summary.get('monthly_return', 0):.2f}%
- Annual Return: {summary.get('annual_return', 0):.2f}%

Trading Statistics:
- Total Trades: {summary.get('total_trades', 0)}
- Win Rate: {summary.get('win_rate', 0):.2f}%
- Profit Factor: {summary.get('profit_factor', 0):.2f}
- Average Trade: ${summary.get('avg_trade', 0):.2f}

Risk Metrics:
- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {risk.get('max_drawdown', 0):.2f}%
- Daily VaR (95%): {risk.get('var_95', 0):.2f}%

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return text