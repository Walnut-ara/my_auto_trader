# reporting/report_generator.py - 완성 버전

"""
Report generation system
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import json
from jinja2 import Template

from config.settings import REPORT_DIR, ReportingConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ReportGenerator:
    """Generate various trading reports"""
    
    def __init__(self):
        self.report_dir = REPORT_DIR
        self.template_dir = Path(__file__).parent / 'templates'
        
    def generate_analysis_report(self, analysis_results: list) -> Path:
        """Generate symbol analysis report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"analysis_report_{timestamp}"
        report_path = self.report_dir / report_name
        report_path.mkdir(exist_ok=True)
        
        # Generate summary
        summary_data = self._summarize_analysis(analysis_results)
        
        # Create visualizations
        self._create_analysis_charts(analysis_results, report_path)
        
        # Generate HTML report
        html_content = self._generate_analysis_html(analysis_results, summary_data)
        
        html_file = report_path / "report.html"
        html_file.write_text(html_content)
        
        # Save raw data
        json_file = report_path / "analysis_data.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report generated: {report_path}")
        return report_path
    
    def generate_optimization_report(self, optimization_results: list) -> Path:
        """Generate optimization report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"optimization_report_{timestamp}"
        report_path = self.report_dir / report_name
        report_path.mkdir(exist_ok=True)
        
        # Create visualizations
        self._create_optimization_charts(optimization_results, report_path)
        
        # Generate summary
        summary = self._summarize_optimization(optimization_results)
        
        # Generate HTML report
        html_content = self._generate_optimization_html(optimization_results, summary)
        
        html_file = report_path / "report.html"
        html_file.write_text(html_content)
        
        logger.info(f"Optimization report generated: {report_path}")
        return report_path
    
    def generate_backtest_report(
        self,
        portfolio_results: list,
        portfolio_performance: dict
    ) -> Path:
        """Generate backtest report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"backtest_report_{timestamp}"
        report_path = self.report_dir / report_name
        report_path.mkdir(exist_ok=True)
        
        # Create visualizations
        self._create_backtest_charts(portfolio_results, portfolio_performance, report_path)
        
        # Generate HTML report
        html_content = self._generate_backtest_html(portfolio_results, portfolio_performance)
        
        html_file = report_path / "report.html"
        html_file.write_text(html_content)
        
        logger.info(f"Backtest report generated: {report_path}")
        return report_path
    
    def generate_trading_report(self, trading_results: dict) -> Path:
        """Generate live trading report"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_name = f"trading_report_{timestamp}"
        report_path = self.report_dir / report_name
        report_path.mkdir(exist_ok=True)
        
        # Process results
        trades_df = trading_results.get('trades', pd.DataFrame())
        equity_df = trading_results.get('equity_curve', pd.DataFrame())
        
        if not trades_df.empty and not equity_df.empty:
            # Create visualizations
            self._create_trading_charts(trades_df, equity_df, report_path)
        
        # Generate summary
        summary = self._summarize_trading(trading_results)
        
        # Generate HTML report
        html_content = self._generate_trading_html(trading_results, summary)
        
        html_file = report_path / "report.html"
        html_file.write_text(html_content)
        
        logger.info(f"Trading report generated: {report_path}")
        return report_path
    
    def generate_daily_report(self) -> Path:
        """Generate daily summary report"""
        
        # Implement daily report logic
        # This would aggregate data from the last 24 hours
        
        timestamp = datetime.now().strftime('%Y%m%d')
        report_name = f"daily_report_{timestamp}"
        report_path = self.report_dir / report_name
        report_path.mkdir(exist_ok=True)
        
        # Placeholder for now
        html_content = "<h1>Daily Report</h1><p>Daily report generation not yet implemented.</p>"
        
        html_file = report_path / "report.html"
        html_file.write_text(html_content)
        
        return report_path
    
    # Visualization methods
    def _create_analysis_charts(self, results: list, output_dir: Path):
        """Create analysis visualization charts"""
        
        if not results:
            return
        
        # Score distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Score distribution
        scores = [r.get('market', {}).get('score', 0) for r in results if 'market' in r]
        if scores:
            axes[0, 0].hist(scores, bins=20, edgecolor='black')
            axes[0, 0].set_title('Symbol Score Distribution')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Count')
        
        # 2. Volatility vs Return
        volatilities = []
        returns = []
        symbols = []
        
        for r in results:
            if 'market' in r and 'performance' in r:
                vol = r['market'].get('volatility', {}).get('daily_volatility', 0)
                ret = r['performance'].get('total_return', 0)
                volatilities.append(vol * 100)
                returns.append(ret)
                symbols.append(r['symbol'])
        
        if volatilities and returns:
            axes[0, 1].scatter(volatilities, returns, alpha=0.6)
            axes[0, 1].set_title('Volatility vs Return')
            axes[0, 1].set_xlabel('Daily Volatility (%)')
            axes[0, 1].set_ylabel('Total Return (%)')
            
            # Add labels for top performers
            for i, symbol in enumerate(symbols[:5]):
                axes[0, 1].annotate(symbol, (volatilities[i], returns[i]))
        
        # 3. Win Rate Distribution
        win_rates = [r.get('performance', {}).get('win_rate', 0) for r in results if 'performance' in r]
        if win_rates:
            axes[1, 0].hist(win_rates, bins=20, edgecolor='black')
            axes[1, 0].set_title('Win Rate Distribution')
            axes[1, 0].set_xlabel('Win Rate (%)')
            axes[1, 0].set_ylabel('Count')
        
        # 4. Recommendation Distribution
        recommendations = [r.get('recommendation', 'Unknown') for r in results]
        rec_counts = pd.Series(recommendations).value_counts()
        if not rec_counts.empty:
            axes[1, 1].bar(rec_counts.index, rec_counts.values)
            axes[1, 1].set_title('Recommendation Distribution')
            axes[1, 1].set_xlabel('Recommendation')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_optimization_charts(self, results: list, output_dir: Path):
        """Create optimization visualization charts"""
        
        if not results:
            return
        
        # Extract successful results
        successful = [r for r in results if r.get('success', False)]
        
        if not successful:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Target vs Achieved Returns
        targets = [r.get('target_return', 0) * 100 for r in successful]
        achieved = [r.get('performance', {}).get('monthly_return', 0) for r in successful]
        symbols = [r.get('symbol', '') for r in successful]
        
        axes[0, 0].scatter(targets, achieved, alpha=0.6)
        axes[0, 0].plot([0, max(targets)], [0, max(targets)], 'r--', label='Target Line')
        axes[0, 0].set_title('Target vs Achieved Monthly Returns')
        axes[0, 0].set_xlabel('Target Return (%)')
        axes[0, 0].set_ylabel('Achieved Return (%)')
        axes[0, 0].legend()
        
        # 2. Risk-Return Scatter
        returns = achieved
        max_dds = [abs(r.get('risk_metrics', {}).get('max_drawdown', 0)) for r in successful]
        
        axes[0, 1].scatter(max_dds, returns, alpha=0.6)
        axes[0, 1].set_title('Risk-Return Profile')
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_ylabel('Monthly Return (%)')
        
        # Add labels for best performers
        if returns and max_dds:
            best_indices = sorted(range(len(returns)), key=lambda i: returns[i] - max_dds[i]/2, reverse=True)[:5]
            for idx in best_indices:
                axes[0, 1].annotate(symbols[idx], (max_dds[idx], returns[idx]))
        
        # 3. Parameter Distribution (RSI Period example)
        rsi_periods = [r.get('best_params', {}).get('rsi_period', 0) for r in successful]
        if rsi_periods:
            axes[1, 0].hist(rsi_periods, bins=15, edgecolor='black')
            axes[1, 0].set_title('Optimized RSI Period Distribution')
            axes[1, 0].set_xlabel('RSI Period')
            axes[1, 0].set_ylabel('Count')
        
        # 4. Performance Metrics Heatmap
        metrics_data = []
        metric_names = ['monthly_return', 'win_rate', 'profit_factor', 'sharpe_ratio']
        
        for r in successful[:20]:  # Top 20
            row = [r.get('symbol', '')]
            perf = r.get('performance', {})
            risk = r.get('risk_metrics', {})
            
            row.append(perf.get('monthly_return', 0))
            row.append(perf.get('win_rate', 0))
            row.append(perf.get('profit_factor', 0))
            row.append(risk.get('sharpe_ratio', 0))
            
            metrics_data.append(row)
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data, columns=['Symbol'] + metric_names)
            df_metrics.set_index('Symbol', inplace=True)
            
            # Normalize for heatmap
            df_normalized = (df_metrics - df_metrics.min()) / (df_metrics.max() - df_metrics.min())
            
            sns.heatmap(df_normalized.T, cmap='YlOrRd', cbar_kws={'label': 'Normalized Value'}, ax=axes[1, 1])
            axes[1, 1].set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'optimization_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_backtest_charts(self, results: list, performance: dict, output_dir: Path):
        """Create backtest visualization charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Portfolio Equity Curve
        combined_equity = pd.DataFrame()
        for r in results:
            if 'equity_curve' in r and not r['equity_curve'].empty:
                symbol = r['symbol']
                combined_equity[symbol] = r['equity_curve']['equity']
        
        if not combined_equity.empty:
            portfolio_equity = combined_equity.sum(axis=1)
            axes[0, 0].plot(portfolio_equity.index, portfolio_equity.values, linewidth=2)
            axes[0, 0].set_title('Portfolio Equity Curve')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Equity ($)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Monthly Returns Bar Chart
        monthly_data = performance.get('monthly', [])
        if monthly_data:
            months = [m['period'] for m in monthly_data[-12:]]
            returns = [m['return'] for m in monthly_data[-12:]]
            
            colors = ['green' if r > 0 else 'red' for r in returns]
            axes[0, 1].bar(months, returns, color=colors)
            axes[0, 1].set_title('Monthly Returns (Last 12 Months)')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Return (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Drawdown Chart
        if not combined_equity.empty:
            portfolio_equity = combined_equity.sum(axis=1)
            cumulative = portfolio_equity / portfolio_equity.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max * 100
            
            axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            axes[1, 0].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            axes[1, 0].set_title('Portfolio Drawdown')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Symbol Performance Comparison
        symbol_returns = []
        symbol_names = []
        
        for r in results:
            if 'performance' in r:
                symbol_names.append(r['symbol'])
                symbol_returns.append(r['performance']['summary'].get('total_return', 0))
        
        if symbol_returns:
            # Sort by returns
            sorted_data = sorted(zip(symbol_names, symbol_returns), key=lambda x: x[1], reverse=True)
            symbol_names, symbol_returns = zip(*sorted_data[:15])  # Top 15
            
            colors = ['green' if r > 0 else 'red' for r in symbol_returns]
            axes[1, 1].barh(symbol_names, symbol_returns, color=colors)
            axes[1, 1].set_title('Symbol Performance Comparison')
            axes[1, 1].set_xlabel('Total Return (%)')
            axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'backtest_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed charts
        self._create_detailed_performance_chart(performance, output_dir)
    
    def _create_detailed_performance_chart(self, performance: dict, output_dir: Path):
        """Create detailed performance analysis chart"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Trade Distribution
        trades = performance.get('trades', {})
        if trades:
            trade_dist = trades.get('pnl_distribution', {})
            if trade_dist:
                percentiles = ['p10', 'p25', 'p50', 'p75', 'p90']
                values = [trade_dist.get(p, 0) for p in percentiles]
                
                axes[0, 0].bar(percentiles, values)
                axes[0, 0].set_title('Trade PnL Distribution')
                axes[0, 0].set_xlabel('Percentile')
                axes[0, 0].set_ylabel('PnL ($)')
                axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. Win/Loss Analysis
        summary = performance.get('summary', {})
        if summary:
            wins = summary.get('winning_trades', 0)
            losses = summary.get('losing_trades', 0)
            
            if wins + losses > 0:
                axes[0, 1].pie([wins, losses], labels=['Wins', 'Losses'], 
                             autopct='%1.1f%%', startangle=90,
                             colors=['green', 'red'])
                axes[0, 1].set_title('Win/Loss Distribution')
        
        # 3. Risk Metrics
        risk = performance.get('risk', {})
        if risk:
            metrics = ['Sharpe', 'Sortino', 'Calmar']
            values = [
                risk.get('sharpe_ratio', 0),
                risk.get('sortino_ratio', 0),
                risk.get('calmar_ratio', 0)
            ]
            
            axes[0, 2].bar(metrics, values)
            axes[0, 2].set_title('Risk-Adjusted Returns')
            axes[0, 2].set_ylabel('Ratio')
            axes[0, 2].axhline(y=1, color='red', linestyle='--', linewidth=1, label='Benchmark')
            axes[0, 2].legend()
        
        # 4. Pattern Analysis - Best Trading Hours
        patterns = performance.get('patterns', {})
        if patterns and 'best_trading_hours' in patterns:
            hours = patterns['best_trading_hours']
            if hours:
                hour_performance = {h: 0 for h in range(24)}
                # This would need actual hour performance data
                axes[1, 0].bar(range(24), [0]*24)  # Placeholder
                axes[1, 0].set_title('Performance by Hour')
                axes[1, 0].set_xlabel('Hour (UTC)')
                axes[1, 0].set_ylabel('Avg Return (%)')
        
        # 5. Consecutive Streaks
        if trades:
            max_wins = trades.get('max_consecutive_wins', 0)
            max_losses = trades.get('max_consecutive_losses', 0)
            avg_wins = trades.get('avg_consecutive_wins', 0)
            avg_losses = trades.get('avg_consecutive_losses', 0)
            
            x = ['Max Wins', 'Max Losses', 'Avg Wins', 'Avg Losses']
            y = [max_wins, max_losses, avg_wins, avg_losses]
            colors = ['green', 'red', 'lightgreen', 'lightcoral']
            
            axes[1, 1].bar(x, y, color=colors)
            axes[1, 1].set_title('Consecutive Trade Streaks')
            axes[1, 1].set_ylabel('Number of Trades')
        
        # 6. Monthly Win Rate Heatmap
        monthly_data = performance.get('monthly', [])
        if len(monthly_data) >= 12:
            # Create month x metric heatmap
            months = [m['period'] for m in monthly_data[-12:]]
            win_rates = [m.get('win_rate', 0) for m in monthly_data[-12:]]
            returns = [m.get('return', 0) for m in monthly_data[-12:]]
            
            heatmap_data = pd.DataFrame({
                'Win Rate': win_rates,
                'Return': returns
            }, index=months).T
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=axes[1, 2])
            axes[1, 2].set_title('Monthly Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detailed_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trading_charts(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame, output_dir: Path):
        """Create live trading charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        axes[0, 0].plot(equity_df.index, equity_df['equity'], label='Total Equity', linewidth=2)
        axes[0, 0].plot(equity_df.index, equity_df['cash'], label='Cash', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('Account Equity Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative PnL
        if not trades_df.empty:
            trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
            axes[0, 1].plot(trades_df.index, trades_df['cumulative_pnl'], linewidth=2)
            axes[0, 1].set_title('Cumulative P&L')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('Cumulative P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. Trade Distribution
        if not trades_df.empty:
            axes[1, 0].hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
            
            # Add statistics
            mean_pnl = trades_df['pnl'].mean()
            axes[1, 0].axvline(x=mean_pnl, color='green', linestyle='--', linewidth=1, label=f'Mean: ${mean_pnl:.2f}')
            axes[1, 0].legend()
        
        # 4. Symbol Performance
        if not trades_df.empty and 'symbol' in trades_df.columns:
            symbol_pnl = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=True)
            
            colors = ['green' if pnl > 0 else 'red' for pnl in symbol_pnl.values]
            axes[1, 1].barh(symbol_pnl.index, symbol_pnl.values, color=colors)
            axes[1, 1].set_title('P&L by Symbol')
            axes[1, 1].set_xlabel('Total P&L ($)')
            axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'trading_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary methods
    def _summarize_analysis(self, results: list) -> dict:
        """Summarize analysis results"""
        
        if not results:
            return {}
        
        total_symbols = len(results)
        
        # Grade distribution
        grades = [r.get('recommendation', 'Unknown') for r in results]
        grade_counts = pd.Series(grades).value_counts().to_dict()
        
        # Average metrics
        avg_score = sum(r.get('market', {}).get('score', 0) for r in results) / total_symbols if total_symbols > 0 else 0
        avg_return = sum(r.get('performance', {}).get('total_return', 0) for r in results) / total_symbols if total_symbols > 0 else 0
        avg_win_rate = sum(r.get('performance', {}).get('win_rate', 0) for r in results) / total_symbols if total_symbols > 0 else 0
        
        # Top performers
        top_symbols = sorted(results, key=lambda x: x.get('performance', {}).get('total_return', 0), reverse=True)[:5]
        
        return {
            'total_symbols': total_symbols,
            'grade_distribution': grade_counts,
            'average_score': avg_score,
            'average_return': avg_return,
            'average_win_rate': avg_win_rate,
            'top_performers': [s['symbol'] for s in top_symbols],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _summarize_optimization(self, results: list) -> dict:
        """Summarize optimization results"""
        
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        if not successful:
            return {
                'total_optimizations': len(results),
                'successful': 0,
                'failed': len(failed),
                'average_return': 0,
                'target_achievement_rate': 0
            }
        
        # Calculate averages
        avg_return = sum(r.get('performance', {}).get('monthly_return', 0) for r in successful) / len(successful)
        avg_sharpe = sum(r.get('risk_metrics', {}).get('sharpe_ratio', 0) for r in successful) / len(successful)
        
        # Target achievement
        achieved_target = sum(1 for r in successful if r.get('target_achieved', False))
        achievement_rate = achieved_target / len(successful) * 100 if successful else 0
        
        # Best performer
        best = max(successful, key=lambda x: x.get('performance', {}).get('monthly_return', 0))
        
        return {
            'total_optimizations': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'target_achievement_rate': achievement_rate,
            'best_performer': {
                'symbol': best.get('symbol', 'Unknown'),
                'return': best.get('performance', {}).get('monthly_return', 0),
                'parameters': best.get('best_params', {})
            },
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _summarize_trading(self, results: dict) -> dict:
        """Summarize trading results"""
        
        trades_df = results.get('trades', pd.DataFrame())
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'running_hours': results.get('running_time', 0)
            }
        
        # Calculate metrics
        total_trades = len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'average_pnl': trades_df['pnl'].mean(),
            'best_trade': {
                'symbol': best_trade.get('symbol', 'Unknown'),
                'pnl': best_trade['pnl'],
                'time': str(best_trade.name)
            },
            'worst_trade': {
                'symbol': worst_trade.get('symbol', 'Unknown'),
                'pnl': worst_trade['pnl'],
                'time': str(worst_trade.name)
            },
            'running_hours': results.get('running_time', 0),
            'active_positions': len(results.get('positions', {}))
        }
    
    # HTML generation methods
    def _generate_analysis_html(self, results: list, summary: dict) -> str:
        """Generate HTML report for analysis"""
        
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Symbol Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
# reporting/report_generator.py 계속

        h1, h2 { color: #333; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .chart { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .grade-A { color: #28a745; font-weight: bold; }
        .grade-B { color: #17a2b8; font-weight: bold; }
        .grade-C { color: #ffc107; font-weight: bold; }
        .grade-D { color: #fd7e14; font-weight: bold; }
        .grade-F { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Symbol Analysis Report</h1>
        <p>Generated: {{ summary.analysis_date }}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value">{{ summary.total_symbols }}</div>
                <div class="metric-label">Total Symbols</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f" | format(summary.average_score) }}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f%%" | format(summary.average_return) }}</div>
                <div class="metric-label">Average Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f%%" | format(summary.average_win_rate) }}</div>
                <div class="metric-label">Average Win Rate</div>
            </div>
        </div>
        
        <div class="chart">
            <h2>Analysis Summary</h2>
            <img src="analysis_summary.png" alt="Analysis Summary">
        </div>
        
        <h2>Symbol Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Grade</th>
                    <th>Score</th>
                    <th>Volatility</th>
                    <th>Trend</th>
                    <th>Return</th>
                    <th>Win Rate</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results[:50] %}
                <tr>
                    <td><strong>{{ result.symbol }}</strong></td>
                    <td class="grade-{{ result.recommendation.split()[0] }}">{{ result.recommendation.split()[0] }}</td>
                    <td>{{ "%.1f" | format(result.market.score|default(0)) }}</td>
                    <td>{{ "%.2f%%" | format(result.market.volatility.daily_volatility|default(0) * 100) }}</td>
                    <td>{{ result.market.trend.current|default('N/A') }}</td>
                    <td>{{ "%.2f%%" | format(result.performance.total_return|default(0)) }}</td>
                    <td>{{ "%.1f%%" | format(result.performance.win_rate|default(0)) }}</td>
                    <td>{{ result.recommendation }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        from jinja2 import Template
        template = Template(template)
        return template.render(summary=summary, results=results)
    
    def _generate_optimization_html(self, results: list, summary: dict) -> str:
        """Generate HTML report for optimization"""
        
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimization Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; }
        .success { color: #28a745; }
        .failed { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .chart { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .params { font-size: 12px; color: #666; }
        .achieved { color: #28a745; font-weight: bold; }
        .not-achieved { color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Report</h1>
        <p>Generated: {{ summary.optimization_date }}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value">{{ summary.total_optimizations }}</div>
                <div class="metric-label">Total Optimizations</div>
            </div>
            <div class="metric">
                <div class="metric-value success">{{ summary.successful }}</div>
                <div class="metric-label">Successful</div>
            </div>
            <div class="metric">
                <div class="metric-value failed">{{ summary.failed }}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f%%" | format(summary.average_return) }}</div>
                <div class="metric-label">Average Return</div>
            </div>
        </div>
        
        <div class="chart">
            <h2>Optimization Results</h2>
            <img src="optimization_summary.png" alt="Optimization Summary">
        </div>
        
        {% if summary.best_performer %}
        <h2>Best Performer</h2>
        <div class="summary">
            <p><strong>Symbol:</strong> {{ summary.best_performer.symbol }}</p>
            <p><strong>Monthly Return:</strong> {{ "%.2f%%" | format(summary.best_performer.return) }}</p>
            <p><strong>Parameters:</strong></p>
            <ul>
                {% for param, value in summary.best_performer.parameters.items() %}
                <li>{{ param }}: {{ value }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Target Return</th>
                    <th>Achieved Return</th>
                    <th>Win Rate</th>
                    <th>Sharpe Ratio</th>
                    <th>Max Drawdown</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td><strong>{{ result.symbol }}</strong></td>
                    <td>{{ "%.1f%%" | format(result.target_return * 100) }}</td>
                    <td>{{ "%.2f%%" | format(result.performance.monthly_return|default(0)) }}</td>
                    <td>{{ "%.1f%%" | format(result.performance.win_rate|default(0)) }}</td>
                    <td>{{ "%.2f" | format(result.risk_metrics.sharpe_ratio|default(0)) }}</td>
                    <td>{{ "%.2f%%" | format(result.risk_metrics.max_drawdown|default(0)) }}</td>
                    <td class="{% if result.target_achieved %}achieved{% else %}not-achieved{% endif %}">
                        {% if result.target_achieved %}✓ Achieved{% else %}✗ Not Achieved{% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        from jinja2 import Template
        template = Template(template)
        return template.render(summary=summary, results=results)
    
    def _generate_backtest_html(self, results: list, performance: dict) -> str:
        """Generate HTML report for backtest"""
        
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .chart { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .monthly-table { margin-top: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Backtest Report</h1>
        <p>Generated: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
        
        <div class="summary">
            <h2>Portfolio Summary</h2>
            <div class="metric">
                <div class="metric-value {% if performance.summary.total_return >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "%.2f%%" | format(performance.summary.total_return) }}
                </div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f%%" | format(performance.summary.monthly_return) }}</div>
                <div class="metric-label">Monthly Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ performance.summary.total_trades }}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f%%" | format(performance.summary.win_rate) }}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.2f" | format(performance.risk.sharpe_ratio) }}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{{ "%.2f%%" | format(performance.risk.max_drawdown) }}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
        
        <div class="chart">
            <h2>Performance Charts</h2>
            <img src="backtest_summary.png" alt="Backtest Summary">
        </div>
        
        <div class="chart">
            <h2>Detailed Analysis</h2>
            <img src="detailed_performance.png" alt="Detailed Performance">
        </div>
        
        <h2>Monthly Performance</h2>
        <table class="monthly-table">
            <thead>
                <tr>
                    <th>Month</th>
                    <th>Return</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>PnL</th>
                </tr>
            </thead>
            <tbody>
                {% for month in performance.monthly[-12:] %}
                <tr>
                    <td>{{ month.period }}</td>
                    <td class="{% if month.return >= 0 %}positive{% else %}negative{% endif %}">
                        {{ "%.2f%%" | format(month.return) }}
                    </td>
                    <td>{{ month.trades }}</td>
                    <td>{{ "%.1f%%" | format(month.win_rate) }}</td>
                    <td class="{% if month.pnl >= 0 %}positive{% else %}negative{% endif %}">
                        ${{ "%.2f" | format(month.pnl) }}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h2>Symbol Performance</h2>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Total Return</th>
                    <th>Monthly Return</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Sharpe Ratio</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td><strong>{{ result.symbol }}</strong></td>
                    <td class="{% if result.performance.summary.total_return >= 0 %}positive{% else %}negative{% endif %}">
                        {{ "%.2f%%" | format(result.performance.summary.total_return) }}
                    </td>
                    <td>{{ "%.2f%%" | format(result.performance.summary.monthly_return) }}</td>
                    <td>{{ result.performance.summary.total_trades }}</td>
                    <td>{{ "%.1f%%" | format(result.performance.summary.win_rate) }}</td>
                    <td>{{ "%.2f" | format(result.performance.risk.sharpe_ratio|default(0)) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        from jinja2 import Template
        from datetime import datetime
        template = Template(template)
        return template.render(
            results=results, 
            performance=performance,
            datetime=datetime
        )
    
    def _generate_trading_html(self, results: dict, summary: dict) -> str:
        """Generate HTML report for trading"""
        
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2 { color: #333; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 14px; color: #666; }
        .positive { color: #28a745; }
        .negative { color: #dc3545; }
        .chart { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .trade-info { background-color: #e9ecef; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live Trading Report</h1>
        <p>Generated: {{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</p>
        
        <div class="summary">
            <h2>Trading Summary</h2>
            <div class="metric">
                <div class="metric-value">{{ summary.total_trades }}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value {% if summary.total_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ${{ "%.2f" | format(summary.total_pnl) }}
                </div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f%%" | format(summary.win_rate) }}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ summary.active_positions }}</div>
                <div class="metric-label">Active Positions</div>
            </div>
        </div>
        
        {% if summary.best_trade %}
        <div class="trade-info">
            <h3>Best Trade</h3>
            <p><strong>Symbol:</strong> {{ summary.best_trade.symbol }}</p>
            <p><strong>P&L:</strong> <span class="positive">${{ "%.2f" | format(summary.best_trade.pnl) }}</span></p>
            <p><strong>Time:</strong> {{ summary.best_trade.time }}</p>
        </div>
        {% endif %}
        
        {% if summary.worst_trade %}
        <div class="trade-info">
            <h3>Worst Trade</h3>
            <p><strong>Symbol:</strong> {{ summary.worst_trade.symbol }}</p>
            <p><strong>P&L:</strong> <span class="negative">${{ "%.2f" | format(summary.worst_trade.pnl) }}</span></p>
            <p><strong>Time:</strong> {{ summary.worst_trade.time }}</p>
        </div>
        {% endif %}
        
        <div class="chart">
            <h2>Trading Performance</h2>
            <img src="trading_summary.png" alt="Trading Summary">
        </div>
    </div>
</body>
</html>
        """
        
        from jinja2 import Template
        from datetime import datetime
        template = Template(template)
        return template.render(
            results=results,
            summary=summary,
            datetime=datetime
        )