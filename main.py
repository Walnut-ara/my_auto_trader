#!/usr/bin/env python3
"""
Crypto Trading System - Main Entry Point
Advanced automated cryptocurrency trading system with AI optimization
"""

import sys
import click
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    TradingConfig, OptimizationConfig, SymbolConfig, 
    LoggingConfig, FeatureFlags
)
from core.data_manager import DataManager
from analysis.performance import PerformanceAnalyzer
from analysis.market_analyzer import MarketAnalyzer
from analysis.optimizer import AdvancedOptimizer
from trading.backtest import BacktestEngine
from trading.live_trader import LiveTrader
from reporting.report_generator import ReportGenerator
from dashboard.app import create_app
from utils.logger import setup_logging

# Setup logging
logger = setup_logging()


class CryptoTradingSystem:
    """Main trading system controller"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.data_manager = DataManager()
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_analyzer = MarketAnalyzer(self.data_manager)
        self.optimizer = AdvancedOptimizer(self.data_manager)
        self.backtest_engine = BacktestEngine()
        self.report_generator = ReportGenerator()
        
        logger.info("Crypto Trading System initialized")
    
    def analyze_symbols(
        self,
        symbols: List[str],
        period_days: int = 180,
        save_report: bool = True
    ):
        """Comprehensive symbol analysis"""
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Analyzing {len(symbols)} symbols...")
        click.echo(f"{'='*60}\n")
        
        analysis_results = []
        
        with click.progressbar(symbols, label='Analyzing symbols') as bar:
            for symbol in bar:
                try:
                    # Get historical data
                    data = self.data_manager.get_historical_data(
                        symbol,
                        TradingConfig.DEFAULT_TIMEFRAME,
                        datetime.now() - timedelta(days=period_days),
                        datetime.now()
                    )
                    
                    if data.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Market analysis
                    market_analysis = self.market_analyzer.analyze_symbol(symbol, data)
                    
                    # Quick backtest with default parameters
                    default_params = {
                        'rsi_period': 14,
                        'rsi_oversold': 30,
                        'rsi_overbought': 70,
                        'stop_loss': market_analysis['volatility']['atr_percentage'] * 1.5,
                        'take_profit': market_analysis['volatility']['atr_percentage'] * 3,
                        'position_size_pct': 0.1
                    }
                    
                    backtest_results = self.backtest_engine.run(
                        data, default_params, 
                        initial_capital=TradingConfig.BACKTEST_INITIAL_CAPITAL
                    )
                    
                    # Performance analysis
                    performance = self.performance_analyzer.analyze_performance(
                        backtest_results['trades'],
                        backtest_results['equity_curve']
                    )
                    
                    analysis_results.append({
                        'symbol': symbol,
                        'market': market_analysis,
                        'performance': performance['summary'],
                        'recommendation': self._generate_recommendation(
                            market_analysis, performance['summary']
                        )
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
                    continue
        
        # Display results
        self._display_analysis_results(analysis_results)
        
        # Save report
        if save_report:
            report_path = self.report_generator.generate_analysis_report(
                analysis_results
            )
            click.echo(f"\nReport saved to: {report_path}")
    
    def optimize_symbols(
        self,
        symbols: List[str],
        target_return: Optional[float] = None,
        aggressive: bool = False,
        trials: Optional[int] = None
    ):
        """Optimize trading parameters for symbols"""
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Optimizing {len(symbols)} symbols...")
        if target_return:
            click.echo(f"Target monthly return: {target_return:.1%}")
        click.echo(f"Mode: {'Aggressive' if aggressive else 'Conservative'}")
        click.echo(f"{'='*60}\n")
        
        optimization_results = []
        
        for symbol in symbols:
            click.echo(f"\nOptimizing {symbol}...")
            
            try:
                # Market analysis first
                data = self.data_manager.get_historical_data(
                    symbol,
                    TradingConfig.DEFAULT_TIMEFRAME,
                    datetime.now() - timedelta(days=180),
                    datetime.now()
                )
                
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                market_analysis = self.market_analyzer.analyze_symbol(symbol, data)
                
                # Determine target return if not specified
                if target_return is None:
                    recommended_return = self._get_recommended_return(
                        market_analysis, aggressive
                    )
                else:
                    recommended_return = target_return
                
                # Run optimization
                result = self.optimizer.optimize(
                    symbol=symbol,
                    data=data,
                    target_return=recommended_return,
                    n_trials=trials or (
                        OptimizationConfig.AGGRESSIVE_N_TRIALS if aggressive 
                        else OptimizationConfig.DEFAULT_N_TRIALS
                    ),
                    param_ranges=OptimizationConfig.PARAMETER_RANGES[
                        'aggressive' if aggressive else 'moderate'
                    ]
                )
                
                # Save optimized parameters
                if result['success']:
                    self.data_manager.save_optimized_parameters(
                        symbol,
                        result['best_params'],
                        result['performance']
                    )
                
                optimization_results.append(result)
                
                # Display result
                self._display_optimization_result(result)
                
            except Exception as e:
                logger.error(f"Failed to optimize {symbol}: {e}")
                continue
        
        # Summary report
        self._display_optimization_summary(optimization_results)
        
        # Save detailed report
        report_path = self.report_generator.generate_optimization_report(
            optimization_results
        )
        click.echo(f"\nOptimization report saved to: {report_path}")
    
    def backtest_portfolio(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_optimized: bool = True
    ):
        """Run backtest on portfolio"""
        
        if symbols is None:
            # Load optimized symbols
            symbols = self._get_optimized_symbols()
            if not symbols:
                click.echo("No optimized symbols found. Please run optimization first.")
                return
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Backtesting portfolio with {len(symbols)} symbols")
        click.echo(f"{'='*60}\n")
        
        # Date range
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
        
        if start_date is None:
            start_date = end_date - timedelta(days=90)
        else:
            start_date = pd.to_datetime(start_date)
        
        portfolio_results = []
        
        with click.progressbar(symbols, label='Backtesting symbols') as bar:
            for symbol in bar:
                try:
                    # Get data
                    data = self.data_manager.get_historical_data(
                        symbol,
                        TradingConfig.DEFAULT_TIMEFRAME,
                        start_date,
                        end_date
                    )
                    
                    if data.empty:
                        continue
                    
                    # Get parameters
                    if use_optimized:
                        params_data = self.data_manager.load_optimized_parameters(symbol)
                        if params_data:
                            params = params_data['parameters']
                        else:
                            logger.warning(f"No optimized parameters for {symbol}, using defaults")
                            params = self._get_default_params()
                    else:
                        params = self._get_default_params()
                    
                    # Run backtest
                    results = self.backtest_engine.run(
                        data, params,
                        initial_capital=TradingConfig.BACKTEST_INITIAL_CAPITAL / len(symbols)
                    )
                    
                    # Analyze performance
                    performance = self.performance_analyzer.analyze_performance(
                        results['trades'],
                        results['equity_curve'],
                        initial_capital=TradingConfig.BACKTEST_INITIAL_CAPITAL / len(symbols)
                    )
                    
                    portfolio_results.append({
                        'symbol': symbol,
                        'parameters': params,
                        'performance': performance,
                        'trades': results['trades'],
                        'equity_curve': results['equity_curve']
                    })
                    
                except Exception as e:
                    logger.error(f"Backtest failed for {symbol}: {e}")
                    continue
        
        # Portfolio analysis
        portfolio_performance = self._analyze_portfolio_performance(portfolio_results)
        
        # Display results
        self._display_portfolio_results(portfolio_performance)
        
        # Generate detailed report
        report_path = self.report_generator.generate_backtest_report(
            portfolio_results,
            portfolio_performance
        )
        click.echo(f"\nBacktest report saved to: {report_path}")
    
    def recommend_portfolio(self, top_n: int = 50):
        """Analyze and recommend optimal portfolio"""
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Analyzing top {top_n} symbols for portfolio recommendation")
        click.echo(f"{'='*60}\n")
        
        # Get top volume symbols
        symbols = self.data_manager.get_top_volume_symbols(top_n)
        
        # Analyze all symbols
        symbol_scores = []
        
        with click.progressbar(symbols, label='Analyzing symbols') as bar:
            for symbol in bar:
                try:
                    score = self.market_analyzer.score_symbol(symbol)
                    if score:
                        symbol_scores.append(score)
                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
                    continue
        
        # Rank and select symbols
        symbol_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Portfolio construction
        portfolio = self._construct_portfolio(symbol_scores)
        
        # Display recommendations
        self._display_portfolio_recommendations(portfolio)
        
        # Save recommendations
        self.data_manager.save_analysis_data(
            'portfolio',
            portfolio,
            'recommendations'
        )
        
        # Ask if user wants to optimize
        if click.confirm("\nWould you like to optimize these symbols?"):
            selected_symbols = [s['symbol'] for s in portfolio['selected']]
            self.optimize_symbols(selected_symbols)
    
    def start_trading(
        self,
        symbols: Optional[List[str]] = None,
        paper_trading: bool = True
    ):
        """Start live or paper trading"""
        
        if not paper_trading and not FeatureFlags.ENABLE_LIVE_TRADING:
            click.echo("Live trading is disabled. Please enable it in settings.")
            return
        
        if symbols is None:
            symbols = self._get_optimized_symbols()
            if not symbols:
                click.echo("No optimized symbols found. Please run optimization first.")
                return
        
        click.echo(f"\n{'='*60}")
        click.echo(f"Starting {'paper' if paper_trading else 'live'} trading")
        click.echo(f"Symbols: {', '.join(symbols)}")
        click.echo(f"{'='*60}\n")
        
        # Initialize trader
        trader = LiveTrader(
            data_manager=self.data_manager,
            paper_trading=paper_trading
        )
        
        # Load parameters for each symbol
        for symbol in symbols:
            params_data = self.data_manager.load_optimized_parameters(symbol)
            if params_data:
                trader.add_symbol(symbol, params_data['parameters'])
            else:
                logger.warning(f"No parameters for {symbol}, skipping")
        
        # Start trading
        try:
            trader.start()
            click.echo("\nTrading started. Press Ctrl+C to stop.")
            
            # Keep running
            while True:
                trader.update_status()
                time.sleep(60)
                
        except KeyboardInterrupt:
            click.echo("\nStopping trader...")
            trader.stop()
            
            # Generate final report
            report_path = self.report_generator.generate_trading_report(
                trader.get_results()
            )
            click.echo(f"\nTrading report saved to: {report_path}")
    
    def launch_dashboard(self):
        """Launch web dashboard"""
        
        click.echo(f"\n{'='*60}")
        click.echo("Launching Trading Dashboard")
        click.echo(f"{'='*60}\n")
        
        app = create_app(self)
        
        click.echo(f"Dashboard starting at http://localhost:{DashboardConfig.PORT}")
        click.echo("Press Ctrl+C to stop")
        
        app.run(
            host=DashboardConfig.HOST,
            port=DashboardConfig.PORT,
            debug=DashboardConfig.DEBUG
        )
    
    # Helper methods
    def _generate_recommendation(self, market_analysis: dict, performance: dict) -> str:
        """Generate trading recommendation"""
        
        score = 0
        
        # Volatility score
        if 0.02 <= market_analysis['volatility']['daily_volatility'] <= 0.05:
            score += 30
        elif market_analysis['volatility']['daily_volatility'] < 0.02:
            score += 10
        else:
            score += 20
        
        # Trend score
        if market_analysis['trend']['strength'] == 'Strong':
            score += 30
        else:
            score += 15
        
        # Performance score
        if performance['profit_factor'] > 1.5:
            score += 40
        elif performance['profit_factor'] > 1.2:
            score += 25
        else:
            score += 10
        
        if score >= 80:
            return "Highly Recommended"
        elif score >= 60:
            return "Recommended"
        elif score >= 40:
            return "Moderate"
        else:
            return "Not Recommended"
    
    def _get_recommended_return(self, market_analysis: dict, aggressive: bool) -> float:
        """Get recommended target return based on market analysis"""
        
        volatility = market_analysis['volatility']['daily_volatility']
        
        if volatility < 0.02:  # Low volatility
            base_return = 0.01
        elif volatility < 0.03:  # Medium volatility
            base_return = 0.02
        elif volatility < 0.05:  # High volatility
            base_return = 0.03
        else:  # Very high volatility
            base_return = 0.05
        
        if aggressive:
            return min(base_return * 1.5, 0.15)
        else:
            return base_return
    
    def _get_optimized_symbols(self) -> List[str]:
        """Get list of optimized symbols"""
        
        optimized_dir = DATA_DIR / 'optimized'
        if not optimized_dir.exists():
            return []
        
        symbols = []
        for file in optimized_dir.glob('*_params.json'):
            symbol = file.stem.replace('_params', '')
            symbols.append(symbol)
        
        return sorted(symbols)
    
    def _get_default_params(self) -> dict:
        """Get default trading parameters"""
        
        return {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stop_loss': TradingConfig.DEFAULT_STOP_LOSS_PCT,
            'take_profit': TradingConfig.DEFAULT_TAKE_PROFIT_PCT,
            'position_size_pct': 0.1
        }
    
    def _display_analysis_results(self, results: List[dict]):
        """Display analysis results in table format"""
        
        if not results:
            click.echo("No analysis results to display")
            return
        
        # Prepare data for table
        table_data = []
        for r in results[:20]:  # Top 20
            table_data.append([
                r['symbol'],
                f"{r['market']['trend']['current']}",
                f"{r['market']['volatility']['daily_volatility']:.2%}",
                f"{r['performance']['total_return']:.2f}%",
                f"{r['performance']['win_rate']:.1f}%",
                f"{r['performance']['profit_factor']:.2f}",
                r['recommendation']
            ])
        
        headers = ['Symbol', 'Trend', 'Volatility', 'Return', 'Win Rate', 'PF', 'Recommendation']
        
        click.echo("\nAnalysis Results:")
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def _display_optimization_result(self, result: dict):
        """Display single optimization result"""
        
        if result['success']:
            click.echo(f"✓ {result['symbol']} optimization complete:")
            click.echo(f"  Target return: {result['target_return']:.1%}")
            click.echo(f"  Achieved return: {result['performance']['monthly_return']:.2%}")
            click.echo(f"  Win rate: {result['performance']['win_rate']:.1f}%")
            click.echo(f"  Best parameters: {result['best_params']}")
        else:
            click.echo(f"✗ {result['symbol']} optimization failed: {result.get('error', 'Unknown error')}")
    
    def _display_optimization_summary(self, results: List[dict]):
        """Display optimization summary"""
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        click.echo(f"\n{'='*60}")
        click.echo("Optimization Summary")
        click.echo(f"{'='*60}")
        click.echo(f"Total: {len(results)}")
        click.echo(f"Successful: {len(successful)}")
        click.echo(f"Failed: {len(failed)}")
        
        if successful:
            avg_return = sum(r['performance']['monthly_return'] for r in successful) / len(successful)
            click.echo(f"\nAverage monthly return: {avg_return:.2%}")
            
            # Top performers
            top_performers = sorted(successful, 
                                  key=lambda x: x['performance']['monthly_return'], 
                                  reverse=True)[:5]
            
            click.echo("\nTop 5 performers:")
            for r in top_performers:
                click.echo(f"  {r['symbol']}: {r['performance']['monthly_return']:.2%}")
    
    def _analyze_portfolio_performance(self, results: List[dict]) -> dict:
        """Analyze combined portfolio performance"""
        
        if not results:
            return {}
        
        # Combine all trades
        all_trades = pd.concat([r['trades'] for r in results if not r['trades'].empty])
        
        # Combine equity curves
        combined_equity = pd.DataFrame()
        for r in results:
            if not r['equity_curve'].empty:
                combined_equity[r['symbol']] = r['equity_curve']['equity']
        
        # Calculate portfolio equity
        portfolio_equity = combined_equity.sum(axis=1)
        portfolio_equity_df = pd.DataFrame({
            'equity': portfolio_equity,
            'timestamp': combined_equity.index
        })
        
        # Analyze combined performance
        portfolio_performance = self.performance_analyzer.analyze_performance(
            all_trades,
            portfolio_equity_df,
            initial_capital=TradingConfig.BACKTEST_INITIAL_CAPITAL
        )
        
        # Add correlation analysis
        if len(combined_equity.columns) > 1:
            correlation_matrix = combined_equity.pct_change().corr()
            avg_correlation = correlation_matrix.values[
                ~np.eye(correlation_matrix.shape[0], dtype=bool)
            ].mean()
        else:
            avg_correlation = 0
        
        portfolio_performance['correlation'] = {
            'average': round(avg_correlation, 3),
            'interpretation': 'Low' if avg_correlation < 0.3 else 'High'
        }
        
        return portfolio_performance
    
    def _display_portfolio_results(self, performance: dict):
        """Display portfolio backtest results"""
        
        summary = performance['summary']
        
        click.echo(f"\n{'='*60}")
        click.echo("Portfolio Performance Summary")
        click.echo(f"{'='*60}")
        
        click.echo(f"\nReturns:")
        click.echo(f"  Total Return: {summary['total_return']:.2f}%")
        click.echo(f"  Monthly Return: {summary['monthly_return']:.2f}%")
        click.echo(f"  Annual Return: {summary['annual_return']:.2f}%")
        
        click.echo(f"\nRisk Metrics:")
        click.echo(f"  Sharpe Ratio: {performance['risk']['sharpe_ratio']:.2f}")
        click.echo(f"  Max Drawdown: {performance['risk']['max_drawdown']:.2f}%")
        click.echo(f"  Daily VaR (95%): {performance['risk']['var_95']:.2f}%")
        
        click.echo(f"\nTrading Statistics:")
        click.echo(f"  Total Trades: {summary['total_trades']}")
        click.echo(f"  Win Rate: {summary['win_rate']:.1f}%")
        click.echo(f"  Profit Factor: {summary['profit_factor']:.2f}")
        
        click.echo(f"\nPortfolio Metrics:")
        click.echo(f"  Average Correlation: {performance['correlation']['average']:.3f}")
        
        # Monthly breakdown
        click.echo(f"\nMonthly Performance:")
        monthly_data = []
        for month in performance['monthly'][-6:]:  # Last 6 months
            monthly_data.append([
                month['period'],
                f"{month['return']:.2f}%",
                month['trades'],
                f"{month['win_rate']:.1f}%",
                f"${month['pnl']:.2f}"
            ])
        
        headers = ['Month', 'Return', 'Trades', 'Win Rate', 'PnL']
        click.echo(tabulate(monthly_data, headers=headers, tablefmt='simple'))
    
    def _construct_portfolio(self, symbol_scores: List[dict]) -> dict:
        """Construct optimal portfolio from scored symbols"""
        
        # Select top symbols by grade
        selected = []
        
        # Grade A symbols (all)
        grade_a = [s for s in symbol_scores if s['grade'] == 'A']
        selected.extend(grade_a)
        
        # Grade B symbols (up to 10)
        grade_b = [s for s in symbol_scores if s['grade'] == 'B'][:10]
        selected.extend(grade_b)
        
        # Grade C symbols (up to 5, if total < 20)
        if len(selected) < 20:
            grade_c = [s for s in symbol_scores if s['grade'] == 'C'][:5]
            selected.extend(grade_c)
        
        # Calculate allocations
        total_score = sum(s['total_score'] for s in selected)
        
        for symbol in selected:
            symbol['allocation'] = (symbol['total_score'] / total_score) * 100
            symbol['suggested_capital'] = (symbol['allocation'] / 100) * TradingConfig.BACKTEST_INITIAL_CAPITAL
        
        return {
            'selected': selected,
            'total_symbols': len(selected),
            'grade_distribution': {
                'A': len([s for s in selected if s['grade'] == 'A']),
                'B': len([s for s in selected if s['grade'] == 'B']),
                'C': len([s for s in selected if s['grade'] == 'C'])
            },
            'recommended_capital_per_symbol': TradingConfig.BACKTEST_INITIAL_CAPITAL / len(selected),
            'timestamp': datetime.now().isoformat()
        }
    
    def _display_portfolio_recommendations(self, portfolio: dict):
        """Display portfolio recommendations"""
        
        click.echo(f"\n{'='*60}")
        click.echo("Portfolio Recommendations")
        click.echo(f"{'='*60}")
        
        click.echo(f"\nSelected {portfolio['total_symbols']} symbols")
        click.echo(f"Grade distribution: A={portfolio['grade_distribution']['A']}, "
                  f"B={portfolio['grade_distribution']['B']}, "
                  f"C={portfolio['grade_distribution']['C']}")
        
        # Display selected symbols
        table_data = []
        for s in portfolio['selected'][:20]:  # Top 20
            table_data.append([
                s['symbol'],
                s['grade'],
                f"{s['total_score']:.1f}",
                f"{s['volatility']:.2%}",
                f"{s['expected_return']:.1%}",
                f"{s['allocation']:.1f}%",
                f"${s['suggested_capital']:.0f}"
            ])
        
        headers = ['Symbol', 'Grade', 'Score', 'Volatility', 'Expected Return', 'Allocation', 'Capital']
        click.echo("\nTop 20 Symbols:")
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))


# CLI Commands
@click.group()
@click.pass_context
def cli(ctx):
    """Crypto Trading System - Advanced automated trading with AI optimization"""
    ctx.obj = CryptoTradingSystem()


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to analyze')
@click.option('--top', '-t', default=30, help='Analyze top N symbols by volume')
@click.option('--period', '-p', default=180, help='Analysis period in days')
@click.option('--save', is_flag=True, help='Save detailed report')
@click.pass_obj
def analyze(system, symbols, top, period, save):
    """Analyze symbols with market and performance metrics"""
    
    if not symbols:
        symbols = system.data_manager.get_top_volume_symbols(top)
    
    system.analyze_symbols(list(symbols), period, save)


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to optimize')
@click.option('--target', '-t', type=float, help='Target monthly return (e.g., 0.02 for 2%)')
@click.option('--aggressive', '-a', is_flag=True, help='Use aggressive optimization')
@click.option('--trials', '-n', type=int, help='Number of optimization trials')
@click.pass_obj
def optimize(system, symbols, target, aggressive, trials):
    """Optimize trading parameters using AI"""
    
    if not symbols:
        if click.confirm("No symbols specified. Optimize all analyzed symbols?"):
            symbols = system._get_optimized_symbols()
            if not symbols:
                click.echo("No symbols found. Please run 'analyze' first.")
                return
        else:
            return
    
    system.optimize_symbols(list(symbols), target, aggressive, trials)


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to backtest')
@click.option('--start', help='Start date (YYYY-MM-DD)')
@click.option('--end', help='End date (YYYY-MM-DD)')
@click.option('--use-optimized/--use-default', default=True, help='Use optimized parameters')
@click.pass_obj
def backtest(system, symbols, start, end, use_optimized):
    """Run portfolio backtest"""
    
    system.backtest_portfolio(
        list(symbols) if symbols else None,
        start,
        end,
        use_optimized
    )


@cli.command()
@click.option('--top', '-t', default=50, help='Analyze top N symbols')
@click.pass_obj
def recommend(system, top):
    """Get portfolio recommendations based on market analysis"""
    
    system.recommend_portfolio(top)


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to trade')
@click.option('--paper/--live', default=True, help='Paper trading or live trading')
@click.pass_obj
def trade(system, symbols, paper):
    """Start automated trading"""
    
    if not paper and not click.confirm("Are you sure you want to start LIVE trading?"):
        return
    
    system.start_trading(list(symbols) if symbols else None, paper)


@cli.command()
@click.pass_obj
def dashboard(system):
    """Launch web dashboard"""
    
    system.launch_dashboard()


@cli.command()
@click.option('--type', '-t', 
              type=click.Choice(['daily', 'weekly', 'monthly', 'full']),
              default='daily',
              help='Report type')
@click.pass_obj
def report(system, type):
    """Generate trading reports"""
    
    click.echo(f"Generating {type} report...")
    
    if type == 'daily':
        report_path = system.report_generator.generate_daily_report()
    elif type == 'weekly':
        report_path = system.report_generator.generate_weekly_report()
    elif type == 'monthly':
        report_path = system.report_generator.generate_monthly_report()
    else:
        report_path = system.report_generator.generate_full_report()
    
    click.echo(f"Report saved to: {report_path}")


@cli.command()
@click.option('--days', '-d', default=7, help='Delete cache older than N days')
@click.pass_obj
def clean(system, days):
    """Clean old cache files"""
    
    click.echo(f"Cleaning cache files older than {days} days...")
    system.data_manager.clean_cache(days)
    click.echo("Cache cleaned successfully")


if __name__ == '__main__':
    cli()