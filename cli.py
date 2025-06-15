"""
Command-line interface for the trading system
"""

import click
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path

from config.settings import Config, load_config, save_config
from main import TradingSystem
from utils.logger import get_logger
from utils.helpers import create_dirs

logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', default='config/config.yaml', help='Config file path')
@click.pass_context
def cli(ctx, config):
    """AI Trading System CLI"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    
    # Create necessary directories
    create_dirs()


@cli.command()
@click.option('--symbols', '-s', multiple=True, required=True, help='Symbols to trade')
@click.option('--timeframes', '-t', multiple=True, default=['5'], help='Timeframes')
@click.option('--strategy', default='adaptive', help='Strategy to use')
@click.option('--capital', type=float, default=100000, help='Starting capital')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--optimize', is_flag=True, help='Run optimization first')
@click.pass_context
def backtest(ctx, symbols, timeframes, strategy, capital, start_date, end_date, optimize):
    """Run backtest"""
    
    click.echo(f"Running backtest for {', '.join(symbols)}")
    
    # Load config
    config = load_config(ctx.obj['config_path'])
    
    # Initialize system
    system = TradingSystem(config)
    
    # Convert dates
    start = pd.to_datetime(start_date) if start_date else datetime.now() - timedelta(days=90)
    end = pd.to_datetime(end_date) if end_date else datetime.now()
    
    # Run optimization if requested
    if optimize:
        click.echo("Running optimization...")
        
        with click.progressbar(symbols) as bar:
            for symbol in bar:
                asyncio.run(system.optimize_parameters(
                    symbol=symbol,
                    timeframes=list(timeframes),
                    target_return=0.02  # 2% monthly
                ))
    
    # Run backtest
    click.echo("Running backtest...")
    
    results = system.run_backtest(
        symbols=list(symbols),
        timeframes=list(timeframes),
        strategy=strategy,
        start_date=start,
        end_date=end,
        initial_capital=capital
    )
    
    # Display results
    click.echo("\n=== Backtest Results ===")
    click.echo(f"Total Return: {results['total_return']:.2f}%")
    click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    click.echo(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    click.echo(f"Win Rate: {results['win_rate']:.2f}%")
    click.echo(f"Total Trades: {results['total_trades']}")
    
    # Save report
    if click.confirm("Save detailed report?"):
        report_path = system.generate_report(results)
        click.echo(f"Report saved to: {report_path}")


@cli.command()
@click.option('--symbols', '-s', multiple=True, required=True, help='Symbols to trade')
@click.option('--timeframes', '-t', multiple=True, default=['5'], help='Timeframes')
@click.option('--paper', is_flag=True, help='Use paper trading')
@click.option('--strategy', default='adaptive', help='Strategy to use')
@click.pass_context
def trade(ctx, symbols, timeframes, paper, strategy):
    """Start live trading"""
    
    mode = "paper" if paper else "live"
    click.echo(f"Starting {mode} trading for {', '.join(symbols)}")
    
    if not paper and not click.confirm("Are you sure you want to start LIVE trading?"):
        return
    
    # Load config
    config = load_config(ctx.obj['config_path'])
    
    # Initialize system
    system = TradingSystem(config)
    
    # Start trading
    try:
        asyncio.run(system.start_trading(
            symbols=list(symbols),
            timeframes=list(timeframes),
            paper_trading=paper,
            strategy=strategy
        ))
    except KeyboardInterrupt:
        click.echo("\nStopping trading...")
        system.stop_trading()


@cli.command()
@click.argument('symbol')
@click.option('--timeframes', '-t', multiple=True, default=['5', '15'], help='Timeframes')
@click.option('--target-return', type=float, default=0.02, help='Target monthly return')
@click.option('--max-iterations', type=int, default=100, help='Max optimization iterations')
@click.pass_context
def optimize(ctx, symbol, timeframes, target_return, max_iterations):
    """Optimize strategy parameters"""
    
    click.echo(f"Optimizing parameters for {symbol}")
    
    # Load config
    config = load_config(ctx.obj['config_path'])
    
    # Initialize system
    system = TradingSystem(config)
    
    # Run optimization
    with click.progressbar(length=max_iterations) as bar:
        def callback(iteration, score):
            bar.update(1)
            if iteration % 10 == 0:
                click.echo(f"\nIteration {iteration}: Score = {score:.4f}")
        
        results = asyncio.run(system.optimize_parameters(
            symbol=symbol,
            timeframes=list(timeframes),
            target_return=target_return,
            max_iterations=max_iterations,
            callback=callback
        ))
    
    # Display results
    click.echo("\n=== Optimization Results ===")
    click.echo(f"Best Score: {results['best_score']:.4f}")
    click.echo(f"Target Return: {target_return*100:.1f}%")
    click.echo(f"Achieved Return: {results['performance']['monthly_return']*100:.2f}%")
    click.echo(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    
    click.echo("\nBest Parameters:")
    for key, value in results['best_params'].items():
        click.echo(f"  {key}: {value}")
    
    # Save parameters
    if click.confirm("Save optimized parameters?"):
        system.save_optimized_parameters(symbol, results['best_params'])
        click.echo("Parameters saved!")


@cli.command()
@click.option('--symbols', '-s', multiple=True, help='Symbols to analyze')
@click.option('--days', type=int, default=30, help='Days to analyze')
@click.pass_context
def analyze(ctx, symbols, days):
    """Analyze market data"""
    
    click.echo(f"Analyzing market data for last {days} days")
    
    # Load config
    config = load_config(ctx.obj['config_path'])
    
    # Initialize system
    system = TradingSystem(config)
    
    # Get symbols to analyze
    if not symbols:
        # Use default watchlist
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Analyze each symbol
    analysis_results = {}
    
    with click.progressbar(symbols) as bar:
        for symbol in bar:
            results = asyncio.run(system.analyze_symbol(
                symbol=symbol,
                lookback_days=days
            ))
            analysis_results[symbol] = results
    
    # Display results
    click.echo("\n=== Market Analysis ===")
    
    for symbol, analysis in analysis_results.items():
        click.echo(f"\n{symbol}:")
        click.echo(f"  Current Price: ${analysis['current_price']:.2f}")
        click.echo(f"  {days}-day Return: {analysis['period_return']*100:.2f}%")
        click.echo(f"  Volatility: {analysis['volatility']*100:.1f}%")
        click.echo(f"  RSI: {analysis['rsi']:.1f}")
        click.echo(f"  Trend: {analysis['trend']}")
        click.echo(f"  Signal: {analysis['signal']}")


@cli.command()
@click.option('--export', is_flag=True, help='Export to CSV')
@click.option('--limit', type=int, default=50, help='Number of trades to show')
@click.pass_context
def history(ctx, export, limit):
    """View trade history"""
    
    # Load config
    config = load_config(ctx.obj['config_path'])
    
    # Initialize system
    system = TradingSystem(config)
    
    # Get trade history
    trades = system.get_trade_history(limit=limit)
    
    if trades.empty:
        click.echo("No trades found")
        return
    
    # Display trades
    click.echo("\n=== Trade History ===")
    
    for _, trade in trades.iterrows():
        click.echo(f"\n{trade['exit_time'].strftime('%Y-%m-%d %H:%M')} - {trade['symbol']}:")
       click.echo(f"  Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}")
       click.echo(f"  Size: {trade['size']:.2f}")
       click.echo(f"  P&L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
       click.echo(f"  Reason: {trade['exit_reason']}")
   
   # Summary statistics
   click.echo(f"\n=== Summary ===")
   click.echo(f"Total Trades: {len(trades)}")
   click.echo(f"Win Rate: {(trades['pnl'] > 0).mean() * 100:.1f}%")
   click.echo(f"Average P&L: ${trades['pnl'].mean():.2f}")
   click.echo(f"Total P&L: ${trades['pnl'].sum():.2f}")
   
   # Export if requested
   if export:
       filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
       trades.to_csv(filename, index=False)
       click.echo(f"\nExported to {filename}")


@cli.command()
@click.pass_context
def status(ctx):
   """Show system status"""
   
   # Load config
   config = load_config(ctx.obj['config_path'])
   
   # Initialize system
   system = TradingSystem(config)
   
   # Get status
   status_info = system.get_status()
   
   click.echo("\n=== System Status ===")
   click.echo(f"Status: {status_info['status']}")
   click.echo(f"Uptime: {status_info['uptime']}")
   
   if status_info['trading_active']:
       click.echo(f"\nTrading Mode: {status_info['trading_mode']}")
       click.echo(f"Active Symbols: {', '.join(status_info['active_symbols'])}")
       click.echo(f"Current Equity: ${status_info['current_equity']:,.2f}")
       click.echo(f"Today's P&L: ${status_info['daily_pnl']:,.2f}")
       
       click.echo("\nPositions:")
       for symbol, pos in status_info['positions'].items():
           click.echo(f"  {symbol}: {pos['size']:.2f} @ ${pos['avg_price']:.2f} (P&L: ${pos['unrealized_pnl']:.2f})")
   
   click.echo(f"\nData Sources: {', '.join(status_info['data_sources'])}")
   click.echo(f"Database: {status_info['database_status']}")


@cli.command()
@click.option('--broker', type=click.Choice(['alpaca', 'ibkr', 'paper']), default='paper')
@click.option('--api-key', help='Broker API key')
@click.option('--secret-key', help='Broker secret key')
@click.option('--base-url', help='Broker base URL')
@click.pass_context
def configure(ctx, broker, api_key, secret_key, base_url):
   """Configure system settings"""
   
   click.echo("Configuring trading system...")
   
   # Load existing config
   config = load_config(ctx.obj['config_path'])
   
   # Update broker settings
   if broker:
       config['broker']['name'] = broker
       click.echo(f"Broker set to: {broker}")
   
   if api_key:
       config['broker']['api_key'] = api_key
       click.echo("API key updated")
   
   if secret_key:
       config['broker']['secret_key'] = secret_key
       click.echo("Secret key updated")
   
   if base_url:
       config['broker']['base_url'] = base_url
       click.echo(f"Base URL set to: {base_url}")
   
   # Interactive configuration
   if click.confirm("Configure data sources?"):
       config['data']['yahoo_finance'] = click.confirm("Enable Yahoo Finance?", default=True)
       config['data']['alpha_vantage'] = click.confirm("Enable Alpha Vantage?", default=False)
       
       if config['data']['alpha_vantage']:
           av_key = click.prompt("Alpha Vantage API key", hide_input=True)
           config['data']['alpha_vantage_key'] = av_key
   
   if click.confirm("Configure risk management?"):
       config['risk']['max_position_size'] = click.prompt(
           "Max position size (%)", 
           type=float, 
           default=10.0
       ) / 100
       
       config['risk']['max_portfolio_risk'] = click.prompt(
           "Max portfolio risk (%)", 
           type=float, 
           default=20.0
       ) / 100
       
       config['risk']['max_daily_loss'] = click.prompt(
           "Max daily loss (%)", 
           type=float, 
           default=5.0
       ) / 100
   
   # Save config
   save_config(config, ctx.obj['config_path'])
   click.echo("\nConfiguration saved!")


@cli.command()
@click.option('--period', type=click.Choice(['today', 'week', 'month', 'all']), default='week')
@click.option('--format', type=click.Choice(['terminal', 'html', 'pdf']), default='terminal')
@click.pass_context
def report(ctx, period, format):
   """Generate performance report"""
   
   click.echo(f"Generating {period} report...")
   
   # Load config
   config = load_config(ctx.obj['config_path'])
   
   # Initialize system
   system = TradingSystem(config)
   
   # Determine date range
   end_date = datetime.now()
   if period == 'today':
       start_date = end_date.replace(hour=0, minute=0, second=0)
   elif period == 'week':
       start_date = end_date - timedelta(days=7)
   elif period == 'month':
       start_date = end_date - timedelta(days=30)
   else:  # all
       start_date = None
   
   # Generate report
   report_data = system.generate_performance_report(
       start_date=start_date,
       end_date=end_date
   )
   
   if format == 'terminal':
       # Display in terminal
       click.echo("\n=== Performance Report ===")
       click.echo(f"Period: {report_data['period']}")
       
       click.echo(f"\nReturns:")
       click.echo(f"  Total: {report_data['total_return']:.2f}%")
       click.echo(f"  Daily Avg: {report_data['daily_return']:.2f}%")
       click.echo(f"  Monthly: {report_data['monthly_return']:.2f}%")
       
       click.echo(f"\nRisk Metrics:")
       click.echo(f"  Sharpe Ratio: {report_data['sharpe_ratio']:.2f}")
       click.echo(f"  Sortino Ratio: {report_data['sortino_ratio']:.2f}")
       click.echo(f"  Max Drawdown: {report_data['max_drawdown']:.2f}%")
       click.echo(f"  Volatility: {report_data['volatility']:.2f}%")
       
       click.echo(f"\nTrading Statistics:")
       click.echo(f"  Total Trades: {report_data['total_trades']}")
       click.echo(f"  Win Rate: {report_data['win_rate']:.1f}%")
       click.echo(f"  Avg Win: ${report_data['avg_win']:.2f}")
       click.echo(f"  Avg Loss: ${report_data['avg_loss']:.2f}")
       click.echo(f"  Profit Factor: {report_data['profit_factor']:.2f}")
       
   else:
       # Generate file report
       output_path = system.save_report(report_data, format=format)
       click.echo(f"Report saved to: {output_path}")
       
       if format == 'html' and click.confirm("Open in browser?"):
           import webbrowser
           webbrowser.open(output_path)


@cli.command()
@click.argument('symbol')
@click.option('--interval', type=int, default=1, help='Update interval in seconds')
@click.pass_context
def monitor(ctx, symbol, interval):
   """Monitor symbol in real-time"""
   
   click.echo(f"Monitoring {symbol} (Press Ctrl+C to stop)")
   
   # Load config
   config = load_config(ctx.obj['config_path'])
   
   # Initialize system
   system = TradingSystem(config)
   
   async def monitor_loop():
       while True:
           try:
               # Get real-time data
               data = await system.data_manager.get_realtime_data(symbol, ['1'])
               
               if data:
                   current = data.get('close', 0)
                   volume = data.get('volume', 0)
                   
                   # Get indicators
                   analysis = await system.analyze_symbol(symbol, lookback_days=1)
                   
                   # Clear screen
                   click.clear()
                   
                   # Display data
                   click.echo(f"=== {symbol} Monitor ===")
                   click.echo(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                   click.echo(f"Price: ${current:.2f}")
                   click.echo(f"Volume: {volume:,}")
                   click.echo(f"RSI: {analysis.get('rsi', 0):.1f}")
                   click.echo(f"Signal: {analysis.get('signal', 'N/A')}")
                   
                   # Check for position
                   positions = system.get_positions()
                   if symbol in positions:
                       pos = positions[symbol]
                       click.echo(f"\nPosition: {pos['size']:.2f} @ ${pos['avg_price']:.2f}")
                       click.echo(f"P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
               
               await asyncio.sleep(interval)
               
           except KeyboardInterrupt:
               break
           except Exception as e:
               click.echo(f"Error: {e}")
               await asyncio.sleep(interval)
   
   try:
       asyncio.run(monitor_loop())
   except KeyboardInterrupt:
       click.echo("\nMonitoring stopped")


@cli.command()
@click.pass_context
def doctor(ctx):
   """Check system health and dependencies"""
   
   click.echo("Running system diagnostics...\n")
   
   issues = []
   
   # Check Python version
   import sys
   python_version = sys.version_info
   if python_version.major < 3 or python_version.minor < 8:
       issues.append(f"Python 3.8+ required (found {python_version.major}.{python_version.minor})")
   else:
       click.echo("✓ Python version OK")
   
   # Check required packages
   required_packages = [
       'pandas', 'numpy', 'scikit-learn', 'torch', 'yfinance',
       'plotly', 'alpaca-trade-api', 'asyncio', 'flask'
   ]
   
   for package in required_packages:
       try:
           __import__(package.replace('-', '_'))
           click.echo(f"✓ {package} installed")
       except ImportError:
           issues.append(f"{package} not installed")
           click.echo(f"✗ {package} missing")
   
   # Check directories
   from pathlib import Path
   required_dirs = ['data', 'logs', 'reports', 'models']
   
   for dir_name in required_dirs:
       if Path(dir_name).exists():
           click.echo(f"✓ {dir_name}/ directory exists")
       else:
           Path(dir_name).mkdir(parents=True, exist_ok=True)
           click.echo(f"✓ {dir_name}/ directory created")
   
   # Check configuration
   try:
       config = load_config(ctx.obj['config_path'])
       click.echo("✓ Configuration loaded")
       
       # Check broker config
       if 'broker' in config and config['broker'].get('api_key'):
           click.echo("✓ Broker credentials configured")
       else:
           issues.append("Broker credentials not configured")
           click.echo("✗ Broker credentials missing")
           
   except Exception as e:
       issues.append(f"Configuration error: {e}")
       click.echo(f"✗ Configuration error: {e}")
   
   # Check database
   try:
       from utils.database import DatabaseManager
       db = DatabaseManager()
       # Test query
       db.get_trades(limit=1)
       click.echo("✓ Database connection OK")
   except Exception as e:
       issues.append(f"Database error: {e}")
       click.echo(f"✗ Database error: {e}")
   
   # Check network connectivity
   try:
       import requests
       response = requests.get('https://api.github.com', timeout=5)
       if response.status_code == 200:
           click.echo("✓ Internet connection OK")
       else:
           issues.append("Internet connectivity issue")
           click.echo("✗ Internet connectivity issue")
   except Exception:
       issues.append("No internet connection")
       click.echo("✗ No internet connection")
   
   # Summary
   click.echo(f"\n{'='*40}")
   if issues:
       click.echo(f"Found {len(issues)} issue(s):")
       for issue in issues:
           click.echo(f"  - {issue}")
       click.echo("\nRun 'pip install -r requirements.txt' to install missing packages")
   else:
       click.echo("✅ All systems operational!")


if __name__ == '__main__':
   cli()