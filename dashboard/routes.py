"""
API routes for the dashboard
"""

from flask import Blueprint, jsonify, request, render_template
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger(__name__)


def register_routes(app):
    """Register all routes with the Flask app"""
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/api/status')
    def api_status():
        """System status endpoint"""
        from dashboard.app import get_system_status
        status = get_system_status(app.trading_system)
        return jsonify(status)
    
    @app.route('/api/symbols')
    def api_symbols():
        """Get available symbols"""
        if not app.trading_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        try:
            # Get optimized symbols
            symbols = app.trading_system._get_optimized_symbols()
            
            # Get symbol info
            symbol_info = []
            for symbol in symbols[:20]:  # Limit to 20
                params = app.trading_system.data_manager.load_optimized_parameters(symbol)
                if params:
                    symbol_info.append({
                        'symbol': symbol,
                        'monthly_return': params['performance'].get('monthly_return', 0),
                        'win_rate': params['performance'].get('win_rate', 0),
                        'optimization_date': params.get('optimization_date', '')
                    })
            
            return jsonify({
                'symbols': symbol_info,
                'total': len(symbols)
            })
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analysis/<symbol>')
    def api_symbol_analysis(symbol):
        """Get symbol analysis"""
        if not app.trading_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        try:
            # Run analysis
            analysis = app.trading_system.market_analyzer.analyze_symbol(symbol)
            
            return jsonify(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/optimize', methods=['POST'])
    def api_optimize():
        """Run optimization"""
        if not app.trading_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        data = request.json
        symbol = data.get('symbol')
        target_return = data.get('target_return', 0.02)
        aggressive = data.get('aggressive', False)
        
        if not symbol:
            return jsonify({'error': 'Symbol required'}), 400
        
        try:
            # Run optimization
            app.trading_system.optimize_symbols(
                [symbol],
                target_return=target_return,
                aggressive=aggressive
            )
            
            return jsonify({'success': True, 'message': f'Optimization started for {symbol}'})
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/backtest', methods=['POST'])
    def api_backtest():
        """Run backtest"""
        if not app.trading_system:
            return jsonify({'error': 'System not initialized'}), 500
        
        data = request.json
        symbols = data.get('symbols', [])
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        try:
            # Run backtest
            app.trading_system.backtest_portfolio(
                symbols=symbols if symbols else None,
                start_date=start_date,
                end_date=end_date
            )
            
            return jsonify({'success': True, 'message': 'Backtest completed'})
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/performance')
    def api_performance():
        """Get performance data"""
        from dashboard.app import get_performance
        performance = get_performance(app.trading_system)
        return jsonify(performance)
    
    @app.route('/api/positions')
    def api_positions():
        """Get current positions"""
        from dashboard.app import get_positions
        positions = get_positions(app.trading_system)
        return jsonify(positions)
    
    @app.route('/api/trades')
    def api_trades():
        """Get recent trades"""
        if not hasattr(app.trading_system, 'live_trader'):
            return jsonify({'trades': []})
        
        try:
            results = app.trading_system.live_trader.get_results()
            trades = results.get('trades', pd.DataFrame())
            
            if not trades.empty:
                # Convert to list of dicts
                trades_list = trades.tail(50).to_dict('records')
                
                # Format dates
                for trade in trades_list:
                    for key in ['entry_time', 'exit_time']:
                        if key in trade and pd.notna(trade[key]):
                            if isinstance(trade[key], pd.Timestamp):
                               trade[key] = trade[key].isoformat()
               
               return jsonify({'trades': trades_list})
           
           return jsonify({'trades': []})
           
       except Exception as e:
           logger.error(f"Error getting trades: {e}")
           return jsonify({'trades': []})
   
   @app.route('/api/charts/<chart_type>')
   def api_charts(chart_type):
       """Get chart data"""
       if not app.trading_system:
           return jsonify({'error': 'System not initialized'}), 500
       
       try:
           if chart_type == 'equity':
               # Get equity curve data
               if hasattr(app.trading_system, 'live_trader'):
                   results = app.trading_system.live_trader.get_results()
                   equity_curve = results.get('equity_curve', pd.DataFrame())
                   
                   if not equity_curve.empty:
                       data = {
                           'timestamps': equity_curve.index.tolist(),
                           'equity': equity_curve['equity'].tolist()
                       }
                       return jsonify(data)
               
               return jsonify({'timestamps': [], 'equity': []})
           
           elif chart_type == 'pnl':
               # Get P&L data
               if hasattr(app.trading_system, 'live_trader'):
                   trades = app.trading_system.live_trader.trades
                   
                   if trades:
                       df = pd.DataFrame(trades)
                       df['cumulative_pnl'] = df['pnl'].cumsum()
                       
                       data = {
                           'timestamps': df.index.tolist(),
                           'pnl': df['pnl'].tolist(),
                           'cumulative_pnl': df['cumulative_pnl'].tolist()
                       }
                       return jsonify(data)
               
               return jsonify({'timestamps': [], 'pnl': [], 'cumulative_pnl': []})
           
           else:
               return jsonify({'error': 'Unknown chart type'}), 400
               
       except Exception as e:
           logger.error(f"Error getting chart data: {e}")
           return jsonify({'error': str(e)}), 500
   
   @app.route('/api/trading/<action>', methods=['POST'])
   def api_trading_control(action):
       """Control trading (start/stop)"""
       if not app.trading_system:
           return jsonify({'error': 'System not initialized'}), 500
       
       try:
           if action == 'start':
               data = request.json
               symbols = data.get('symbols', [])
               paper_trading = data.get('paper_trading', True)
               
               app.trading_system.start_trading(
                   symbols=symbols if symbols else None,
                   paper_trading=paper_trading
               )
               
               return jsonify({'success': True, 'message': 'Trading started'})
           
           elif action == 'stop':
               if hasattr(app.trading_system, 'live_trader'):
                   app.trading_system.live_trader.stop()
                   return jsonify({'success': True, 'message': 'Trading stopped'})
               else:
                   return jsonify({'error': 'No active trading session'}), 400
           
           else:
               return jsonify({'error': 'Invalid action'}), 400
               
       except Exception as e:
           logger.error(f"Trading control error: {e}")
           return jsonify({'error': str(e)}), 500
   
   @app.route('/api/reports')
   def api_reports():
       """Get available reports"""
       try:
           from pathlib import Path
           report_dir = Path('data/reports')
           
           if not report_dir.exists():
               return jsonify({'reports': []})
           
           reports = []
           for report_path in report_dir.glob('*/report.html'):
               report_info = {
                   'name': report_path.parent.name,
                   'path': str(report_path),
                   'created': datetime.fromtimestamp(
                       report_path.stat().st_mtime
                   ).isoformat()
               }
               reports.append(report_info)
           
           # Sort by creation date
           reports.sort(key=lambda x: x['created'], reverse=True)
           
           return jsonify({'reports': reports[:20]})  # Last 20 reports
           
       except Exception as e:
           logger.error(f"Error getting reports: {e}")
           return jsonify({'reports': []})