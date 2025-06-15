"""
Flask web application for the trading system
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path
import threading

from config.settings import Config, load_config
from main import TradingSystem
from utils.logger import get_logger
from utils.web_ui import create_app

logger = get_logger(__name__)

# Initialize Flask app
app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")

# Global trading system instance
trading_system = None
background_thread = None
thread_lock = threading.Lock()


def background_task():
    """Background task for real-time updates"""
    
    while True:
        if trading_system and hasattr(trading_system, 'live_trader'):
            try:
                # Get current status
                status = trading_system.get_status()
                
                # Emit status update
                socketio.emit('status_update', status)
                
                # Get latest trades
                trades = trading_system.get_trade_history(limit=10)
                if not trades.empty:
                    socketio.emit('trades_update', trades.to_dict('records'))
                
                # Get positions
                positions = trading_system.get_positions()
                socketio.emit('positions_update', positions)
                
            except Exception as e:
                logger.error(f"Background task error: {e}")
        
        socketio.sleep(5)  # Update every 5 seconds


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    global background_thread
    
    with thread_lock:
        if background_thread is None:
            background_thread = socketio.start_background_task(background_task)
    
    emit('connected', {'data': 'Connected to trading system'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')


@socketio.on('start_trading')
def handle_start_trading(data):
    """Start trading via WebSocket"""
    global trading_system
    
    symbols = data.get('symbols', [])
    paper_trading = data.get('paper_trading', True)
    
    if not trading_system:
        emit('error', {'message': 'Trading system not initialized'})
        return
    
    try:
        # Start trading in background
        asyncio.run(trading_system.start_trading(
            symbols=symbols,
            paper_trading=paper_trading
        ))
        
        emit('trading_started', {'symbols': symbols, 'mode': 'paper' if paper_trading else 'live'})
        
    except Exception as e:
        logger.error(f"Failed to start trading: {e}")
        emit('error', {'message': str(e)})


@socketio.on('stop_trading')
def handle_stop_trading():
    """Stop trading via WebSocket"""
    global trading_system
    
    if trading_system and hasattr(trading_system, 'live_trader'):
        trading_system.stop_trading()
        emit('trading_stopped', {})
    else:
        emit('error', {'message': 'No active trading session'})


@socketio.on('get_chart_data')
def handle_get_chart_data(data):
    """Get chart data via WebSocket"""
    
    symbol = data.get('symbol')
    timeframe = data.get('timeframe', '5')
    period = data.get('period', '1d')
    
    if not symbol:
        emit('error', {'message': 'Symbol required'})
        return
    
    try:
        # Calculate date range
        end_date = datetime.now()
        if period == '1d':
            start_date = end_date - timedelta(days=1)
        elif period == '1w':
            start_date = end_date - timedelta(weeks=1)
        elif period == '1m':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=90)
        
        # Get data
        data_manager = trading_system.data_manager if trading_system else None
        if data_manager:
            df = asyncio.run(data_manager.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            ))
            
            if not df.empty:
                # Prepare chart data
                chart_data = {
                    'timestamps': df.index.strftime('%Y-%m-%d %H:%M').tolist(),
                    'open': df['open'].tolist(),
                    'high': df['high'].tolist(),
                    'low': df['low'].tolist(),
                    'close': df['close'].tolist(),
                    'volume': df['volume'].tolist()
                }
                
                emit('chart_data', {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data': chart_data
                })
            else:
                emit('error', {'message': 'No data available'})
        
    except Exception as e:
        logger.error(f"Failed to get chart data: {e}")
        emit('error', {'message': str(e)})


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize trading system"""
    global trading_system
    
    try:
        config_path = request.json.get('config_path', 'config/config.yaml')
        config = load_config(config_path)
        
        trading_system = TradingSystem(config)
        app.trading_system = trading_system  # Store reference for API routes
        
        return jsonify({'success': True, 'message': 'System initialized'})
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/watchlist', methods=['GET', 'POST'])
def watchlist():
    """Manage watchlist"""
    
    if request.method == 'GET':
        # Get watchlist
        watchlist_path = Path('data/watchlist.json')
        if watchlist_path.exists():
            with open(watchlist_path, 'r') as f:
                symbols = json.load(f)
        else:
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        return jsonify({'symbols': symbols})
    
    else:  # POST
        # Update watchlist
        symbols = request.json.get('symbols', [])
        
        watchlist_path = Path('data/watchlist.json')
        with open(watchlist_path, 'w') as f:
            json.dump(symbols, f)
        
        return jsonify({'success': True, 'symbols': symbols})


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest"""
    global trading_system
    
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        # Get parameters
        data = request.json
        symbols = data.get('symbols', ['AAPL'])
        start_date = pd.to_datetime(data.get('start_date', datetime.now() - timedelta(days=90)))
        end_date = pd.to_datetime(data.get('end_date', datetime.now()))
        initial_capital = data.get('initial_capital', 100000)
        strategy = data.get('strategy', 'adaptive')
        
        # Run backtest
        results = trading_system.run_backtest(
            symbols=symbols,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimize/<symbol>', methods=['POST'])
def optimize_symbol(symbol):
    """Optimize strategy for symbol"""
    global trading_system
    
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        # Get parameters
        data = request.json
        target_return = data.get('target_return', 0.02)
        timeframes = data.get('timeframes', ['5', '15'])
        
        # Run optimization
        results = asyncio.run(trading_system.optimize_parameters(
            symbol=symbol,
            timeframes=timeframes,
            target_return=target_return
        ))
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/<format>')
def export_data(format):
    """Export data in various formats"""
    
    if not trading_system:
        return jsonify({'error': 'System not initialized'}), 400
    
    try:
        # Get data type
        data_type = request.args.get('type', 'trades')
        
        if data_type == 'trades':
            df = trading_system.get_trade_history()
        elif data_type == 'performance':
            df = pd.DataFrame(trading_system.get_performance_history())
        else:
            return jsonify({'error': 'Invalid data type'}), 400
        
        # Export based on format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            filename = f"{data_type}_{timestamp}.csv"
            filepath = Path('data/exports') / filename
            filepath.parent.mkdir(exist_ok=True)
            
            df.to_csv(filepath, index=False)
            return send_file(filepath, as_attachment=True)
            
        elif format == 'json':
            return jsonify(df.to_dict('records'))
            
        elif format == 'excel':
            filename = f"{data_type}_{timestamp}.xlsx"
            filepath = Path('data/exports') / filename
            filepath.parent.mkdir(exist_ok=True)
            
            df.to_excel(filepath, index=False)
            return send_file(filepath, as_attachment=True)
            
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e)}), 500


def run_web_app(config: Config = None, debug: bool = False, port: int = 5000):
    """Run the web application"""
    
    global trading_system
    
    # Initialize trading system if config provided
    if config:
        trading_system = TradingSystem(config)
        app.trading_system = trading_system
    
    # Run app
    socketio.run(app, debug=debug, port=port, host='0.0.0.0')


if __name__ == '__main__':
    # Load config and run
    config = load_config('config/config.yaml')
    run_web_app(config, debug=True)