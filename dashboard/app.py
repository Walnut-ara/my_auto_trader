"""
Flask web dashboard for the trading system
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
from datetime import datetime
import threading
import time

from config.settings import DashboardConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app(trading_system=None):
    """Create Flask application"""
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = DashboardConfig.SECRET_KEY
    
    # Initialize SocketIO for real-time updates
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Store reference to trading system
    app.trading_system = trading_system
    
    # Import routes
    from dashboard.routes import register_routes
    register_routes(app)
    
    # WebSocket events
    @socketio.on('connect')
    def handle_connect():
        logger.info(f"Client connected: {request.sid}")
        emit('connected', {'data': 'Connected to trading system'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info(f"Client disconnected: {request.sid}")
    
    @socketio.on('request_update')
    def handle_update_request(data):
        """Handle real-time update requests"""
        update_type = data.get('type', 'status')
        
        if update_type == 'status':
            status = get_system_status(app.trading_system)
            emit('status_update', status)
        elif update_type == 'positions':
            positions = get_positions(app.trading_system)
            emit('positions_update', positions)
        elif update_type == 'performance':
            performance = get_performance(app.trading_system)
            emit('performance_update', performance)
    
    # Background thread for periodic updates
    def background_updates():
        while True:
            try:
                # Send updates to all connected clients
                with app.app_context():
                    status = get_system_status(app.trading_system)
                    socketio.emit('status_update', status)
                    
                    if hasattr(app.trading_system, 'live_trader'):
                        positions = get_positions(app.trading_system)
                        socketio.emit('positions_update', positions)
                
                time.sleep(DashboardConfig.PORTFOLIO_UPDATE_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in background updates: {e}")
                time.sleep(5)
    
    # Start background thread
    if DashboardConfig.WEBSOCKET_ENABLED:
        update_thread = threading.Thread(target=background_updates)
        update_thread.daemon = True
        update_thread.start()
    
    return app


def get_system_status(trading_system):
    """Get current system status"""
    
    if not trading_system:
        return {'status': 'Not initialized'}
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'system_status': 'Active',
        'mode': 'Paper Trading',  # or Live Trading
        'active_symbols': len(trading_system.symbols) if hasattr(trading_system, 'symbols') else 0,
        'data_manager': 'Connected' if hasattr(trading_system, 'data_manager') else 'Disconnected',
        'optimizer': 'Ready' if hasattr(trading_system, 'optimizer') else 'Not initialized'
    }
    
    # Add live trading status if available
    if hasattr(trading_system, 'live_trader'):
        trader = trading_system.live_trader
        status.update({
            'trading_active': trader.running,
            'positions': len(trader.positions),
            'total_trades': len(trader.trades),
            'uptime_hours': trader.get_uptime_hours() if hasattr(trader, 'get_uptime_hours') else 0
        })
    
    return status


def get_positions(trading_system):
    """Get current positions"""
    
    if not hasattr(trading_system, 'live_trader'):
        return {'positions': {}}
    
    trader = trading_system.live_trader
    positions = {}
    
    for symbol, position in trader.positions.items():
        if position.get('size', 0) != 0:
            positions[symbol] = {
                'size': position['size'],
                'entry_price': position.get('entry_price', 0),
                'current_price': position.get('current_price', 0),
                'pnl': position.get('unrealized_pnl', 0),
                'pnl_pct': position.get('unrealized_pnl_pct', 0),
                'entry_time': position.get('entry_time', '').isoformat() if isinstance(position.get('entry_time'), datetime) else ''
            }
    
    return {'positions': positions, 'count': len(positions)}


def get_performance(trading_system):
    """Get performance metrics"""
    
    if not hasattr(trading_system, 'performance_analyzer'):
        return {'error': 'Performance analyzer not available'}
    
    # Get recent trades if available
    if hasattr(trading_system, 'live_trader'):
        results = trading_system.live_trader.get_results()
        
        if not results['trades'].empty:
            performance = trading_system.performance_analyzer.analyze_performance(
                results['trades'],
                results['equity_curve']
            )
            return performance
    
    return {'summary': {}, 'risk': {}, 'trades': {}}