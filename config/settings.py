"""
Global configuration settings for the crypto trading system
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
REPORT_DIR = DATA_DIR / "reports"

# Ensure directories exist
for dir_path in [DATA_DIR, LOG_DIR, REPORT_DIR, 
                 DATA_DIR / "historical", 
                 DATA_DIR / "optimized",
                 DATA_DIR / "analysis"]:
    dir_path.mkdir(parents=True, exist_ok=True)
    

class BrokerConfig:
    API_KEY = "your_api_key"
    API_SECRET = "your_api_secret"
    BASE_URL = "https://api-testnet.bybit.com"

# Trading settings
class TradingConfig:
    """Trading configuration"""
    
    # API Configuration
    API_KEY = os.getenv('BYBIT_API_KEY', '')
    API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    TESTNET = os.getenv('BYBIT_TESTNET', 'True').lower() == 'true'
    
    # Default trading parameters
    DEFAULT_POSITION_SIZE_USD = 100
    DEFAULT_LEVERAGE = 1
    MAX_POSITIONS = 10
    
    # Risk management
    MAX_DAILY_LOSS_PCT = 0.05  # 5%
    MAX_PORTFOLIO_RISK_PCT = 0.20  # 20%
    DEFAULT_STOP_LOSS_PCT = 0.02  # 2%
    DEFAULT_TAKE_PROFIT_PCT = 0.03  # 3%
    
    # Timeframes for analysis
    TIMEFRAMES = ['1', '5', '15', '60', '240', '1D']
    DEFAULT_TIMEFRAME = '15'
    
    # Backtesting
    BACKTEST_INITIAL_CAPITAL = 10000
    BACKTEST_COMMISSION = 0.0006  # 0.06%
    BACKTEST_SLIPPAGE = 0.0001  # 0.01%
    


class BybitConfig:
    """Bybit-specific configuration"""
    
    # API Configuration
    API_KEY = os.getenv('BYBIT_API_KEY', '')
    API_SECRET = os.getenv('BYBIT_API_SECRET', '')
    TESTNET = os.getenv('BYBIT_TESTNET', 'True').lower() == 'true'
    
    # Bybit API settings
    RECV_WINDOW = 5000
    BASE_URL = "https://api-testnet.bybit.com" if TESTNET else "https://api.bybit.com"
    WS_URL = "wss://stream-testnet.bybit.com" if TESTNET else "wss://stream.bybit.com"
    
    # Rate limits (requests per second)
    RATE_LIMIT = {
        'order': 10,
        'query': 50,
        'other': 20
    }
    
    # Order settings
    TIME_IN_FORCE = 'GTC'  # Good Till Cancel
    
    # Symbol settings
    DEFAULT_CATEGORY = 'linear'  # USDT perpetual
    
    
class MonthlyTargetConfig:
    """Configuration for achieving monthly targets"""
    
    # Target settings
    MONTHLY_TARGET = 0.01  # 1% monthly target
    DAILY_TARGET = MONTHLY_TARGET / 22  # ~22 trading days per month
    
    # Risk limits for 1% monthly target
    MAX_DAILY_DRAWDOWN = 0.003  # 0.3% max daily loss
    MAX_CONSECUTIVE_LOSSES = 3  # Stop after 3 consecutive losses
    POSITION_SIZE_MULTIPLIER = 0.5  # Conservative position sizing
    
    # Market condition adjustments
    VOLATILITY_THRESHOLDS = {
        'low': 0.01,      # < 1% daily volatility
        'medium': 0.02,   # 1-2% daily volatility  
        'high': 0.03,     # 2-3% daily volatility
        'extreme': 0.05   # > 5% daily volatility
    }
    
    # Position sizing by volatility
    POSITION_SIZE_BY_VOLATILITY = {
        'low': 0.10,      # 10% position size
        'medium': 0.08,   # 8% position size
        'high': 0.05,     # 5% position size
        'extreme': 0.02   # 2% position size
    }
    
    # Stop/Target by volatility
    STOPS_BY_VOLATILITY = {
        'low': {'stop': 0.005, 'target': 0.01},     # 0.5% stop, 1% target
        'medium': {'stop': 0.01, 'target': 0.02},   # 1% stop, 2% target
        'high': {'stop': 0.015, 'target': 0.03},    # 1.5% stop, 3% target
        'extreme': {'stop': 0.02, 'target': 0.04}   # 2% stop, 4% target
    }

class OptimizationConfig:
    """Optimization configuration"""
    
    # Optuna settings
    DEFAULT_N_TRIALS = 100
    AGGRESSIVE_N_TRIALS = 300
    N_JOBS = -1  # Use all CPU cores
    
    # Parameter ranges
    PARAMETER_RANGES = {
        'conservative': {
            'rsi_period': [14, 20, 30],
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'stop_loss': [0.01, 0.015, 0.02],
            'take_profit': [0.02, 0.025, 0.03],
            'position_size_pct': [0.05, 0.1]
        },
        'moderate': {
            'rsi_period': [10, 14, 20, 30],
            'rsi_oversold': [20, 25, 30, 35],
            'rsi_overbought': [65, 70, 75, 80],
            'stop_loss': [0.01, 0.015, 0.02, 0.025],
            'take_profit': [0.02, 0.03, 0.04, 0.05],
            'position_size_pct': [0.05, 0.1, 0.15]
        },
        'aggressive': {
            'rsi_period': [8, 10, 14, 20, 30],
            'rsi_oversold': [15, 20, 25, 30, 35, 40],
            'rsi_overbought': [60, 65, 70, 75, 80, 85],
            'stop_loss': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
            'take_profit': [0.02, 0.03, 0.04, 0.05, 0.08, 0.10],
            'position_size_pct': [0.05, 0.1, 0.15, 0.2]
        }
    }
    
    # Target returns by volatility
    TARGET_RETURNS = {
        'low_volatility': [0.005, 0.01, 0.015],
        'medium_volatility': [0.01, 0.02, 0.03],
        'high_volatility': [0.02, 0.05, 0.08],
        'very_high_volatility': [0.05, 0.08, 0.10, 0.15]
    }

class AnalysisConfig:
    """Analysis configuration"""
    
    # Market analysis periods
    SHORT_TERM_DAYS = 30
    MEDIUM_TERM_DAYS = 90
    LONG_TERM_DAYS = 180
    
    # Technical indicators
    INDICATORS = {
        'trend': ['SMA', 'EMA', 'ADX', 'MACD'],
        'momentum': ['RSI', 'STOCH', 'CCI', 'MFI'],
        'volatility': ['ATR', 'BB', 'KC', 'DC'],
        'volume': ['OBV', 'VWAP', 'CMF', 'FI']
    }
    
    # Market regime thresholds
    VOLATILITY_THRESHOLDS = {
        'very_low': 0.01,
        'low': 0.02,
        'medium': 0.03,
        'high': 0.05,
        'very_high': 0.08
    }
    
    # Performance metrics
    PERFORMANCE_METRICS = [
        'total_return', 'monthly_return', 'sharpe_ratio',
        'sortino_ratio', 'max_drawdown', 'win_rate',
        'profit_factor', 'avg_win', 'avg_loss',
        'trades_per_month', 'consecutive_wins', 'consecutive_losses'
    ]

class ReportingConfig:
    """Reporting configuration"""
    
    # Report types
    REPORT_TYPES = ['daily', 'weekly', 'monthly', 'optimization', 'backtest']
    
    # Chart settings
    CHART_STYLE = 'darkgrid'
    CHART_DPI = 100
    CHART_SIZE = (12, 8)
    
    # Email settings (optional)
    EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
    EMAIL_SMTP_SERVER = os.getenv('EMAIL_SMTP_SERVER', '')
    EMAIL_SMTP_PORT = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '').split(',')

class DatabaseConfig:
    """Database configuration"""
    
    # SQLite by default, can be changed to PostgreSQL/MySQL
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        f"sqlite:///{DATA_DIR}/trading.db"
    )
    
    # Connection settings
    POOL_SIZE = 5
    MAX_OVERFLOW = 10
    POOL_TIMEOUT = 30

class LoggingConfig:
    """Logging configuration"""
    
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Separate logs for different components
    LOG_FILES = {
        'trading': LOG_DIR / 'trading.log',
        'optimization': LOG_DIR / 'optimization.log',
        'analysis': LOG_DIR / 'analysis.log',
        'errors': LOG_DIR / 'errors.log'
    }

# Symbol lists
class SymbolConfig:
    """Symbol configuration"""
    
    # Top trading pairs by volume
    TOP_SYMBOLS = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'ADAUSDT', 'AVAXUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT',
        'LINKUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
        'XLMUSDT', 'NEARUSDT', 'ALGOUSDT', 'FILUSDT', 'VETUSDT',
        'ICPUSDT', 'TRXUSDT', 'FTMUSDT', 'MANAUSDT', 'HBARUSDT',
        'AAVEUSDT', 'SANDUSDT', 'AXSUSDT', 'EGLDUSDT', 'THETAUSDT'
    ]
    
    # Stable coins (excluded from trading)
    STABLE_COINS = ['USDTUSDT', 'BUSDUSDT', 'USDCUSDT', 'DAIUSDT']
    
    # Categories
    SYMBOL_CATEGORIES = {
        'large_cap': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'],
        'defi': ['UNIUSDT', 'AAVEUSDT', 'LINKUSDT', 'MATICUSDT'],
        'gaming': ['SANDUSDT', 'AXSUSDT', 'MANAUSDT'],
        'layer1': ['SOLUSDT', 'AVAXUSDT', 'NEARUSDT', 'ALGOUSDT'],
        'meme': ['DOGEUSDT', 'SHIBUSDT']
    }

# Dashboard settings
class DashboardConfig:
    """Dashboard configuration"""
    
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
    
    # WebSocket settings
    WEBSOCKET_PING_INTERVAL = 25
    WEBSOCKET_PING_TIMEOUT = 120
    
    # Update intervals (seconds)
    PRICE_UPDATE_INTERVAL = 1
    PORTFOLIO_UPDATE_INTERVAL = 5
    PERFORMANCE_UPDATE_INTERVAL = 60

# Feature flags
class FeatureFlags:
    """Feature toggles"""
    
    ENABLE_LIVE_TRADING = os.getenv('ENABLE_LIVE_TRADING', 'False').lower() == 'true'
    ENABLE_PAPER_TRADING = os.getenv('ENABLE_PAPER_TRADING', 'True').lower() == 'true'
    ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'True').lower() == 'true'
    ENABLE_AUTO_OPTIMIZATION = os.getenv('ENABLE_AUTO_OPTIMIZATION', 'False').lower() == 'true'
    ENABLE_MULTI_STRATEGY = os.getenv('ENABLE_MULTI_STRATEGY', 'False').lower() == 'true'

# Export all configs
__all__ = [
    'TradingConfig',
    'OptimizationConfig',
    'AnalysisConfig',
    'ReportingConfig',
    'DatabaseConfig',
    'LoggingConfig',
    'SymbolConfig',
    'DashboardConfig',
    'FeatureFlags',
    'BASE_DIR',
    'DATA_DIR',
    'LOG_DIR',
    'REPORT_DIR'
]