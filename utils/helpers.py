"""
Utility helper functions
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)


def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size


def calculate_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float
) -> float:
    """Calculate position size based on risk"""
    
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_price) / entry_price
    
    if price_risk > 0:
        position_value = risk_amount / price_risk
        position_size = position_value / entry_price
        return position_size
    
    return 0


def format_number(
    value: Union[int, float],
    decimal_places: int = 2,
    prefix: str = '',
    suffix: str = ''
) -> str:
    """Format number for display"""
    
    if pd.isna(value):
        return 'N/A'
    
    if isinstance(value, (int, float)):
        if abs(value) >= 1e9:
            return f"{prefix}{value/1e9:.{decimal_places}f}B{suffix}"
        elif abs(value) >= 1e6:
            return f"{prefix}{value/1e6:.{decimal_places}f}M{suffix}"
        elif abs(value) >= 1e3:
            return f"{prefix}{value/1e3:.{decimal_places}f}K{suffix}"
        else:
            return f"{prefix}{value:.{decimal_places}f}{suffix}"
    
    return str(value)


def calculate_returns(
    prices: pd.Series,
    method: str = 'simple'
) -> pd.Series:
    """Calculate returns from price series"""
    
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method: {method}")


def resample_data(
    data: pd.DataFrame,
    source_timeframe: str,
    target_timeframe: str
) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe"""
    
    # Convert timeframe strings to pandas offset
    timeframe_map = {
        '1': '1T',
        '5': '5T',
        '15': '15T',
        '30': '30T',
        '60': '1H',
        '240': '4H',
        '1D': '1D'
    }
    
    source_offset = timeframe_map.get(source_timeframe, source_timeframe)
    target_offset = timeframe_map.get(target_timeframe, target_timeframe)
    
    # Resample
    resampled = data.resample(target_offset).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


def validate_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV data integrity"""
    
    if data.empty:
        return False
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        logger.error(f"Missing required columns. Found: {data.columns.tolist()}")
        return False
    
    # Check for NaN values
    if data[required_columns].isna().any().any():
        logger.warning("Data contains NaN values")
        return False
    
    # Check OHLC relationships
    invalid_bars = (
        (data['high'] < data['low']) |
        (data['high'] < data['open']) |
        (data['high'] < data['close']) |
        (data['low'] > data['open']) |
        (data['low'] > data['close'])
    )
    
    if invalid_bars.any():
        logger.warning(f"Found {invalid_bars.sum()} invalid OHLC bars")
        return False
    
    return True


def merge_timeframes(
    data_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Merge multiple timeframe data"""
    
    if not data_dict:
        return pd.DataFrame()
    
    # Use highest resolution data as base
    base_tf = min(data_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    merged = data_dict[base_tf].copy()
    
    # Add indicators from other timeframes
    for tf, data in data_dict.items():
        if tf == base_tf:
            continue
        
        # Resample to base timeframe
        for col in data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                # Forward fill higher timeframe indicators
                merged[f'{col}_{tf}'] = data[col].reindex(merged.index, method='ffill')
    
    return merged


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio"""
    
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if returns.std() == 0:
        return 0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio"""
    
    if len(returns) < 2:
        return 0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
    """Calculate maximum drawdown and duration"""
    
    if len(equity_curve) < 2:
        return {'max_drawdown': 0, 'max_duration': 0}
    
    # Calculate running maximum
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    
    # Calculate drawdown duration
    drawdown_start = None
    max_duration = 0
    current_duration = 0
    
    for i in range(len(drawdown)):
        if drawdown.iloc[i] < 0:
            if drawdown_start is None:
                drawdown_start = i
            current_duration = i - drawdown_start
        else:
            if current_duration > max_duration:
                max_duration = current_duration
            drawdown_start = None
            current_duration = 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_duration': max_duration
    }


def hash_params(params: Dict[str, Any]) -> str:
    """Create hash of parameters for caching"""
    
    # Sort keys for consistent hashing
    sorted_params = json.dumps(params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


def create_dirs():
    """Create necessary directories"""
    
    from config.settings import DATA_DIR, LOG_DIR, REPORT_DIR
    
    dirs = [
        DATA_DIR,
        LOG_DIR,
        REPORT_DIR,
        DATA_DIR / 'historical',
        DATA_DIR / 'optimized',
        DATA_DIR / 'analysis',
        DATA_DIR / 'reports'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files
    for dir_path in dirs:
        gitkeep = Path(dir_path) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()


def load_json_config(filepath: Union[str, Path]) -> Dict:
    """Load JSON configuration file"""
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {filepath}: {e}")
        return {}


def save_json_config(data: Dict, filepath: Union[str, Path]):
    """Save configuration to JSON file"""
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Config saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving config to {filepath}: {e}")