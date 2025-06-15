"""
Input validation utilities
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom validation error"""
    pass


class Validators:
    """Collection of validation functions"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol format"""
        
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        
        # Check format (e.g., BTCUSDT, ETHUSDT)
        pattern = r'^[A-Z]{2,10}USDT$'
        
        if not re.match(pattern, symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")
        
        return True
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        """Validate timeframe string"""
        
        valid_timeframes = ['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', '1D', '1W', '1M']
        
        if timeframe not in valid_timeframes:
            raise ValidationError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        
        return True
    
    @staticmethod
    def validate_price(price: float, min_price: float = 0.0) -> bool:
        """Validate price value"""
        
        if not isinstance(price, (int, float)):
            raise ValidationError(f"Price must be numeric, got {type(price)}")
        
        if price <= min_price:
            raise ValidationError(f"Price must be greater than {min_price}")
        
        if np.isnan(price) or np.isinf(price):
            raise ValidationError("Price cannot be NaN or infinite")
        
        return True
    
    @staticmethod
    def validate_quantity(quantity: float, min_quantity: float = 0.0) -> bool:
        """Validate quantity value"""
        
        if not isinstance(quantity, (int, float)):
            raise ValidationError(f"Quantity must be numeric, got {type(quantity)}")
        
        if quantity <= min_quantity:
            raise ValidationError(f"Quantity must be greater than {min_quantity}")
        
        return True
    
    @staticmethod
    def validate_percentage(value: float, name: str = "Value") -> bool:
        """Validate percentage value (0-1 range)"""
        
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric")
        
        if value < 0 or value > 1:
            raise ValidationError(f"{name} must be between 0 and 1")
        
        return True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame structure"""
        
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise ValidationError("DataFrame is empty")
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN in critical columns
        for col in required_columns:
            if df[col].isna().any():
                raise ValidationError(f"Column '{col}' contains NaN values")
        
        return True
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> bool:
        """Validate OHLCV data integrity"""
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Basic structure validation
        Validators.validate_dataframe(df, required_columns)
        
        # OHLC relationship validation
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_rows.any():
            raise ValidationError(f"Found {invalid_rows.sum()} rows with invalid OHLC relationships")
        
        # Check for negative prices
        if (df[['open', 'high', 'low', 'close']] < 0).any().any():
            raise ValidationError("Negative prices found in data")
        
        # Check for zero volume (warning only)
        zero_volume_count = (df['volume'] == 0).sum()
        if zero_volume_count > len(df) * 0.1:  # More than 10% zero volume
            print(f"Warning: {zero_volume_count} rows with zero volume")
        
        return True
    
    @staticmethod
    def validate_date_range(start_date: Any, end_date: Any) -> bool:
        """Validate date range"""
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        if not isinstance(start_date, (datetime, pd.Timestamp)):
            raise ValidationError("start_date must be a datetime object")
        
        if not isinstance(end_date, (datetime, pd.Timestamp)):
            raise ValidationError("end_date must be a datetime object")
        
        if start_date >= end_date:
            raise ValidationError("start_date must be before end_date")
        
        # Check if dates are not too far in the past
        if start_date < pd.Timestamp('2010-01-01'):
            raise ValidationError("start_date is too far in the past")
        
        # Check if end_date is not in the future
        if end_date > pd.Timestamp.now() + pd.Timedelta(days=1):
            raise ValidationError("end_date cannot be in the future")
        
        return True
    
    @staticmethod
    def validate_strategy_params(params: Dict[str, Any], strategy_type: str) -> bool:
        """Validate strategy parameters"""
        
        if not isinstance(params, dict):
            raise ValidationError("Parameters must be a dictionary")
        
        # Common parameter validation
        if 'position_size_pct' in params:
            Validators.validate_percentage(params['position_size_pct'], 'position_size_pct')
        
        if 'stop_loss' in params:
            Validators.validate_percentage(params['stop_loss'], 'stop_loss')
        
        if 'take_profit' in params:
            Validators.validate_percentage(params['take_profit'], 'take_profit')
        
        # Strategy-specific validation
        if strategy_type == 'rsi_mean_reversion':
            required = ['rsi_period', 'rsi_oversold', 'rsi_overbought']
            for param in required:
                if param not in params:
                    raise ValidationError(f"Missing required parameter: {param}")
            
            if params['rsi_oversold'] >= params['rsi_overbought']:
                raise ValidationError("rsi_oversold must be less than rsi_overbought")
        
        elif strategy_type == 'trend_following':
            required = ['ema_fast', 'ema_slow']
            for param in required:
                if param not in params:
                    raise ValidationError(f"Missing required parameter: {param}")
            
            if params['ema_fast'] >= params['ema_slow']:
                raise ValidationError("ema_fast must be less than ema_slow")
        
        return True
    
    @staticmethod
    def validate_api_credentials(api_key: str, api_secret: str) -> bool:
        """Validate API credentials format"""
        
        if not api_key or not isinstance(api_key, str):
            raise ValidationError("API key must be a non-empty string")
        
        if not api_secret or not isinstance(api_secret, str):
            raise ValidationError("API secret must be a non-empty string")
        
        # Check minimum length
        if len(api_key) < 10:
            raise ValidationError("API key seems too short")
        
        if len(api_secret) < 10:
            raise ValidationError("API secret seems too short")
        
        # Check for placeholder values
        if api_key.lower() in ['your_api_key', 'api_key_here', 'xxx']:
            raise ValidationError("API key appears to be a placeholder")
        
        return True
    
    @staticmethod
    def validate_order_params(order_type: str, side: str, quantity: float, 
                            price: Optional[float] = None) -> bool:
        """Validate order parameters"""
        
        # Validate order type
        valid_order_types = ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
        if order_type.upper() not in valid_order_types:
            raise ValidationError(f"Invalid order type: {order_type}")
        
        # Validate side
        valid_sides = ['BUY', 'SELL']
        if side.upper() not in valid_sides:
            raise ValidationError(f"Invalid order side: {side}")
        
        # Validate quantity
        Validators.validate_quantity(quantity)
        
        # Validate price for limit orders
        if order_type.upper() in ['LIMIT', 'STOP_LIMIT'] and price is None:
            raise ValidationError(f"Price is required for {order_type} orders")
        
        if price is not None:
            Validators.validate_price(price)
        
        return True
    
    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize string input"""
        
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\"\'&]', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized


class DataValidator:
    """Validate data integrity and quality"""
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality check"""
        
        report = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min(),
                'end': df.index.max()
            },
            'missing_data': {},
            'outliers': {},
            'issues': []
        }
        
        # Check for missing data
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Check for outliers (using IQR method)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Check for data gaps
        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(df.index)
            if expected_freq:
                date_range = pd.date_range(
                    start=df.index.min(),
                    end=df.index.max(),
                    freq=expected_freq
                )
                missing_dates = date_range.difference(df.index)
                if len(missing_dates) > 0:
                    report['issues'].append({
                        'type': 'missing_dates',
                        'count': len(missing_dates),
                        'description': f"Found {len(missing_dates)} missing timestamps"
                    })
        
        # Check for duplicate indices
        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            report['issues'].append({
                'type': 'duplicate_indices',
                'count': duplicate_count,
                'description': f"Found {duplicate_count} duplicate indices"
            })
        
        return report
    
    @staticmethod
    def validate_backtest_results(results: Dict[str, Any]) -> bool:
        """Validate backtest results structure"""
        
        required_keys = ['trades', 'equity_curve', 'metrics']
        
        for key in required_keys:
            if key not in results:
                raise ValidationError(f"Missing required key in results: {key}")
        
        # Validate trades DataFrame
        if not results['trades'].empty:
            required_trade_columns = ['entry_time', 'exit_time', 'side', 'entry_price', 
                                     'exit_price', 'quantity', 'pnl']
            
            missing_columns = set(required_trade_columns) - set(results['trades'].columns)
            if missing_columns:
                raise ValidationError(f"Missing trade columns: {missing_columns}")
        
        # Validate equity curve
        if results['equity_curve'].empty:
            raise ValidationError("Equity curve is empty")
        
        if 'equity' not in results['equity_curve'].columns:
            raise ValidationError("Equity curve missing 'equity' column")
        
        # Check for negative equity
        if (results['equity_curve']['equity'] < 0).any():
            raise ValidationError("Negative equity detected in results")
        
        return True