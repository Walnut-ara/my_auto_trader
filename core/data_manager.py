"""
Unified data management system for historical and real-time data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pybit.unified_trading import HTTP

from config.settings import DATA_DIR, TradingConfig

logger = logging.getLogger(__name__)


class DataManager:
    """Centralized data management"""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize data manager"""
        self.api_key = api_key or TradingConfig.API_KEY
        self.api_secret = api_secret or TradingConfig.API_SECRET
        self.client = None
        
        if self.api_key and self.api_secret:
            self.client = HTTP(
                testnet=TradingConfig.TESTNET,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
        
        self.cache_dir = DATA_DIR / "historical"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache for frequently accessed data
        self._cache = {}
        self._cache_expiry = {}
        
    def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get historical data with intelligent caching"""
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Check cache first
        cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
        
        if use_cache and cache_key in self._cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                logger.debug(f"Using memory cache for {cache_key}")
                return self._cache[cache_key].copy()
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded from cache: {cache_file.name}")
                
                # Update memory cache
                self._cache[cache_key] = df
                self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Download fresh data
        logger.info(f"Downloading {symbol} {interval} data from {start_date.date()} to {end_date.date()}")
        
        try:
            df = self._download_bybit_data(symbol, interval, start_date, end_date)
            
            if not df.empty:
                # Save to cache
                df.to_parquet(cache_file)
                
                # Update memory cache
                self._cache[cache_key] = df
                self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                
                logger.info(f"Downloaded {len(df)} bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return pd.DataFrame()
    
    def _download_bybit_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Download data from Bybit API"""
        
        if not self.client:
            raise ValueError("API client not initialized")
        
        # Convert to milliseconds
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        # Calculate interval in minutes
        interval_minutes = self._get_interval_minutes(interval)
        
        while current_ts < end_ts:
            try:
                response = self.client.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    start=current_ts,
                    end=min(current_ts + 200 * interval_minutes * 60 * 1000, end_ts),
                    limit=200
                )
                
                if response['retCode'] == 0:
                    data = response['result']['list']
                    if not data:
                        break
                    
                    all_data.extend(data)
                    
                    # Update timestamp for next batch
                    last_timestamp = int(data[0][0])
                    if last_timestamp <= current_ts:
                        break
                    current_ts = last_timestamp + 1
                    
                    time.sleep(0.1)  # Rate limiting
                else:
                    logger.error(f"API error: {response['retMsg']}")
                    break
                    
            except Exception as e:
                logger.error(f"Download error: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Process data
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert to float
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = df[col].astype(float)
        
        # Sort by time
        df.sort_index(inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def get_multiple_timeframes(
        self,
        symbol: str,
        intervals: List[str],
        lookback_days: int = 180
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        data_dict = {}
        
        with ThreadPoolExecutor(max_workers=len(intervals)) as executor:
            future_to_interval = {
                executor.submit(
                    self.get_historical_data,
                    symbol, interval, start_date, end_date
                ): interval
                for interval in intervals
            }
            
            for future in as_completed(future_to_interval):
                interval = future_to_interval[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data_dict[interval] = data
                        logger.info(f"Loaded {symbol} {interval}: {len(data)} bars")
                except Exception as e:
                    logger.error(f"Failed to load {symbol} {interval}: {e}")
        
        return data_dict
    
    def get_symbol_info(self, symbols: List[str] = None) -> pd.DataFrame:
        """Get symbol information"""
        
        if not self.client:
            raise ValueError("API client not initialized")
        
        try:
            response = self.client.get_instruments_info(category="linear")
            
            if response['retCode'] == 0:
                data = response['result']['list']
                df = pd.DataFrame(data)
                
                if symbols:
                    df = df[df['symbol'].isin(symbols)]
                
                # Select relevant columns
                columns = ['symbol', 'baseCoin', 'quoteCoin', 'status', 
                          'lotSizeFilter', 'priceFilter']
                df = df[columns]
                
                return df
            else:
                logger.error(f"Failed to get symbol info: {response['retMsg']}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return pd.DataFrame()
    
    def get_top_volume_symbols(self, n: int = 30, quote_currency: str = 'USDT') -> List[str]:
        """Get top symbols by 24h volume"""
        
        if not self.client:
            logger.warning("API client not initialized, using default symbols")
            from config.settings import SymbolConfig
            return SymbolConfig.TOP_SYMBOLS[:n]
        
        try:
            response = self.client.get_tickers(category="linear")
            
            if response['retCode'] == 0:
                tickers = response['result']['list']
                
                # Filter by quote currency
                tickers = [t for t in tickers if t['symbol'].endswith(quote_currency)]
                
                # Sort by volume
                tickers.sort(key=lambda x: float(x.get('volume24h', 0)), reverse=True)
                
                # Get top N symbols
                symbols = [t['symbol'] for t in tickers[:n]]
                
                logger.info(f"Found {len(symbols)} top volume symbols")
                return symbols
            else:
                logger.error(f"Failed to get tickers: {response['retMsg']}")
                from config.settings import SymbolConfig
                return SymbolConfig.TOP_SYMBOLS[:n]
                
        except Exception as e:
            logger.error(f"Error getting top symbols: {e}")
            from config.settings import SymbolConfig
            return SymbolConfig.TOP_SYMBOLS[:n]
    
    def calculate_indicators(self, df: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """Calculate technical indicators"""
        
        if indicators is None:
            indicators = ['RSI', 'SMA', 'EMA', 'ATR', 'BB']
        
        df = df.copy()
        
        # RSI
        if 'RSI' in indicators:
            df['RSI'] = self._calculate_rsi(df['close'])
        
        # Simple Moving Averages
        if 'SMA' in indicators:
            for period in [20, 50, 200]:
                df[f'SMA_{period}'] = df['close'].rolling(period).mean()
        
        # Exponential Moving Averages
        if 'EMA' in indicators:
            for period in [12, 26]:
                df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
        
        # Average True Range
        if 'ATR' in indicators:
            df['ATR'] = self._calculate_atr(df)
        
        # Bollinger Bands
        if 'BB' in indicators:
            df['BB_middle'] = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * std
            df['BB_lower'] = df['BB_middle'] - 2 * std
        
        # MACD
        if 'MACD' in indicators:
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Volume indicators
        if 'OBV' in indicators:
            df['OBV'] = self._calculate_obv(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        return pd.Series(obv, index=df.index)
    
    def _get_interval_minutes(self, interval: str) -> int:
        """Convert interval string to minutes"""
        if interval.endswith('D'):
            return int(interval[:-1]) * 1440
        elif interval.endswith('W'):
            return int(interval[:-1]) * 10080
        elif interval.endswith('M'):
            return int(interval[:-1]) * 43200
        else:
            return int(interval)
    
    def get_market_data_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Get market summary for multiple symbols"""
        
        if not self.client:
            return pd.DataFrame()
        
        try:
            response = self.client.get_tickers(category="linear")
            
            if response['retCode'] == 0:
                tickers = response['result']['list']
                
                # Filter for requested symbols
                tickers = [t for t in tickers if t['symbol'] in symbols]
                
                # Create DataFrame
                df = pd.DataFrame(tickers)
                
                # Select and rename columns
                columns_map = {
                    'symbol': 'Symbol',
                    'lastPrice': 'Price',
                    'price24hPcnt': 'Change_24h',
                    'volume24h': 'Volume_24h',
                    'turnover24h': 'Turnover_24h',
                    'highPrice24h': 'High_24h',
                    'lowPrice24h': 'Low_24h'
                }
                
                df = df[list(columns_map.keys())]
                df.rename(columns=columns_map, inplace=True)
                
                # Convert to numeric
                numeric_columns = ['Price', 'Change_24h', 'Volume_24h', 
                                 'Turnover_24h', 'High_24h', 'Low_24h']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Calculate additional metrics
                df['Volatility'] = (df['High_24h'] - df['Low_24h']) / df['Price']
                df['Change_24h'] = df['Change_24h'] * 100  # Convert to percentage
                
                return df.sort_values('Volume_24h', ascending=False)
            
        except Exception as e:
            logger.error(f"Error getting market summary: {e}")
            return pd.DataFrame()
    
    def save_analysis_data(self, symbol: str, data: Dict, analysis_type: str = 'general'):
        """Save analysis results"""
        
        analysis_dir = DATA_DIR / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        filename = f"{symbol}_{analysis_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = analysis_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved analysis to {filepath}")
    
    def load_optimized_parameters(self, symbol: str) -> Optional[Dict]:
        """Load optimized parameters for a symbol"""
        
        optimized_dir = DATA_DIR / 'optimized'
        param_file = optimized_dir / f"{symbol}_params.json"
        
        if param_file.exists():
            try:
                with open(param_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded optimized parameters for {symbol}")
                    return data
            except Exception as e:
                logger.error(f"Failed to load parameters for {symbol}: {e}")
        
        return None
    
    def save_optimized_parameters(self, symbol: str, params: Dict, performance: Dict):
        """Save optimized parameters"""
        
        optimized_dir = DATA_DIR / 'optimized'
        optimized_dir.mkdir(exist_ok=True)
        
        data = {
            'symbol': symbol,
            'parameters': params,
            'performance': performance,
            'optimization_date': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        param_file = optimized_dir / f"{symbol}_params.json"
        with open(param_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved optimized parameters for {symbol}")
    
    def clean_cache(self, days_old: int = 7):
        """Clean old cache files"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            if cache_file.stat().st_mtime < cutoff_date.timestamp():
                cache_file.unlink()
                logger.info(f"Deleted old cache file: {cache_file.name}")
        
        # Clear memory cache
        self._cache.clear()
        self._cache_expiry.clear()