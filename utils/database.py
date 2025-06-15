"""
Database management for trade history and system state
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

from config.settings import DatabaseConfig, DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """SQLite database manager"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DATA_DIR / 'trading.db')
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        
        with self.get_connection() as conn:
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    pnl REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    strategy TEXT,
                    parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Positions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    strategy TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Performance snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    daily_pnl REAL,
                    total_pnl REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    metadata TEXT
                )
            """)
            
            # System events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    metadata TEXT
                )
            """)
            
            # Optimization results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    target_return REAL,
                    achieved_return REAL,
                    parameters TEXT NOT NULL,
                    performance_metrics TEXT,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_time ON performance_snapshots(timestamp)")
            
            conn.commit()
            
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_trade(self, trade: dict):
        """Save trade to database"""
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (
                    symbol, entry_time, exit_time, entry_price, exit_price,
                    size, side, pnl, pnl_pct, exit_reason, strategy, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['symbol'],
                trade['entry_time'],
                trade.get('exit_time'),
                trade['entry_price'],
                trade.get('exit_price'),
                trade['size'],
                trade.get('side', 'long'),
                trade.get('pnl'),
                trade.get('pnl_pct'),
                trade.get('exit_reason'),
                trade.get('strategy'),
                json.dumps(trade.get('parameters', {}))
            ))
            conn.commit()
    
    def get_trades(
        self,
        symbol: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """Get trades from database"""
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY entry_time DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        # Parse JSON parameters
        if 'parameters' in df.columns:
            df['parameters'] = df['parameters'].apply(
                lambda x: json.loads(x) if x else {}
            )
        
        return df
    
    def save_position(self, position: dict):
        """Save or update position"""
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions (
                    symbol, size, entry_price, entry_time,
                    stop_loss, take_profit, strategy, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position['symbol'],
                position['size'],
                position['entry_price'],
                position['entry_time'],
                position.get('stop_loss'),
                position.get('take_profit'),
                position.get('strategy'),
                datetime.now()
            ))
            conn.commit()
    
    def get_positions(self) -> pd.DataFrame:
        """Get all open positions"""
        
        query = "SELECT * FROM positions WHERE size != 0"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
        
        return df
    
    def delete_position(self, symbol: str):
        """Delete position (when closed)"""
        
        with self.get_connection() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()
    
    def save_performance_snapshot(self, snapshot: dict):
        """Save performance snapshot"""
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO performance_snapshots (
                    timestamp, equity, cash, positions_value,
                    daily_pnl, total_pnl, win_rate, sharpe_ratio,
                    max_drawdown, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot['timestamp'],
                snapshot['equity'],
                snapshot['cash'],
                snapshot['positions_value'],
                snapshot.get('daily_pnl'),
                snapshot.get('total_pnl'),
                snapshot.get('win_rate'),
                snapshot.get('sharpe_ratio'),
                snapshot.get('max_drawdown'),
                json.dumps(snapshot.get('metadata', {}))
            ))
            conn.commit()
    
    def get_performance_history(
        self,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """Get performance history"""
        
        query = "SELECT * FROM performance_snapshots WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def log_event(self, event_type: str, message: str, metadata: dict = None):
        """Log system event"""
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO system_events (event_type, message, metadata)
                VALUES (?, ?, ?)
            """, (
                event_type,
                message,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
    
    def save_optimization_result(self, result: dict):
        """Save optimization result"""
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO optimization_results (
                    symbol, timestamp, target_return, achieved_return,
                    parameters, performance_metrics, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result['symbol'],
                datetime.now(),
                result.get('target_return'),
                result.get('performance', {}).get('monthly_return'),
                json.dumps(result.get('best_params', {})),
                json.dumps(result.get('performance', {})),
                result.get('success', False)
            ))
            conn.commit()
    
    def get_optimization_history(
        self,
        symbol: str = None,
        limit: int = 10
    ) -> pd.DataFrame:
        """Get optimization history"""
        
        query = "SELECT * FROM optimization_results WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Parse JSON fields
        for col in ['parameters', 'performance_metrics']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if x else {}
                )
        
        return df