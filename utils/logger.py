"""
Logging configuration for the trading system
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

from config.settings import LoggingConfig, LOG_DIR

colorama.init()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(name: str = None) -> logging.Logger:
    """Setup logging configuration"""
    
    logger = logging.getLogger(name or 'crypto_trading')
    logger.setLevel(getattr(logging, LoggingConfig.LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handlers
    for log_type, log_file in LoggingConfig.LOG_FILES.items():
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        if log_type == 'errors':
            file_handler.setLevel(logging.ERROR)
        else:
            file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Daily rotating handler
    daily_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / f"trading_{datetime.now().strftime('%Y%m')}.log",
        when='midnight',
        interval=1,
        backupCount=30
    )
    daily_handler.setLevel(logging.DEBUG)
    daily_handler.setFormatter(file_formatter)
    logger.addHandler(daily_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)