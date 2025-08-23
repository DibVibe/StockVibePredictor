"""
Chart Configuration Module
===========================
This module contains all configuration settings for chart data fetching,
timeframe definitions, and display parameters.

Author: StockVibePredictor Team
Created: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ChartType(Enum):
    """Supported chart types"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    OHLC = "ohlc"
    AREA = "area"
    HEIKIN_ASHI = "heikin_ashi"
    RENKO = "renko"


class Interval(Enum):
    """Available data intervals"""
    ONE_MIN = "1m"
    TWO_MIN = "2m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    THIRTY_MIN = "30m"
    SIXTY_MIN = "60m"
    NINETY_MIN = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTH = "3mo"


class Period(Enum):
    """Available data periods"""
    ONE_DAY = "1d"
    FIVE_DAY = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTH = "3mo"
    SIX_MONTH = "6mo"
    ONE_YEAR = "1y"
    TWO_YEAR = "2y"
    FIVE_YEAR = "5y"
    TEN_YEAR = "10y"
    YTD = "ytd"
    MAX = "max"


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class TimeframeConfig:
    """
    Configuration for a specific timeframe

    Attributes:
        period: The period of historical data to fetch
        interval: The interval between data points
        model_suffix: Suffix for model files related to this timeframe
        cache_time: Cache duration in seconds
        display_limit: Maximum number of data points to display
        description: Human-readable description
        business_days_only: Whether to filter for business days only
        min_data_points: Minimum required data points for valid analysis
        max_data_age: Maximum age of data in seconds before refresh
        indicators: Default indicators to calculate for this timeframe
    """
    period: str
    interval: str
    model_suffix: str
    cache_time: int
    display_limit: int
    description: str
    business_days_only: bool = False
    min_data_points: int = 10
    max_data_age: int = 86400  # 24 hours default
    indicators: List[str] = field(default_factory=list)

    def validate_data_freshness(self, latest_timestamp, current_time) -> bool:
        """Check if data is fresh enough for this timeframe"""
        if latest_timestamp is None or current_time is None:
            return False

        time_diff = (current_time - latest_timestamp).total_seconds()
        is_fresh = time_diff < self.max_data_age

        if not is_fresh:
            logger.warning(
                f"Data staleness detected: {time_diff:.0f}s old "
                f"(max allowed: {self.max_data_age}s)"
            )

        return is_fresh

    def get_cache_key_prefix(self) -> str:
        """Generate cache key prefix for this timeframe"""
        return f"chart_{self.period}_{self.interval}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "period": self.period,
            "interval": self.interval,
            "model_suffix": self.model_suffix,
            "cache_time": self.cache_time,
            "display_limit": self.display_limit,
            "description": self.description,
            "business_days_only": self.business_days_only,
            "min_data_points": self.min_data_points,
            "max_data_age": self.max_data_age,
            "indicators": self.indicators
        }


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    display_name: str
    period: int
    color: str = "#0000FF"
    line_width: int = 1
    overlay: bool = True  # True if overlays on price chart
    parameters: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TIMEFRAME CONFIGURATIONS
# ============================================================================

TIMEFRAME_CONFIGS = {
    "1d": TimeframeConfig(
        period="7d",
        interval="5m",
        model_suffix="_1d",
        cache_time=300,  # 5 minutes
        display_limit=288,  # 24 hours * 12 (5-min intervals)
        description="1 Day - Intraday predictions with 5-minute intervals",
        business_days_only=True,
        min_data_points=50,
        max_data_age=900,  # 15 minutes for intraday
        indicators=["VWAP", "RSI", "MACD", "Volume"]
    ),

    "5d": TimeframeConfig(
        period="1mo",
        interval="15m",
        model_suffix="_1w",
        cache_time=600,  # 10 minutes
        display_limit=480,  # 5 days * 24 hours * 4 (15-min intervals)
        description="5 Days - Short-term with 15-minute intervals",
        business_days_only=True,
        min_data_points=100,
        max_data_age=3600,  # 1 hour
        indicators=["SMA20", "RSI", "MACD", "BollingerBands", "Volume"]
    ),

    "1w": TimeframeConfig(
        period="2mo",
        interval="1h",
        model_suffix="_1w",
        cache_time=1800,  # 30 minutes
        display_limit=168,  # 7 days * 24 hours
        description="1 Week - 7 days of hourly data",
        business_days_only=False,
        min_data_points=50,
        max_data_age=7200,  # 2 hours
        indicators=["SMA20", "SMA50", "RSI", "MACD", "Volume"]
    ),

    "1mo": TimeframeConfig(
        period="3mo",
        interval="1d",
        model_suffix="_1mo",
        cache_time=3600,  # 1 hour
        display_limit=30,  # ~30 days
        description="1 Month - 30 days of daily data",
        business_days_only=False,
        min_data_points=20,
        max_data_age=86400,  # 24 hours
        indicators=["SMA20", "SMA50", "RSI", "MACD", "BollingerBands"]
    ),

    "3mo": TimeframeConfig(
        period="6mo",
        interval="1d",
        model_suffix="_1mo",
        cache_time=5400,  # 1.5 hours
        display_limit=90,  # ~90 days
        description="3 Months - 90 days of daily data",
        business_days_only=False,
        min_data_points=60,
        max_data_age=86400,  # 24 hours
        indicators=["SMA50", "SMA200", "RSI", "MACD", "ADX"]
    ),

    "6mo": TimeframeConfig(
        period="1y",
        interval="1d",
        model_suffix="_1mo",
        cache_time=7200,  # 2 hours
        display_limit=180,  # ~180 days
        description="6 Months - 180 days of daily data",
        business_days_only=False,
        min_data_points=100,
        max_data_age=86400,  # 24 hours
        indicators=["SMA50", "SMA200", "RSI", "MACD", "ADX", "OBV"]
    ),

    "1y": TimeframeConfig(
        period="2y",
        interval="1d",
        model_suffix="_1y",
        cache_time=21600,  # 6 hours
        display_limit=365,  # 365 days
        description="1 Year - 365 days of daily data",
        business_days_only=False,
        min_data_points=200,
        max_data_age=172800,  # 48 hours
        indicators=["SMA50", "SMA200", "RSI", "MACD", "Stochastic"]
    ),

    "2y": TimeframeConfig(
        period="3y",
        interval="1wk",
        model_suffix="_1y",
        cache_time=28800,  # 8 hours
        display_limit=104,  # 2 years * 52 weeks
        description="2 Years - 104 weeks of weekly data",
        business_days_only=False,
        min_data_points=50,
        max_data_age=259200,  # 72 hours
        indicators=["SMA20", "SMA50", "RSI", "MACD"]
    ),

    "5y": TimeframeConfig(
        period="max",
        interval="1mo",
        model_suffix="_1y",
        cache_time=43200,  # 12 hours
        display_limit=60,  # 5 years * 12 months
        description="5 Years - 60 months of monthly data",
        business_days_only=False,
        min_data_points=30,
        max_data_age=604800,  # 1 week
        indicators=["SMA12", "SMA24", "RSI", "ROC"]
    ),
}


# ============================================================================
# INDICATOR CONFIGURATIONS
# ============================================================================

INDICATOR_CONFIGS = {
    "SMA20": IndicatorConfig(
        name="SMA20",
        display_name="Simple Moving Average (20)",
        period=20,
        color="#FFA500",
        overlay=True
    ),
    "SMA50": IndicatorConfig(
        name="SMA50",
        display_name="Simple Moving Average (50)",
        period=50,
        color="#FF0000",
        overlay=True
    ),
    "SMA200": IndicatorConfig(
        name="SMA200",
        display_name="Simple Moving Average (200)",
        period=200,
        color="#800080",
        overlay=True
    ),
    "EMA12": IndicatorConfig(
        name="EMA12",
        display_name="Exponential Moving Average (12)",
        period=12,
        color="#00FF00",
        overlay=True
    ),
    "EMA26": IndicatorConfig(
        name="EMA26",
        display_name="Exponential Moving Average (26)",
        period=26,
        color="#0000FF",
        overlay=True
    ),
    "RSI": IndicatorConfig(
        name="RSI",
        display_name="Relative Strength Index",
        period=14,
        color="#FF00FF",
        overlay=False,
        parameters={"overbought": 70, "oversold": 30}
    ),
    "MACD": IndicatorConfig(
        name="MACD",
        display_name="MACD",
        period=0,
        color="#008080",
        overlay=False,
        parameters={"fast": 12, "slow": 26, "signal": 9}
    ),
    "BollingerBands": IndicatorConfig(
        name="BollingerBands",
        display_name="Bollinger Bands",
        period=20,
        color="#808080",
        overlay=True,
        parameters={"std_dev": 2}
    ),
    "Volume": IndicatorConfig(
        name="Volume",
        display_name="Volume",
        period=0,
        color="#4169E1",
        overlay=False
    ),
    "VWAP": IndicatorConfig(
        name="VWAP",
        display_name="Volume Weighted Average Price",
        period=0,
        color="#FF1493",
        overlay=True
    ),
    "ADX": IndicatorConfig(
        name="ADX",
        display_name="Average Directional Index",
        period=14,
        color="#DAA520",
        overlay=False
    ),
    "Stochastic": IndicatorConfig(
        name="Stochastic",
        display_name="Stochastic Oscillator",
        period=14,
        color="#00CED1",
        overlay=False,
        parameters={"k_period": 14, "d_period": 3}
    ),
    "OBV": IndicatorConfig(
        name="OBV",
        display_name="On Balance Volume",
        period=0,
        color="#8B4513",
        overlay=False
    ),
    "ROC": IndicatorConfig(
        name="ROC",
        display_name="Rate of Change",
        period=12,
        color="#FF6347",
        overlay=False
    ),
}


# ============================================================================
# MARKET CONFIGURATIONS
# ============================================================================

MARKET_HOURS = {
    "US": {
        "pre_market_start": "04:00",
        "pre_market_end": "09:30",
        "market_open": "09:30",
        "market_close": "16:00",
        "after_hours_start": "16:00",
        "after_hours_end": "20:00",
        "timezone": "America/New_York",
        "trading_days": [0, 1, 2, 3, 4],  # Monday to Friday
    },
    "IN": {
        "pre_market_start": "09:00",
        "pre_market_end": "09:15",
        "market_open": "09:15",
        "market_close": "15:30",
        "after_hours_start": "15:30",
        "after_hours_end": "16:00",
        "timezone": "Asia/Kolkata",
        "trading_days": [0, 1, 2, 3, 4],
    },
    "UK": {
        "pre_market_start": "07:00",
        "pre_market_end": "08:00",
        "market_open": "08:00",
        "market_close": "16:30",
        "after_hours_start": "16:30",
        "after_hours_end": "17:00",
        "timezone": "Europe/London",
        "trading_days": [0, 1, 2, 3, 4],
    },
    "JP": {
        "pre_market_start": "08:00",
        "pre_market_end": "09:00",
        "market_open": "09:00",
        "market_close": "15:00",
        "after_hours_start": "15:00",
        "after_hours_end": "16:00",
        "timezone": "Asia/Tokyo",
        "trading_days": [0, 1, 2, 3, 4],
    },
}


# ============================================================================
# CHART DISPLAY SETTINGS
# ============================================================================

CHART_THEMES = {
    "light": {
        "background": "#FFFFFF",
        "grid": "#E0E0E0",
        "text": "#000000",
        "up_candle": "#26A69A",
        "down_candle": "#EF5350",
        "volume": "#1976D2",
        "crosshair": "#758696",
    },
    "dark": {
        "background": "#131722",
        "grid": "#363C4E",
        "text": "#D9D9D9",
        "up_candle": "#26A69A",
        "down_candle": "#EF5350",
        "volume": "#1976D2",
        "crosshair": "#758696",
    },
    "tradingview": {
        "background": "#1E222D",
        "grid": "#2A2E39",
        "text": "#B2B5BE",
        "up_candle": "#53B987",
        "down_candle": "#EB4D5C",
        "volume": "#4A90E2",
        "crosshair": "#9598A1",
    }
}


DEFAULT_CHART_OPTIONS = {
    "theme": "dark",
    "type": ChartType.CANDLESTICK.value,
    "show_volume": True,
    "show_grid": True,
    "show_crosshair": True,
    "show_legend": True,
    "show_toolbar": True,
    "enable_zoom": True,
    "enable_pan": True,
}


# ============================================================================
# VALIDATION AND HELPER FUNCTIONS
# ============================================================================

def validate_timeframe(timeframe: str) -> bool:
    """
    Validate if a timeframe is supported

    Args:
        timeframe: Timeframe string to validate

    Returns:
        bool: True if valid, False otherwise
    """
    return timeframe in TIMEFRAME_CONFIGS


def get_timeframe_config(timeframe: str) -> Optional[TimeframeConfig]:
    """
    Get configuration for a specific timeframe

    Args:
        timeframe: Timeframe string

    Returns:
        TimeframeConfig or None if not found
    """
    return TIMEFRAME_CONFIGS.get(timeframe)


def get_available_timeframes() -> List[str]:
    """Get list of all available timeframes"""
    return list(TIMEFRAME_CONFIGS.keys())


def get_timeframe_descriptions() -> Dict[str, str]:
    """Get descriptions for all timeframes"""
    return {key: config.description for key, config in TIMEFRAME_CONFIGS.items()}


def get_indicator_config(indicator: str) -> Optional[IndicatorConfig]:
    """
    Get configuration for a specific indicator

    Args:
        indicator: Indicator name

    Returns:
        IndicatorConfig or None if not found
    """
    return INDICATOR_CONFIGS.get(indicator)


def get_available_indicators() -> List[str]:
    """Get list of all available indicators"""
    return list(INDICATOR_CONFIGS.keys())


def get_market_hours(market: str = "US") -> Dict[str, Any]:
    """
    Get market hours for a specific market

    Args:
        market: Market code (US, IN, UK, JP)

    Returns:
        Dictionary with market hours information
    """
    return MARKET_HOURS.get(market, MARKET_HOURS["US"])


def is_market_open(market: str = "US") -> bool:
    """
    Check if a market is currently open

    Args:
        market: Market code

    Returns:
        bool: True if market is open
    """
    from datetime import datetime
    import pytz

    market_info = get_market_hours(market)
    tz = pytz.timezone(market_info["timezone"])
    now = datetime.now(tz)

    # Check if it's a trading day
    if now.weekday() not in market_info["trading_days"]:
        return False

    # Check if within market hours
    current_time = now.strftime("%H:%M")
    return market_info["market_open"] <= current_time <= market_info["market_close"]


def get_optimal_timeframe_for_period(period_days: int) -> str:
    """
    Get the optimal timeframe based on the period in days

    Args:
        period_days: Number of days

    Returns:
        str: Optimal timeframe key
    """
    if period_days <= 1:
        return "1d"
    elif period_days <= 5:
        return "5d"
    elif period_days <= 7:
        return "1w"
    elif period_days <= 30:
        return "1mo"
    elif period_days <= 90:
        return "3mo"
    elif period_days <= 180:
        return "6mo"
    elif period_days <= 365:
        return "1y"
    elif period_days <= 730:
        return "2y"
    else:
        return "5y"


def calculate_data_points_needed(timeframe: str, hours: int = 24) -> int:
    """
    Calculate number of data points needed for a given timeframe and hours

    Args:
        timeframe: Timeframe key
        hours: Number of hours of data needed

    Returns:
        int: Number of data points
    """
    config = get_timeframe_config(timeframe)
    if not config:
        return 0

    interval = config.interval

    # Map intervals to minutes
    interval_minutes = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
        "60m": 60, "90m": 90, "1h": 60, "1d": 1440,
        "5d": 7200, "1wk": 10080, "1mo": 43200, "3mo": 129600
    }

    if interval in interval_minutes:
        minutes_per_point = interval_minutes[interval]
        total_minutes = hours * 60
        return total_minutes // minutes_per_point

    return config.display_limit


# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================

DATA_QUALITY_THRESHOLDS = {
    "min_volume": 1000,  # Minimum volume to consider data valid
    "max_spread_percent": 5.0,  # Maximum bid-ask spread as percentage
    "max_gap_hours": 48,  # Maximum gap between data points in hours
    "min_liquidity_score": 0.3,  # Minimum liquidity score (0-1)
    "outlier_std_dev": 3,  # Number of standard deviations for outlier detection
}


# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_FORMATS = {
    "csv": {
        "extension": ".csv",
        "mime_type": "text/csv",
        "separator": ",",
    },
    "excel": {
        "extension": ".xlsx",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "json": {
        "extension": ".json",
        "mime_type": "application/json",
        "indent": 2,
    },
    "parquet": {
        "extension": ".parquet",
        "mime_type": "application/octet-stream",
    }
}


# ============================================================================
# API RATE LIMITS
# ============================================================================

RATE_LIMITS = {
    "yfinance": {
        "requests_per_second": 2,
        "requests_per_minute": 60,
        "requests_per_hour": 2000,
        "concurrent_connections": 5,
    },
    "alpha_vantage": {
        "requests_per_minute": 5,  # Free tier
        "requests_per_day": 500,
        "concurrent_connections": 1,
    },
    "polygon": {
        "requests_per_second": 5,  # Free tier
        "requests_per_minute": 100,
        "concurrent_connections": 3,
    }
}


# ============================================================================
# CACHE SETTINGS
# ============================================================================

CACHE_PREFIXES = {
    "chart_data": "chart_",
    "prediction": "pred_",
    "indicator": "ind_",
    "market_info": "mkt_",
    "ticker_info": "tick_",
}

CACHE_VERSIONS = {
    "chart_data": "v2",
    "prediction": "v1",
    "indicator": "v1",
}


def get_cache_key(prefix: str, *args, version: str = "v1") -> str:
    """
    Generate a cache key with versioning

    Args:
        prefix: Cache prefix
        *args: Arguments to include in key
        version: Cache version

    Returns:
        str: Cache key
    """
    import hashlib

    key_parts = [prefix, version] + [str(arg) for arg in args]
    key_string = "_".join(key_parts)

    # Hash long keys to avoid cache key length limits
    if len(key_string) > 200:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}{version}_{key_hash}"

    return key_string


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_chart_logging():
    """Setup logging for chart module"""
    import logging.config

    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': 'logs/chart_config.log',
                'mode': 'a'
            },
        },
        'loggers': {
            'chart_config': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        }
    }

    logging.config.dictConfig(LOGGING_CONFIG)


# Initialize logging when module is imported
# Uncomment the line below if you want automatic logging setup
# setup_chart_logging()

# Log module initialization
logger.info("Chart configuration module initialized")
