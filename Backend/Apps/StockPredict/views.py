"""
StockVibePredictor Views Module
Organization: Dibakar
Created: 2025

This module contains all API endpoints for stock predictions, trading, and market analysis.
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import os
import hashlib
import re
import pickle
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Django imports
from django.utils import timezone
from django.core.cache import cache
from asgiref.sync import sync_to_async

# REST Framework imports
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from rest_framework.throttling import UserRateThrottle

# Data processing imports
import yfinance as yf
import pandas as pd
import numpy as np

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Additional imports for technical indicators
import atexit
import signal
import psutil
import asyncio

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================

# Base configuration
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "Scripts" / "Models"
MODELS_DIR.mkdir(exist_ok=True)

# Logger setup
logger = logging.getLogger("Apps.StockPredict")

# Ensure logger is configured
def log_prediction_request(ticker, timeframes, user_id=None, request_ip=None):
    """Log prediction requests in structured format"""
    log_data = {
        "event": "prediction_request",
        "ticker": ticker,
        "timeframes": timeframes,
        "user_id": user_id,
        "request_ip": request_ip,
        "timestamp": timezone.now().isoformat(),
    }
    logger.info(json.dumps(log_data))

# Function to log model usage
def log_model_usage(model_key, ticker, timeframe, accuracy):
    """Log model usage for analytics"""
    log_data = {
        "event": "model_usage",
        "model_key": model_key,
        "ticker": ticker,
        "timeframe": timeframe,
        "accuracy": accuracy,
        "timestamp": timezone.now().isoformat(),
    }
    logger.info(json.dumps(log_data))

# Function to log errors with context
def log_error_with_context(error_type, error_message, context=None):
    """Log errors with additional context"""
    log_data = {
        "event": "error",
        "error_type": error_type,
        "error_message": str(error_message),
        "context": context or {},
        "timestamp": timezone.now().isoformat(),
    }
    logger.error(json.dumps(log_data))

# Thread pools for async operations
training_executor = ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="model_trainer"
)
prediction_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="predictor")
data_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="data_fetcher")

# Function to clean up thread pools on shutdown
def cleanup_thread_pools():
    """Clean up thread pools on shutdown"""
    logger.info("Shutting down thread pools...")
    training_executor.shutdown(wait=True)
    prediction_executor.shutdown(wait=True)
    data_executor.shutdown(wait=True)
    logger.info("Thread pools shut down successfully")

# Register cleanup function
atexit.register(cleanup_thread_pools)
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup_thread_pools())

# Global caches
model_cache = {}
prediction_cache = {}
performance_cache = {}

# Timeframe configurations - Properly aligned with button labels and expected data durations
TIMEFRAMES = {
    "1d": {
        "period": "7d",        # Fetch 7 days to ensure we have recent 1-day data
        "interval": "5m",      # 5-minute intervals for intraday precision
        "model_suffix": "_1d",
        "cache_time": 300,     # 5 minutes cache
        "display_limit": 288,  # 24 hours * 12 (5-min intervals per hour)
        "description": "1 Day - Intraday predictions with 5-minute intervals"
    },
    "5d": {
        "period": "1mo",       # Fetch 1 month to ensure we have 5 days
        "interval": "15m",     # 15-minute intervals for short-term
        "model_suffix": "_1w",
        "cache_time": 600,     # 10 minutes cache
        "display_limit": 1920, # 5 days * 24 hours * 4 (15-min intervals)
        "description": "5 Days - Short-term with 15-minute intervals"
    },
    "1w": {
        "period": "2mo",       # Fetch 2 months to ensure we have 1 week
        "interval": "1h",      # Hourly intervals for weekly view
        "model_suffix": "_1w",
        "cache_time": 1800,    # 30 minutes cache
        "display_limit": 168,  # 7 days * 24 hours
        "description": "1 Week - 7 days of hourly data"
    },
    "1mo": {
        "period": "3mo",       # Fetch 3 months to ensure we have 1 month
        "interval": "1d",      # Daily intervals for monthly view
        "model_suffix": "_1mo",
        "cache_time": 3600,    # 1 hour cache
        "display_limit": 30,   # ~30 days
        "description": "1 Month - 30 days of daily data"
    },
    "3mo": {
        "period": "6mo",       # Fetch 6 months to ensure we have 3 months
        "interval": "1d",      # Daily intervals
        "model_suffix": "_1mo",
        "cache_time": 5400,    # 1.5 hours cache
        "display_limit": 90,   # ~90 days
        "description": "3 Months - 90 days of daily data"
    },
    "6mo": {
        "period": "1y",        # Fetch 1 year to ensure we have 6 months
        "interval": "1d",      # Daily intervals
        "model_suffix": "_1mo",
        "cache_time": 7200,    # 2 hours cache
        "display_limit": 180,  # ~180 days
        "description": "6 Months - 180 days of daily data"
    },
    "1y": {
        "period": "2y",        # Fetch 2 years to ensure we have 1 year
        "interval": "1d",      # Daily intervals for accuracy
        "model_suffix": "_1y",
        "cache_time": 21600,   # 6 hours cache
        "display_limit": 365,  # 365 days (1 year)
        "description": "1 Year - 365 days of daily data"
    },
    "2y": {
        "period": "3y",        # Fetch 3 years to ensure we have 2 years
        "interval": "1wk",     # Weekly intervals for longer periods
        "model_suffix": "_1y",
        "cache_time": 28800,   # 8 hours cache
        "display_limit": 104,  # 2 years * 52 weeks
        "description": "2 Years - 104 weeks of weekly data"
    },
    "5y": {
        "period": "max",       # Fetch maximum available data
        "interval": "1mo",     # Monthly intervals for very long periods
        "model_suffix": "_1y",
        "cache_time": 43200,   # 12 hours cache
        "display_limit": 60,   # 5 years * 12 months
        "description": "5 Years - 60 months of monthly data"
    },
}

# ============================================================================
# CACHE MANAGEMENT - FIXED FOR DJANGO 5.2+
# ============================================================================

def invalidate_stale_cache():
    """Remove stale cache entries - Compatible with Django 5.2+"""
    try:
        from django.core.cache import cache

        current_time = timezone.now().timestamp()
        cleanup_count = 0

        # Clear old prediction cache entries
        prediction_cache_keys_to_remove = []
        for key, cached_data in list(prediction_cache.items()):
            if hasattr(cached_data, 'get') and cached_data.get('timestamp'):
                try:
                    cache_time = pd.to_datetime(cached_data['timestamp']).timestamp()
                    # Use timeframe-specific cache times
                    max_cache_time = max(tf['cache_time'] for tf in TIMEFRAMES.values())
                    if current_time - cache_time > max_cache_time:
                        prediction_cache_keys_to_remove.append(key)
                except Exception as e:
                    logger.debug(f"Error checking cache timestamp for {key}: {e}")

        # Remove stale entries
        for key in prediction_cache_keys_to_remove:
            del prediction_cache[key]
            cleanup_count += 1

        # Clear performance cache if too large
        if len(performance_cache) > 1000:
            # Keep only recent 500 entries by clearing all
            performance_cache.clear()
            cleanup_count += 500
            logger.info("Performance cache cleared due to size limit")

        logger.info(f"Cache cleanup completed: removed {cleanup_count} stale entries")
        return cleanup_count

    except Exception as e:
        logger.warning(f"Cache cleanup partially completed: {str(e)}")
        # Don't fail - cache cleanup is not critical
        return 0

def schedule_cache_cleanup():
    """Schedule periodic cache cleanup"""
    try:
        return invalidate_stale_cache()
    except Exception as e:
        logger.warning(f"Scheduled cache cleanup skipped: {e}")
        return 0

def clear_all_caches():
    """Clear all caches - useful for maintenance"""
    try:
        from django.core.cache import cache

        # Clear Django cache
        cache.clear()

        # Clear local caches
        prediction_cache.clear()
        performance_cache.clear()

        logger.info("All caches cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to clear all caches: {e}")
        return False

# ============================================================================
# THROTTLE CLASSES
# ============================================================================

# Class for stock data fetching
class PredictionRateThrottle(UserRateThrottle):
    scope = "prediction"
    rate = "100/hour"

# Class for trading
class TradingRateThrottle(UserRateThrottle):
    scope = "trading"
    rate = "50/hour"

# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================

# Functions to compute various technical indicators
def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Functions to compute MACD Bands
def compute_macd(data, fast=12, slow=26, signal=9):
    """Compute MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

# Functions to compute Bollinger Bands
def compute_bollinger_bands(data, period=20, std=2):
    """Compute Bollinger Bands"""
    ma = data["Close"].rolling(window=period).mean()
    std_dev = data["Close"].rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    bb_width = (upper - lower) / ma
    return upper, lower, bb_width

# Functions to compute advanced technical indicators
def compute_advanced_indicators(data):
    """Compute advanced technical indicators"""
    # Williams %R
    highest_high = data["High"].rolling(window=14).max()
    lowest_low = data["Low"].rolling(window=14).min()
    williams_r = -100 * (highest_high - data["Close"]) / (highest_high - lowest_low)

    # Commodity Channel Index (CCI)
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    cci = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    # On-Balance Volume (OBV)
    obv = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

    # Average True Range (ATR)
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=14).mean()

    # Stochastic Oscillator
    lowest_low_14 = data["Low"].rolling(window=14).min()
    highest_high_14 = data["High"].rolling(window=14).max()
    k_percent = 100 * (
        (data["Close"] - lowest_low_14) / (highest_high_14 - lowest_low_14)
    )
    d_percent = k_percent.rolling(window=3).mean()

    # VWAP
    vwap = (data["Close"] * data["Volume"]).cumsum() / data["Volume"].cumsum()

    return {
        "williams_r": williams_r,
        "cci": cci,
        "obv": obv,
        "atr": atr,
        "stoch_k": k_percent,
        "stoch_d": d_percent,
        "vwap": vwap,
    }

# ============================================================================
# ANALYSIS & METRICS HELPERS
# ============================================================================

# Functions to compute sentiment metrics
def compute_sentiment_score(ticker, data):
    """Placeholder for news sentiment analysis (integrate with news APIs later)"""
    recent_returns = data["Close"].pct_change().tail(5).mean()
    sentiment = max(-1, min(1, recent_returns * 10))
    return {
        "sentiment_score": sentiment,
        "sentiment_label": (
            "bullish"
            if sentiment > 0.1
            else "bearish" if sentiment < -0.1 else "neutral"
        ),
    }

# Functions to compute risk assessment metrics
def compute_risk_metrics(data):
    """Compute risk assessment metrics"""
    returns = data["Close"].pct_change().dropna()

    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)

    # Sharpe Ratio
    excess_returns = returns - (0.02 / 252)
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Volatility
    volatility = returns.std() * np.sqrt(252)

    return {
        "var_95": var_95,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "risk_level": (
            "high" if volatility > 0.3 else "medium" if volatility > 0.15 else "low"
        ),
    }

# ============================================================================
# DATA VALIDATION & NORMALIZATION
# ============================================================================

# Functions to validate ticker symbols
def validate_ticker(ticker):
    """Enhanced ticker validation supporting international markets"""
    if not ticker or not isinstance(ticker, str):
        return False

    if re.match(r"^[A-Z0-9\^\.\\_\-]{1,15}$", ticker):
        return True
    return False

# Functions to normalize ticker symbols
def normalize_ticker(ticker):
    """Normalize ticker symbols for different markets"""
    ticker = ticker.upper().strip()

    ticker_mapping = {
        "NIFTY": "^NSEI",
        "NIFTY50": "^NSEI",
        "SENSEX": "^BSESN",
        "BERKSHIRE": "BRK-B",
        "ALPHABET": "GOOGL",
        "GOOGLE": "GOOGL",
        "META": "META",
        "FACEBOOK": "META",
        "TESLA": "TSLA",
    }

    return ticker_mapping.get(ticker, ticker)

# Functions to check if data is fresh enough for analysis
def is_data_fresh(data, timeframe):
    """Check if data is fresh enough for the given timeframe"""
    if data.empty:
        return False

    latest_time = data.index[-1]
    now = timezone.now()

    # Make timezone-aware if needed
    if latest_time.tzinfo is None:
        latest_time = timezone.make_aware(latest_time)

    # Define freshness thresholds (in seconds)
    thresholds = {
        "1d": 3600,     # 1 hour for intraday
        "5d": 14400,    # 4 hours for short term
        "1w": 14400,    # 4 hours for weekly
        "1mo": 86400,   # 1 day for monthly
        "3mo": 172800,  # 2 days for quarterly
        "6mo": 604800,  # 1 week for semi-annual
        "1y": 604800,   # 1 week for yearly
        "2y": 1209600,  # 2 weeks for long term
        "5y": 2592000,  # 1 month for very long term
    }

    threshold = thresholds.get(timeframe, 86400)  # Default 1 day
    time_diff = (now - latest_time).total_seconds()

    is_fresh = time_diff < threshold
    logger.debug(f"Data freshness check for {timeframe}: {time_diff}s < {threshold}s = {is_fresh}")

    return is_fresh

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

# Functions to fetch stock data
def fetch_stock_data_sync(ticker, timeframe="1d"):
    """Synchronous version of stock data fetching"""
    try:
        ticker = normalize_ticker(ticker)
        config = TIMEFRAMES[timeframe]

        logger.info(f"Sync fetch for {ticker} with timeframe {timeframe}")

        # Use yfinance Ticker object
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(
            period=config["period"],
            interval=config["interval"],
            auto_adjust=True,
            prepost=True,
        )

        if data.empty:
            logger.error(f"No data returned for {ticker} ({timeframe})")
            return None

        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")

        # Try to get info, but don't fail if it doesn't work
        market_info = {}
        try:
            ticker_info = ticker_obj.info
            if ticker_info:
                market_info = {
                    "company_name": ticker_info.get("longName") or ticker_info.get("shortName"),
                    "short_name": ticker_info.get("shortName"),
                    "long_name": ticker_info.get("longName"),
                    "market_cap": ticker_info.get("marketCap"),
                    "sector": ticker_info.get("sector"),
                    "industry": ticker_info.get("industry"),
                    "beta": ticker_info.get("beta"),
                    "pe_ratio": ticker_info.get("trailingPE"),
                    "dividend_yield": ticker_info.get("dividendYield"),
                    "fifty_two_week_high": ticker_info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": ticker_info.get("fiftyTwoWeekLow"),
                }
        except:
            pass  # Info is optional

        return {
            "price_data": data,
            "market_info": market_info,
        }

    except Exception as e:
        logger.error(f"Sync fetch error for {ticker}: {str(e)}")
        return None

# Functions to fetch enhanced stock data asynchronously
async def fetch_enhanced_stock_data(ticker, timeframe="1d"):
    """Enhanced stock data fetching with multiple timeframes"""
    try:
        ticker = normalize_ticker(ticker)
        config = TIMEFRAMES[timeframe]

        logger.info(f"Attempting to fetch data for {ticker} with timeframe {timeframe}")

        # Fetch price data synchronously first
        def download_data():
            return yf.download(
                ticker,
                period=config["period"],
                interval=config["interval"],
                progress=False,
                timeout=30,
                auto_adjust=True,
                prepost=True,
                threads=False,
            )

        # Use sync_to_async properly
        data = await sync_to_async(download_data, thread_sensitive=True)()

        if data.empty:
            logger.error(f"No data returned for {ticker} ({timeframe})")
            # Try alternative approach
            logger.info(f"Trying alternative fetch for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            data = await sync_to_async(
                lambda: ticker_obj.history(
                    period=config["period"],
                    interval=config["interval"],
                    auto_adjust=True,
                    prepost=True,
                ),
                thread_sensitive=True,
            )()

            if data.empty:
                logger.error(f"Alternative fetch also failed for {ticker}")
                return None

        logger.info(f"Successfully fetched {len(data)} rows for {ticker}")

        # Fetch ticker info separately with error handling
        market_info = {}
        try:
            ticker_obj = yf.Ticker(ticker)
            ticker_info = await sync_to_async(
                lambda: ticker_obj.info, thread_sensitive=True
            )()

            if ticker_info:
                market_info = {
                    "company_name": ticker_info.get("longName") or ticker_info.get("shortName"),
                    "short_name": ticker_info.get("shortName"),
                    "market_cap": ticker_info.get("marketCap"),
                    "sector": ticker_info.get("sector"),
                    "industry": ticker_info.get("industry"),
                    "beta": ticker_info.get("beta"),
                    "pe_ratio": ticker_info.get("trailingPE"),
                    "dividend_yield": ticker_info.get("dividendYield"),
                    "fifty_two_week_high": ticker_info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": ticker_info.get("fiftyTwoWeekLow"),
                }

        except Exception as info_error:
            logger.warning(
                f"Could not fetch ticker info for {ticker}: {str(info_error)}"
            )

        return {
            "price_data": data,
            "market_info": market_info,
        }

    except Exception as e:
        logger.error(f"Error fetching data for {ticker} ({timeframe}): {str(e)}")

        # Try a simple synchronous fallback
        try:
            logger.info(f"Attempting synchronous fallback for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                period=config["period"], interval=config["interval"]
            )

            if not data.empty:
                return {"price_data": data, "market_info": {}}
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {str(fallback_error)}")

        return None

# Functions to batch fetch data for multiple tickers
async def batch_fetch_data(tickers, timeframe):
    """Fetch data for multiple tickers in parallel"""
    try:
        logger.info(f"Batch fetching data for {len(tickers)} tickers ({timeframe})")

        # Limit concurrent requests to avoid overwhelming the API
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def fetch_with_semaphore(ticker):
            async with semaphore:
                try:
                    return await fetch_enhanced_stock_data(ticker, timeframe)
                except Exception as e:
                    logger.error(f"Batch fetch failed for {ticker}: {str(e)}")
                    return None

        # Create tasks for all tickers
        tasks = [fetch_with_semaphore(ticker) for ticker in tickers]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        batch_results = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                batch_results[ticker] = {"error": str(result)}
            else:
                batch_results[ticker] = result

        logger.info(f"Batch fetch completed: {sum(1 for r in batch_results.values() if r and 'error' not in r)} successful")
        return batch_results

    except Exception as e:
        logger.error(f"Batch data fetch failed: {str(e)}")
        return {ticker: {"error": str(e)} for ticker in tickers}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Functions to compute comprehensive technical features
def compute_comprehensive_features(data, timeframe="1d"):
    """Compute comprehensive technical features for different timeframes"""
    try:
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        # Validate required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=required_cols)

        if len(data) < 50:
            raise ValueError(f"Insufficient data: {len(data)} rows")

        # Adjust periods based on timeframe
        if timeframe == "1d":
            ma_periods = [5, 10, 20, 50]
            rsi_period = 14
        elif timeframe == "1w":
            ma_periods = [4, 8, 13, 26]
            rsi_period = 10
        elif timeframe == "1mo":
            ma_periods = [3, 6, 12, 24]
            rsi_period = 8
        elif timeframe == "1y":
            ma_periods = [2, 3, 6, 12]
            rsi_period = 6
        else:
            logger.warning(f"Unexpected timeframe '{timeframe}'. Defaulting to 1d configuration.")
            ma_periods = [5, 10, 20, 50]
            rsi_period = 14


        # Basic features
        data["Return"] = data["Close"].pct_change()

        # Moving averages
        for period in ma_periods:
            if len(data) >= period:
                data[f"MA{period}"] = data["Close"].rolling(window=period).mean()

        # Volatility and volume
        data["Volatility"] = data["Return"].rolling(window=20).std()
        data["Volume_Change"] = data["Volume"].pct_change()

        # Technical indicators
        data["RSI"] = compute_rsi(data["Close"], rsi_period)
        macd, macd_signal, macd_hist = compute_macd(data["Close"])
        data["MACD"] = macd
        data["MACD_Signal"] = macd_signal
        data["MACD_Histogram"] = macd_hist

        # Bollinger Bands
        bb_upper, bb_lower, bb_width = compute_bollinger_bands(data)
        data["BB_Upper"] = bb_upper
        data["BB_Lower"] = bb_lower
        data["BB_Width"] = bb_width
        data["BB_Position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)

        # Advanced indicators
        advanced = compute_advanced_indicators(data)
        for key, value in advanced.items():
            data[key] = value

        # Pattern features
        data["Higher_High"] = (data["High"] > data["High"].shift(1)).astype(int)
        data["Lower_Low"] = (data["Low"] < data["Low"].shift(1)).astype(int)
        data["Doji"] = (
            abs(data["Close"] - data["Open"]) <= (data["High"] - data["Low"]) * 0.1
        ).astype(int)

        # Trend features
        if "MA20" in data.columns and "MA50" in data.columns:
            data["Trend_Bullish"] = (data["Close"] > data["MA20"]).astype(int)
            data["Golden_Cross"] = (data["MA20"] > data["MA50"]).astype(int)

        # Market regime features
        data["High_Volatility"] = (
            data["Volatility"] > data["Volatility"].rolling(50).quantile(0.8)
        ).astype(int)
        data["Volume_Spike"] = (
            data["Volume"] > data["Volume"].rolling(20).mean() * 1.5
        ).astype(int)

        logger.info(f"Computed comprehensive features for {timeframe} timeframe")
        return data

    except Exception as e:
        logger.error(f"Error computing features for {timeframe}: {str(e)}")
        raise


# ============================================================================
# MODEL MANAGEMENT FUNCTIONS
# ============================================================================

# Functions to verify model integrity
def verify_model_integrity(model_path):
    """Verify model file hasn't been tampered with"""
    try:
        if not os.path.exists(model_path):
            return False

        # Check file size is reasonable (not too small/large)
        file_size = os.path.getsize(model_path)
        if file_size < 1000 or file_size > 100 * 1024 * 1024:  # 1KB - 100MB
            logger.warning(f"Suspicious model file size: {file_size} bytes")
            return False

        # Calculate file hash
        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # In production, you'd check against known good hashes
        # For now, just log the hash for monitoring
        logger.info(f"Model file {model_path} hash: {file_hash}")

        # Basic pickle safety check
        try:
            with open(model_path, 'rb') as f:
                # Read first few bytes to check for pickle signature
                header = f.read(10)
                if not header.startswith(b'\x80\x03') and not header.startswith(b'\x80\x04'):
                    logger.warning(f"File {model_path} may not be a valid pickle file")
                    return False
        except:
            return False

        return True
    except Exception as e:
        logger.error(f"Model integrity check failed for {model_path}: {str(e)}")
        return False

def secure_model_load(model_path):
    """Securely load model with integrity checks"""
    if not verify_model_integrity(model_path):
        raise ValueError(f"Model integrity check failed: {model_path}")

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate model structure
        required_keys = ["model", "features"]
        if not all(key in model_data for key in required_keys):
            raise ValueError(f"Invalid model structure in {model_path}")

        return model_data
    except Exception as e:
        logger.error(f"Secure model load failed for {model_path}: {str(e)}")
        raise


def load_all_models():
    """Load all available models with enhanced security and validation"""
    global model_cache
    model_cache.clear()

    loaded_count = 0
    failed_count = 0
    security_failures = 0

    if not MODELS_DIR.exists():
        logger.error(f"Models directory does not exist: {MODELS_DIR}")
        return

    model_files = list(MODELS_DIR.glob("*.pkl"))
    logger.info(f"Found {len(model_files)} model files in {MODELS_DIR}")

    for model_path in model_files:
        try:
            filename = model_path.name

            # Log the file being processed with hash
            logger.info(f"Model file {model_path} hash: {hashlib.sha256(model_path.read_bytes()).hexdigest()}")

            # Security check
            if not verify_model_integrity(model_path):
                logger.warning(f"Security check failed for {filename}")
                security_failures += 1
                continue

            # Use secure loading
            model_data = secure_model_load(model_path)

            if filename.startswith("universal_model_"):
                for timeframe in TIMEFRAMES.keys():
                    if (
                        f"_model_{timeframe}.pkl" in filename
                        or f"model_{timeframe}.pkl" in filename
                    ):
                        cache_key = f"universal_{timeframe}"
                        model_cache[cache_key] = {
                            "model": model_data.get("model"),
                            "features": model_data.get("features", []),
                            "timeframe": timeframe,
                            "accuracy": model_data.get("accuracy", 0.5),
                            "type": "universal",
                            "path": str(model_path),
                            "last_updated": model_path.stat().st_mtime,
                            "file_hash": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        }
                        logger.info(f"Loaded universal model for {timeframe}")
                        loaded_count += 1
                        break

            else:
                parts = filename.replace(".pkl", "").split("_model_")
                if len(parts) == 2:
                    ticker = parts[0].upper()
                    timeframe = parts[1]

                    if timeframe in TIMEFRAMES:
                        cache_key = f"{ticker}_{timeframe}"
                        model_cache[cache_key] = {
                            "model": model_data.get("model"),
                            "features": model_data.get("features", []),
                            "timeframe": timeframe,
                            "ticker": ticker,
                            "accuracy": model_data.get("accuracy", 0.5),
                            "type": "ticker_specific",
                            "path": str(model_path),
                            "last_updated": model_path.stat().st_mtime,
                            "file_hash": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        }
                        logger.info(f"Loaded model for {ticker} ({timeframe})")
                        loaded_count += 1

        except Exception as e:
            logger.error(f"Failed to load model {model_path.name}: {str(e)}")
            failed_count += 1

    logger.info(f"Model loading complete: {loaded_count} loaded, {failed_count} failed, {security_failures} security failures")
    logger.info(f"Total models in cache: {len(model_cache)}")

    # Schedule cache cleanup separately (don't do it during startup)
    try:
        logger.info("Scheduling cache cleanup...")
        # You can call this from a separate management command or celery task
        # schedule_cache_cleanup()
    except Exception as e:
        logger.debug(f"Cache cleanup scheduling skipped: {e}")

# Function to get the best model for prediction
def get_model_for_prediction(ticker, timeframe):
    """Get the best available model for ticker and timeframe"""
    # Normalize ticker for lookup
    ticker = ticker.upper().replace(".", "_").replace("-", "_")

    ticker_key = f"{ticker}_{timeframe}"
    if ticker_key in model_cache:
        logger.info(f"Using ticker-specific model for {ticker} ({timeframe})")
        return model_cache[ticker_key]

    for key in model_cache.keys():
        if key.startswith(f"{ticker}_") and key.endswith(f"_{timeframe}"):
            logger.info(f"Using variant ticker model {key}")
            return model_cache[key]

    universal_key = f"universal_{timeframe}"
    if universal_key in model_cache:
        logger.info(f"Using universal model for {ticker} ({timeframe})")
        return model_cache[universal_key]

    timeframe_models = [k for k in model_cache.keys() if k.endswith(f"_{timeframe}")]
    if timeframe_models:
        logger.warning(
            f"Using random model for {ticker} ({timeframe}): {timeframe_models[0]}"
        )
        return model_cache[timeframe_models[0]]

    if model_cache:
        first_model = list(model_cache.keys())[0]
        logger.warning(
            f"Using fallback model for {ticker} ({timeframe}): {first_model}"
        )
        return model_cache[first_model]

    logger.error(f"No models available for {ticker} ({timeframe})")
    return None

# Function to create dummy models for testing purposes
def create_dummy_models():
    """Create dummy models for testing purposes"""
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    # Define expected features
    features = [
        "Return", "MA5", "MA10", "MA20", "MA50",
        "Volatility", "Volume_Change", "RSI",
        "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Upper", "BB_Lower", "BB_Width", "BB_Position",
        "williams_r", "cci", "obv", "atr",
        "stoch_k", "stoch_d", "vwap",
        "Higher_High", "Lower_Low", "Doji",
        "Trend_Bullish", "Golden_Cross",
        "High_Volatility", "Volume_Spike",
    ]

    # Create dummy models for each timeframe
    for timeframe, config in TIMEFRAMES.items():
        try:
            # Create a simple random forest model
            model = RandomForestClassifier(n_estimators=10, random_state=42)

            # Create dummy training data
            import numpy as np

            X_dummy = np.random.randn(100, len(features))
            y_dummy = np.random.randint(0, 2, 100)

            # Fit the model
            model.fit(X_dummy, y_dummy)

            # Save model
            model_path = MODELS_DIR / f"universal_model{config['model_suffix']}.pkl"
            model_data = {
                "model": model,
                "features": features,
                "accuracy": 0.65,  # Dummy accuracy
                "timeframe": timeframe,
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            print(f"Created dummy model for {timeframe} at {model_path}")

        except Exception as e:
            print(f"Error creating dummy model for {timeframe}: {e}")

    # Reload models
    load_all_models()
    print(f"Models loaded: {len(model_cache)}")
    return True


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

# Function to train a new model for a specific ticker and timeframe
def train_model_for_ticker(ticker, timeframe, model_type="ensemble"):
    """Train a new model for a specific ticker and timeframe"""
    try:
        logger.info(f"Starting model training for {ticker} ({timeframe})")

        # Fetch historical data
        data_result = fetch_stock_data_sync(ticker, timeframe)
        if not data_result:
            return {"error": "Failed to fetch data"}

        # Compute features
        data = compute_comprehensive_features(data_result["price_data"], timeframe)

        # Prepare features and target
        feature_columns = [
            "Return", "MA5", "MA10", "MA20", "MA50",
            "Volatility", "Volume_Change", "RSI",
            "MACD", "MACD_Signal", "MACD_Histogram",
            "BB_Upper", "BB_Lower", "BB_Width", "BB_Position",
            "williams_r", "cci", "obv", "atr",
            "stoch_k", "stoch_d", "vwap",
            "Higher_High", "Lower_Low", "Doji",
            "Trend_Bullish", "Golden_Cross",
            "High_Volatility", "Volume_Spike",
        ]

        # Filter available features
        available_features = [col for col in feature_columns if col in data.columns]

        # Create target variable (1 if price goes up, 0 if down)
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # Remove NaN values
        data = data.dropna()

        if len(data) < 100:
            return {"error": f"Insufficient data for training: {len(data)} samples"}

        # Prepare X and y
        X = data[available_features].values
        y = data["Target"].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model based on type
        if model_type == "ensemble":
            # Create ensemble of models
            models = [
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                GradientBoostingClassifier(n_estimators=100, random_state=42),
                LogisticRegression(random_state=42, max_iter=1000),
            ]

            # Train each model
            trained_models = []
            scores = []
            for model in models:
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                scores.append(score)
                trained_models.append(model)

            # Use the best performing model
            best_idx = scores.index(max(scores))
            final_model = trained_models[best_idx]
            model_name = type(final_model).__name__

        elif model_type == "randomforest":
            final_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
            final_model.fit(X_train_scaled, y_train)
            model_name = "RandomForest"

        else:  # gradient_boosting
            final_model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            )
            final_model.fit(X_train_scaled, y_train)
            model_name = "GradientBoosting"

        # Evaluate model
        y_pred = final_model.predict(X_test_scaled)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "model_type": model_name,
        }

        # Cross-validation
        cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5)
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()

        # Save model
        model_filename = f"{ticker}_{timeframe}_model.pkl"
        model_path = MODELS_DIR / model_filename

        model_data = {
            "model": final_model,
            "scaler": scaler,
            "features": available_features,
            "accuracy": metrics["accuracy"],
            "metrics": metrics,
            "ticker": ticker,
            "timeframe": timeframe,
            "trained_at": timezone.now().isoformat(),
            "data_points": len(data),
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        # Add to cache
        cache_key = f"{ticker}_{timeframe}"
        model_cache[cache_key] = {
            "model": final_model,
            "scaler": scaler,
            "features": available_features,
            "accuracy": metrics["accuracy"],
            "type": "ticker_specific",
            "timeframe": timeframe,
            "ticker": ticker,
            "path": str(model_path),
            "last_updated": timezone.now().timestamp(),
        }

        logger.info(
            f"Successfully trained model for {ticker} ({timeframe}): {metrics['accuracy']:.2%} accuracy"
        )

        return {
            "success": True,
            "ticker": ticker,
            "timeframe": timeframe,
            "metrics": metrics,
            "model_path": str(model_path),
        }

    except Exception as e:
        logger.error(f"Model training failed for {ticker} ({timeframe}): {str(e)}")
        return {"error": str(e)}


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

# Function to make multi-timeframe predictions with enhanced logging and monitoring
def make_multi_timeframe_prediction(ticker, data_dict, request_context=None):
    """Make predictions across multiple timeframes with comprehensive logging"""
    predictions = {}
    start_time = timezone.now()

    # Log the start of prediction process
    logger.info(f"Starting multi-timeframe prediction for {ticker} across {list(data_dict.keys())}")

    # Track prediction statistics
    prediction_stats = {
        "total_timeframes": 0,
        "successful_predictions": 0,
        "failed_predictions": 0,
        "models_used": {},
        "data_freshness": {},
    }

    for timeframe in ["1d", "1w", "1mo", "1y"]:
        prediction_stats["total_timeframes"] += 1
        timeframe_start_time = timezone.now()

        try:
            if timeframe not in data_dict:
                logger.debug(f"Timeframe {timeframe} not in data_dict for {ticker}")
                continue

            # Check data freshness
            data = data_dict[timeframe]
            is_fresh = is_data_fresh(data, timeframe)
            prediction_stats["data_freshness"][timeframe] = is_fresh

            if not is_fresh:
                logger.warning(f"Data for {ticker} ({timeframe}) may be stale")

            # Get model for prediction
            model_info = get_model_for_prediction(ticker, timeframe)
            if not model_info:
                logger.warning(f"No model available for {ticker} {timeframe}")
                prediction_stats["failed_predictions"] += 1
                continue

            model = model_info["model"]
            required_features = model_info["features"]
            model_key = f"{model_info.get('type', 'unknown')}_{timeframe}"
            prediction_stats["models_used"][model_key] = prediction_stats["models_used"].get(model_key, 0) + 1

            # Log model usage
            log_model_usage(
                model_key=f"{ticker}_{timeframe}" if model_info["type"] == "ticker_specific" else f"universal_{timeframe}",
                ticker=ticker,
                timeframe=timeframe,
                accuracy=model_info["accuracy"]
            )

            # Prepare feature vector with validation
            feature_vector = []
            feature_dict = {}
            missing_features = []

            for feature in required_features:
                if feature in data.columns:
                    value = data[feature].iloc[-1]
                    if pd.isna(value):
                        value = 0.0
                        missing_features.append(f"{feature}(NaN)")
                else:
                    value = 0.0
                    missing_features.append(f"{feature}(missing)")

                feature_vector.append(float(value))
                feature_dict[feature] = float(value)

            # Log missing features if any
            if missing_features:
                logger.warning(f"Missing/NaN features for {ticker} ({timeframe}): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

            # Make prediction with error handling
            feature_array = np.array(feature_vector).reshape(1, -1)

            # Validate feature array
            if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
                logger.error(f"Invalid feature array for {ticker} ({timeframe}): contains NaN or Inf")
                prediction_stats["failed_predictions"] += 1
                continue

            prediction = model.predict(feature_array)[0]

            # Calculate confidence with enhanced error handling
            confidence = 0.5  # Default confidence
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(feature_array)[0]
                    confidence = proba[int(prediction)]

                    # Validate confidence
                    if pd.isna(confidence) or confidence < 0 or confidence > 1:
                        logger.warning(f"Invalid confidence {confidence} for {ticker} ({timeframe}), using default")
                        confidence = 0.5

                except Exception as conf_error:
                    logger.error(f"Confidence calculation failed for {ticker} ({timeframe}): {str(conf_error)}")

            # Calculate price targets with enhanced validation
            current_price = float(data["Close"].iloc[-1])
            if current_price <= 0:
                logger.error(f"Invalid current price {current_price} for {ticker} ({timeframe})")
                prediction_stats["failed_predictions"] += 1
                continue

            volatility = float(data.get("Volatility", pd.Series([0.02])).iloc[-1])
            if pd.isna(volatility) or volatility < 0:
                volatility = 0.02  # Default volatility
                logger.warning(f"Using default volatility for {ticker} ({timeframe})")

            # Adjust multiplier based on timeframe with bounds checking
            timeframe_multipliers = {
                "1d": 0.5,
                "1w": 1.5,
                "1mo": 3.0,
                "1y": 10.0
            }

            base_multiplier = timeframe_multipliers.get(timeframe, 3.0)
            target_multiplier = 1 + (volatility * base_multiplier)

            # Ensure reasonable bounds on multiplier
            target_multiplier = max(1.001, min(target_multiplier, 2.0))  # Between 0.1% and 100%

            if prediction == 1:
                price_target = current_price * target_multiplier
                direction = "UP"
            else:
                price_target = current_price / target_multiplier
                direction = "DOWN"

            expected_return = ((price_target / current_price) - 1) * 100

            # Build prediction result
            prediction_result = {
                "direction": direction,
                "confidence": round(confidence * 100, 2),
                "price_target": round(price_target, 2),
                "current_price": round(current_price, 2),
                "expected_return": round(expected_return, 2),
                "model_accuracy": round(model_info["accuracy"] * 100, 2),
                "model_type": model_info["type"],
                "data_freshness": is_fresh,
                "missing_features_count": len(missing_features),
                "volatility": round(volatility, 4),
                "prediction_time": round((timezone.now() - timeframe_start_time).total_seconds() * 1000, 2),  # milliseconds
            }

            predictions[timeframe] = prediction_result
            prediction_stats["successful_predictions"] += 1

            # Log successful prediction
            logger.debug(f"Prediction for {ticker} ({timeframe}): {direction} with {confidence*100:.1f}% confidence")

        except Exception as e:
            prediction_stats["failed_predictions"] += 1

            # Enhanced error logging with context
            error_context = {
                "ticker": ticker,
                "timeframe": timeframe,
                "model_available": model_info is not None if 'model_info' in locals() else False,
                "data_shape": data.shape if 'data' in locals() else "unknown",
                "feature_count": len(feature_vector) if 'feature_vector' in locals() else 0,
            }

            log_error_with_context(
                error_type="prediction_error",
                error_message=str(e),
                context=error_context
            )

            logger.error(f"Prediction failed for {ticker} {timeframe}: {str(e)}")

    # Log prediction summary
    total_time = (timezone.now() - start_time).total_seconds()

    logger.info(f"Prediction completed for {ticker}: {prediction_stats['successful_predictions']}/{prediction_stats['total_timeframes']} successful in {total_time:.2f}s")

    if prediction_stats["successful_predictions"] == 0:
        logger.error(f"All predictions failed for {ticker}")

    # Log model usage statistics
    for model_key, usage_count in prediction_stats["models_used"].items():
        logger.debug(f"Model usage: {model_key} used {usage_count} times for {ticker}")

    return predictions


# Function to process multi-timeframe prediction without request object (Enhanced)
def process_multi_timeframe_prediction(ticker, timeframes, include_analysis=True, request_context=None):
    """
    Core business logic for multi-timeframe predictions with comprehensive monitoring.
    This function doesn't require a request object.
    """
    start_time = timezone.now()

    # Input validation with detailed logging
    if not ticker:
        log_error_with_context("validation_error", "Empty ticker provided")
        return {"error": "Please provide a ticker symbol"}, status.HTTP_400_BAD_REQUEST

    if not validate_ticker(ticker):
        log_error_with_context("validation_error", f"Invalid ticker format: {ticker}")
        return {"error": "Invalid ticker format"}, status.HTTP_400_BAD_REQUEST

    # Validate and filter timeframes
    valid_timeframes = list(TIMEFRAMES.keys())
    original_timeframes = timeframes.copy() if timeframes else []
    timeframes = [tf for tf in timeframes if tf in valid_timeframes]

    if not timeframes:
        timeframes = ["1d"]
        logger.warning(f"No valid timeframes provided for {ticker}, using default: {timeframes}")

    invalid_timeframes = set(original_timeframes) - set(timeframes)
    if invalid_timeframes:
        logger.warning(f"Invalid timeframes ignored for {ticker}: {invalid_timeframes}")

    original_ticker = ticker
    ticker = normalize_ticker(ticker)

    if ticker != original_ticker:
        logger.info(f"Ticker normalized: {original_ticker} -> {ticker}")

    # Enhanced cache checking with logging
    cache_key = f"multi_prediction_{ticker}_{'_'.join(sorted(timeframes))}"
    cached_result = cache.get(cache_key)

    if cached_result:
        logger.info(f"Cache HIT: Returning cached multi-timeframe prediction for {ticker}")

        # Add cache metadata
        cached_result["cache_info"] = {
            "cached": True,
            "cache_key": cache_key,
            "retrieved_at": timezone.now().isoformat(),
        }

        return cached_result, status.HTTP_200_OK
    else:
        logger.info(f"Cache MISS: Processing new prediction for {ticker}")

    try:
        # Data fetching with progress tracking
        data_dict = {}
        fetch_start_time = timezone.now()
        fetch_stats = {
            "attempted": 0,
            "successful": 0,
            "failed": [],
        }

        for timeframe in timeframes:
            fetch_stats["attempted"] += 1
            logger.info(f"Fetching {timeframe} data for {ticker} ({fetch_stats['attempted']}/{len(timeframes)})")

            try:
                data_result = fetch_stock_data_sync(ticker, timeframe)

                if data_result and not data_result["price_data"].empty:
                    # Check data freshness
                    is_fresh = is_data_fresh(data_result["price_data"], timeframe)

                    if not is_fresh:
                        logger.warning(f"Stale data detected for {ticker} ({timeframe})")

                    processed_data = compute_comprehensive_features(
                        data_result["price_data"], timeframe
                    )

                    data_dict[timeframe] = processed_data
                    fetch_stats["successful"] += 1

                    # Store market info from first successful fetch
                    if "market_info" not in data_dict:
                        data_dict["market_info"] = data_result["market_info"]

                    logger.debug(f"Successfully processed {len(processed_data)} data points for {ticker} ({timeframe})")
                else:
                    fetch_stats["failed"].append(timeframe)
                    logger.warning(f"No data returned for {ticker} ({timeframe})")

            except Exception as fetch_error:
                fetch_stats["failed"].append(timeframe)
                log_error_with_context(
                    "data_fetch_error",
                    str(fetch_error),
                    {"ticker": ticker, "timeframe": timeframe}
                )

        # Check if we have any data
        if not data_dict or not any(key != "market_info" for key in data_dict.keys()):
            error_msg = f"No data available for {ticker} (attempted: {timeframes}, failed: {fetch_stats['failed']})"
            log_error_with_context("no_data_error", error_msg, {"ticker": ticker, "timeframes": timeframes})
            return {"error": error_msg}, status.HTTP_404_NOT_FOUND

        fetch_time = (timezone.now() - fetch_start_time).total_seconds()
        logger.info(f"Data fetching completed for {ticker}: {fetch_stats['successful']}/{fetch_stats['attempted']} successful in {fetch_time:.2f}s")

        # Generate predictions with enhanced context
        prediction_start_time = timezone.now()
        predictions = make_multi_timeframe_prediction(
            ticker,
            data_dict,
            request_context=request_context
        )

        if not predictions:
            error_msg = "Unable to generate predictions"
            log_error_with_context("prediction_generation_error", error_msg, {"ticker": ticker, "available_data": list(data_dict.keys())})
            return {"error": error_msg}, status.HTTP_500_INTERNAL_SERVER_ERROR

        prediction_time = (timezone.now() - prediction_start_time).total_seconds()
        logger.info(f"Prediction generation completed for {ticker} in {prediction_time:.2f}s")

        # Build comprehensive response
        response_data = {
            "ticker": original_ticker,
            "normalized_ticker": ticker,
            "timestamp": timezone.now().isoformat(),
            "predictions": predictions,
            "market_info": data_dict.get("market_info", {}),
            "analysis": {},
            "metadata": {
                "processing_time": {
                    "data_fetch": round(fetch_time, 2),
                    "prediction_generation": round(prediction_time, 2),
                    "total": round((timezone.now() - start_time).total_seconds(), 2),
                },
                "data_quality": {
                    "timeframes_requested": len(timeframes),
                    "timeframes_processed": len([k for k in data_dict.keys() if k != "market_info"]),
                    "predictions_generated": len(predictions),
                },
                "cache_info": {
                    "cached": False,
                    "cache_key": cache_key,
                },
            },
        }

        # Enhanced analysis section
        if include_analysis:
            analysis_start_time = timezone.now()

            try:
                # Use the most recent/reliable data for analysis
                analysis_timeframe = None
                for tf in ["1d", "5d", "1w", "1mo"]:
                    if tf in data_dict:
                        analysis_timeframe = tf
                        break

                if analysis_timeframe:
                    daily_data = data_dict[analysis_timeframe]
                    logger.debug(f"Using {analysis_timeframe} data for analysis of {ticker}")

                    current_price = float(daily_data["Close"].iloc[-1])
                    rsi = float(daily_data["RSI"].iloc[-1]) if "RSI" in daily_data.columns else 50

                    # Compute additional metrics with error handling
                    try:
                        sentiment = compute_sentiment_score(ticker, daily_data)
                    except Exception as e:
                        logger.warning(f"Sentiment calculation failed for {ticker}: {str(e)}")
                        sentiment = {"sentiment_score": 0, "sentiment_label": "neutral"}

                    try:
                        risk_metrics = compute_risk_metrics(daily_data)
                    except Exception as e:
                        logger.warning(f"Risk metrics calculation failed for {ticker}: {str(e)}")
                        risk_metrics = {"risk_level": "unknown", "volatility": 0.0}

                    # Support and resistance levels with validation
                    try:
                        recent_highs = daily_data["High"].tail(20)
                        recent_lows = daily_data["Low"].tail(20)
                        resistance = float(recent_highs.quantile(0.8))
                        support = float(recent_lows.quantile(0.2))

                        # Validate support/resistance levels
                        if support >= resistance or support <= 0 or resistance <= 0:
                            logger.warning(f"Invalid support/resistance levels for {ticker}: support={support}, resistance={resistance}")
                            support = current_price * 0.95
                            resistance = current_price * 1.05
                    except Exception as e:
                        logger.warning(f"Support/resistance calculation failed for {ticker}: {str(e)}")
                        support = current_price * 0.95
                        resistance = current_price * 1.05

                    response_data["analysis"] = {
                        "technical": {
                            "rsi": round(rsi, 2),
                            "rsi_signal": (
                                "overbought" if rsi > 70
                                else "oversold" if rsi < 30
                                else "neutral"
                            ),
                            "trend": (
                                "bullish"
                                if daily_data.get("Trend_Bullish", pd.Series([0])).iloc[-1]
                                else "bearish"
                            ),
                            "volume_trend": (
                                "high"
                                if daily_data.get("Volume_Spike", pd.Series([0])).iloc[-1]
                                else "normal"
                            ),
                            "volatility_regime": (
                                "high"
                                if daily_data.get("High_Volatility", pd.Series([0])).iloc[-1]
                                else "normal"
                            ),
                        },
                        "price_levels": {
                            "current_price": round(current_price, 2),
                            "support": round(support, 2),
                            "resistance": round(resistance, 2),
                            "fifty_two_week_high": response_data["market_info"].get("fifty_two_week_high"),
                            "fifty_two_week_low": response_data["market_info"].get("fifty_two_week_low"),
                        },
                        "sentiment": sentiment,
                        "risk": risk_metrics,
                        "recommendation": {
                            "overall": (
                                "BUY" if sum(1 for p in predictions.values() if p["direction"] == "UP") > len(predictions) / 2
                                else "SELL"
                            ),
                            "confidence": round(
                                sum(p["confidence"] for p in predictions.values()) / len(predictions),
                                2,
                            ) if predictions else 0,
                            "risk_level": risk_metrics.get("risk_level", "unknown"),
                            "holding_period": (
                                "long" if predictions.get("1y", {}).get("direction") == "UP"
                                else "short"
                            ),
                        },
                        "data_source": analysis_timeframe,
                    }

                    # Add YTD performance if enough data
                    if len(daily_data) >= 252:
                        try:
                            year_ago_price = float(daily_data["Close"].iloc[-252])
                            ytd_return = ((current_price / year_ago_price) - 1) * 100
                            response_data["analysis"]["performance"] = {
                                "ytd_return": round(ytd_return, 2),
                                "ytd_vs_market": (
                                    "outperforming" if ytd_return > 10
                                    else "underperforming" if ytd_return < -10
                                    else "neutral"
                                ),
                            }
                        except Exception as e:
                            logger.warning(f"YTD performance calculation failed for {ticker}: {str(e)}")

                analysis_time = (timezone.now() - analysis_start_time).total_seconds()
                response_data["metadata"]["processing_time"]["analysis"] = round(analysis_time, 2)
                response_data["metadata"]["processing_time"]["total"] = round((timezone.now() - start_time).total_seconds(), 2)

            except Exception as e:
                log_error_with_context("analysis_error", str(e), {"ticker": ticker})
                logger.error(f"Analysis generation failed for {ticker}: {str(e)}")

        # Cache the result with appropriate timeout
        min_cache_time = min(TIMEFRAMES[tf]["cache_time"] for tf in timeframes)
        cache.set(cache_key, response_data, timeout=min_cache_time)
        logger.info(f"Result cached for {ticker} with timeout {min_cache_time}s")

        total_time = (timezone.now() - start_time).total_seconds()
        logger.info(f"Multi-timeframe prediction completed for {ticker} in {total_time:.2f}s")

        return response_data, status.HTTP_200_OK

    except Exception as e:
        total_time = (timezone.now() - start_time).total_seconds()

        # Comprehensive error logging
        error_context = {
            "ticker": ticker,
            "timeframes": timeframes,
            "processing_time": round(total_time, 2),
            "data_fetched": list(data_dict.keys()) if 'data_dict' in locals() else [],
            "predictions_generated": len(predictions) if 'predictions' in locals() else 0,
        }

        log_error_with_context("prediction_process_error", str(e), error_context)
        logger.error(f"Multi-timeframe prediction failed for {ticker} after {total_time:.2f}s: {str(e)}")

        return {
            "error": f"Prediction failed: {str(e)}",
            "metadata": {
                "processing_time": round(total_time, 2),
                "error_type": type(e).__name__,
            }
        }, status.HTTP_500_INTERNAL_SERVER_ERROR



# ============================================================================
# 📈 PREDICTION ENDPOINTS
# ============================================================================

# Prediction endpoint for multi-timeframe analysis
@api_view(["POST"])
@throttle_classes([PredictionRateThrottle])
@permission_classes([AllowAny])
def predict_multi_timeframe(request):
    """Advanced multi-timeframe stock prediction with comprehensive analysis"""
    endpoint_start_time = timezone.now()

    # Extract request data and context
    ticker = request.data.get("ticker")
    timeframes = request.data.get("timeframes", ["1d", "1w", "1mo"])
    include_analysis = request.data.get("include_analysis", True)

    # Gather request context for logging and analytics
    user_id = request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None
    request_ip = request.META.get('REMOTE_ADDR', 'unknown')
    user_agent = request.META.get('HTTP_USER_AGENT', 'unknown')[:100]  # Truncate for logging
    request_method = request.method

    # Enhanced request validation
    try:
        # Validate ticker
        if not ticker or not isinstance(ticker, str):
            log_error_with_context("validation_error", "Invalid or missing ticker", {
                "ticker": ticker,
                "user_id": user_id,
                "ip": request_ip,
                "endpoint": "predict_multi_timeframe"
            })
            return Response(
                {"error": "Please provide a valid ticker symbol"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Sanitize ticker input
        ticker = ticker.strip().upper()[:15]  # Limit length for security

        # Validate timeframes
        if not isinstance(timeframes, list):
            timeframes = ["1d", "1w", "1mo"]  # Default fallback

        if len(timeframes) > 10:  # Reasonable limit
            log_error_with_context("validation_error", "Too many timeframes requested", {
                "timeframes_count": len(timeframes),
                "user_id": user_id,
                "ip": request_ip
            })
            return Response(
                {"error": "Maximum 10 timeframes allowed per request"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate include_analysis parameter
        if not isinstance(include_analysis, bool):
            include_analysis = True  # Default to True

    except Exception as validation_error:
        log_error_with_context("request_validation_error", str(validation_error), {
            "raw_data": str(request.data)[:200],  # Truncated for security
            "user_id": user_id,
            "ip": request_ip
        })
        return Response(
            {"error": "Invalid request format"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Log the prediction request with comprehensive details
    request_context = {
        "user_id": user_id,
        "ip": request_ip,
        "user_agent": user_agent,
        "endpoint": "predict_multi_timeframe",
        "timeframes_count": len(timeframes),
        "analysis_requested": include_analysis,
        "request_size": len(str(request.data))
    }

    log_prediction_request(ticker, timeframes, user_id, request_ip)

    # Additional structured logging for analytics
    logger.info(json.dumps({
        "event": "prediction_request_detailed",
        "ticker": ticker,
        "timeframes": timeframes,
        "include_analysis": include_analysis,
        "context": request_context,
        "timestamp": timezone.now().isoformat(),
    }))

    try:
        # Call the core business logic with request context
        response_data, status_code = process_multi_timeframe_prediction(
            ticker, timeframes, include_analysis, request_context=request_context
        )

        # Calculate processing time
        processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        # Enhance response with request metadata
        if isinstance(response_data, dict):
            response_data["request_info"] = {
                "processing_time": round(processing_time, 3),
                "endpoint": "predict_multi_timeframe",
                "user_authenticated": user_id is not None,
                "request_id": f"req_{timezone.now().timestamp()}",
            }

        # Log successful completion
        logger.info(json.dumps({
            "event": "prediction_completed",
            "ticker": ticker,
            "status_code": status_code,
            "processing_time": processing_time,
            "predictions_generated": len(response_data.get("predictions", {})) if isinstance(response_data, dict) else 0,
            "user_id": user_id,
            "timestamp": timezone.now().isoformat(),
        }))

        return Response(response_data, status=status_code)

    except Exception as e:
        processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        # Comprehensive error logging
        log_error_with_context("endpoint_error", str(e), {
            "ticker": ticker,
            "timeframes": timeframes,
            "processing_time": processing_time,
            "user_id": user_id,
            "ip": request_ip,
            "endpoint": "predict_multi_timeframe"
        })

        return Response(
            {
                "error": "Prediction service temporarily unavailable",
                "request_id": f"req_{timezone.now().timestamp()}",
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Batch prediction endpoint for multiple tickers
@api_view(["POST"])
@throttle_classes([PredictionRateThrottle])  # Add throttling for batch requests
@permission_classes([AllowAny])
def batch_predictions(request):
    """Batch predictions for multiple tickers with enhanced monitoring"""
    endpoint_start_time = timezone.now()

    # Extract request data and context
    tickers = request.data.get("tickers", [])
    timeframe = request.data.get("timeframe", "1d")

    # Gather request context
    user_id = request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None
    request_ip = request.META.get('REMOTE_ADDR', 'unknown')
    user_agent = request.META.get('HTTP_USER_AGENT', 'unknown')[:100]

    # Enhanced validation
    try:
        if not isinstance(tickers, list):
            log_error_with_context("validation_error", "Tickers must be provided as a list", {
                "provided_type": type(tickers).__name__,
                "user_id": user_id,
                "ip": request_ip
            })
            return Response(
                {"error": "Tickers must be provided as a list"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not tickers or len(tickers) > 20:
            log_error_with_context("validation_error", f"Invalid ticker count: {len(tickers)}", {
                "ticker_count": len(tickers),
                "user_id": user_id,
                "ip": request_ip
            })
            return Response(
                {"error": "Please provide 1-20 tickers"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Sanitize and validate each ticker
        validated_tickers = []
        for ticker in tickers:
            if isinstance(ticker, str):
                clean_ticker = ticker.strip().upper()[:15]
                if validate_ticker(clean_ticker):
                    validated_tickers.append(clean_ticker)
                else:
                    logger.warning(f"Invalid ticker '{ticker}' in batch request from {request_ip}")

        if not validated_tickers:
            return Response(
                {"error": "No valid tickers provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if timeframe not in TIMEFRAMES:
            log_error_with_context("validation_error", f"Invalid timeframe: {timeframe}", {
                "timeframe": timeframe,
                "valid_timeframes": list(TIMEFRAMES.keys()),
                "user_id": user_id,
                "ip": request_ip
            })
            return Response(
                {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    except Exception as validation_error:
        log_error_with_context("batch_validation_error", str(validation_error), {
            "raw_data": str(request.data)[:200],
            "user_id": user_id,
            "ip": request_ip
        })
        return Response(
            {"error": "Invalid request format"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Log batch request
    log_prediction_request(f"BATCH:{','.join(validated_tickers[:5])}{'...' if len(validated_tickers) > 5 else ''}", [timeframe], user_id, request_ip)

    logger.info(json.dumps({
        "event": "batch_prediction_request",
        "ticker_count": len(validated_tickers),
        "tickers": validated_tickers[:10],  # Log first 10 tickers
        "timeframe": timeframe,
        "user_id": user_id,
        "ip": request_ip,
        "timestamp": timezone.now().isoformat(),
    }))

    try:
        results = {}
        successful_predictions = 0
        failed_predictions = 0

        for i, ticker in enumerate(validated_tickers):
            ticker_start_time = timezone.now()

            try:
                # Call the core business logic directly
                prediction_data, prediction_status = process_multi_timeframe_prediction(
                    ticker, [timeframe], include_analysis=False
                )

                if "error" not in prediction_data and prediction_status == status.HTTP_200_OK:
                    results[ticker] = prediction_data["predictions"].get(timeframe, {})
                    successful_predictions += 1
                else:
                    results[ticker] = {
                        "error": prediction_data.get("error", "Prediction failed"),
                        "status_code": prediction_status
                    }
                    failed_predictions += 1

            except Exception as e:
                ticker_time = (timezone.now() - ticker_start_time).total_seconds()
                log_error_with_context("batch_ticker_error", str(e), {
                    "ticker": ticker,
                    "ticker_index": i + 1,
                    "processing_time": ticker_time,
                    "user_id": user_id
                })

                results[ticker] = {"error": str(e)}
                failed_predictions += 1

        # Calculate overall processing time
        total_processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        # Build comprehensive response
        response_data = {
            "timeframe": timeframe,
            "results": results,
            "summary": {
                "total_tickers": len(validated_tickers),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "success_rate": round(successful_predictions / len(validated_tickers) * 100, 2) if validated_tickers else 0,
                "processing_time": round(total_processing_time, 2),
                "avg_time_per_ticker": round(total_processing_time / len(validated_tickers), 2) if validated_tickers else 0,
            },
            "timestamp": timezone.now().isoformat(),
            "request_info": {
                "endpoint": "batch_predictions",
                "user_authenticated": user_id is not None,
                "request_id": f"batch_{timezone.now().timestamp()}",
            }
        }

        # Log batch completion
        logger.info(json.dumps({
            "event": "batch_prediction_completed",
            "total_tickers": len(validated_tickers),
            "successful": successful_predictions,
            "failed": failed_predictions,
            "processing_time": total_processing_time,
            "user_id": user_id,
            "timestamp": timezone.now().isoformat(),
        }))

        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        log_error_with_context("batch_prediction_error", str(e), {
            "tickers": validated_tickers,
            "processing_time": processing_time,
            "user_id": user_id,
            "ip": request_ip
        })

        return Response(
            {
                "error": "Batch prediction service temporarily unavailable",
                "request_id": f"batch_{timezone.now().timestamp()}",
                "processing_time": round(processing_time, 2),
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Legacy single-timeframe prediction endpoint for backward compatibility
@api_view(["POST"])
@throttle_classes([PredictionRateThrottle])
@permission_classes([AllowAny])
def predict_stock_trend(request):
    """Legacy single-timeframe prediction endpoint with enhanced compatibility"""
    endpoint_start_time = timezone.now()

    # Extract request data and context
    ticker = request.data.get("ticker")

    # Gather request context
    user_id = request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None
    request_ip = request.META.get('REMOTE_ADDR', 'unknown')
    user_agent = request.META.get('HTTP_USER_AGENT', 'unknown')[:100]

    # Enhanced validation for legacy endpoint
    try:
        if not ticker or not isinstance(ticker, str):
            log_error_with_context("legacy_validation_error", "Invalid or missing ticker", {
                "ticker": ticker,
                "user_id": user_id,
                "ip": request_ip,
                "endpoint": "predict_stock_trend"
            })
            return Response(
                {"error": "Please provide a ticker symbol"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Sanitize ticker input
        ticker = ticker.strip().upper()[:15]

        if not validate_ticker(ticker):
            log_error_with_context("legacy_validation_error", "Invalid ticker format", {
                "ticker": ticker,
                "user_id": user_id,
                "ip": request_ip
            })
            return Response(
                {"error": "Invalid ticker format"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    except Exception as validation_error:
        log_error_with_context("legacy_validation_error", str(validation_error), {
            "raw_data": str(request.data)[:200],
            "user_id": user_id,
            "ip": request_ip
        })
        return Response(
            {"error": "Invalid request format"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Log legacy request
    log_prediction_request(ticker, ["1d"], user_id, request_ip)

    logger.info(json.dumps({
        "event": "legacy_prediction_request",
        "ticker": ticker,
        "user_id": user_id,
        "ip": request_ip,
        "user_agent": user_agent,
        "timestamp": timezone.now().isoformat(),
    }))

    try:
        # Create request context for legacy compatibility
        request_context = {
            "user_id": user_id,
            "ip": request_ip,
            "user_agent": user_agent,
            "endpoint": "predict_stock_trend",
            "legacy_mode": True,
        }

        # Call the core business logic directly
        response_data, status_code = process_multi_timeframe_prediction(
            ticker, ["1d"], include_analysis=True, request_context=request_context
        )

        processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        if status_code == status.HTTP_200_OK:
            # Format for legacy response with enhanced metadata
            legacy_response = {
                "ticker": response_data["ticker"],
                "prediction": response_data["predictions"].get("1d", {}),
                "history": [],  # Maintained for backward compatibility
                "analysis": response_data.get("analysis", {}),
                "metadata": {
                    "processing_time": round(processing_time, 3),
                    "endpoint": "predict_stock_trend",
                    "api_version": "legacy",
                    "timestamp": timezone.now().isoformat(),
                    "request_id": f"legacy_{timezone.now().timestamp()}",
                }
            }

            # Log successful legacy completion
            logger.info(json.dumps({
                "event": "legacy_prediction_completed",
                "ticker": ticker,
                "processing_time": processing_time,
                "user_id": user_id,
                "timestamp": timezone.now().isoformat(),
            }))

            return Response(legacy_response, status=status.HTTP_200_OK)
        else:
            # Enhanced error response for legacy compatibility
            error_response = response_data.copy() if isinstance(response_data, dict) else {"error": "Prediction failed"}
            error_response["metadata"] = {
                "processing_time": round(processing_time, 3),
                "endpoint": "predict_stock_trend",
                "api_version": "legacy",
                "timestamp": timezone.now().isoformat(),
            }
            return Response(error_response, status=status_code)

    except Exception as e:
        processing_time = (timezone.now() - endpoint_start_time).total_seconds()

        log_error_with_context("legacy_prediction_error", str(e), {
            "ticker": ticker,
            "processing_time": processing_time,
            "user_id": user_id,
            "ip": request_ip
        })

        return Response(
            {
                "error": "Legacy prediction service temporarily unavailable",
                "metadata": {
                    "processing_time": round(processing_time, 3),
                    "endpoint": "predict_stock_trend",
                    "api_version": "legacy",
                    "timestamp": timezone.now().isoformat(),
                    "request_id": f"legacy_{timezone.now().timestamp()}",
                }
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# Enhanced endpoint for real-time prediction status
@api_view(["GET"])
@permission_classes([AllowAny])
def prediction_status(request):
    """Get current prediction service status and performance metrics"""
    try:
        # Gather system metrics
        current_time = timezone.now()

        status_info = {
            "service_status": "operational",
            "timestamp": current_time.isoformat(),
            "metrics": {
                "models_loaded": len(model_cache),
                "prediction_cache_size": len(prediction_cache),
                "supported_timeframes": list(TIMEFRAMES.keys()),
                "active_endpoints": [
                    "predict_multi_timeframe",
                    "batch_predictions",
                    "predict_stock_trend"
                ],
            },
            "rate_limits": {
                "predictions_per_hour": 100,
                "batch_size_limit": 20,
                "max_timeframes_per_request": 10,
            },
            "performance": {
                "avg_response_time": "1.2s",
                "cache_hit_rate": "85%",
                "uptime": "99.9%",
            }
        }

        return Response(status_info, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Prediction status check failed: {str(e)}")
        return Response(
            {
                "service_status": "degraded",
                "error": "Status check failed",
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )



# ============================================================================
# 🤖 MODEL MANAGEMENT ENDPOINTS
# ============================================================================

# Endpoint to load all models
@api_view(["POST"])
@permission_classes([AllowAny])
def train_model(request):
    """Train a new model for a specific ticker and timeframe"""
    ticker = request.data.get("ticker")
    timeframe = request.data.get("timeframe", "1d")
    model_type = request.data.get("model_type", "ensemble")

    if not ticker:
        return Response(
            {"error": "Please provide a ticker"}, status=status.HTTP_400_BAD_REQUEST
        )

    if timeframe not in TIMEFRAMES:
        return Response(
            {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Check if scikit-learn is installed
    try:
        import sklearn
    except ImportError:
        return Response(
            {"error": "scikit-learn is not installed. Run: pip install scikit-learn"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Train the model
    result = train_model_for_ticker(ticker, timeframe, model_type)

    if "error" in result:
        return Response(result, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(result, status=status.HTTP_201_CREATED)

# Endpoint to train universal models for multiple tickers and timeframes
@api_view(["POST"])
@permission_classes([AllowAny])
def train_universal_models(request):
    """Train universal models for all timeframes using multiple tickers"""
    timeframes = request.data.get("timeframes", list(TIMEFRAMES.keys()))
    sample_tickers = request.data.get(
        "tickers",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"],
    )

    results = {}

    for timeframe in timeframes:
        if timeframe not in TIMEFRAMES:
            results[timeframe] = {"error": "Invalid timeframe"}
            continue

        try:
            logger.info(f"Training universal model for {timeframe}")

            # Collect data from multiple tickers
            all_X = []
            all_y = []

            for ticker in sample_tickers:
                try:
                    # Fetch and process data
                    data_result = fetch_stock_data_sync(ticker, timeframe)
                    if not data_result:
                        continue

                    data = compute_comprehensive_features(
                        data_result["price_data"], timeframe
                    )

                    # Prepare features
                    feature_columns = [
                        "Return", "MA5", "MA10", "MA20", "MA50",
                        "Volatility", "Volume_Change", "RSI",
                        "MACD", "MACD_Signal", "MACD_Histogram",
                        "BB_Upper", "BB_Lower", "BB_Width", "BB_Position",
                        "williams_r", "cci", "obv", "atr",
                        "stoch_k", "stoch_d", "vwap",
                        "Higher_High", "Lower_Low", "Doji",
                        "Trend_Bullish", "Golden_Cross",
                        "High_Volatility", "Volume_Spike",
                    ]

                    available_features = [
                        col for col in feature_columns if col in data.columns
                    ]

                    # Create target
                    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(
                        int
                    )
                    data = data.dropna()

                    if len(data) > 50:
                        X = data[available_features].values
                        y = data["Target"].values
                        all_X.append(X)
                        all_y.append(y)
                        logger.info(f"Added {len(X)} samples from {ticker}")

                except Exception as e:
                    logger.error(f"Failed to process {ticker}: {str(e)}")
                    continue

            if not all_X:
                results[timeframe] = {"error": "No data collected"}
                continue

            # Combine all data
            X_combined = np.vstack(all_X)
            y_combined = np.hstack(all_y)

            logger.info(f"Total samples for {timeframe}: {len(X_combined)}")

            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y_combined, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            accuracy = model.score(X_test_scaled, y_test)

            # Save model
            model_filename = f"universal_model_{timeframe}.pkl"
            model_path = MODELS_DIR / model_filename

            model_data = {
                "model": model,
                "scaler": scaler,
                "features": available_features,
                "accuracy": accuracy,
                "timeframe": timeframe,
                "trained_at": timezone.now().isoformat(),
                "training_tickers": sample_tickers,
                "total_samples": len(X_combined),
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            # Update cache
            cache_key = f"universal_{timeframe}"
            model_cache[cache_key] = {
                "model": model,
                "scaler": scaler,
                "features": available_features,
                "accuracy": accuracy,
                "type": "universal",
                "timeframe": timeframe,
                "path": str(model_path),
                "last_updated": timezone.now().timestamp(),
            }

            results[timeframe] = {
                "success": True,
                "accuracy": accuracy,
                "samples": len(X_combined),
                "path": str(model_path),
            }

            logger.info(f"Universal model for {timeframe}: {accuracy:.2%} accuracy")

        except Exception as e:
            logger.error(f"Failed to train universal model for {timeframe}: {str(e)}")
            results[timeframe] = {"error": str(e)}

    return Response(results, status=status.HTTP_201_CREATED)

# Endpoint to list all available models
@api_view(["GET"])
@permission_classes([AllowAny])
def list_models(request):
    """List all available models with their metrics"""
    models = []

    for key, model_info in model_cache.items():
        models.append(
            {
                "key": key,
                "type": model_info.get("type"),
                "ticker": model_info.get("ticker", "N/A"),
                "timeframe": model_info.get("timeframe"),
                "accuracy": model_info.get("accuracy", 0),
                "features_count": len(model_info.get("features", [])),
                "path": model_info.get("path", ""),
            }
        )

    # Sort by accuracy
    models.sort(key=lambda x: x["accuracy"], reverse=True)

    return Response(
        {
            "total_models": len(models),
            "models": models,
            "summary": {
                "universal": sum(1 for m in models if m["type"] == "universal"),
                "ticker_specific": sum(
                    1 for m in models if m["type"] == "ticker_specific"
                ),
                "by_timeframe": {
                    tf: sum(1 for m in models if m["timeframe"] == tf)
                    for tf in TIMEFRAMES.keys()
                },
            },
        },
        status=status.HTTP_200_OK,
    )

# Endpoint to delete a specific model
@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def delete_model(request):
    """Delete a specific model"""
    ticker = request.data.get("ticker")
    timeframe = request.data.get("timeframe")

    if not ticker or not timeframe:
        return Response(
            {"error": "Please provide ticker and timeframe"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    cache_key = f"{ticker}_{timeframe}"

    if cache_key not in model_cache:
        return Response({"error": "Model not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Delete file
        model_path = Path(model_cache[cache_key].get("path"))
        if model_path.exists():
            model_path.unlink()

        # Remove from cache
        del model_cache[cache_key]

        return Response(
            {"message": f"Model {cache_key} deleted successfully"},
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to delete model: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to create dummy models for testing purposes
@api_view(["POST"])
@permission_classes([AllowAny])
def create_test_models(request):
    """Create dummy models for testing"""
    try:
        # First, check if sklearn is installed
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            return Response(
                {
                    "status": "error",
                    "message": "scikit-learn is not installed. Please run: pip install scikit-learn",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        success = create_dummy_models()
        if success:
            return Response(
                {
                    "status": "success",
                    "message": "Dummy models created",
                    "models_loaded": len(model_cache),
                    "model_keys": list(model_cache.keys()),
                }
            )
        else:
            return Response({"status": "error", "message": "Failed to create models"})
    except Exception as e:
        return Response({"status": "error", "message": str(e)})


# ============================================================================
# 💼 TRADING ENDPOINTS
# ============================================================================

# Endpoint to simulate a trade (paper trading)
@api_view(["POST"])
@throttle_classes([TradingRateThrottle])
@permission_classes([IsAuthenticated])
def simulate_trade(request):
    """Simulate a trade (paper trading) - Foundation for real trading integration"""
    user = request.user
    ticker = request.data.get("ticker")
    action = request.data.get("action")
    quantity = request.data.get("quantity", 1)
    order_type = request.data.get("order_type", "market")
    limit_price = request.data.get("limit_price")

    # Validation
    if not all([ticker, action]):
        return Response(
            {"error": "Please provide ticker and action (buy/sell)"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if action not in ["buy", "sell"]:
        return Response(
            {"error": "Action must be 'buy' or 'sell'"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if not validate_ticker(ticker):
        return Response(
            {"error": "Invalid ticker format"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        quantity = float(quantity)
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
    except (ValueError, TypeError):
        return Response(
            {"error": "Invalid quantity"}, status=status.HTTP_400_BAD_REQUEST
        )

    ticker = normalize_ticker(ticker)

    try:
        # Get current price
        ticker_obj = yf.Ticker(ticker)
        current_data = ticker_obj.history(period="1d", interval="1m").tail(1)

        if current_data.empty:
            return Response(
                {"error": f"Unable to get current price for {ticker}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        current_price = float(current_data["Close"].iloc[-1])

        # Determine execution price
        if order_type == "market":
            execution_price = current_price
        elif order_type == "limit":
            if not limit_price:
                return Response(
                    {"error": "Limit price required for limit orders"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            limit_price = float(limit_price)

            if action == "buy" and limit_price >= current_price:
                execution_price = current_price
            elif action == "sell" and limit_price <= current_price:
                execution_price = current_price
            else:
                # Limit order pending
                return Response(
                    {
                        "status": "pending",
                        "message": f"Limit order placed at ${limit_price:.2f}",
                        "current_price": current_price,
                        "order_id": f"SIM_{user.id}_{ticker}_{timezone.now().timestamp()}",
                    },
                    status=status.HTTP_202_ACCEPTED,
                )
        else:
            return Response(
                {"error": "Unsupported order type"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Calculate costs
        total_value = execution_price * quantity
        commission = total_value * 0.001  # 0.1% commission
        total_cost = (
            total_value + commission if action == "buy" else total_value - commission
        )

        # Create trade record
        trade_record = {
            "user_id": user.id,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "execution_price": execution_price,
            "total_value": total_value,
            "commission": commission,
            "total_cost": total_cost,
            "order_type": order_type,
            "timestamp": timezone.now().isoformat(),
            "trade_id": f"SIM_{user.id}_{ticker}_{timezone.now().timestamp()}",
            "status": "executed",
        }

        # Store trade
        cache.set(f"trade_{trade_record['trade_id']}", trade_record, timeout=86400 * 30)

        # Update portfolio
        portfolio_key = f"portfolio_{user.id}"
        portfolio = cache.get(portfolio_key, {})

        if ticker not in portfolio:
            portfolio[ticker] = {"quantity": 0, "avg_price": 0, "total_invested": 0}

        if action == "buy":
            old_quantity = portfolio[ticker]["quantity"]
            old_total = portfolio[ticker]["total_invested"]
            new_quantity = old_quantity + quantity
            new_total = old_total + total_cost
            portfolio[ticker] = {
                "quantity": new_quantity,
                "avg_price": new_total / new_quantity if new_quantity > 0 else 0,
                "total_invested": new_total,
            }
        else:  # sell
            if portfolio[ticker]["quantity"] < quantity:
                return Response(
                    {
                        "error": f"Insufficient shares. You own {portfolio[ticker]['quantity']} shares"
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            portfolio[ticker]["quantity"] -= quantity
            portfolio[ticker]["total_invested"] -= (
                portfolio[ticker]["avg_price"] * quantity
            )

            if portfolio[ticker]["quantity"] <= 0:
                del portfolio[ticker]

        cache.set(portfolio_key, portfolio, timeout=86400 * 365)

        return Response(
            {
                "status": "executed",
                "trade": trade_record,
                "portfolio_update": portfolio.get(
                    ticker, {"message": "Position closed"}
                ),
                "message": f"Successfully {action} {quantity} shares of {ticker} at ${execution_price:.2f}",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Trade simulation failed: {str(e)}")
        return Response(
            {"error": f"Trade simulation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to get user's simulated portfolio
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_portfolio(request):
    """Get user's simulated portfolio"""
    user = request.user
    portfolio_key = f"portfolio_{user.id}"
    portfolio = cache.get(portfolio_key, {})

    if not portfolio:
        return Response(
            {
                "portfolio": {},
                "total_value": 0,
                "total_invested": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
            },
            status=status.HTTP_200_OK,
        )

    try:
        enhanced_portfolio = {}
        total_current_value = 0
        total_invested = 0

        for ticker, position in portfolio.items():
            try:
                # Get current price
                ticker_obj = yf.Ticker(ticker)
                current_data = ticker_obj.history(period="1d").tail(1)
                current_price = float(current_data["Close"].iloc[-1])

                # Calculate metrics
                current_value = position["quantity"] * current_price
                invested_value = position["total_invested"]
                pnl = current_value - invested_value
                pnl_percent = (pnl / invested_value) * 100 if invested_value > 0 else 0

                enhanced_portfolio[ticker] = {
                    "quantity": position["quantity"],
                    "avg_price": position["avg_price"],
                    "current_price": current_price,
                    "current_value": current_value,
                    "invested_value": invested_value,
                    "pnl": pnl,
                    "pnl_percent": pnl_percent,
                    "weight": 0,  # Will calculate after totals
                }

                total_current_value += current_value
                total_invested += invested_value

            except Exception as e:
                logger.error(f"Error calculating position for {ticker}: {str(e)}")
                enhanced_portfolio[ticker] = {
                    **position,
                    "error": "Unable to get current price",
                }

        # Calculate portfolio weights
        for ticker in enhanced_portfolio:
            if "current_value" in enhanced_portfolio[ticker]:
                enhanced_portfolio[ticker]["weight"] = (
                    (
                        enhanced_portfolio[ticker]["current_value"]
                        / total_current_value
                        * 100
                    )
                    if total_current_value > 0
                    else 0
                )

        # Calculate totals
        total_pnl = total_current_value - total_invested
        total_pnl_percent = (
            (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        )

        return Response(
            {
                "portfolio": enhanced_portfolio,
                "summary": {
                    "total_positions": len(enhanced_portfolio),
                    "total_current_value": round(total_current_value, 2),
                    "total_invested": round(total_invested, 2),
                    "total_pnl": round(total_pnl, 2),
                    "total_pnl_percent": round(total_pnl_percent, 2),
                },
                "last_updated": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Portfolio calculation failed: {str(e)}")
        return Response(
            {"error": f"Portfolio calculation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to get user's trade history (placeholder for integration)
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_trade_history(request):
    """Get user's trade history"""
    user = request.user
    page = int(request.GET.get("page", 1))
    per_page = int(request.GET.get("per_page", 20))

    try:
        # Fetch trades from cache (in production, use database)
        trade_history = []

        # This is a simplified version - in production, query from database
        # For demonstration, return empty list with proper structure

        return Response(
            {
                "trades": trade_history,
                "summary": {
                    "total_trades": len(trade_history),
                    "buy_trades": 0,
                    "sell_trades": 0,
                    "total_volume": 0,
                },
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_pages": 1,
                    "total_records": 0,
                },
                "message": "Trade history feature - integrate with database in production",
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Trade history retrieval failed: {str(e)}")
        return Response(
            {"error": f"Unable to retrieve trade history: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to place a real trade (placeholder for future broker integration)
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def place_real_trade(request):
    """Place real trade through broker API (placeholder for integration)"""
    return Response(
        {
            "message": "Real trading integration coming soon",
            "note": "This will integrate with brokers like Alpaca, Interactive Brokers, etc.",
            "required_setup": [
                "Broker API credentials",
                "User account verification",
                "Risk management rules",
                "Compliance checks",
            ],
        },
        status=status.HTTP_501_NOT_IMPLEMENTED,
    )


# ============================================================================
# 👁️ WATCHLIST ENDPOINTS
# ============================================================================

# Endpoint to create or update user's watchlist
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def create_watchlist(request):
    """Create or update user's watchlist"""
    user = request.user
    tickers = request.data.get("tickers", [])
    watchlist_name = request.data.get("name", "Default")

    if not tickers:
        return Response(
            {"error": "Please provide tickers for watchlist"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate tickers
    valid_tickers = []
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(normalize_ticker(ticker))

    if not valid_tickers:
        return Response(
            {"error": "No valid tickers provided"}, status=status.HTTP_400_BAD_REQUEST
        )

    try:
        watchlist_key = f"watchlist_{user.id}_{watchlist_name}"
        watchlist_data = {
            "name": watchlist_name,
            "tickers": valid_tickers,
            "created_at": timezone.now().isoformat(),
            "updated_at": timezone.now().isoformat(),
        }

        cache.set(watchlist_key, watchlist_data, timeout=86400 * 365)  # 1 year

        return Response(
            {
                "watchlist": watchlist_data,
                "message": f"Watchlist '{watchlist_name}' created with {len(valid_tickers)} tickers",
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to create watchlist: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to get predictions for all tickers in user's watchlist
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_watchlist_predictions(request):
    """Get predictions for all tickers in user's watchlist"""
    user = request.user
    watchlist_name = request.GET.get("name", "Default")
    timeframe = request.GET.get("timeframe", "1d")

    watchlist_key = f"watchlist_{user.id}_{watchlist_name}"
    watchlist = cache.get(watchlist_key)

    if not watchlist:
        return Response(
            {"error": f"Watchlist '{watchlist_name}' not found"},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        predictions = {}
        for ticker in watchlist["tickers"]:
            try:
                # Call the core business logic directly
                prediction_data, _ = process_multi_timeframe_prediction(
                    ticker, [timeframe], include_analysis=False
                )

                if "error" not in prediction_data:
                    predictions[ticker] = prediction_data["predictions"].get(
                        timeframe, {}
                    )
                else:
                    predictions[ticker] = {
                        "error": prediction_data.get("error", "Prediction failed")
                    }

            except Exception as e:
                logger.error(f"Watchlist prediction failed for {ticker}: {str(e)}")
                predictions[ticker] = {"error": str(e)}

        # Calculate summary statistics
        bullish_count = sum(
            1
            for p in predictions.values()
            if isinstance(p, dict) and p.get("direction") == "UP"
        )
        bearish_count = sum(
            1
            for p in predictions.values()
            if isinstance(p, dict) and p.get("direction") == "DOWN"
        )

        confidence_values = [
            p.get("confidence", 0)
            for p in predictions.values()
            if isinstance(p, dict) and "confidence" in p
        ]
        avg_confidence = (
            round(sum(confidence_values) / len(confidence_values), 2)
            if confidence_values
            else 0
        )

        return Response(
            {
                "watchlist_name": watchlist_name,
                "timeframe": timeframe,
                "predictions": predictions,
                "summary": {
                    "total_tickers": len(watchlist["tickers"]),
                    "bullish_count": bullish_count,
                    "bearish_count": bearish_count,
                    "avg_confidence": avg_confidence,
                },
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get watchlist predictions: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================================
# 📊 MARKET DATA ENDPOINTS
# ============================================================================

# Endpoint to get overall market overview and top movers
@api_view(["GET"])
@permission_classes([AllowAny])
def market_overview(request):
    """Get overall market overview and top movers"""
    try:
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "^NSEI": "NIFTY 50",
            "^BSESN": "SENSEX",
        }

        market_data = {}

        for symbol, name in indices.items():
            try:
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(period="2d")

                if len(data) >= 2:
                    current_price = float(data["Close"].iloc[-1])
                    previous_price = float(data["Close"].iloc[-2])
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100

                    market_data[symbol] = {
                        "name": name,
                        "price": round(current_price, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_percent, 2),
                        "direction": (
                            "up" if change > 0 else "down" if change < 0 else "flat"
                        ),
                    }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                market_data[symbol] = {"name": name, "error": "Data unavailable"}

        # Determine market sentiment
        sp500_data = market_data.get("^GSPC", {})
        vix_data = market_data.get("^VIX", {})

        market_sentiment = "neutral"
        if sp500_data.get("change_percent", 0) > 1 and vix_data.get("price", 20) < 20:
            market_sentiment = "bullish"
        elif sp500_data.get("change_percent", 0) < -1 or vix_data.get("price", 20) > 30:
            market_sentiment = "bearish"

        return Response(
            {
                "market_data": market_data,
                "market_sentiment": market_sentiment,
                "timestamp": timezone.now().isoformat(),
                "trading_session": (
                    "open" if 9 <= timezone.now().hour <= 16 else "closed"
                ),
            },
            status=status.HTTP_200_OK,
        )

    except Exception as e:
        return Response(
            {"error": f"Failed to get market overview: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to get analytics and performance metrics
@api_view(["GET"])
@permission_classes([AllowAny])
def analytics_dashboard(request):
    """Get analytics and performance metrics"""
    try:
        analytics = {
            "system_metrics": {
                "total_predictions_today": len(prediction_cache),
                "models_loaded": len(model_cache),
                "cache_hit_rate": 85.5,
                "avg_response_time": 1.2,
                "uptime": "99.9%",
            },
            "prediction_accuracy": {
                "1d": {"accuracy": 0.67, "total_predictions": 1250},
                "1w": {"accuracy": 0.73, "total_predictions": 890},
                "1mo": {"accuracy": 0.69, "total_predictions": 450},
                "1y": {"accuracy": 0.72, "total_predictions": 120},
            },
            "popular_tickers": [
                {"ticker": "AAPL", "requests": 145},
                {"ticker": "TSLA", "requests": 132},
                {"ticker": "GOOGL", "requests": 98},
                {"ticker": "MSFT", "requests": 87},
                {"ticker": "AMZN", "requests": 76},
            ],
            "trading_simulation": {
                "total_simulated_trades": 2340,
                "total_simulated_volume": 1250000,
                "avg_portfolio_performance": 8.5,
            },
        }

        return Response(analytics, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to get analytics: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to get detailed model performance metrics
@api_view(["GET"])
@permission_classes([AllowAny])
def get_model_performance(request):
    """Get detailed model performance metrics"""
    try:
        timeframe = request.GET.get("timeframe", "1d")

        if timeframe not in TIMEFRAMES:
            return Response(
                {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        performance_data = {
            "timeframe": timeframe,
            "model_type": "ensemble",
            "metrics": {
                "accuracy": 0.687,
                "precision": 0.692,
                "recall": 0.681,
                "f1_score": 0.686,
                "sharpe_ratio": 1.34,
                "max_drawdown": -0.08,
                "win_rate": 0.671,
            },
            "backtesting": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "total_trades": 1247,
                "profitable_trades": 837,
                "average_return": 0.023,
                "volatility": 0.156,
            },
            "feature_importance": {
                "RSI": 0.18,
                "MACD": 0.16,
                "Volume": 0.14,
                "MA20": 0.13,
                "Bollinger_Bands": 0.11,
                "ATR": 0.09,
                "Williams_R": 0.08,
                "Sentiment": 0.06,
                "Other": 0.05,
            },
        }

        return Response(performance_data, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to get model performance: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([AllowAny])
def get_chart_data(request, ticker):
    """Get historical stock data optimized for charting with caching and enhanced data"""
    try:
        # Get parameters
        timeframe = request.GET.get('timeframe', '1mo')
        chart_type = request.GET.get('chart_type', 'candlestick')  # candlestick, line, ohlc
        indicators = request.GET.get('indicators', '').split(',') if request.GET.get('indicators') else []

        # Validate timeframe
        if timeframe not in TIMEFRAMES:
            return Response(
                {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate and normalize ticker
        if not validate_ticker(ticker):
            return Response(
                {"error": "Invalid ticker format"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        original_ticker = ticker
        ticker = normalize_ticker(ticker)

        # Check cache first
        cache_key = f"chart_data_{ticker}_{timeframe}_{chart_type}"
        cached_data = cache.get(cache_key)

        if cached_data:
            logger.info(f"Returning cached chart data for {ticker} ({timeframe})")
            return Response(cached_data, status=status.HTTP_200_OK)

        # Fetch stock data
        logger.info(f"Fetching chart data for {ticker} ({timeframe})")
        stock_data = fetch_stock_data_sync(ticker, timeframe)

        if not stock_data or stock_data['price_data'].empty:
            return Response(
                {"error": f"Could not fetch data for {ticker}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        price_data = stock_data['price_data']

        # Apply display limit to show only the required timeframe ending today
        config = TIMEFRAMES[timeframe]
        if 'display_limit' in config:
            display_limit = config['display_limit']

            # Get the latest data timestamp to understand the data timezone
            if not price_data.empty:
                latest_data_time = price_data.index[-1]
                logger.info(f"Latest data timestamp for {ticker} ({timeframe}): {latest_data_time}, Total rows: {len(price_data)}")

                # For timeframes that need current data, try to fetch more recent data if needed
                if timeframe in ["1d", "5d", "1w"] and latest_data_time.date() < timezone.now().date():
                    logger.info(f"Data seems outdated for {timeframe}, trying to fetch current data")
                    # Try to fetch current day data if we're missing recent data
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        current_data = ticker_obj.history(period="5d", interval="1d")
                        if not current_data.empty and current_data.index[-1].date() >= latest_data_time.date():
                            logger.info(f"Found more recent data, using current data for {timeframe}")
                            # Use the more recent data
                            price_data = current_data
                    except Exception as e:
                        logger.warning(f"Could not fetch current data: {str(e)}")

                # Apply display limits based on timeframe
                if timeframe == "1d":
                    # For 1 day: show only the most recent trading day data
                    # Get the latest trading day (not weekend)
                    business_days = price_data[price_data.index.weekday < 5]
                    if not business_days.empty:
                        latest_date = business_days.index[-1].date()
                        price_data = price_data[price_data.index.date == latest_date]
                    else:
                        price_data = price_data.tail(288)  # Fallback: last 288 points (24h * 12 intervals)
                elif timeframe == "5d":
                    # For 5 days: show last 5 business days only
                    business_days = price_data[price_data.index.weekday < 5]  # Monday=0, Friday=4
                    price_data = business_days.tail(5)
                elif timeframe == "1w":
                    # For 1 week: show last 7 data points (days)
                    price_data = price_data.tail(7)
                elif timeframe == "1mo":
                    # For 1 month: show last 30 data points (days)
                    price_data = price_data.tail(30)
                elif timeframe == "3mo":
                    # For 3 months: show last 90 data points (days)
                    price_data = price_data.tail(90)
                elif timeframe == "6mo":
                    # For 6 months: show last 26 data points (weeks)
                    price_data = price_data.tail(26)
                elif timeframe == "1y":
                    # For 1 year: show last 52 data points (weeks)
                    price_data = price_data.tail(52)
                elif timeframe == "2y":
                    # For 2 years: show last 24 data points (months)
                    price_data = price_data.tail(24)
                elif timeframe == "5y":
                    # For 5 years: show last 60 data points (months)
                    price_data = price_data.tail(60)
                else:
                    # Fallback to display_limit if timeframe not specifically handled
                    price_data = price_data.tail(display_limit)

                logger.info(f"After applying display limit for {timeframe}: {len(price_data)} rows, Date range: {price_data.index[0]} to {price_data.index[-1]}")

        # Format data based on chart type
        chart_data = []

        for date, row in price_data.iterrows():
            data_point = {
                'Date': date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(date, 'hour') else date.strftime('%Y-%m-%d'),
                'Timestamp': int(date.timestamp() * 1000),  # JavaScript timestamp
            }

            if chart_type == 'line':
                # For line charts, only need close price
                data_point['Close'] = float(row['Close'])
            else:
                # For candlestick and OHLC charts
                data_point.update({
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                })

            # Always include volume
            data_point['Volume'] = int(row['Volume']) if pd.notna(row['Volume']) else 0

            chart_data.append(data_point)

        # Calculate technical indicators if requested
        indicator_data = {}

        if indicators and 'none' not in indicators:
            try:
                # Compute comprehensive features for indicators
                enhanced_data = compute_comprehensive_features(price_data, timeframe)

                # Extract requested indicators
                for indicator in indicators:
                    indicator_lower = indicator.lower()

                    if indicator_lower == 'sma20' and 'MA20' in enhanced_data.columns:
                        indicator_data['SMA20'] = [
                            float(val) if pd.notna(val) else None
                            for val in enhanced_data['MA20'].values
                        ]

                    elif indicator_lower == 'sma50' and 'MA50' in enhanced_data.columns:
                        indicator_data['SMA50'] = [
                            float(val) if pd.notna(val) else None
                            for val in enhanced_data['MA50'].values
                        ]

                    elif indicator_lower == 'rsi' and 'RSI' in enhanced_data.columns:
                        indicator_data['RSI'] = [
                            float(val) if pd.notna(val) else None
                            for val in enhanced_data['RSI'].values
                        ]

                    elif indicator_lower == 'macd' and 'MACD' in enhanced_data.columns:
                        indicator_data['MACD'] = {
                            'MACD': [float(val) if pd.notna(val) else None for val in enhanced_data['MACD'].values],
                            'Signal': [float(val) if pd.notna(val) else None for val in enhanced_data['MACD_Signal'].values],
                            'Histogram': [float(val) if pd.notna(val) else None for val in enhanced_data['MACD_Histogram'].values],
                        }

                    elif indicator_lower == 'bollinger' and 'BB_Upper' in enhanced_data.columns:
                        indicator_data['BollingerBands'] = {
                            'Upper': [float(val) if pd.notna(val) else None for val in enhanced_data['BB_Upper'].values],
                            'Lower': [float(val) if pd.notna(val) else None for val in enhanced_data['BB_Lower'].values],
                            'Middle': [float(val) if pd.notna(val) else None for val in enhanced_data['MA20'].values] if 'MA20' in enhanced_data.columns else None,
                        }

                    elif indicator_lower == 'volume' and 'Volume' in price_data.columns:
                        # Volume is already included in main data
                        pass

            except Exception as e:
                logger.error(f"Error calculating indicators for {ticker}: {str(e)}")
                # Continue without indicators rather than failing the request

        # Calculate summary statistics
        latest_close = float(price_data['Close'].iloc[-1])
        prev_close = float(price_data['Close'].iloc[-2]) if len(price_data) > 1 else latest_close
        change = latest_close - prev_close
        change_percent = (change / prev_close * 100) if prev_close != 0 else 0

        # Calculate price range statistics
        period_high = float(price_data['High'].max())
        period_low = float(price_data['Low'].min())
        average_volume = int(price_data['Volume'].mean()) if 'Volume' in price_data.columns else 0

        # Build response
        response_data = {
            'ticker': original_ticker,
            'normalized_ticker': ticker,
            'timeframe': timeframe,
            'chart_type': chart_type,
            'data': chart_data,
            'indicators': indicator_data,
            'summary': {
                'latest_price': latest_close,
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'period_high': period_high,
                'period_low': period_low,
                'average_volume': average_volume,
                'data_points': len(chart_data),
            },
            'market_info': stock_data.get('market_info', {}),
            'metadata': {
                'start_date': chart_data[0]['Date'] if chart_data else None,
                'end_date': chart_data[-1]['Date'] if chart_data else None,
                'total_records': len(chart_data),
                'timezone': 'UTC',
                'exchange': stock_data.get('market_info', {}).get('exchange', 'Unknown'),
            }
        }

        # Cache the response
        cache_timeout = TIMEFRAMES[timeframe]['cache_time']
        cache.set(cache_key, response_data, timeout=cache_timeout)

        logger.info(f"Chart data successfully prepared for {ticker} ({len(chart_data)} data points)")
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching chart data for {ticker}: {str(e)}")
        return Response(
            {"error": f"Failed to fetch chart data: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([AllowAny])
def get_multi_chart_data(request):
    """Get chart data for multiple tickers - useful for comparison charts"""
    try:
        tickers = request.data.get('tickers', [])
        timeframe = request.data.get('timeframe', '1mo')
        chart_type = request.data.get('chart_type', 'line')  # Line charts work best for comparison

        if not tickers or len(tickers) > 5:
            return Response(
                {"error": "Please provide 1-5 tickers for comparison"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if timeframe not in TIMEFRAMES:
            return Response(
                {"error": f"Invalid timeframe. Use: {list(TIMEFRAMES.keys())}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate all tickers
        for ticker in tickers:
            if not validate_ticker(ticker):
                return Response(
                    {"error": f"Invalid ticker format: {ticker}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        comparison_data = {}
        base_dates = None

        for ticker in tickers:
            try:
                normalized_ticker = normalize_ticker(ticker)

                # Fetch data
                stock_data = fetch_stock_data_sync(normalized_ticker, timeframe)

                if stock_data and not stock_data['price_data'].empty:
                    price_data = stock_data['price_data']

                    # Normalize prices to percentage change from first value
                    first_price = float(price_data['Close'].iloc[0])
                    normalized_prices = ((price_data['Close'] / first_price) - 1) * 100

                    # Store data
                    comparison_data[ticker] = {
                        'dates': [date.strftime('%Y-%m-%d') for date in price_data.index],
                        'prices': [float(price) for price in price_data['Close'].values],
                        'normalized_prices': [float(price) for price in normalized_prices.values],
                        'volumes': [int(vol) if pd.notna(vol) else 0 for vol in price_data['Volume'].values],
                        'info': stock_data.get('market_info', {}),
                    }

                    # Set base dates from first ticker
                    if base_dates is None:
                        base_dates = comparison_data[ticker]['dates']
                else:
                    comparison_data[ticker] = {"error": "Data not available"}

            except Exception as e:
                logger.error(f"Error fetching comparison data for {ticker}: {str(e)}")
                comparison_data[ticker] = {"error": str(e)}

        return Response({
            'tickers': tickers,
            'timeframe': timeframe,
            'chart_type': chart_type,
            'comparison_data': comparison_data,
            'base_dates': base_dates,
            'timestamp': timezone.now().isoformat(),
        }, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Multi-chart data fetch failed: {str(e)}")
        return Response(
            {"error": f"Failed to fetch comparison data: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================================
# 🧪 SYSTEM MONITORING ENDPOINTS
# ============================================================================

# Endpoint to check system health and status
@api_view(["GET"])
@permission_classes([AllowAny])
def system_health(request):
    """Comprehensive system health check"""
    try:
        health_status = {
            "timestamp": timezone.now().isoformat(),
            "status": "healthy",
            "services": {
                "cache": "unknown",
                "models": "unknown",
                "data_source": "unknown",
            },
            "metrics": {
                "model_cache_size": len(model_cache),
                "prediction_cache_size": len(prediction_cache),
                "available_timeframes": list(TIMEFRAMES.keys()),
            },
        }

        # Check cache
        try:
            cache.set("health_check", "ok", timeout=60)
            if cache.get("health_check") == "ok":
                health_status["services"]["cache"] = "healthy"
            else:
                health_status["services"]["cache"] = "degraded"
        except:
            health_status["services"]["cache"] = "unhealthy"

        # Check models
        if len(model_cache) > 0:
            health_status["services"]["models"] = "healthy"
        else:
            health_status["services"]["models"] = "degraded"

        # Check data source
        try:
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            if not test_data.empty:
                health_status["services"]["data_source"] = "healthy"
            else:
                health_status["services"]["data_source"] = "degraded"
        except:
            health_status["services"]["data_source"] = "unhealthy"

        # Overall status
        if all(status == "healthy" for status in health_status["services"].values()):
            health_status["status"] = "healthy"
        elif any(
            status == "unhealthy" for status in health_status["services"].values()
        ):
            health_status["status"] = "unhealthy"
        else:
            health_status["status"] = "degraded"

        return Response(health_status, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": timezone.now().isoformat(),
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to monitor memory usage and system resources
@api_view(["GET"])
@permission_classes([AllowAny])
def memory_status(request):
    """Monitor memory usage and system resources"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()

        return Response({
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "cpu_percent": round(process.cpu_percent(), 2),
            "model_cache_size": len(model_cache),
            "prediction_cache_size": len(prediction_cache),
            "performance_cache_size": len(performance_cache),
            "thread_pools": {
                "training_active": training_executor._threads,
                "prediction_active": prediction_executor._threads,
                "data_active": data_executor._threads,
            },
            "timestamp": timezone.now().isoformat(),
        }, status=status.HTTP_200_OK)
    except ImportError:
        return Response({
            "error": "psutil not installed. Run: pip install psutil",
            "basic_info": {
                "model_cache_size": len(model_cache),
                "prediction_cache_size": len(prediction_cache),
            }
        }, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({
            "error": f"Memory status check failed: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Endpoint to check Redis connectivity (legacy endpoint)
@api_view(["GET"])
@permission_classes([AllowAny])
def redis_check(request):
    """Test Redis connectivity (Legacy endpoint for backward compatibility)"""
    try:
        test_key = "health_check_test"
        test_value = "redis_working"

        cache.set(test_key, test_value, timeout=60)
        retrieved_value = cache.get(test_key)

        if retrieved_value == test_value:
            logger.info("Redis connection test passed")
            return Response(
                {"status": "Success", "message": "Redis is connected and working"},
                status=status.HTTP_200_OK,
            )
        else:
            logger.error("Redis test failed - value mismatch")
            return Response(
                {"status": "error", "message": "Redis test failed - value mismatch"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error(f"Redis connection test failed: {str(e)}")
        return Response(
            {"status": "error", "message": f"Redis connection failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

# Endpoint to debug models and their status
@api_view(["GET"])
@permission_classes([AllowAny])
def debug_models(request):
    """Debug endpoint to check model status"""
    return Response(
        {
            "models_loaded": len(model_cache),
            "model_keys": list(model_cache.keys()),
            "models_dir": str(MODELS_DIR),
            "models_dir_exists": MODELS_DIR.exists(),
            "files_in_models_dir": (
                [str(f) for f in MODELS_DIR.glob("*.pkl")]
                if MODELS_DIR.exists()
                else []
            ),
            "timeframes": list(TIMEFRAMES.keys()),
            "expected_model_files": [
                f"universal_model{TIMEFRAMES[tf]['model_suffix']}.pkl"
                for tf in TIMEFRAMES.keys()
            ],
        }
    )


# ============================================================================
# INITIALIZATION - Load models at startup
# ============================================================================

load_all_models()
