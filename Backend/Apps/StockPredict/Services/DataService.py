import pandas as pd
from typing import Optional
import logging
from django.utils import timezone
import yfinance as yf

logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Service for processing and filtering stock data"""

    @staticmethod
    def apply_timeframe_filter(
        data: pd.DataFrame,
        timeframe: str,
        config
    ) -> pd.DataFrame:
        """Apply timeframe-specific filtering logic"""

        if data.empty:
            return data

        logger.info(f"Processing {len(data)} rows for timeframe {timeframe}")

        # First, handle data freshness for short timeframes
        processed_data = StockDataProcessor._ensure_data_freshness(data, timeframe)

        # Then apply business days filtering
        if config.business_days_only:
            processed_data = StockDataProcessor._filter_business_days(processed_data, timeframe)

        # Finally apply display limit
        if config.display_limit and len(processed_data) > config.display_limit:
            processed_data = StockDataProcessor._apply_display_limit(processed_data, timeframe, config.display_limit)

        logger.info(f"After processing: {len(processed_data)} rows, Date range: {processed_data.index[0] if not processed_data.empty else 'N/A'} to {processed_data.index[-1] if not processed_data.empty else 'N/A'}")

        return processed_data

    @staticmethod
    def _ensure_data_freshness(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Ensure data is fresh for short timeframes"""
        if data.empty:
            return data

        if timeframe in ["1d", "5d", "1w"]:
            latest_data_time = data.index[-1]
            current_time = timezone.now()

            # Check if data is from today for intraday timeframes
            if timeframe == "1d" and latest_data_time.date() < current_time.date():
                logger.info(f"Data seems outdated for {timeframe}, latest: {latest_data_time.date()}, current: {current_time.date()}")

        return data

    @staticmethod
    def _filter_business_days(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Filter for business days only"""
        if data.empty:
            return data

        # Filter for business days (Monday=0 to Friday=4)
        business_days = data[data.index.weekday < 5]

        if timeframe == "1d" and not business_days.empty:
            # Get latest business day only
            latest_date = business_days.index[-1].date()
            filtered_data = data[data.index.date == latest_date]
            logger.info(f"Filtered to latest business day {latest_date}: {len(filtered_data)} rows")
            return filtered_data

        elif timeframe == "5d" and not business_days.empty:
            # Get last 5 unique business days
            unique_business_days = business_days.index.normalize().unique()
            if len(unique_business_days) >= 5:
                last_5_days = unique_business_days[-5:]
            else:
                last_5_days = unique_business_days
            filtered_data = data[data.index.normalize().isin(last_5_days)]
            logger.info(f"Filtered to last {len(last_5_days)} business days: {len(filtered_data)} rows")
            return filtered_data

        return business_days if not business_days.empty else data

    @staticmethod
    def _apply_display_limit(
        data: pd.DataFrame,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Apply display limit based on timeframe"""

        if data.empty or len(data) <= limit:
            return data

        logger.info(f"Applying display limit of {limit} to {len(data)} rows for {timeframe}")

        # Smart limiting based on timeframe
        if timeframe in ["1d"]:
            # For intraday, take the most recent data points
            return data.tail(limit)

        elif timeframe in ["5d", "1w"]:
            # For short timeframes, take recent data
            return data.tail(limit)

        elif timeframe in ["1mo", "3mo", "6mo"]:
            # For medium timeframes, ensure even distribution if we have too much data
            if len(data) > limit * 2:
                # Use step sampling to get even distribution
                step = max(1, len(data) // limit)
                indices = range(0, len(data), step)[:limit]
                # Ensure we always include the last data point
                if indices[-1] != len(data) - 1:
                    indices = list(indices[:-1]) + [len(data) - 1]
                return data.iloc[indices]
            else:
                return data.tail(limit)

        else:
            # For long timeframes (1y, 2y, 5y), use tail
            return data.tail(limit)

    @staticmethod
    def validate_data_quality(data: pd.DataFrame, config) -> tuple[bool, str]:
        """Validate data quality and return status with message"""
        if data.empty:
            return False, "No data available"

        if len(data) < config.min_data_points:
            return False, f"Insufficient data points: {len(data)} < {config.min_data_points}"

        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"

        # Check for data freshness
        latest_timestamp = data.index[-1]
        current_time = timezone.now()

        if config.validate_data_freshness(latest_timestamp, current_time):
            return True, "Data quality is good"
        else:
            return False, f"Data is too old: latest {latest_timestamp}, current {current_time}"
