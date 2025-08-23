from django.core.cache import cache
import hashlib
import json
from typing import Any, Optional

class CacheManager:
    """Centralized cache management"""

    @staticmethod
    def generate_key(*args) -> str:
        """Generate a consistent cache key"""
        key_data = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_data.encode()).hexdigest()

    @staticmethod
    def get_chart_data(ticker: str, timeframe: str, chart_type: str) -> Optional[dict]:
        """Get cached chart data"""
        key = f"chart_v2_{CacheManager.generate_key(ticker, timeframe, chart_type)}"
        return cache.get(key)

    @staticmethod
    def set_chart_data(
        ticker: str,
        timeframe: str,
        chart_type: str,
        data: dict,
        timeout: int
    ):
        """Cache chart data with versioning"""
        key = f"chart_v2_{CacheManager.generate_key(ticker, timeframe, chart_type)}"
        cache.set(key, data, timeout=timeout)
