import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
import yfinance as yf
from asgiref.sync import sync_to_async

# Import the config from the config module
from ..Config.ChartConfig import TIMEFRAME_CONFIGS

# Initialize logger
logger = logging.getLogger(__name__)


class AsyncDataFetcher:
    """Async data fetching for improved performance"""

    @staticmethod
    async def fetch_multiple_timeframes(ticker: str, timeframes: List[str]) -> Dict:
        """Fetch data for multiple timeframes concurrently"""
        tasks = []
        for timeframe in timeframes:
            task = AsyncDataFetcher._fetch_single_timeframe(ticker, timeframe)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict = {}
        for timeframe, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {ticker} {timeframe}: {result}")
                data_dict[timeframe] = None
            else:
                data_dict[timeframe] = result

        return data_dict

    @staticmethod
    async def _fetch_single_timeframe(ticker: str, timeframe: str):
        """Fetch data for a single timeframe"""
        config = TIMEFRAME_CONFIGS.get(timeframe)
        if not config:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Use sync_to_async for yfinance calls
        fetch_func = sync_to_async(
            lambda: yf.download(
                ticker,
                period=config.period,
                interval=config.interval,
                progress=False,
                auto_adjust=True,
                prepost=True
            ),
            thread_sensitive=True
        )

        return await fetch_func()
