from celery import shared_task
from ..Services import DataService

@shared_task
def fetch_stock_data_async(ticker, timeframe):
    """Async task for fetching stock data"""
    service = DataService()
    return service.fetch_stock_data(ticker, timeframe)

@shared_task
def cleanup_old_predictions():
    """Clean up old prediction data"""
    pass
