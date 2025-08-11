from django.urls import path, include
from .views import (
    predict_multi_timeframe,
    batch_predictions,
    predict_stock_trend,
    simulate_trade,
    get_portfolio,
    get_trade_history,
    place_real_trade,
    create_watchlist,
    get_watchlist_predictions,
    market_overview,
    analytics_dashboard,
    get_model_performance,
    system_health,
    redis_check,
    debug_models,
    train_model,
    train_universal_models,
    list_models,
    delete_model,
)
from .company_essentials import company_essentials

app_name = "StockPredict"

urlpatterns = [
    path("predict/multi/", predict_multi_timeframe, name="predict_multi_timeframe"),
    path("predict/batch/", batch_predictions, name="batch_predictions"),
    path("predict/", predict_stock_trend, name="predict_stock_trend"),
    path("company-essentials/<str:ticker>/", company_essentials, name="company_essentials"),
    path("models/train/", train_model, name="train_model"),
    path(
        "models/train-universal/", train_universal_models, name="train_universal_models"
    ),
    path("models/list/", list_models, name="list_models"),
    path("models/delete/", delete_model, name="delete_model"),
    path("trading/simulate/", simulate_trade, name="simulate_trade"),
    path("trading/portfolio/", get_portfolio, name="get_portfolio"),
    path("trading/history/", get_trade_history, name="get_trade_history"),
    path("trading/real/", place_real_trade, name="place_real_trade"),
    path("watchlist/create/", create_watchlist, name="create_watchlist"),
    path(
        "watchlist/predictions/",
        get_watchlist_predictions,
        name="get_watchlist_predictions",
    ),
    path("market/overview/", market_overview, name="market_overview"),
    path("market/analytics/", analytics_dashboard, name="analytics_dashboard"),
    path("system/health/", system_health, name="system_health"),
    path(
        "system/models/performance/",
        get_model_performance,
        name="get_model_performance",
    ),
    path("debug/models/", debug_models, name="debug_models"),
    path("redis-check/", redis_check, name="redis_check"),
]
