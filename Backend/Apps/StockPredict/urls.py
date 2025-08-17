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
    get_chart_data,
    get_multi_chart_data,
)
from .CompanyEssentials import company_essentials

app_name = "StockPredict"

urlpatterns = [
    # Prediction Endpoints
    path("predict/multi/", predict_multi_timeframe, name="predict_multi_timeframe"),
    path("predict/batch/", batch_predictions, name="batch_predictions"),
    path("predict/", predict_stock_trend, name="predict_stock_trend"),

    # Model Management Endpoints
    path("models/train/", train_model, name="train_model"),
    path(
        "models/train-universal/", train_universal_models, name="train_universal_models"
    ),
    path("models/list/", list_models, name="list_models"),
    path("models/delete/", delete_model, name="delete_model"),

    # Trading Endpoints
    path("trading/simulate/", simulate_trade, name="simulate_trade"),
    path("trading/portfolio/", get_portfolio, name="get_portfolio"),
    path("trading/history/", get_trade_history, name="get_trade_history"),
    path("trading/real/", place_real_trade, name="place_real_trade"),

    # Watchlist Endpoints
    path("watchlist/create/", create_watchlist, name="create_watchlist"),
    path(
        "watchlist/predictions/",
        get_watchlist_predictions,
        name="get_watchlist_predictions",
    ),

    # Market Data Endpoints
    path("market/overview/", market_overview, name="market_overview"),
    path("market/analytics/", analytics_dashboard, name="analytics_dashboard"),

    # Company information Endpoints
    path(
        "company/<str:ticker>/essentials/",
        company_essentials,
        name="company_essentials"
    ),

    # Chart Data Endpoints
    path("chart/<str:ticker>/", get_chart_data, name="get_chart_data"),
    path("chart/multi/", get_multi_chart_data, name="get_multi_chart_data"),

    # System Monitoring Endpoints
    path("system/health/", system_health, name="system_health"),
    path(
        "system/models/performance/",
        get_model_performance,
        name="get_model_performance",
    ),

    # Debug Endpoints
    path("debug/models/", debug_models, name="debug_models"),
    path("redis-check/", redis_check, name="redis_check"),
]
