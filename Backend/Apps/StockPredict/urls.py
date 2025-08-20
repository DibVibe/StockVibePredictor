from django.urls import path, include
from .views import (
    # Prediction Endpoints
    predict_multi_timeframe,
    batch_predictions,
    predict_stock_trend,
    prediction_status,

    # Model Management
    train_model,
    train_universal_models,
    list_models,
    delete_model,
    create_test_models,

    # Trading Endpoints
    simulate_trade,
    get_portfolio,
    get_trade_history,
    place_real_trade,

    # Watchlist Endpoints
    create_watchlist,
    get_watchlist_predictions,

    # Market Data Endpoints
    market_overview,
    analytics_dashboard,
    get_model_performance,
    get_chart_data,
    get_multi_chart_data,

    # System Monitoring
    system_health,
    memory_status,  # ADD THIS
    redis_check,
    debug_models,
)

# Keep your existing CompanyEssentials import
from .CompanyEssentials import company_essentials

app_name = "StockPredict"

urlpatterns = [
    # 📈 Prediction Endpoints (UPDATED PATHS)
    path("predict/multi-timeframe/", predict_multi_timeframe, name="predict_multi_timeframe"),
    path("predict/multi/", predict_multi_timeframe, name="predict_multi_alias"),  # Alias for frontend compatibility
    path("predict/batch/", batch_predictions, name="batch_predictions"),
    path("predict/trend/", predict_stock_trend, name="predict_stock_trend"),
    path("predict/status/", prediction_status, name="prediction_status"),

    # 🤖 Model Management Endpoints
    path("models/train/", train_model, name="train_model"),
    path("models/train-universal/", train_universal_models, name="train_universal_models"),
    path("models/list/", list_models, name="list_models"),
    path("models/delete/", delete_model, name="delete_model"),
    path("models/create-test/", create_test_models, name="create_test_models"),

    # 💼 Trading Endpoints (OK as is)
    path("trading/simulate/", simulate_trade, name="simulate_trade"),
    path("trading/portfolio/", get_portfolio, name="get_portfolio"),
    path("trading/history/", get_trade_history, name="get_trade_history"),
    path("trading/real/", place_real_trade, name="place_real_trade"),

    # 👁️ Watchlist Endpoints (OK as is)
    path("watchlist/create/", create_watchlist, name="create_watchlist"),
    path("watchlist/predictions/", get_watchlist_predictions, name="get_watchlist_predictions"),

    # 📊 Market Data Endpoints (REORGANIZED)
    path("market/overview/", market_overview, name="market_overview"),
    path("market/analytics/", analytics_dashboard, name="analytics_dashboard"),
    path("market/performance/", get_model_performance, name="get_model_performance"),
    path("market/chart/<str:ticker>/", get_chart_data, name="get_chart_data"),
    path("market/chart/multi/", get_multi_chart_data, name="get_multi_chart_data"),
    # Test endpoint for chart debugging
    path("chart/<str:ticker>/", get_chart_data, name="chart_simple"),

    # 🏢 Company Information (Keep your existing endpoint)
    path("company/<str:ticker>/essentials/", company_essentials, name="company_essentials"),

    # 🧪 System Monitoring Endpoints (REORGANIZED)
    path("system/health/", system_health, name="system_health"),
    path("system/memory/", memory_status, name="memory_status"),
    path("system/redis/", redis_check, name="redis_check"),
    path("system/debug/models/", debug_models, name="debug_models"),
]
