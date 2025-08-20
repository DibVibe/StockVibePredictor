from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.views.decorators.cache import cache_page

@cache_page(60 * 5)  #
def api_root(request):
    """Enhanced API root endpoint with real-time status"""
    return JsonResponse({
        "name": "StockVibePredictor API",
        "version": "2.0.0",
        "status": "operational",
        "api_documentation": "https://your-docs-url.com",
        "rate_limits": {
            "predictions": "100/hour",
            "trading": "50/hour",
            "anonymous": "20/hour"
        },
        "endpoints": {
            "predictions": {
                "multi_timeframe": "/api/v1/predict/multi-timeframe/",
                "batch": "/api/v1/predict/batch/",
                "trend": "/api/v1/predict/trend/",
                "status": "/api/v1/predict/status/"
            },
            "models": {
                "train": "/api/v1/models/train/",
                "train_universal": "/api/v1/models/train-universal/",
                "list": "/api/v1/models/list/",
                "delete": "/api/v1/models/delete/",
                "create_test": "/api/v1/models/create-test/"
            },
            "trading": {
                "simulate": "/api/v1/trading/simulate/",
                "portfolio": "/api/v1/trading/portfolio/",
                "history": "/api/v1/trading/history/",
                "real": "/api/v1/trading/real/"
            },
            "market": {
                "overview": "/api/v1/market/overview/",
                "analytics": "/api/v1/market/analytics/",
                "performance": "/api/v1/market/performance/",
                "chart": "/api/v1/market/chart/{ticker}/",
                "multi_chart": "/api/v1/market/chart/multi/"
            },
            "system": {
                "health": "/api/v1/system/health/",
                "memory": "/api/v1/system/memory/",
                "metrics": "/api/v1/system/metrics/"
            }
        },
        "websocket_endpoints": {
            "real_time_predictions": "ws://your-domain/ws/predictions/",
            "market_data_stream": "ws://your-domain/ws/market/"
        }
    })

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("Apps.StockPredict.urls")),
    path("api/", include("Apps.StockPredict.urls")),
    path("", api_root, name="api_root"),
]
