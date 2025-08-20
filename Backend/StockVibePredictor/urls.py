from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

def api_root(request):
    """API root endpoint with comprehensive information"""
    return JsonResponse({
        "name": "StockVibePredictor API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predictions": {
                "multi_timeframe": "/api/predict/multi-timeframe/",
                "batch": "/api/predict/batch/",
                "trend": "/api/predict/trend/",
                "status": "/api/predict/status/"
            },
            "models": {
                "train": "/api/models/train/",
                "train_universal": "/api/models/train-universal/",
                "list": "/api/models/list/",
                "delete": "/api/models/delete/",
                "create_test": "/api/models/create-test/"
            },
            "trading": {
                "simulate": "/api/trading/simulate/",
                "portfolio": "/api/trading/portfolio/",
                "history": "/api/trading/history/",
                "real": "/api/trading/real/"
            },
            "watchlist": {
                "create": "/api/watchlist/create/",
                "predictions": "/api/watchlist/predictions/"
            },
            "market": {
                "overview": "/api/market/overview/",
                "analytics": "/api/market/analytics/",
                "performance": "/api/market/performance/",
                "chart": "/api/market/chart/{ticker}/",
                "multi_chart": "/api/market/chart/multi/"
            },
            "company": {
                "essentials": "/api/company/{ticker}/essentials/"
            },
            "system": {
                "health": "/api/system/health/",
                "memory": "/api/system/memory/",
                "redis": "/api/system/redis/",
                "debug_models": "/api/system/debug/models/"
            }
        },
        "documentation": "Use Postman collection for detailed API documentation",
        "authentication": "Bearer token required for protected endpoints"
    })

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("Apps.StockPredict.urls")),
    path("", api_root, name="api_root"),
]
