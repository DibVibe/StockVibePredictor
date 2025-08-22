from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
from django.views.decorators.cache import cache_page

@cache_page(60 * 5)
def api_root(request):
    """Enhanced API root endpoint with real-time status"""
    return JsonResponse({
        "name": "StockVibePredictor API",
        "version": "2.1.0",
        "status": "operational",
        "api_documentation": "https://your-docs-url.com",
        "changelog": {
            "2.1.0": "Added simplified alias endpoints for frontend compatibility",
            "2.0.0": "Added API versioning and enhanced monitoring"
        },
        "rate_limits": {
            "predictions": "100/hour",
            "trading": "50/hour",
            "anonymous": "20/hour"
        },
        "endpoints": {
            "predictions": {
                "multi_timeframe": "/api/v1/predict/multi-timeframe/",
                "multi_timeframe_alias": "/api/v1/predict/multi/",
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
            "watchlist": {  # 🆕 Added missing watchlist section
                "create": "/api/v1/watchlist/create/",
                "predictions": "/api/v1/watchlist/predictions/"
            },
            "market": {
                "overview": "/api/v1/market/overview/",
                "analytics": "/api/v1/market/analytics/",
                "performance": "/api/v1/market/performance/",
                "chart": "/api/v1/market/chart/{ticker}/",
                "chart_simple": "/api/v1/chart/{ticker}/",  # 🆕 Simple chart alias
                "multi_chart": "/api/v1/market/chart/multi/"
            },
            "company": {  # 🆕 Added missing company section
                "essentials": "/api/v1/company/{ticker}/essentials/"
            },
            "system": {
                "health": "/api/v1/system/health/",
                "memory": "/api/v1/system/memory/",
                "redis": "/api/v1/system/redis/",
                "debug_models": "/api/v1/system/debug/models/"
            }
        },
        "aliases": {  # 🆕 New section documenting all aliases
            "description": "Simplified endpoints for easier integration",
            "mappings": {
                "/api/v1/predict/multi/": {
                    "alias_for": "/api/v1/predict/multi-timeframe/",
                    "purpose": "Shorter URL for frontend compatibility"
                },
                "/api/v1/chart/{ticker}/": {
                    "alias_for": "/api/v1/market/chart/{ticker}/",
                    "purpose": "Quick chart access for debugging"
                },
                "/api/predict/multi/": {
                    "alias_for": "/api/predict/multi-timeframe/",
                    "purpose": "Non-versioned alias for backward compatibility"
                },
                "/api/chart/{ticker}/": {
                    "alias_for": "/api/market/chart/{ticker}/",
                    "purpose": "Non-versioned quick chart access"
                }
            }
        },
        "websocket_endpoints": {
            "real_time_predictions": "ws://your-domain/ws/predictions/",
            "market_data_stream": "ws://your-domain/ws/market/"
        },
        "quick_start": {  # 🆕 Added quick start section
            "test_health": "/api/v1/system/health/",
            "simple_prediction": "/api/v1/predict/multi/",
            "simple_chart": "/api/v1/chart/AAPL/"
        },
        "supported_versions": {  # 🆕 Version support info
            "current": "v1",
            "deprecated": [],
            "sunset_dates": {},
            "notes": "Both /api/ and /api/v1/ paths are supported"
        }
    })

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("Apps.StockPredict.urls")),  # Versioned API
    path("api/", include("Apps.StockPredict.urls")),      # Non-versioned (backward compatibility)
    path("", api_root, name="api_root"),
]
