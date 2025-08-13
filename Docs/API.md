# <p align="center">📡 StockVibePredictor API Documentation</p>

<p align="center">
<img src="https://img.shields.io/badge/API_Version-1.0.0-blue?style=for-the-badge" alt="API Version" />
<img src="https://img.shields.io/badge/Type-RESTful-green?style=for-the-badge" alt="REST" />
<img src="https://img.shields.io/badge/Framework-Django_REST-red?style=for-the-badge" alt="Django" />
</p>

<p align="center"><b>Complete API Reference for StockVibePredictor</b></p>

<p align="center"><i>Last Updated: August 2025</i></p>

---

<h2 align="center">📋 Table of Contents</h2>

<p align="center">
<a href="#-authentication">🔑 Authentication</a> •
<a href="#-base-url">🌐 Base URL</a> •
<a href="#-response-format">📊 Response Format</a> •
<a href="#️-error-codes">⚠️ Error Codes</a> •
<a href="#-endpoints">🎯 Endpoints</a>
</p>

<p align="center">
<a href="#-predictions">📈 Predictions</a> •
<a href="#-model-management">🤖 Model Management</a> •
<a href="#-trading">💼 Trading</a> •
<a href="#️-watchlist">👁️ Watchlist</a> •
<a href="#-market-data">📊 Market Data</a> •
<a href="#-system-monitoring">🧪 System Monitoring</a>
</p>

<p align="center">
<code>Authorization: Bearer &lt;your-token-here&gt;</code>
</p>

---

## 🔑 Authentication

Most endpoints are **publicly accessible** (`AllowAny`). Some endpoints require authentication:

```http
Authorization: Bearer <your-token-here>
```

</td></tr>
</table>

| Permission Level  | Description                |
| :---------------- | :------------------------- |
| `AllowAny`        | No authentication required |
| `IsAuthenticated` | Requires valid auth token  |

---

## 🌐 Base URL

```
Development: http://127.0.0.1:8000/api
Production: https://your-domain.com/api
```

---

## 📊 Response Format

All responses follow this structure:

### ✅ Success Response

```json
{
    "status": "success",
    "data": { ... },
    "timestamp": "2024-12-07T10:30:00Z"
}
```

### ❌ Error Response

```json
{
  "error": "Error message",
  "status": "error",
  "timestamp": "2024-12-07T10:30:00Z"
}
```

---

## ⚠️ Error Codes

| Status Code | Description           |
| :---------- | :-------------------- |
| `200`       | Success               |
| `201`       | Created               |
| `400`       | Bad Request           |
| `401`       | Unauthorized          |
| `403`       | Forbidden             |
| `404`       | Not Found             |
| `429`       | Too Many Requests     |
| `500`       | Internal Server Error |
| `501`       | Not Implemented       |

---

# 🎯 API Endpoints

## 📈 Predictions

### 1️⃣ **Single Stock Prediction** (Legacy)

```http
POST /api/predict/
```

**Description:** Get stock prediction for single timeframe (backward compatibility)

**Authentication:** Not required

**Request Body:**

```json
{
  "ticker": "AAPL"
}
```

**Response:**

```json
{
    "ticker": "AAPL",
    "prediction": {
        "direction": "UP",
        "confidence": 75.5,
        "price_target": 195.50,
        "current_price": 189.25,
        "expected_return": 3.30,
        "model_accuracy": 68.7
    },
    "history": [],
    "analysis": { ... }
}
```

---

### 2️⃣ **Multi-Timeframe Prediction**

```http
POST /api/predict/multi/
```

**Description:** Get predictions across multiple timeframes with comprehensive analysis

**Authentication:** Not required

**Rate Limit:** 100 requests/hour

**Request Body:**

```json
{
  "ticker": "AAPL",
  "timeframes": ["1d", "1w", "1mo", "1y"],
  "include_analysis": true
}
```

**Response:**

```json
{
    "ticker": "AAPL",
    "normalized_ticker": "AAPL",
    "timestamp": "2024-12-07T10:30:00Z",
    "predictions": {
        "1d": {
            "direction": "UP",
            "confidence": 72.5,
            "price_target": 195.50,
            "current_price": 189.25,
            "expected_return": 3.30,
            "model_accuracy": 68.7,
            "model_type": "universal"
        },
        "1w": { ... },
        "1mo": { ... },
        "1y": { ... }
    },
    "market_info": {
        "market_cap": 3000000000000,
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "beta": 1.24,
        "pe_ratio": 31.5
    },
    "analysis": {
        "technical": {
            "rsi": 58.3,
            "rsi_signal": "neutral",
            "trend": "bullish",
            "volume_trend": "high"
        },
        "price_levels": {
            "support": 185.20,
            "resistance": 195.80
        },
        "sentiment": {
            "sentiment_score": 0.65,
            "sentiment_label": "bullish"
        },
        "risk": {
            "volatility": 0.23,
            "risk_level": "medium"
        },
        "recommendation": {
            "overall": "BUY",
            "confidence": 70.5,
            "risk_level": "medium",
            "holding_period": "long"
        }
    }
}
```

---

### 3️⃣ **Batch Predictions**

```http
POST /api/predict/batch/
```

**Description:** Get predictions for multiple tickers at once

**Authentication:** Not required

**Request Body:**

```json
{
  "tickers": ["AAPL", "GOOGL", "TSLA", "MSFT"],
  "timeframe": "1d"
}
```

**Response:**

```json
{
    "timeframe": "1d",
    "results": {
        "AAPL": {
            "direction": "UP",
            "confidence": 72.5,
            "price_target": 195.50
        },
        "GOOGL": { ... },
        "TSLA": { ... },
        "MSFT": { ... }
    },
    "timestamp": "2024-12-07T10:30:00Z"
}
```

**Limitations:**

- Maximum 20 tickers per request
- Single timeframe only

---

## 🤖 Model Management

### 4️⃣ **List All Models**

```http
GET /api/models/list/
```

**Description:** Get list of all available models with their metrics

**Authentication:** Not required

**Response:**

```json
{
    "total_models": 8,
    "models": [
        {
            "key": "AAPL_1d",
            "type": "ticker_specific",
            "ticker": "AAPL",
            "timeframe": "1d",
            "accuracy": 0.687,
            "features_count": 29,
            "path": "/Models/AAPL_1d_model.pkl"
        },
        { ... }
    ],
    "summary": {
        "universal": 4,
        "ticker_specific": 4,
        "by_timeframe": {
            "1d": 2,
            "1w": 2,
            "1mo": 2,
            "1y": 2
        }
    }
}
```

---

### 5️⃣ **Train Model for Ticker**

```http
POST /api/models/train/
```

**Description:** Train a new model for specific ticker and timeframe

**Authentication:** Not required

**Request Body:**

```json
{
  "ticker": "NVDA",
  "timeframe": "1d",
  "model_type": "ensemble"
}
```

**Model Types:**

- `ensemble` (default)
- `randomforest`
- `gradient_boosting`

**Response:**

```json
{
  "success": true,
  "ticker": "NVDA",
  "timeframe": "1d",
  "metrics": {
    "accuracy": 0.723,
    "precision": 0.715,
    "recall": 0.708,
    "f1_score": 0.711,
    "cv_mean": 0.698,
    "cv_std": 0.032
  },
  "model_path": "/Models/NVDA_1d_model.pkl"
}
```

---

### 6️⃣ **Train Universal Models**

```http
POST /api/models/train-universal/
```

**Description:** Train universal models for all timeframes using multiple tickers

**Authentication:** Not required

**Request Body:**

```json
{
  "timeframes": ["1d", "1w", "1mo", "1y"],
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
}
```

**Response:**

```json
{
    "1d": {
        "success": true,
        "accuracy": 0.687,
        "samples": 5000,
        "path": "/Models/universal_model_1d.pkl"
    },
    "1w": { ... },
    "1mo": { ... },
    "1y": { ... }
}
```

---

### 7️⃣ **Delete Model**

```http
DELETE /api/models/delete/
```

**Description:** Delete a specific model

**Authentication:** Required ✅

**Request Body:**

```json
{
  "ticker": "AAPL",
  "timeframe": "1d"
}
```

**Response:**

```json
{
  "message": "Model AAPL_1d deleted successfully"
}
```

---

### 8️⃣ **Create Test Models**

```http
POST /api/models/create-test/
```

**Description:** Create dummy models for testing

**Authentication:** Not required

**Response:**

```json
{
  "status": "success",
  "message": "Dummy models created",
  "models_loaded": 4,
  "model_keys": [
    "universal_1d",
    "universal_1w",
    "universal_1mo",
    "universal_1y"
  ]
}
```

---

## 💼 Trading

### 9️⃣ **Simulate Trade**

```http
POST /api/trading/simulate/
```

**Description:** Execute paper trading simulation

**Authentication:** Required ✅

**Rate Limit:** 50 requests/hour

**Request Body:**

```json
{
  "ticker": "AAPL",
  "action": "buy",
  "quantity": 10,
  "order_type": "market",
  "limit_price": null
}
```

**Order Types:**

- `market` - Execute at current price
- `limit` - Execute at specified price or better

**Response:**

```json
{
  "status": "executed",
  "trade": {
    "trade_id": "SIM_1_AAPL_1701936600",
    "ticker": "AAPL",
    "action": "buy",
    "quantity": 10,
    "execution_price": 189.5,
    "total_value": 1895.0,
    "commission": 1.9,
    "total_cost": 1896.9,
    "timestamp": "2024-12-07T10:30:00Z"
  },
  "portfolio_update": {
    "quantity": 10,
    "avg_price": 189.5,
    "total_invested": 1896.9
  },
  "message": "Successfully bought 10 shares of AAPL at $189.50"
}
```

---

### 🔟 **Get Portfolio**

```http
GET /api/trading/portfolio/
```

**Description:** Get user's simulated portfolio with current values

**Authentication:** Required ✅

**Response:**

```json
{
    "portfolio": {
        "AAPL": {
            "quantity": 10,
            "avg_price": 189.50,
            "current_price": 195.25,
            "current_value": 1952.50,
            "invested_value": 1896.90,
            "pnl": 55.60,
            "pnl_percent": 2.93,
            "weight": 45.2
        },
        "GOOGL": { ... }
    },
    "summary": {
        "total_positions": 3,
        "total_current_value": 4320.50,
        "total_invested": 4150.00,
        "total_pnl": 170.50,
        "total_pnl_percent": 4.11
    },
    "last_updated": "2024-12-07T10:30:00Z"
}
```

---

### 1️⃣1️⃣ **Get Trade History**

```http
GET /api/trading/history/
```

**Description:** Get user's trade history with pagination

**Authentication:** Required ✅

**Query Parameters:**

- `page` (int): Page number (default: 1)
- `per_page` (int): Items per page (default: 20)

**Response:**

```json
{
  "trades": [
    {
      "trade_id": "SIM_1_AAPL_1701936600",
      "ticker": "AAPL",
      "action": "buy",
      "quantity": 10,
      "execution_price": 189.5,
      "timestamp": "2024-12-07T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total_pages": 5,
    "total_records": 95
  }
}
```

---

### 1️⃣2️⃣ **Place Real Trade** (Future)

```http
POST /api/trading/real/
```

**Description:** Place real trade through broker API

**Authentication:** Required ✅

**Status:** 🚧 Not Implemented

**Response:**

```json
{
  "message": "Real trading integration coming soon",
  "note": "This will integrate with brokers like Alpaca, Interactive Brokers, etc.",
  "required_setup": [
    "Broker API credentials",
    "User account verification",
    "Risk management rules",
    "Compliance checks"
  ]
}
```

---

## 👁️ Watchlist

### 1️⃣3️⃣ **Create Watchlist**

```http
POST /api/watchlist/create/
```

**Description:** Create or update user's watchlist

**Authentication:** Required ✅

**Request Body:**

```json
{
  "name": "Tech Giants",
  "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
}
```

**Response:**

```json
{
  "watchlist": {
    "name": "Tech Giants",
    "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "created_at": "2024-12-07T10:30:00Z",
    "updated_at": "2024-12-07T10:30:00Z"
  },
  "message": "Watchlist 'Tech Giants' created with 5 tickers"
}
```

---

### 1️⃣4️⃣ **Get Watchlist Predictions**

```http
GET /api/watchlist/predictions/
```

**Description:** Get predictions for all tickers in watchlist

**Authentication:** Required ✅

**Query Parameters:**

- `name` (string): Watchlist name (default: "Default")
- `timeframe` (string): Timeframe for predictions (default: "1d")

**Response:**

```json
{
    "watchlist_name": "Tech Giants",
    "timeframe": "1d",
    "predictions": {
        "AAPL": {
            "direction": "UP",
            "confidence": 72.5,
            "price_target": 195.50
        },
        "GOOGL": { ... }
    },
    "summary": {
        "total_tickers": 5,
        "bullish_count": 3,
        "bearish_count": 2,
        "avg_confidence": 68.4
    },
    "timestamp": "2024-12-07T10:30:00Z"
}
```

---

## 📊 Market Data

### 1️⃣5️⃣ **Market Overview**

```http
GET /api/market/overview/
```

**Description:** Get overall market overview with major indices

**Authentication:** Not required

**Response:**

```json
{
    "market_data": {
        "^GSPC": {
            "name": "S&P 500",
            "price": 4567.80,
            "change": 23.45,
            "change_percent": 0.52,
            "direction": "up"
        },
        "^DJI": { ... },
        "^IXIC": { ... },
        "^VIX": { ... },
        "^NSEI": { ... },
        "^BSESN": { ... }
    },
    "market_sentiment": "bullish",
    "trading_session": "open",
    "timestamp": "2024-12-07T10:30:00Z"
}
```

---

### 1️⃣6️⃣ **Analytics Dashboard**

```http
GET /api/market/analytics/
```

**Description:** Get system analytics and performance metrics

**Authentication:** Not required

**Response:**

```json
{
    "system_metrics": {
        "total_predictions_today": 342,
        "models_loaded": 8,
        "cache_hit_rate": 85.5,
        "avg_response_time": 1.2,
        "uptime": "99.9%"
    },
    "prediction_accuracy": {
        "1d": {
            "accuracy": 0.67,
            "total_predictions": 1250
        },
        "1w": { ... }
    },
    "popular_tickers": [
        {
            "ticker": "AAPL",
            "requests": 145
        },
        { ... }
    ],
    "trading_simulation": {
        "total_simulated_trades": 2340,
        "total_simulated_volume": 1250000,
        "avg_portfolio_performance": 8.5
    }
}
```

---

### 1️⃣7️⃣ **Model Performance**

```http
GET /api/system/models/performance/
```

**Description:** Get detailed model performance metrics

**Authentication:** Not required

**Query Parameters:**

- `timeframe` (string): Specific timeframe (default: "1d")

**Response:**

```json
{
  "timeframe": "1d",
  "model_type": "ensemble",
  "metrics": {
    "accuracy": 0.687,
    "precision": 0.692,
    "recall": 0.681,
    "f1_score": 0.686,
    "sharpe_ratio": 1.34,
    "max_drawdown": -0.08,
    "win_rate": 0.671
  },
  "backtesting": {
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "total_trades": 1247,
    "profitable_trades": 837,
    "average_return": 0.023,
    "volatility": 0.156
  },
  "feature_importance": {
    "RSI": 0.18,
    "MACD": 0.16,
    "Volume": 0.14,
    "MA20": 0.13
  }
}
```

---

## 🧪 System Monitoring

### 1️⃣8️⃣ **System Health Check**

```http
GET /api/system/health/
```

**Description:** Comprehensive system health check

**Authentication:** Not required

**Response:**

```json
{
  "timestamp": "2024-12-07T10:30:00Z",
  "status": "healthy",
  "services": {
    "cache": "healthy",
    "models": "healthy",
    "data_source": "healthy"
  },
  "metrics": {
    "model_cache_size": 8,
    "prediction_cache_size": 42,
    "available_timeframes": ["1d", "1w", "1mo", "1y"]
  }
}
```

**Status Values:**

- `healthy` - All systems operational
- `degraded` - Partial functionality
- `unhealthy` - Major issues

---

### 1️⃣9️⃣ **Redis Connection Check**

```http
GET /api/redis-check/
```

**Description:** Test Redis cache connectivity

**Authentication:** Not required

**Response:**

```json
{
  "status": "Success",
  "message": "Redis is connected and working"
}
```

---

### 2️⃣0️⃣ **Debug Models**

```http
GET /api/debug/models/
```

**Description:** Debug endpoint to check model status

**Authentication:** Not required

**Response:**

```json
{
  "models_loaded": 8,
  "model_keys": ["universal_1d", "universal_1w", "AAPL_1d"],
  "models_dir": "/Backend/Scripts/Models",
  "models_dir_exists": true,
  "files_in_models_dir": ["universal_model_1d.pkl", "universal_model_1w.pkl"],
  "timeframes": ["1d", "1w", "1mo", "1y"]
}
```

---

## 🎯 Rate Limiting

| Endpoint Category | Rate Limit   | Per  |
| :---------------- | :----------- | :--- |
| Predictions       | 100 requests | Hour |
| Trading           | 50 requests  | Hour |
| Others            | Unlimited    | -    |

---

## 🔍 Example Requests

### cURL Example

```bash
curl -X POST http://127.0.0.1:8000/api/predict/multi/ \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "timeframes": ["1d", "1w"],
    "include_analysis": true
  }'
```

### Python Example

```python
import requests

url = "http://127.0.0.1:8000/api/predict/multi/"
payload = {
    "ticker": "AAPL",
    "timeframes": ["1d", "1w"],
    "include_analysis": True
}

response = requests.post(url, json=payload)
data = response.json()
print(data)
```

### JavaScript Example

```javascript
fetch("http://127.0.0.1:8000/api/predict/multi/", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    ticker: "AAPL",
    timeframes: ["1d", "1w"],
    include_analysis: true,
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

---

## 📝 Ticker Format

### Supported Ticker Formats

| Market        | Format       | Example          |
| :------------ | :----------- | :--------------- |
| US Stocks     | `TICKER`     | `AAPL`, `GOOGL`  |
| Indian Stocks | `TICKER.NS`  | `RELIANCE.NS`    |
| Indices       | `^SYMBOL`    | `^NSEI`, `^GSPC` |
| Crypto        | `SYMBOL-USD` | `BTC-USD`        |

### Ticker Aliases

| Alias      | Actual Ticker |
| :--------- | :------------ |
| `NIFTY`    | `^NSEI`       |
| `SENSEX`   | `^BSESN`      |
| `GOOGLE`   | `GOOGL`       |
| `FACEBOOK` | `META`        |

---

## 🚨 Error Handling

### Common Error Responses

#### Invalid Ticker

```json
{
  "error": "Invalid ticker format",
  "status": "error"
}
```

#### No Data Available

```json
{
  "error": "No data available for TICKER",
  "status": "error"
}
```

#### Rate Limit Exceeded

```json
{
  "error": "Rate limit exceeded. Please try again later.",
  "status": "error",
  "retry_after": 3600
}
```

#### Model Not Found

```json
{
  "error": "No model available for TICKER",
  "status": "error"
}
```

---

## 📊 Timeframe Options

| Timeframe | Description | Historical Data | Prediction Horizon |
| :-------- | :---------- | :-------------- | :----------------- |
| `1d`      | Daily       | 3 months        | Next trading day   |
| `1w`      | Weekly      | 1 year          | Next week          |
| `1mo`     | Monthly     | 2 years         | Next month         |
| `1y`      | Yearly      | 10 years        | Next year          |

---

## 🔐 Security Headers

Recommended headers for production:

```http
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

---

## 📞 Support & Contact

- **GitHub Issues:** [github.com/ThisIsDibakar/StockVibePredictor/issues](https://github.com/ThisIsDibakar/StockVibePredictor/issues)
- **Email:** support@stockvibepredictor.com
- **Documentation:** [docs.stockvibepredictor.com](https://docs.stockvibepredictor.com)

---

<div align="center">

**© 2025 StockVibePredictor API Documentation**

_Built with ❤️ by Dibakar_

</div>
