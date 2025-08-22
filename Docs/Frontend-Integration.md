# Frontend Integration Guide

## Quick Access Endpoints

For easier frontend integration, we provide simplified alias endpoints:

### 1. Simple Chart Endpoint

**Original:** `/api/market/chart/{ticker}/` <br />
**Alias:** `/api/chart/{ticker}/` <br />

```javascript
// React/Frontend Example
const getChartData = async (ticker) => {
  const response = await fetch(`${API_URL}/chart/${ticker}/`);
  return response.json();
};

// Usage
getChartData("AAPL").then((data) => {
  console.log("Chart data:", data);
});
```

### 2. Multi-Prediction Endpoint

**Original:** `⁠/api/predict/multi-timeframe/` <br />
**Alias:** `⁠/api/predict/multi/` <br />

```javascript
// React/Frontend Example
const getPredictions = async (ticker, timeframes) => {
  const response = await fetch(`${API_URL}/predict/multi/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ticker,
      timeframes,
      include_analysis: true,
    }),
  });
  return response.json();
};

// Usage
getPredictions("AAPL", ["1d", "1w"]).then((predictions) => {
  console.log("Predictions:", predictions);
});
```

## Why Use Aliases?

1. Cleaner URLs - Easier to remember and type
2. Backward Compatibility - Support old frontend code
3. Debugging - Simpler URLs for testing
4. Frontend Friendly - No special characters like hyphens

## Complete URL Mapping :

| Purpose     | Full URL                        | Alias URL              |
| :---------- | :------------------------------ | :--------------------- |
| Predictions | `/api/predict/multi-timeframe/` | `/api/predict/multi/`  |
| Chart Data  | `/api/market/chart/{ticker}/`   | `/api/chart/{ticker}/` |
