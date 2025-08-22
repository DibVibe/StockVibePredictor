# API Migration Notes

## Version 1.0.1 - Alias Endpoints Added

### New Endpoints (Aliases)

1. **`/api/predict/multi/`** - Alias for `/api/predict/multi-timeframe/`
2. **`/api/chart/{ticker}/`** - Alias for `/api/market/chart/{ticker}/`

### Why Added?

- Frontend team requested simpler URLs
- Better compatibility with React Router
- Easier debugging in development

### Migration Guide

#### No breaking changes. Both old and new URLs work:

```javascript
// Both work identically:
fetch('/api/predict/multi-timeframe/', {...})  // Original
fetch('/api/predict/multi/', {...})            // New alias

// Both work identically:
fetch('/api/market/chart/AAPL/')  // Original
fetch('/api/chart/AAPL/')         // New alias
```

### Frontend Updates Required :<br><small>None — aliases are optional. Update at your convenience.</small>

---
