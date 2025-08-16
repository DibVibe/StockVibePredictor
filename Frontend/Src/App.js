import React, {
  useState,
  useEffect,
  useCallback,
  useMemo,
  Suspense,
  lazy,
} from "react";
import axios from "axios";
import "./App.css";

// Regular Components
import StockInput from "./Components/Javascript/StockInput";
import LoadingSpinner from "./Components/Javascript/LoadingSpinner";
import SystemStatus from "./Components/Javascript/SystemStatus";
import TimeframeSelector from "./Components/Javascript/TimeframeSelector";
import ErrorMessage from "./Components/Javascript/ErrorMessage";
import TabNavigation from "./Components/Javascript/TabNavigation";

// Lazy load heavy components
const StockChart = lazy(() => import("./Components/Javascript/StockChart"));
const CompanyEssentials = lazy(() =>
  import("./Components/Javascript/CompanyEssentials")
);
const PredictionsTab = lazy(() =>
  import("./Components/Javascript/PredictionTab")
);
const AnalysisTab = lazy(() => import("./Components/Javascript/AnalysisTab"));
const SentimentAnalysis = lazy(() =>
  import("./Components/Javascript/SentimentAnalysis")
);

// Utility: Debounce function
const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Chart Loading Skeleton Component
const ChartSkeleton = () => (
  <div className="chart-skeleton">
    <div className="skeleton-header"></div>
    <div className="skeleton-chart">
      <div className="skeleton-bar"></div>
      <div className="skeleton-bar"></div>
      <div className="skeleton-bar"></div>
      <div className="skeleton-bar"></div>
      <div className="skeleton-bar"></div>
    </div>
    <div className="skeleton-footer"></div>
  </div>
);

// Component wrapper with Suspense
const SuspenseWrapper = ({ children, fallback }) => (
  <Suspense fallback={fallback || <LoadingSpinner />}>{children}</Suspense>
);

function App() {
  // ==================== STATE MANAGEMENT ====================
  const [stockData, setStockData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [marketInfo, setMarketInfo] = useState(null);
  const [companyData, setCompanyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartLoading, setChartLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTicker, setCurrentTicker] = useState("");
  const [selectedTimeframes, setSelectedTimeframes] = useState([
    "1d",
    "1w",
    "1mo",
  ]);
  const [chartTimeframe, setChartTimeframe] = useState("1mo");
  const [chartType, setChartType] = useState("candlestick");
  const [selectedIndicators, setSelectedIndicators] = useState(["sma20"]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [activeTab, setActiveTab] = useState("predictions");

  // Cache for chart data
  const [chartCache, setChartCache] = useState({});
  const [requestQueue, setRequestQueue] = useState([]);

  // ==================== CONFIGURATION ====================
  const API_BASE_URL = "http://localhost:8000/api";

  const TIMEFRAMES = useMemo(
    () => ({
      "1d": "1 Day",
      "1w": "1 Week",
      "1mo": "1 Month",
      "1y": "1 Year",
    }),
    []
  );

  const CHART_TIMEFRAMES = useMemo(
    () => ({
      "1d": "1 Day",
      "5d": "5 Days",
      "1mo": "1 Month",
      "3mo": "3 Months",
      "6mo": "6 Months",
      "1y": "1 Year",
      "2y": "2 Years",
      "5y": "5 Years",
    }),
    []
  );

  const CHART_TYPES = useMemo(
    () => ({
      candlestick: "Candlestick",
      line: "Line",
      ohlc: "OHLC",
    }),
    []
  );

  const AVAILABLE_INDICATORS = useMemo(
    () => ({
      sma20: "SMA 20",
      sma50: "SMA 50",
      rsi: "RSI",
      macd: "MACD",
      bollinger: "Bollinger Bands",
      volume: "Volume",
    }),
    []
  );

  // ==================== AXIOS CONFIGURATION ====================
  useEffect(() => {
    // Set default axios config for better performance
    axios.defaults.timeout = 15000; // 15 second timeout (increased)
    axios.defaults.headers.common["Content-Type"] = "application/json";

    // Add request interceptor for caching headers
    const requestInterceptor = axios.interceptors.request.use(
      (config) => {
        config.headers["Cache-Control"] = "max-age=300";
        return config;
      },
      (error) => Promise.reject(error)
    );

    return () => {
      axios.interceptors.request.eject(requestInterceptor);
    };
  }, []);

  // ==================== SYSTEM HEALTH CHECK ====================
  useEffect(() => {
    checkSystemHealth();
    const healthInterval = setInterval(checkSystemHealth, 60000);
    return () => clearInterval(healthInterval);
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/system/health/`);
      setSystemHealth(response.data);

      if (response.data.status !== "healthy") {
        console.warn("System health issues detected:", response.data);
      }
    } catch (err) {
      console.error("System health check failed:", err);
      setSystemHealth({ status: "unhealthy", error: "Cannot connect to API" });
    }
  };

  // ==================== COMPANY DATA HANDLER ====================
  const handleCompanyDataReceived = useCallback((data) => {
    setCompanyData(data);

    if (data && data.company_info) {
      setMarketInfo((prevMarketInfo) => ({
        ...prevMarketInfo,
        sector: data.company_info.sector,
        industry: data.company_info.industry,
        market_cap: data.essentials?.market_cap?.value,
        pe_ratio: data.essentials?.pe_ratio?.value,
        dividend_yield: data.essentials?.dividend_yield?.value,
        exchange: data.company_info.exchange,
        week_52_high: data.price_summary?.week_52_high,
        week_52_low: data.price_summary?.week_52_low,
        current_price: data.current_price,
      }));
    }
  }, []);

  // ==================== OPTIMIZED CHART DATA FETCHER ====================
  const fetchChartData = useCallback(
    async (
      ticker,
      timeframe = chartTimeframe,
      type = chartType,
      indicators = selectedIndicators
    ) => {
      // Create cache key
      const cacheKey = `${ticker}_${timeframe}_${type}_${indicators.join(",")}`;

      // Check cache first
      if (chartCache[cacheKey]) {
        console.log("Using cached chart data for:", cacheKey);
        setChartData(chartCache[cacheKey]);
        return chartCache[cacheKey];
      }

      // Check if request is already in queue
      if (requestQueue.includes(cacheKey)) {
        console.log("Request already in progress for:", cacheKey);
        return;
      }

      setChartLoading(true);
      setRequestQueue((prev) => [...prev, cacheKey]);

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 12000); // 12 second timeout (increased)

        const response = await axios.get(`${API_BASE_URL}/chart/${ticker}/`, {
          params: {
            timeframe: timeframe,
            chart_type: type,
            indicators: indicators.join(","),
            optimize: true,
            points_limit: 100,
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        // Store in cache with timestamp
        const cacheData = {
          ...response.data,
          _timestamp: Date.now(),
        };

        setChartCache((prev) => ({
          ...prev,
          [cacheKey]: cacheData,
        }));

        setChartData(response.data);

        console.log("Chart data fetched successfully:", {
          ticker: response.data.ticker,
          dataPoints: response.data.data?.length || 0,
          indicators: Object.keys(response.data.indicators || {}),
        });

        return response.data;
      } catch (err) {
        if (err.name === "AbortError" || err.code === "ECONNABORTED") {
          console.error("Chart request timeout");
          // Don't set error for chart timeout - it's not critical
        } else {
          console.warn("Chart data fetch failed:", err);
        }
        setChartData(null);
        return null;
      } finally {
        setChartLoading(false);
        setRequestQueue((prev) => prev.filter((key) => key !== cacheKey));
      }
    },
    [chartCache, requestQueue, chartTimeframe, chartType, selectedIndicators]
  );

  // Debounced version for indicator changes
  const debouncedFetchChart = useMemo(
    () => debounce(fetchChartData, 500),
    [fetchChartData]
  );

  // ==================== DEBUGGING STOCK DATA FETCHER ====================
  const fetchStockData = async (ticker) => {
    if (!ticker.trim()) {
      setError("Please enter a stock ticker symbol! 📈");
      resetData();
      return;
    }

    setLoading(true);
    setError(null);
    setCurrentTicker(ticker.toUpperCase());

    console.log("=== DEBUG INFO ===");
    console.log("API Base URL:", API_BASE_URL);
    console.log("Ticker:", ticker.toUpperCase());
    console.log("Selected timeframes:", selectedTimeframes);

    try {
      const requestUrl = `${API_BASE_URL}/predict/multi/`;
      const requestPayload = {
        ticker: ticker.toUpperCase(),
        timeframes: selectedTimeframes,
        include_analysis: true,
      };

      console.log("Request URL:", requestUrl);
      console.log("Request payload:", requestPayload);

      // Test basic connectivity first
      console.log("Testing basic connectivity...");

      const predictionsResponse = await axios.post(requestUrl, requestPayload, {
        timeout: 30000, // 30 seconds for debugging
      });

      console.log("Response received:", predictionsResponse.status);
      console.log("Response data:", predictionsResponse.data);

      const data = predictionsResponse.data;
      setPredictions(data.predictions);
      setAnalysis(data.analysis || null);
      setMarketInfo(data.market_info || null);
      setStockData(data.history || null);

      console.log("Data processed successfully");
      setActiveTab("predictions");

      // Chart data (separate - non-critical)
      fetchChartData(
        ticker.toUpperCase(),
        chartTimeframe,
        chartType,
        selectedIndicators.slice(0, 1)
      ).catch((chartErr) => {
        console.warn("Chart data failed (non-critical):", chartErr);
      });
    } catch (err) {
      console.log("=== ERROR DETAILS ===");
      console.log("Error object:", err);
      console.log("Error message:", err.message);
      console.log("Error code:", err.code);
      console.log("Response status:", err.response?.status);
      console.log("Response data:", err.response?.data);
      console.log("Response headers:", err.response?.headers);

      handleFetchError(err, ticker);
    } finally {
      setLoading(false);
    }
  };

  // ==================== ENHANCED ERROR HANDLER ====================
  const handleFetchError = (err, ticker) => {
    console.log("=== HANDLING ERROR ===");
    console.log("Error type:", typeof err);
    console.log("Error details:", err);

    let errorMessage = "Unable to fetch stock data. Please try again.";
    let debugInfo = "";

    if (err.code === "ECONNREFUSED") {
      errorMessage = "❌ Cannot connect to backend server";
      debugInfo = `Make sure Django server is running on ${API_BASE_URL}`;
    } else if (err.code === "ENOTFOUND") {
      errorMessage = "❌ Network error - cannot resolve hostname";
      debugInfo = "Check your internet connection and API URL";
    } else if (err.code === "ECONNABORTED" || err.name === "AbortError") {
      errorMessage = "❌ Request timeout (30+ seconds)";
      debugInfo = "Server is taking too long to respond";
    } else if (err.response?.status === 404) {
      errorMessage = `❌ Stock ticker "${ticker}" not found`;
      debugInfo = "Please check the ticker symbol";
    } else if (err.response?.status === 400) {
      errorMessage = err.response.data?.error || "❌ Invalid request format";
      debugInfo = "Check the request parameters";
    } else if (err.response?.status >= 500) {
      errorMessage = "❌ Server error";
      debugInfo = err.response.data?.error || "Backend server issue";
    } else if (err.response?.data?.error) {
      errorMessage = `❌ ${err.response.data.error}`;
      debugInfo = "Server returned an error";
    } else {
      errorMessage = `❌ Unknown error: ${err.message}`;
      debugInfo = `Error code: ${err.code || "UNKNOWN"}`;
    }

    console.log("Final error message:", errorMessage);
    console.log("Debug info:", debugInfo);

    setError(`${errorMessage}\n\n🔧 Debug: ${debugInfo}`);
    resetData();
  };

  // ==================== RESET DATA ====================
  const resetData = () => {
    setStockData(null);
    setChartData(null);
    setPredictions(null);
    setAnalysis(null);
    setMarketInfo(null);
    setCompanyData(null);
    // Keep cache for performance
  };

  // ==================== TIMEFRAME HANDLER ====================
  const handleTimeframeChange = (timeframe) => {
    setSelectedTimeframes((prevSelected) => {
      if (prevSelected.includes(timeframe)) {
        return prevSelected.length > 1
          ? prevSelected.filter((tf) => tf !== timeframe)
          : prevSelected;
      } else {
        return [...prevSelected, timeframe];
      }
    });
  };

  // ==================== OPTIMIZED CHART CONTROLS ====================
  const handleChartTimeframeChange = useCallback(
    (timeframe) => {
      setChartTimeframe(timeframe);
      if (currentTicker) {
        fetchChartData(currentTicker, timeframe, chartType, selectedIndicators);
      }
    },
    [currentTicker, chartType, selectedIndicators, fetchChartData]
  );

  const handleChartTypeChange = useCallback(
    (type) => {
      setChartType(type);
      if (currentTicker) {
        fetchChartData(currentTicker, chartTimeframe, type, selectedIndicators);
      }
    },
    [currentTicker, chartTimeframe, selectedIndicators, fetchChartData]
  );

  const handleIndicatorChange = useCallback(
    (indicator) => {
      let newIndicators;

      if (selectedIndicators.includes(indicator)) {
        newIndicators = selectedIndicators.filter((ind) => ind !== indicator);
      } else {
        // Limit to 2 indicators for performance
        if (selectedIndicators.length >= 2) {
          newIndicators = [selectedIndicators[1], indicator];
        } else {
          newIndicators = [...selectedIndicators, indicator];
        }
      }

      setSelectedIndicators(newIndicators);

      if (currentTicker) {
        // Use debounced version to avoid rapid API calls
        debouncedFetchChart(
          currentTicker,
          chartTimeframe,
          chartType,
          newIndicators
        );
      }
    },
    [
      selectedIndicators,
      currentTicker,
      chartTimeframe,
      chartType,
      debouncedFetchChart,
    ]
  );

  // ==================== TAB CONFIGURATION ====================
  const tabConfig = useMemo(
    () => [
      {
        id: "all",
        label: "Overview",
        icon: "📋",
        tooltip: "Complete overview of all data",
        disabled: !currentTicker,
      },
      {
        id: "predictions",
        label: "Predictions",
        icon: "📈",
        tooltip: "View AI-powered price predictions",
        badge: predictions ? Object.keys(predictions).length : 0,
        disabled: !predictions,
      },
      {
        id: "analysis",
        label: "Analysis",
        icon: "🔍",
        tooltip: "Technical and market analysis",
        badge: null,
        disabled: !analysis && !marketInfo && !companyData,
      },
      {
        id: "chart",
        label: "Charts",
        icon: "📉",
        tooltip: "Interactive price charts with indicators",
        disabled: !currentTicker,
      },
      {
        id: "essentials",
        label: "Company",
        icon: "📊",
        tooltip: "Company financials and metrics",
        disabled: false,
      },
      {
        id: "sentiment",
        label: "Sentiment",
        icon: "😊",
        tooltip: "Market sentiment analysis",
        disabled: !currentTicker,
      },
    ],
    [currentTicker, predictions, analysis, marketInfo, companyData]
  );

  // ==================== CLEAR OLD CACHE ====================
  useEffect(() => {
    // Clear cache older than 5 minutes
    const clearOldCache = () => {
      const now = Date.now();
      const maxAge = 5 * 60 * 1000; // 5 minutes

      setChartCache((prev) => {
        const newCache = {};
        Object.keys(prev).forEach((key) => {
          if (prev[key]._timestamp && now - prev[key]._timestamp < maxAge) {
            newCache[key] = prev[key];
          }
        });
        return newCache;
      });
    };

    const interval = setInterval(clearOldCache, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  // ==================== RENDER ====================
  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">StockVibePredictor 🚀</h1>
          <p className="app-subtitle">
            Enterprise AI-Powered Multi-Timeframe Stock Predictions with
            Advanced Charting
          </p>
          <SystemStatus systemHealth={systemHealth} />
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          <TimeframeSelector
            timeframes={TIMEFRAMES}
            selectedTimeframes={selectedTimeframes}
            onTimeframeChange={handleTimeframeChange}
            loading={loading}
          />

          <StockInput onSubmit={fetchStockData} loading={loading} />

          {error && <ErrorMessage error={error} />}

          {loading && (
            <div className="loading-container">
              <LoadingSpinner />
              <p className="loading-message">
                Analyzing {currentTicker} across {selectedTimeframes.length}{" "}
                timeframe{selectedTimeframes.length !== 1 ? "s" : ""}...
              </p>
            </div>
          )}

          {currentTicker &&
            !loading &&
            (predictions || companyData || chartData) && (
              <div className="results-tabs">
                <TabNavigation
                  tabs={tabConfig}
                  activeTab={activeTab}
                  onTabChange={setActiveTab}
                />

                <div className="tab-content">
                  {/* Overview Tab */}
                  {activeTab === "all" && (
                    <div className="overview-container">
                      <div className="section-header">
                        <div className="header-content-section">
                          <h2 className="section-title">
                            📋 Complete Overview for {currentTicker}
                          </h2>
                          <p className="section-subtitle">
                            Comprehensive analysis including predictions,
                            charts, fundamentals, and sentiment
                          </p>
                        </div>
                      </div>

                      {/* Quick Stats */}
                      <div className="quick-stats">
                        <div className="stat-card">
                          <span className="stat-label">Current Price</span>
                          <span className="stat-value">
                            {chartData?.summary?.latest_price
                              ? `$${chartData.summary.latest_price.toFixed(2)}`
                              : companyData?.current_price
                              ? `$${companyData.current_price.toFixed(2)}`
                              : "Loading..."}
                          </span>
                          <span
                            className={`stat-change ${
                              chartData?.summary?.change_percent > 0
                                ? "positive"
                                : chartData?.summary?.change_percent < 0
                                ? "negative"
                                : "neutral"
                            }`}
                          >
                            {chartData?.summary?.change_percent !== undefined
                              ? `${
                                  chartData.summary.change_percent > 0
                                    ? "+"
                                    : ""
                                }${chartData.summary.change_percent.toFixed(
                                  2
                                )}%`
                              : ""}
                          </span>
                        </div>

                        <div className="stat-card">
                          <span className="stat-label">Market Cap</span>
                          <span className="stat-value">
                            {companyData?.essentials?.market_cap?.formatted ||
                              "Loading..."}
                          </span>
                        </div>

                        <div className="stat-card">
                          <span className="stat-label">Overall Signal</span>
                          <span
                            className={`stat-value ${
                              analysis?.recommendation?.overall?.toLowerCase() ===
                              "buy"
                                ? "positive"
                                : analysis?.recommendation?.overall?.toLowerCase() ===
                                  "sell"
                                ? "negative"
                                : "neutral"
                            }`}
                          >
                            {analysis?.recommendation?.overall ||
                              "Analyzing..."}
                          </span>
                        </div>
                      </div>

                      {/* Predictions Section */}
                      {predictions && (
                        <section className="overview-section">
                          <h3>📈 AI Predictions</h3>
                          <SuspenseWrapper>
                            <PredictionsTab
                              predictions={predictions}
                              ticker={currentTicker}
                              selectedTimeframes={selectedTimeframes}
                            />
                          </SuspenseWrapper>
                        </section>
                      )}

                      {/* Chart Section */}
                      <section className="overview-section">
                        <h3>📉 Price Chart</h3>
                        {chartData ? (
                          <SuspenseWrapper fallback={<ChartSkeleton />}>
                            <StockChart
                              data={chartData.data}
                              ticker={currentTicker}
                              predictions={predictions}
                              loading={chartLoading}
                            />
                          </SuspenseWrapper>
                        ) : (
                          <ChartSkeleton />
                        )}
                      </section>

                      {/* Analysis Section */}
                      <section className="overview-section">
                        <h3>🔍 Technical Analysis</h3>
                        <SuspenseWrapper>
                          <AnalysisTab
                            analysis={analysis}
                            marketInfo={marketInfo}
                            companyData={companyData}
                            ticker={currentTicker}
                          />
                        </SuspenseWrapper>
                      </section>

                      {/* Company Essentials Section */}
                      <section className="overview-section">
                        <h3>📊 Company Fundamentals</h3>
                        <SuspenseWrapper>
                          <CompanyEssentials
                            ticker={currentTicker}
                            onCompanyDataReceived={handleCompanyDataReceived}
                          />
                        </SuspenseWrapper>
                      </section>
                    </div>
                  )}

                  {/* Predictions Tab */}
                  {activeTab === "predictions" && predictions && (
                    <SuspenseWrapper>
                      <PredictionsTab
                        predictions={predictions}
                        ticker={currentTicker}
                        selectedTimeframes={selectedTimeframes}
                      />
                    </SuspenseWrapper>
                  )}

                  {/* Analysis Tab */}
                  {activeTab === "analysis" && (
                    <SuspenseWrapper>
                      <AnalysisTab
                        analysis={analysis}
                        marketInfo={marketInfo}
                        companyData={companyData}
                        ticker={currentTicker}
                      />
                    </SuspenseWrapper>
                  )}

                  {/* Chart Tab */}
                  {activeTab === "chart" && (
                    <div className="chart-tab-container">
                      <div className="section-header">
                        <div className="header-content-section">
                          <h2 className="section-title">
                            📉 Advanced Stock Charts
                          </h2>
                          <p className="section-subtitle">
                            Interactive charts with technical indicators for{" "}
                            {currentTicker}
                          </p>
                        </div>
                      </div>

                      {/* Chart Controls */}
                      <div className="chart-controls">
                        <div className="control-group">
                          <label>Timeframe:</label>
                          <div className="control-buttons">
                            {Object.entries(CHART_TIMEFRAMES).map(
                              ([key, label]) => (
                                <button
                                  key={key}
                                  className={`control-btn ${
                                    chartTimeframe === key ? "active" : ""
                                  }`}
                                  onClick={() =>
                                    handleChartTimeframeChange(key)
                                  }
                                  disabled={chartLoading}
                                >
                                  {label}
                                </button>
                              )
                            )}
                          </div>
                        </div>

                        <div className="control-group">
                          <label>Chart Type:</label>
                          <div className="control-buttons">
                            {Object.entries(CHART_TYPES).map(([key, label]) => (
                              <button
                                key={key}
                                className={`control-btn ${
                                  chartType === key ? "active" : ""
                                }`}
                                onClick={() => handleChartTypeChange(key)}
                                disabled={chartLoading}
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div className="control-group">
                          <label>Indicators (Max 2 for performance):</label>
                          <div className="control-indicators">
                            {Object.entries(AVAILABLE_INDICATORS).map(
                              ([key, label]) => (
                                <button
                                  key={key}
                                  className={`indicator-btn ${
                                    selectedIndicators.includes(key)
                                      ? "active"
                                      : ""
                                  }`}
                                  onClick={() => handleIndicatorChange(key)}
                                  disabled={
                                    chartLoading ||
                                    (selectedIndicators.length >= 2 &&
                                      !selectedIndicators.includes(key))
                                  }
                                  title={
                                    selectedIndicators.length >= 2 &&
                                    !selectedIndicators.includes(key)
                                      ? "Remove an indicator first"
                                      : ""
                                  }
                                >
                                  {label}
                                </button>
                              )
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Chart Component */}
                      <div
                        className={`chart-wrapper ${
                          chartLoading ? "loading" : ""
                        }`}
                      >
                        {chartData ? (
                          <SuspenseWrapper fallback={<ChartSkeleton />}>
                            <StockChart
                              data={chartData.data}
                              ticker={currentTicker}
                              predictions={predictions}
                              loading={chartLoading}
                              timeframe={chartTimeframe}
                              chartType={chartType}
                              indicators={selectedIndicators}
                            />
                          </SuspenseWrapper>
                        ) : (
                          <ChartSkeleton />
                        )}
                      </div>

                      {/* Chart Info */}
                      {chartData?.summary && (
                        <div className="chart-summary">
                          <div className="summary-stats">
                            <div className="summary-stat">
                              <label>Period High:</label>
                              <span>
                                $
                                {chartData.summary.period_high?.toFixed(2) ||
                                  "N/A"}
                              </span>
                            </div>
                            <div className="summary-stat">
                              <label>Period Low:</label>
                              <span>
                                $
                                {chartData.summary.period_low?.toFixed(2) ||
                                  "N/A"}
                              </span>
                            </div>
                            <div className="summary-stat">
                              <label>Avg Volume:</label>
                              <span>
                                {chartData.summary.average_volume?.toLocaleString() ||
                                  "N/A"}
                              </span>
                            </div>
                            <div className="summary-stat">
                              <label>Data Points:</label>
                              <span>{chartData.summary.data_points || 0}</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Company Essentials Tab */}
                  {activeTab === "essentials" && (
                    <div className="essentials-tab-container">
                      <div className="section-header">
                        <div className="header-content-section">
                          <h2 className="section-title">
                            📊 Company Essentials
                          </h2>
                          <p className="section-subtitle">
                            Comprehensive financial metrics and company
                            information for {currentTicker}
                          </p>
                        </div>
                      </div>

                      <SuspenseWrapper>
                        <CompanyEssentials
                          ticker={currentTicker}
                          onCompanyDataReceived={handleCompanyDataReceived}
                        />
                      </SuspenseWrapper>
                    </div>
                  )}

                  {/* Sentiment Tab */}
                  {activeTab === "sentiment" && (
                    <div className="sentiment-tab-container">
                      <div className="section-header">
                        <div className="header-content-section">
                          <h2 className="section-title">😊 Market Sentiment</h2>
                          <p className="section-subtitle">
                            Real-time sentiment analysis from news and social
                            media for {currentTicker}
                          </p>
                        </div>
                      </div>

                      <SuspenseWrapper>
                        <SentimentAnalysis ticker={currentTicker} />
                      </SuspenseWrapper>
                    </div>
                  )}
                </div>
              </div>
            )}
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>
            🚀 Built with React + Django + Enterprise ML + Advanced Charting
          </p>
          <div className="api-info">
            <span>API Status: {systemHealth?.status || "Unknown"}</span>
            <span>•</span>
            <span>
              Models: {systemHealth?.metrics?.model_cache_size || 0} loaded
            </span>
            <span>•</span>
            <span>Chart Engine: Optimized with Caching</span>
            <span>•</span>
            <span>Cache Size: {Object.keys(chartCache).length} charts</span>
            {companyData && (
              <>
                <span>•</span>
                <span>
                  Last Updated:{" "}
                  {new Date(companyData.last_updated).toLocaleTimeString()}
                </span>
              </>
            )}
          </div>

          <div className="global-disclaimer">
            <p>
              <strong>⚠️ Disclaimer:</strong> These predictions and charts are
              for educational purposes only. Past performance does not guarantee
              future results. Always conduct your own research and consider
              consulting with a financial advisor before making investment
              decisions.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
