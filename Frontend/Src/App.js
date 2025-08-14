import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";

// Components
import StockInput from "./Components/Javascript/StockInput";
import StockChart from "./Components/Javascript/StockChart";
import LoadingSpinner from "./Components/Javascript/LoadingSpinner";
import CompanyEssentials from "./Components/Javascript/CompanyEssentials";
import PredictionsTab from "./Components/Javascript/PredictionTab";
import AnalysisTab from "./Components/Javascript/AnalysisTab";
import SystemStatus from "./Components/Javascript/SystemStatus";
import TimeframeSelector from "./Components/Javascript/TimeframeSelector";
import ErrorMessage from "./Components/Javascript/ErrorMessage";
import TabNavigation from "./Components/Javascript/TabNavigation";
import SentimentAnalysis from "./Components/Javascript/SentimentAnalysis";

function App() {
  // State Management
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
  const [selectedIndicators, setSelectedIndicators] = useState([
    "sma20",
    "rsi",
  ]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [activeTab, setActiveTab] = useState("predictions");

  // Configuration
  const API_BASE_URL = "http://localhost:8000/api";
  const TIMEFRAMES = {
    "1d": "1 Day",
    "1w": "1 Week",
    "1mo": "1 Month",
    "1y": "1 Year",
  };

  const CHART_TIMEFRAMES = {
    "1d": "1 Day",
    "5d": "5 Days",
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
    "2y": "2 Years",
    "5y": "5 Years",
  };

  const CHART_TYPES = {
    candlestick: "Candlestick",
    line: "Line",
    ohlc: "OHLC",
  };

  const AVAILABLE_INDICATORS = {
    sma20: "SMA 20",
    sma50: "SMA 50",
    rsi: "RSI",
    macd: "MACD",
    bollinger: "Bollinger Bands",
    volume: "Volume",
  };

  // System Health Check
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

  // Company Data Handler
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

  // Enhanced Chart Data Fetcher
  const fetchChartData = async (
    ticker,
    timeframe = chartTimeframe,
    type = chartType,
    indicators = selectedIndicators
  ) => {
    setChartLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/chart/${ticker}/`, {
        params: {
          timeframe: timeframe,
          chart_type: type,
          indicators: indicators.join(","),
        },
      });

      setChartData(response.data);
      console.log("Chart data fetched successfully:", {
        ticker: response.data.ticker,
        dataPoints: response.data.data?.length || 0,
        indicators: Object.keys(response.data.indicators || {}),
      });
    } catch (err) {
      console.warn("Chart data fetch failed:", err);
      setChartData(null);
    } finally {
      setChartLoading(false);
    }
  };

  // Stock Data Fetcher
  const fetchStockData = async (ticker) => {
    if (!ticker.trim()) {
      setError("Please enter a stock ticker symbol! 📈");
      resetData();
      return;
    }

    setLoading(true);
    setError(null);
    setCurrentTicker(ticker.toUpperCase());

    try {
      // Fetch predictions and analysis
      const response = await axios.post(`${API_BASE_URL}/predict/multi/`, {
        ticker: ticker.toUpperCase(),
        timeframes: selectedTimeframes,
        include_analysis: true,
      });

      const data = response.data;
      setPredictions(data.predictions);
      setAnalysis(data.analysis || null);
      setMarketInfo(data.market_info || null);
      setStockData(data.history || null);

      // Fetch enhanced chart data separately
      await fetchChartData(ticker.toUpperCase());

      console.log("Prediction successful:", {
        ticker: data.ticker,
        timeframes: Object.keys(data.predictions),
        hasAnalysis: !!data.analysis,
      });

      setActiveTab("predictions");
    } catch (err) {
      console.error("Error fetching stock data:", err);
      handleFetchError(err, ticker);
    } finally {
      setLoading(false);
    }
  };

  // Error Handler
  const handleFetchError = (err, ticker) => {
    let errorMessage = "Unable to fetch stock data. Please try again.";

    if (err.response?.status === 404) {
      errorMessage = `Stock ticker "${ticker}" not found. Please check the symbol.`;
    } else if (err.response?.status === 400) {
      errorMessage = err.response.data?.error || "Invalid ticker format.";
    } else if (err.response?.status >= 500) {
      errorMessage =
        "Server error. The backend might be down or models need training.";
    } else if (err.code === "ECONNREFUSED") {
      errorMessage =
        "Cannot connect to backend. Make sure Django server is running on port 8000.";
    } else if (err.response?.data?.error) {
      errorMessage = err.response.data.error;
    }

    setError(errorMessage);
    resetData();
  };

  // Reset Data
  const resetData = () => {
    setStockData(null);
    setChartData(null);
    setPredictions(null);
    setAnalysis(null);
    setMarketInfo(null);
    setCompanyData(null);
  };

  // Timeframe Handler
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

  // Chart Controls Handlers
  const handleChartTimeframeChange = (timeframe) => {
    setChartTimeframe(timeframe);
    if (currentTicker) {
      fetchChartData(currentTicker, timeframe, chartType, selectedIndicators);
    }
  };

  const handleChartTypeChange = (type) => {
    setChartType(type);
    if (currentTicker) {
      fetchChartData(currentTicker, chartTimeframe, type, selectedIndicators);
    }
  };

  const handleIndicatorChange = (indicator) => {
    const newIndicators = selectedIndicators.includes(indicator)
      ? selectedIndicators.filter((ind) => ind !== indicator)
      : [...selectedIndicators, indicator];

    setSelectedIndicators(newIndicators);
    if (currentTicker) {
      fetchChartData(currentTicker, chartTimeframe, chartType, newIndicators);
    }
  };

  // Tab Configuration
  const tabConfig = [
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
      badge: analysis ? "New" : null,
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
  ];

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
                  {/* Overview Tab - Shows everything */}
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
                          <PredictionsTab
                            predictions={predictions}
                            ticker={currentTicker}
                            selectedTimeframes={selectedTimeframes}
                          />
                        </section>
                      )}

                      {/* Chart Section */}
                      <section className="overview-section">
                        <h3>📉 Price Chart</h3>
                        {chartData ? (
                          <StockChart
                            data={chartData}
                            ticker={currentTicker}
                            predictions={predictions}
                            loading={chartLoading}
                          />
                        ) : (
                          <div className="chart-placeholder">
                            <LoadingSpinner />
                            <p>Loading interactive chart...</p>
                          </div>
                        )}
                      </section>

                      {/* Analysis Section */}
                      <section className="overview-section">
                        <h3>🔍 Technical Analysis</h3>
                        <AnalysisTab
                          analysis={analysis}
                          marketInfo={marketInfo}
                          companyData={companyData}
                          ticker={currentTicker}
                        />
                      </section>

                      {/* Company Essentials Section */}
                      <section className="overview-section">
                        <h3>📊 Company Fundamentals</h3>
                        <CompanyEssentials
                          ticker={currentTicker}
                          onCompanyDataReceived={handleCompanyDataReceived}
                        />
                      </section>
                    </div>
                  )}

                  {/* Individual Tabs */}
                  {activeTab === "predictions" && predictions && (
                    <PredictionsTab
                      predictions={predictions}
                      ticker={currentTicker}
                      selectedTimeframes={selectedTimeframes}
                    />
                  )}

                  {activeTab === "analysis" && (
                    <AnalysisTab
                      analysis={analysis}
                      marketInfo={marketInfo}
                      companyData={companyData}
                      ticker={currentTicker}
                    />
                  )}

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
                          <label>Indicators:</label>
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
                                  disabled={chartLoading}
                                >
                                  {label}
                                </button>
                              )
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Chart Component */}
                      <div className="chart-wrapper">
                        {chartData ? (
                          <StockChart
                            data={chartData}
                            ticker={currentTicker}
                            predictions={predictions}
                            loading={chartLoading}
                            timeframe={chartTimeframe}
                            chartType={chartType}
                            indicators={selectedIndicators}
                          />
                        ) : (
                          <div className="chart-placeholder">
                            <div className="placeholder-content">
                              <h3>📈 Loading Chart Data</h3>
                              <p>
                                Fetching {chartTimeframe} chart data for{" "}
                                {currentTicker}...
                              </p>
                              <LoadingSpinner />
                            </div>
                          </div>
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

                      <CompanyEssentials
                        ticker={currentTicker}
                        onCompanyDataReceived={handleCompanyDataReceived}
                      />
                    </div>
                  )}

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

                      <SentimentAnalysis ticker={currentTicker} />
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
            <span>Chart Engine: TradingView Compatible</span>
            <span>•</span>
            <span>Timeframes: {Object.values(TIMEFRAMES).join(", ")}</span>
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

          {/* Global Disclaimer */}
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
