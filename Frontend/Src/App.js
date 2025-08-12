import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import "./App.css";
import StockInput from "./Components/StockInput";
import PredictionResult from "./Components/PredictionResult";
import StockChart from "./Components/StockChart";
import LoadingSpinner from "./Components/LoadingSpinner";
import CompanyEssentials from "./Components/CompanyEssentials";

function App() {
  const [stockData, setStockData] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [marketInfo, setMarketInfo] = useState(null);
  const [companyData, setCompanyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentTicker, setCurrentTicker] = useState("");
  const [selectedTimeframes, setSelectedTimeframes] = useState([
    "1d",
    "1w",
    "1mo",
  ]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [activeTab, setActiveTab] = useState("predictions"); // For tab navigation

  // API base URL - your Django serves at /api/
  const API_BASE_URL = "http://localhost:8000/api";

  // Available timeframes for user selection
  const TIMEFRAMES = {
    "1d": "1 Day",
    "1w": "1 Week",
    "1mo": "1 Month",
    "1y": "1 Year",
  };

  // Check system health on app load
  useEffect(() => {
    checkSystemHealth();
    const healthInterval = setInterval(checkSystemHealth, 60000); // Check every minute
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

  // Callback for when CompanyEssentials loads data
  const handleCompanyDataReceived = useCallback((data) => {
    setCompanyData(data);

    // Update market info with company essentials data if available
    if (data && data.company_info) {
      setMarketInfo((prevMarketInfo) => ({
        ...prevMarketInfo,
        sector: data.company_info.sector,
        industry: data.company_info.industry,
        market_cap: data.essentials?.market_cap?.value,
        pe_ratio: data.essentials?.pe_ratio?.value,
        dividend_yield: data.essentials?.dividend_yield?.value,
        exchange: data.company_info.exchange,
      }));
    }
  }, []);

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
      // Use the multi-timeframe prediction endpoint
      const response = await axios.post(`${API_BASE_URL}/predict/multi/`, {
        ticker: ticker.toUpperCase(),
        timeframes: selectedTimeframes,
        include_analysis: true,
      });

      // Extract data from the enhanced response
      const data = response.data;

      setPredictions(data.predictions);
      setAnalysis(data.analysis || null);
      setMarketInfo(data.market_info || null);
      setStockData(data.history || null);

      // Log successful prediction for debugging
      console.log("Prediction successful:", {
        ticker: data.ticker,
        timeframes: Object.keys(data.predictions),
        hasAnalysis: !!data.analysis,
      });

      // Switch to predictions tab after successful fetch
      setActiveTab("predictions");
    } catch (err) {
      console.error("Error fetching stock data:", err);

      // Enhanced error handling
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
    } finally {
      setLoading(false);
    }
  };

  const resetData = () => {
    setStockData(null);
    setPredictions(null);
    setAnalysis(null);
    setMarketInfo(null);
    setCompanyData(null);
  };

  const handleTimeframeChange = (timeframe) => {
    setSelectedTimeframes((prevSelected) => {
      if (prevSelected.includes(timeframe)) {
        // Remove timeframe (but always keep at least one)
        return prevSelected.length > 1
          ? prevSelected.filter((tf) => tf !== timeframe)
          : prevSelected;
      } else {
        // Add timeframe
        return [...prevSelected, timeframe];
      }
    });
  };

  const getSystemHealthStatus = () => {
    if (!systemHealth) return { color: "gray", text: "Checking..." };

    switch (systemHealth.status) {
      case "healthy":
        return { color: "#4caf50", text: "● Online" };
      case "degraded":
        return { color: "#ff9800", text: "● Partial" };
      case "unhealthy":
        return { color: "#f44336", text: "● Offline" };
      default:
        return { color: "#9e9e9e", text: "● Unknown" };
    }
  };

  const formatMarketCap = (value) => {
    if (!value) return "N/A";
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toLocaleString()}`;
  };

  const healthStatus = getSystemHealthStatus();

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">StockVibePredictor 🚀</h1>
          <p className="app-subtitle">
            Enterprise AI-Powered Multi-Timeframe Stock Predictions
          </p>

          <div className="system-status">
            <span
              className="status-indicator"
              style={{ color: healthStatus.color }}
              title={
                systemHealth?.services
                  ? JSON.stringify(systemHealth.services, null, 2)
                  : "System status"
              }
            >
              {healthStatus.text}
            </span>
            {systemHealth?.metrics && (
              <span className="system-metrics">
                | {systemHealth.metrics.model_cache_size || 0} models loaded
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="app-main">
        <div className="container">
          {/* Timeframe Selection */}
          <div className="timeframe-selector">
            <h3>Select Prediction Timeframes:</h3>
            <div className="timeframe-buttons">
              {Object.entries(TIMEFRAMES).map(([key, label]) => (
                <button
                  key={key}
                  className={`timeframe-btn ${
                    selectedTimeframes.includes(key) ? "active" : ""
                  }`}
                  onClick={() => handleTimeframeChange(key)}
                  disabled={loading}
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="selected-timeframes">
              Selected:{" "}
              {selectedTimeframes.map((tf) => TIMEFRAMES[tf]).join(", ")}
            </p>
          </div>

          <StockInput onSubmit={fetchStockData} loading={loading} />

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              <div className="error-content">
                <strong>Error:</strong> {error}
                {error.includes("backend") && (
                  <div className="error-help">
                    <p>
                      💡 <strong>Quick Fix:</strong>
                    </p>
                    <p>
                      1. Make sure Django is running:{" "}
                      <code>python manage.py runserver</code>
                    </p>
                    <p>
                      2. Train models if needed:{" "}
                      <code>python TrainModel.py full</code>
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-container">
              <LoadingSpinner />
              <p className="loading-message">
                Analyzing {currentTicker} across {selectedTimeframes.length}{" "}
                timeframe{selectedTimeframes.length !== 1 ? "s" : ""}...
              </p>
            </div>
          )}

          {/* Tab Navigation for Results */}
          {currentTicker && !loading && (predictions || companyData) && (
            <div className="results-tabs">
              <div className="tab-header">
                <button
                  className={`tab-btn ${
                    activeTab === "predictions" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("predictions")}
                >
                  📈 Predictions
                </button>
                <button
                  className={`tab-btn ${
                    activeTab === "essentials" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("essentials")}
                >
                  📊 Company Essentials
                </button>
                <button
                  className={`tab-btn ${activeTab === "chart" ? "active" : ""}`}
                  onClick={() => setActiveTab("chart")}
                >
                  📉 Chart
                </button>
                <button
                  className={`tab-btn ${
                    activeTab === "analysis" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("analysis")}
                >
                  🔍 Analysis
                </button>
              </div>

              <div className="tab-content">
                {/* Prediction Results Tab */}
                {activeTab === "predictions" && predictions && (
                  <div className="results-container">
                    <PredictionResult
                      predictions={predictions}
                      analysis={analysis}
                      marketInfo={marketInfo}
                      ticker={currentTicker}
                      selectedTimeframes={selectedTimeframes}
                    />
                  </div>
                )}

                {/* Company Essentials Tab */}
                {activeTab === "essentials" && (
                  <CompanyEssentials
                    ticker={currentTicker}
                    onCompanyDataReceived={handleCompanyDataReceived}
                  />
                )}

                {/* Stock Chart Tab */}
                {activeTab === "chart" && stockData && (
                  <StockChart
                    data={stockData}
                    ticker={currentTicker}
                    predictions={predictions}
                  />
                )}

                {/* Analysis Tab */}
                {activeTab === "analysis" && (
                  <div className="analysis-container">
                    {/* Market Information Panel */}
                    {(marketInfo || companyData) && (
                      <div className="market-info-panel">
                        <h3>📊 Market Information</h3>
                        <div className="market-stats">
                          {(marketInfo?.sector ||
                            companyData?.company_info?.sector) && (
                            <div className="stat">
                              <label>Sector:</label>
                              <span>
                                {marketInfo?.sector ||
                                  companyData?.company_info?.sector}
                              </span>
                            </div>
                          )}
                          {(marketInfo?.industry ||
                            companyData?.company_info?.industry) && (
                            <div className="stat">
                              <label>Industry:</label>
                              <span>
                                {marketInfo?.industry ||
                                  companyData?.company_info?.industry}
                              </span>
                            </div>
                          )}
                          {(marketInfo?.market_cap ||
                            companyData?.essentials?.market_cap?.value) && (
                            <div className="stat">
                              <label>Market Cap:</label>
                              <span>
                                {companyData?.essentials?.market_cap
                                  ?.formatted ||
                                  formatMarketCap(marketInfo?.market_cap)}
                              </span>
                            </div>
                          )}
                          {(marketInfo?.pe_ratio ||
                            companyData?.essentials?.pe_ratio?.value) && (
                            <div className="stat">
                              <label>P/E Ratio:</label>
                              <span>
                                {companyData?.essentials?.pe_ratio?.formatted ||
                                  marketInfo?.pe_ratio?.toFixed(2)}
                              </span>
                            </div>
                          )}
                          {marketInfo?.beta && (
                            <div className="stat">
                              <label>Beta:</label>
                              <span>{marketInfo.beta.toFixed(2)}</span>
                            </div>
                          )}
                          {(marketInfo?.dividend_yield ||
                            companyData?.essentials?.dividend_yield?.value) && (
                            <div className="stat">
                              <label>Dividend Yield:</label>
                              <span>
                                {companyData?.essentials?.dividend_yield
                                  ?.formatted ||
                                  `${(marketInfo.dividend_yield * 100).toFixed(
                                    2
                                  )}%`}
                              </span>
                            </div>
                          )}
                          {companyData?.company_info?.exchange && (
                            <div className="stat">
                              <label>Exchange:</label>
                              <span>{companyData.company_info.exchange}</span>
                            </div>
                          )}
                        </div>

                        {/* 52-Week Range from Company Essentials */}
                        {companyData?.price_summary && (
                          <div className="price-range">
                            <h4>52-Week Range</h4>
                            <div className="range-info">
                              <span className="range-low">
                                Low: {companyData.currency?.symbol}
                                {companyData.price_summary.week_52_low?.toFixed(
                                  2
                                )}
                              </span>
                              <div className="range-bar-container">
                                <div className="range-bar">
                                  <div
                                    className="current-position"
                                    style={{
                                      left: `${
                                        ((companyData.current_price -
                                          companyData.price_summary
                                            .week_52_low) /
                                          (companyData.price_summary
                                            .week_52_high -
                                            companyData.price_summary
                                              .week_52_low)) *
                                        100
                                      }%`,
                                    }}
                                  />
                                </div>
                              </div>
                              <span className="range-high">
                                High: {companyData.currency?.symbol}
                                {companyData.price_summary.week_52_high?.toFixed(
                                  2
                                )}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Technical Analysis Panel */}
                    {analysis && (
                      <div className="analysis-panel">
                        <h3>🔍 Technical Analysis</h3>

                        {analysis.technical && (
                          <div className="technical-analysis">
                            <h4>Technical Indicators</h4>
                            <div className="indicators">
                              <div className="indicator">
                                <label>RSI:</label>
                                <span
                                  className={`rsi-value ${analysis.technical.rsi_signal}`}
                                >
                                  {analysis.technical.rsi} (
                                  {analysis.technical.rsi_signal})
                                </span>
                              </div>
                              <div className="indicator">
                                <label>Trend:</label>
                                <span
                                  className={`trend ${analysis.technical.trend}`}
                                >
                                  {analysis.technical.trend}
                                </span>
                              </div>
                              <div className="indicator">
                                <label>Volume:</label>
                                <span>{analysis.technical.volume_trend}</span>
                              </div>
                              <div className="indicator">
                                <label>Volatility:</label>
                                <span>
                                  {analysis.technical.volatility_regime}
                                </span>
                              </div>
                            </div>
                          </div>
                        )}

                        {analysis.recommendation && (
                          <div className="recommendation">
                            <h4>📈 Overall Recommendation</h4>
                            <div
                              className={`rec-badge ${analysis.recommendation.overall.toLowerCase()}`}
                            >
                              {analysis.recommendation.overall}
                            </div>
                            <div className="rec-details">
                              <p>
                                Confidence: {analysis.recommendation.confidence}
                                %
                              </p>
                              <p>
                                Risk Level: {analysis.recommendation.risk_level}
                              </p>
                              <p>
                                Suggested Holding:{" "}
                                {analysis.recommendation.holding_period}
                              </p>
                            </div>
                          </div>
                        )}

                        {analysis.sentiment && (
                          <div className="sentiment-analysis">
                            <h4>📰 Market Sentiment</h4>
                            <div
                              className={`sentiment ${analysis.sentiment.sentiment_label}`}
                            >
                              {analysis.sentiment.sentiment_label}
                              <span className="sentiment-score">
                                (
                                {(
                                  analysis.sentiment.sentiment_score * 100
                                ).toFixed(1)}
                                %)
                              </span>
                            </div>
                          </div>
                        )}

                        {/* Additional Metrics from Company Essentials */}
                        {companyData?.additional_metrics && (
                          <div className="additional-analysis">
                            <h4>📊 Additional Metrics</h4>
                            <div className="metrics-grid">
                              {Object.entries(
                                companyData.additional_metrics
                              ).map(([key, metric]) => (
                                <div key={key} className="metric">
                                  <label>{metric.label}:</label>
                                  <span>{metric.formatted}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>🚀 Built with React + Django + Enterprise ML</p>
          <div className="api-info">
            <span>API Status: {healthStatus.text}</span>
            <span>•</span>
            <span>
              Models: {systemHealth?.metrics?.model_cache_size || 0} loaded
            </span>
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
        </div>
      </footer>
    </div>
  );
}

export default App;
