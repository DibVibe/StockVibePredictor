import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
} from "chart.js";
import { Doughnut, Bar, Line } from "react-chartjs-2";
import PropTypes from "prop-types";
import "../CSS/SentimentAnalysis.css";

// Register Chart.js components
ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  Title,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement
);

const SentimentAnalysis = ({ ticker, apiKey, refreshInterval = 300000 }) => {
  // State Management
  const [sentimentData, setSentimentData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [riskMetrics, setRiskMetrics] = useState(null);
  const [riskExposure, setRiskExposure] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);

  // API Configuration
  const API_ENDPOINTS = {
    sentiment: `https://api.example.com/sentiment/${ticker}`,
    news: `https://api.example.com/news/${ticker}`,
    social: `https://api.example.com/social/${ticker}`,
  };

  // ============================================================================
  // API INTEGRATION FUNCTIONS
  // ============================================================================

  const fetchRealSentimentData = useCallback(
    async (symbol) => {
      try {
        // Placeholder for real API integration
        // Replace with your actual API calls

        // Example API call structure:
        /*
      const response = await fetch(`${API_ENDPOINTS.sentiment}?apikey=${apiKey}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        }
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();
      return data;
      */

        // For demonstration, return mock data with more realistic variation
        const mockData = generateMockSentimentData(symbol);
        return mockData;
      } catch (error) {
        console.error("Failed to fetch sentiment data:", error);
        throw error;
      }
    },
    [apiKey]
  );

  const generateMockSentimentData = (symbol) => {
    // Generate more realistic mock data based on ticker
    const basePositive = Math.random() * 60 + 20; // 20-80%
    const baseNegative = Math.random() * 40 + 10; // 10-50%
    const baseNeutral = 100 - basePositive - baseNegative;

    return {
      positive: Math.max(0, basePositive),
      negative: Math.max(0, baseNegative),
      neutral: Math.max(0, baseNeutral),
      sources: ["News APIs", "Twitter", "Reddit", "Financial Forums"],
      totalSources: Math.floor(Math.random() * 1000) + 100,
      confidence: Math.floor(Math.random() * 30) + 70,
      marketCondition: getMarketCondition(basePositive, baseNegative),
      newsContext: generateNewsContext(symbol),
      trend: generateTrendData(),
      socialMentions: Math.floor(Math.random() * 10000) + 1000,
      newsArticles: Math.floor(Math.random() * 100) + 20,
      isData: true,
      dataSource: "Mock Data - Replace with real API",
    };
  };

  const getMarketCondition = (positive, negative) => {
    if (positive > negative + 20) return "Bullish";
    if (negative > positive + 20) return "Bearish";
    return "Neutral";
  };

  const generateNewsContext = (symbol) => {
    const contexts = [
      `${symbol} reports strong quarterly earnings`,
      `Market volatility affects ${symbol} sentiment`,
      `Analyst upgrades ${symbol} price target`,
      `${symbol} announces strategic partnership`,
      `Regulatory concerns impact ${symbol} outlook`,
    ];
    return contexts.slice(0, Math.floor(Math.random() * 3) + 2);
  };

  const generateTrendData = () => {
    const trend = [];
    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      trend.push({
        date: date.toISOString().split("T")[0],
        positive: Math.random() * 60 + 20,
        negative: Math.random() * 40 + 10,
        neutral: Math.random() * 40 + 20,
      });
    }
    return trend;
  };

  // ============================================================================
  // ENHANCED RISK CALCULATION FUNCTIONS
  // ============================================================================

  const calculateAdvancedRiskMetrics = useCallback((sentimentData) => {
    if (!sentimentData) return null;

    const { positive, negative, neutral, confidence, trend } = sentimentData;

    // Enhanced volatility calculation with trend analysis
    const sentimentSpread = Math.abs(positive - negative);
    const trendVolatility = trend ? calculateTrendVolatility(trend) : 0;
    const confidenceAdjustment = (100 - confidence) / 100;

    const volatilityBase = (100 - sentimentSpread) / 100;
    const volatility = Math.min(
      0.8,
      0.05 +
        volatilityBase * 0.45 +
        trendVolatility * 0.2 +
        confidenceAdjustment * 0.1
    );

    // Enhanced Sharpe Ratio calculation
    const riskFreeRate = 0.02;
    const sentimentMomentum = trend ? calculateSentimentMomentum(trend) : 0;
    const expectedReturn =
      (positive - negative) * 0.002 + sentimentMomentum * 0.001;
    const sharpeRatio = expectedReturn / Math.max(volatility, 0.01);

    // Value at Risk with confidence intervals
    const var95 = volatility * 1.645;
    const var99 = volatility * 2.326;

    // Maximum Drawdown with sentiment weighting
    const sentimentRisk = negative / 100;
    const maxDrawdown = Math.min(0.9, volatility * (0.5 + sentimentRisk));

    // Beta estimation (relative to market sentiment)
    const beta = calculateBeta(sentimentData);

    // Risk level with more granular classification
    let riskLevel = "low";
    let riskScore = 0;

    if (volatility > 0.4 || negative > 60) {
      riskLevel = "very-high";
      riskScore = 9;
    } else if (volatility > 0.3 || negative > 45) {
      riskLevel = "high";
      riskScore = 7;
    } else if (volatility > 0.2 || negative > 30) {
      riskLevel = "medium-high";
      riskScore = 6;
    } else if (volatility > 0.15 || negative > 20) {
      riskLevel = "medium";
      riskScore = 5;
    } else if (volatility > 0.1 || negative > 15) {
      riskLevel = "medium-low";
      riskScore = 3;
    } else {
      riskLevel = "low";
      riskScore = 2;
    }

    return {
      volatility,
      sharpeRatio,
      var95,
      var99,
      maxDrawdown,
      beta,
      riskLevel,
      riskScore,
      confidenceScore: confidence || 70,
      sentimentMomentum,
      expectedReturn: expectedReturn * 100, // Convert to percentage
    };
  }, []);

  const calculateTrendVolatility = (trend) => {
    if (!trend || trend.length < 2) return 0;

    const sentimentChanges = [];
    for (let i = 1; i < trend.length; i++) {
      const change = Math.abs(
        trend[i].positive -
          trend[i].negative -
          (trend[i - 1].positive - trend[i - 1].negative)
      );
      sentimentChanges.push(change);
    }

    return (
      sentimentChanges.reduce((a, b) => a + b, 0) /
      sentimentChanges.length /
      100
    );
  };

  const calculateSentimentMomentum = (trend) => {
    if (!trend || trend.length < 2) return 0;

    const first = trend[0].positive - trend[0].negative;
    const last =
      trend[trend.length - 1].positive - trend[trend.length - 1].negative;

    return (last - first) / 100; // Normalized momentum
  };

  const calculateBeta = (sentimentData) => {
    // Simplified beta calculation based on sentiment volatility
    const { positive, negative, confidence } = sentimentData;
    const sentimentRange = Math.abs(positive - negative);
    const confidenceFactor = confidence / 100;

    return Math.min(
      2.0,
      Math.max(0.5, 1 + ((50 - sentimentRange) / 100) * (1 - confidenceFactor))
    );
  };

  const calculateEnhancedRiskExposure = useCallback(
    (riskMetrics, sentimentData) => {
      if (!riskMetrics || !sentimentData) return null;

      const { volatility, sharpeRatio, riskScore, beta } = riskMetrics;
      const { positive, negative, marketCondition, confidence } = sentimentData;

      // Multi-factor risk allocation model
      let highRisk = 0;
      let mediumRisk = 0;
      let lowRisk = 0;

      // Factor 1: Volatility (40% weight)
      if (volatility > 0.4) {
        highRisk += 50;
        mediumRisk += 30;
        lowRisk += 20;
      } else if (volatility > 0.25) {
        highRisk += 35;
        mediumRisk += 40;
        lowRisk += 25;
      } else if (volatility > 0.15) {
        highRisk += 25;
        mediumRisk += 45;
        lowRisk += 30;
      } else {
        highRisk += 15;
        mediumRisk += 35;
        lowRisk += 50;
      }

      // Factor 2: Sentiment Distribution (30% weight)
      const sentimentAdjustment = (negative - positive) * 0.25;
      highRisk += sentimentAdjustment;
      lowRisk -= sentimentAdjustment;

      // Factor 3: Confidence Level (20% weight)
      const confidenceAdjustment = (100 - confidence) * 0.15;
      highRisk += confidenceAdjustment;
      mediumRisk += confidenceAdjustment * 0.5;
      lowRisk -= confidenceAdjustment * 1.5;

      // Factor 4: Beta (10% weight)
      if (beta > 1.5) {
        highRisk += 10;
        lowRisk -= 10;
      } else if (beta < 0.8) {
        lowRisk += 10;
        highRisk -= 10;
      }

      // Normalize and ensure positive values
      highRisk = Math.max(5, Math.min(85, highRisk));
      mediumRisk = Math.max(5, Math.min(75, mediumRisk));
      lowRisk = Math.max(10, Math.min(80, lowRisk));

      const total = highRisk + mediumRisk + lowRisk;
      highRisk = Math.round((highRisk / total) * 100);
      mediumRisk = Math.round((mediumRisk / total) * 100);
      lowRisk = 100 - highRisk - mediumRisk;

      return {
        highRisk,
        mediumRisk,
        lowRisk,
        recommendation: getEnhancedPortfolioRecommendation(
          highRisk,
          mediumRisk,
          lowRisk,
          sentimentData,
          riskMetrics
        ),
        riskAllocation: {
          aggressive: Math.max(0, 100 - highRisk - mediumRisk / 2),
          moderate: mediumRisk + Math.min(highRisk, 20),
          conservative: lowRisk + Math.max(0, highRisk - 20),
        },
      };
    },
    []
  );

  const getEnhancedPortfolioRecommendation = (
    highRisk,
    mediumRisk,
    lowRisk,
    sentimentData,
    riskMetrics
  ) => {
    const { marketCondition, confidence, positive, negative } = sentimentData;
    const { riskLevel, sharpeRatio, expectedReturn } = riskMetrics;

    let strategy, action, riskTolerance, timeHorizon, allocation;

    if (highRisk > 50) {
      strategy = "Defensive";
      action =
        "Reduce exposure, increase cash position, focus on defensive sectors";
      riskTolerance = "Low";
      timeHorizon = "Short-term (1-3 months)";
      allocation = "Bonds: 60%, Large-cap: 25%, Cash: 15%";
    } else if (lowRisk > 50 && sharpeRatio > 0.5) {
      strategy = "Growth-Oriented";
      action =
        "Increase equity allocation, consider growth stocks and emerging markets";
      riskTolerance = "High";
      timeHorizon = "Long-term (1-3 years)";
      allocation = "Equities: 70%, Growth: 20%, International: 10%";
    } else if (mediumRisk > 40) {
      strategy = "Balanced";
      action = "Maintain diversified portfolio, regular rebalancing";
      riskTolerance = "Medium";
      timeHorizon = "Medium-term (6-12 months)";
      allocation = "Equities: 50%, Bonds: 30%, Alternatives: 20%";
    } else {
      strategy = "Opportunistic";
      action = "Tactical allocation based on market conditions";
      riskTolerance = "Variable";
      timeHorizon = "Flexible";
      allocation = "Dynamic allocation based on signals";
    }

    return {
      strategy,
      action,
      riskTolerance,
      timeHorizon,
      allocation,
      confidence: confidence,
      marketOutlook: marketCondition,
      expectedReturn: expectedReturn?.toFixed(1) || "N/A",
    };
  };

  // ============================================================================
  // DATA FETCHING AND EFFECTS
  // ============================================================================

  const fetchSentimentData = useCallback(async () => {
    if (!ticker) return;

    setLoading(true);
    setError(null);

    try {
      const data = await fetchRealSentimentData(ticker);

      if (data) {
        setSentimentData({
          ...data,
          lastUpdated: new Date(),
        });

        // Update historical data
        setHistoricalData((prev) => {
          const newEntry = {
            date: new Date().toISOString(),
            ...data,
          };
          return [...prev.slice(-6), newEntry]; // Keep last 7 entries
        });

        // Calculate enhanced risk metrics
        const calculatedRiskMetrics = calculateAdvancedRiskMetrics(data);
        setRiskMetrics(calculatedRiskMetrics);

        // Calculate enhanced risk exposure
        const calculatedRiskExposure = calculateEnhancedRiskExposure(
          calculatedRiskMetrics,
          data
        );
        setRiskExposure(calculatedRiskExposure);

        setLastRefresh(new Date());
      } else {
        throw new Error("No data received from API");
      }
    } catch (err) {
      setError(`Failed to fetch sentiment data: ${err.message}`);
      console.error("Sentiment analysis error:", err);
    } finally {
      setLoading(false);
    }
  }, [
    ticker,
    fetchRealSentimentData,
    calculateAdvancedRiskMetrics,
    calculateEnhancedRiskExposure,
  ]);

  // Auto-refresh effect
  useEffect(() => {
    let interval;
    if (autoRefresh && ticker) {
      interval = setInterval(fetchSentimentData, refreshInterval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, ticker, refreshInterval, fetchSentimentData]);

  // Initial data fetch
  useEffect(() => {
    fetchSentimentData();
  }, [fetchSentimentData]);

  // ============================================================================
  // CHART DATA AND OPTIONS
  // ============================================================================

  const sentimentChartData = useMemo(() => {
    if (!sentimentData) return null;

    return {
      labels: [
        `😊 Positive (${sentimentData.positive.toFixed(1)}%)`,
        `😐 Neutral (${sentimentData.neutral.toFixed(1)}%)`,
        `😢 Negative (${sentimentData.negative.toFixed(1)}%)`,
      ],
      datasets: [
        {
          data: [
            sentimentData.positive,
            sentimentData.neutral,
            sentimentData.negative,
          ],
          backgroundColor: ["#2ECC71", "#3498DB", "#E74C3C"],
          borderColor: ["#27AE60", "#2980B9", "#C0392B"],
          borderWidth: 3,
          hoverBorderWidth: 4,
          hoverOffset: 8,
        },
      ],
    };
  }, [sentimentData]);

  const trendChartData = useMemo(() => {
    if (!sentimentData?.trend) return null;

    return {
      labels: sentimentData.trend.map((item) =>
        new Date(item.date).toLocaleDateString()
      ),
      datasets: [
        {
          label: "Positive Sentiment",
          data: sentimentData.trend.map((item) => item.positive),
          borderColor: "#2ECC71",
          backgroundColor: "rgba(46, 204, 113, 0.1)",
          tension: 0.4,
          fill: true,
        },
        {
          label: "Negative Sentiment",
          data: sentimentData.trend.map((item) => item.negative),
          borderColor: "#E74C3C",
          backgroundColor: "rgba(231, 76, 60, 0.1)",
          tension: 0.4,
          fill: true,
        },
      ],
    };
  }, [sentimentData]);

  const riskMetricsChartData = useMemo(() => {
    if (!riskMetrics) return null;

    return {
      labels: [
        "Volatility",
        "Sharpe Ratio",
        "VaR (95%)",
        "Max Drawdown",
        "Beta",
      ],
      datasets: [
        {
          label: "Risk Metrics",
          data: [
            riskMetrics.volatility * 100,
            (riskMetrics.sharpeRatio + 2) * 10, // Normalized for display
            riskMetrics.var95 * 100,
            riskMetrics.maxDrawdown * 100,
            riskMetrics.beta * 20, // Normalized for display
          ],
          backgroundColor: [
            "rgba(255, 99, 132, 0.8)",
            "rgba(54, 162, 235, 0.8)",
            "rgba(255, 205, 86, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(153, 102, 255, 0.8)",
          ],
          borderColor: [
            "rgba(255, 99, 132, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 205, 86, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(153, 102, 255, 1)",
          ],
          borderWidth: 2,
        },
      ],
    };
  }, [riskMetrics]);

  // Chart options
  const commonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          padding: 20,
          font: { size: 12 },
          usePointStyle: true,
        },
      },
      tooltip: {
        backgroundColor: "rgba(0, 0, 0, 0.8)",
        titleColor: "white",
        bodyColor: "white",
        borderColor: "rgba(255, 255, 255, 0.2)",
        borderWidth: 1,
      },
    },
    animation: {
      duration: 1000,
      easing: "easeOutQuart",
    },
  };

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  const getSentimentIndicator = () => {
    if (!sentimentData)
      return { emoji: "📊", color: "#95A5A6", text: "No Data" };

    const { positive, negative, marketCondition } = sentimentData;

    if (positive > negative + 15) {
      return {
        emoji: "🚀",
        color: "#27AE60",
        text: `Very Bullish (${marketCondition})`,
      };
    } else if (positive > negative + 5) {
      return {
        emoji: "📈",
        color: "#2ECC71",
        text: `Bullish (${marketCondition})`,
      };
    } else if (negative > positive + 15) {
      return {
        emoji: "📉",
        color: "#C0392B",
        text: `Very Bearish (${marketCondition})`,
      };
    } else if (negative > positive + 5) {
      return {
        emoji: "📊",
        color: "#E74C3C",
        text: `Bearish (${marketCondition})`,
      };
    } else {
      return {
        emoji: "⚖️",
        color: "#3498DB",
        text: `Neutral (${marketCondition})`,
      };
    }
  };

  const formatNumber = (number, decimals = 1) => {
    return Number(number).toLocaleString("en-IN", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  const getTimeSinceLastUpdate = () => {
    if (!lastRefresh) return "Never";
    const now = new Date();
    const diff = now - lastRefresh;
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return "Just now";
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
  };

  const indicator = getSentimentIndicator();

  // ============================================================================
  // RENDER COMPONENT
  // ============================================================================

  return (
    <div className="sentiment-analysis-container">
      {/* Header Section */}
      <div className="sentiment-header">
        <div className="header-content">
          <h2>📊 Advanced Sentiment Analysis</h2>
          <p className="sentiment-subtitle">
            AI-powered sentiment analysis with real-time risk assessment
          </p>

          {ticker && (
            <div className="ticker-info">
              <span className="ticker-symbol">{ticker}</span>
              <div
                className="sentiment-indicator"
                style={{ color: indicator.color }}
              >
                <span className="indicator-emoji">{indicator.emoji}</span>
                <span className="indicator-text">{indicator.text}</span>
              </div>
            </div>
          )}
        </div>

        <div className="header-controls">
          <div className="auto-refresh-toggle">
            <label>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
          </div>

          <button
            className="refresh-btn"
            onClick={fetchSentimentData}
            disabled={loading}
          >
            {loading ? "🔄" : "↻"} Refresh
          </button>

          <div className="last-update">
            Last updated: {getTimeSinceLastUpdate()}
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Analyzing market sentiment...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="error-container">
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            <div>
              <strong>Error:</strong> {error}
              <button className="retry-btn" onClick={fetchSentimentData}>
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      {sentimentData && !loading && (
        <>
          {/* Sentiment Overview */}
          <div className="sentiment-overview">
            <div className="overview-stats">
              <div className="stat-card">
                <h4>Social Mentions</h4>
                <p className="stat-value">
                  {formatNumber(sentimentData.socialMentions, 0)}
                </p>
              </div>
              <div className="stat-card">
                <h4>News Articles</h4>
                <p className="stat-value">
                  {formatNumber(sentimentData.newsArticles, 0)}
                </p>
              </div>
              <div className="stat-card">
                <h4>Confidence Score</h4>
                <p className="stat-value">{sentimentData.confidence}%</p>
              </div>
              <div className="stat-card">
                <h4>Data Sources</h4>
                <p className="stat-value">{sentimentData.totalSources}</p>
              </div>
            </div>
          </div>

          {/* Charts Section */}
          <div className="charts-grid">
            {/* Sentiment Distribution */}
            <div className="chart-card">
              <h3>Sentiment Distribution</h3>
              <div className="chart-container">
                <Doughnut
                  data={sentimentChartData}
                  options={commonChartOptions}
                />
              </div>
            </div>

            {/* Sentiment Trend */}
            {trendChartData && (
              <div className="chart-card">
                <h3>7-Day Sentiment Trend</h3>
                <div className="chart-container">
                  <Line data={trendChartData} options={commonChartOptions} />
                </div>
              </div>
            )}

            {/* Risk Metrics */}
            {riskMetricsChartData && (
              <div className="chart-card">
                <h3>Risk Metrics Overview</h3>
                <div className="chart-container">
                  <Bar
                    data={riskMetricsChartData}
                    options={commonChartOptions}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Risk Analysis Section */}
          {riskMetrics && riskExposure && (
            <div className="risk-analysis">
              <h2>⚖️ Advanced Risk Assessment</h2>

              <div className="risk-metrics-grid">
                <div className="metric-card">
                  <h4>📊 Volatility</h4>
                  <p className="metric-value">
                    {formatNumber(riskMetrics.volatility * 100)}%
                  </p>
                  <p className="metric-desc">Annualized price volatility</p>
                </div>

                <div className="metric-card">
                  <h4>⚡ Sharpe Ratio</h4>
                  <p className="metric-value">
                    {formatNumber(riskMetrics.sharpeRatio, 2)}
                  </p>
                  <p className="metric-desc">Risk-adjusted returns</p>
                </div>

                <div className="metric-card">
                  <h4>📉 Value at Risk</h4>
                  <p className="metric-value">
                    {formatNumber(riskMetrics.var95 * 100)}%
                  </p>
                  <p className="metric-desc">95% confidence interval</p>
                </div>

                <div className="metric-card">
                  <h4>📈 Beta</h4>
                  <p className="metric-value">
                    {formatNumber(riskMetrics.beta, 2)}
                  </p>
                  <p className="metric-desc">Market correlation</p>
                </div>

                <div className="metric-card">
                  <h4>🎯 Expected Return</h4>
                  <p className="metric-value">
                    {formatNumber(riskMetrics.expectedReturn)}%
                  </p>
                  <p className="metric-desc">Sentiment-based projection</p>
                </div>

                <div className="metric-card">
                  <h4>⚠️ Risk Level</h4>
                  <p className={`metric-value risk-${riskMetrics.riskLevel}`}>
                    {riskMetrics.riskLevel.toUpperCase().replace("-", " ")}
                  </p>
                  <p className="metric-desc">Overall risk assessment</p>
                </div>
              </div>

              {/* Portfolio Recommendation */}
              <div className="portfolio-recommendation">
                <h3>💼 Investment Strategy Recommendation</h3>
                <div className="recommendation-card">
                  <div className="rec-header">
                    <span className="strategy-badge">
                      {riskExposure.recommendation.strategy}
                    </span>
                    <span className="confidence-badge">
                      {riskExposure.recommendation.confidence}% Confidence
                    </span>
                  </div>

                  <div className="rec-details-grid">
                    <div className="rec-detail">
                      <strong>Action:</strong>
                      <p>{riskExposure.recommendation.action}</p>
                    </div>
                    <div className="rec-detail">
                      <strong>Risk Tolerance:</strong>
                      <p>{riskExposure.recommendation.riskTolerance}</p>
                    </div>
                    <div className="rec-detail">
                      <strong>Time Horizon:</strong>
                      <p>{riskExposure.recommendation.timeHorizon}</p>
                    </div>
                    <div className="rec-detail">
                      <strong>Suggested Allocation:</strong>
                      <p>{riskExposure.recommendation.allocation}</p>
                    </div>
                    <div className="rec-detail">
                      <strong>Market Outlook:</strong>
                      <p>{riskExposure.recommendation.marketOutlook}</p>
                    </div>
                    <div className="rec-detail">
                      <strong>Expected Return:</strong>
                      <p>{riskExposure.recommendation.expectedReturn}%</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* News Context */}
          <div className="news-context">
            <h3>📰 Recent Market Context</h3>
            <div className="news-items">
              {sentimentData.newsContext.map((item, index) => (
                <div key={index} className="news-item">
                  <span className="news-bullet">•</span>
                  <span className="news-text">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// PropTypes for better development experience
SentimentAnalysis.propTypes = {
  ticker: PropTypes.string.isRequired,
  apiKey: PropTypes.string,
  refreshInterval: PropTypes.number,
};

export default SentimentAnalysis;
