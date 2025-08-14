import React from "react";
import "../CSS/PredictionResult.css";

const PredictionResult = ({
  predictions,
  analysis,
  marketInfo,
  ticker,
  selectedTimeframes,
}) => {
  if (!predictions || Object.keys(predictions).length === 0) {
    return null;
  }

  // Get currency symbol based on ticker
  const getCurrencySymbol = (tickerSymbol) => {
    if (tickerSymbol.includes(".NS") || tickerSymbol.includes(".BO"))
      return "₹";
    if (tickerSymbol.includes("-USD") || tickerSymbol.includes("USD"))
      return "$";
    if (tickerSymbol.includes(".L")) return "£";
    if (tickerSymbol.includes(".EU")) return "€";
    return "$";
  };

  const getCompanyName = (tickerSymbol) => {
    const companyNames = {
      // US Stocks
      AAPL: "Apple Inc.",
      GOOGL: "Alphabet Inc. (Google)",
      GOOG: "Alphabet Inc. (Google)",
      MSFT: "Microsoft Corporation",
      AMZN: "Amazon.com Inc.",
      META: "Meta Platforms Inc. (Facebook)",
      TSLA: "Tesla Inc.",
      NVDA: "NVIDIA Corporation",
      JPM: "JPMorgan Chase & Co.",
      V: "Visa Inc.",
      JNJ: "Johnson & Johnson",
      WMT: "Walmart Inc.",
      PG: "Procter & Gamble Company",
      UNH: "UnitedHealth Group Inc.",
      NFLX: "Netflix Inc.",
      DIS: "The Walt Disney Company",
      PYPL: "PayPal Holdings Inc.",
      INTC: "Intel Corporation",
      AMD: "Advanced Micro Devices Inc.",
      CRM: "Salesforce Inc.",
      ORCL: "Oracle Corporation",
      ADBE: "Adobe Inc.",
      IBM: "International Business Machines Corporation",
      CSCO: "Cisco Systems Inc.",
      AVGO: "Broadcom Inc.",
      BAC: "Bank of America Corporation",
      WFC: "Wells Fargo & Company",
      GS: "The Goldman Sachs Group Inc.",
      MS: "Morgan Stanley",
      C: "Citigroup Inc.",
      MA: "Mastercard Incorporated",
      AXP: "American Express Company",
      PFE: "Pfizer Inc.",
      KO: "The Coca-Cola Company",
      PEP: "PepsiCo Inc.",
      NKE: "NIKE Inc.",
      SBUX: "Starbucks Corporation",
      HD: "The Home Depot Inc.",
      COST: "Costco Wholesale Corporation",
      TGT: "Target Corporation",
      LOW: "Lowe's Companies Inc.",
      EBAY: "eBay Inc.",
      BA: "The Boeing Company",
      CAT: "Caterpillar Inc.",
      DE: "Deere & Company",
      GE: "General Electric Company",
      MMM: "3M Company",
      HON: "Honeywell International Inc.",
      XOM: "Exxon Mobil Corporation",
      CVX: "Chevron Corporation",
      SPY: "SPDR S&P 500 ETF Trust",
      QQQ: "Invesco QQQ Trust ETF",

      // Indian Stocks
      "RELIANCE.NS": "Reliance Industries Limited",
      "TCS.NS": "Tata Consultancy Services Limited",
      "HDFCBANK.NS": "HDFC Bank Limited",
      "INFY.NS": "Infosys Limited",
      "HINDUNILVR.NS": "Hindustan Unilever Limited",
      "ITC.NS": "ITC Limited",
      "SBIN.NS": "State Bank of India",
      "BHARTIARTL.NS": "Bharti Airtel Limited",
      "KOTAKBANK.NS": "Kotak Mahindra Bank Limited",
      "LT.NS": "Larsen & Toubro Limited",
      "HCLTECH.NS": "HCL Technologies Limited",
      "WIPRO.NS": "Wipro Limited",
      "MARUTI.NS": "Maruti Suzuki India Limited",
      "BAJFINANCE.NS": "Bajaj Finance Limited",
      "ASIANPAINT.NS": "Asian Paints Limited",
      "NESTLEIND.NS": "Nestlé India Limited",
      "ULTRACEMCO.NS": "UltraTech Cement Limited",
      "TITAN.NS": "Titan Company Limited",
      "AXISBANK.NS": "Axis Bank Limited",
      "SUNPHARMA.NS": "Sun Pharmaceutical Industries Limited",
      "^NSEI": "NIFTY 50 Index",
      "^BSESN": "BSE SENSEX Index",

      // Crypto and Others
      "BTC-USD": "Bitcoin",
      "ETH-USD": "Ethereum",
      COIN: "Coinbase Global Inc.",
      SQ: "Block Inc. (Square)",
      HOOD: "Robinhood Markets Inc.",
      PLTR: "Palantir Technologies Inc.",
      GME: "GameStop Corp.",
      AMC: "AMC Entertainment Holdings Inc.",
      BB: "BlackBerry Limited",
      NOK: "Nokia Corporation",
      SPCE: "Virgin Galactic Holdings Inc.",
      SOFI: "SoFi Technologies Inc.",
      UPST: "Upstart Holdings Inc.",
      AFRM: "Affirm Holdings Inc.",
      DKNG: "DraftKings Inc.",
      UBER: "Uber Technologies Inc.",
      LYFT: "Lyft Inc.",
      DASH: "DoorDash Inc.",
      ABNB: "Airbnb Inc.",
      SNOW: "Snowflake Inc.",
      ZM: "Zoom Video Communications Inc.",
      DOCU: "DocuSign Inc.",
      TWLO: "Twilio Inc.",
      OKTA: "Okta Inc.",

      // Additional stocks I suggested
      RIVN: "Rivian Automotive Inc.",
      LCID: "Lucid Group Inc.",
      NIO: "NIO Inc.",
      XPEV: "XPeng Inc.",
      LI: "Li Auto Inc.",
      F: "Ford Motor Company",
      GM: "General Motors Company",
      BABA: "Alibaba Group Holding Limited",
      JD: "JD.com Inc.",
      PDD: "Pinduoduo Inc.",
      TSM: "Taiwan Semiconductor Manufacturing Company",
      ASML: "ASML Holding N.V.",
      SAP: "SAP SE",
      TM: "Toyota Motor Corporation",
      SONY: "Sony Group Corporation",
      "BNB-USD": "Binance Coin",
      "ADA-USD": "Cardano",
      "SOL-USD": "Solana",
      "DOT-USD": "Polkadot",
    };

    return companyNames[tickerSymbol.toUpperCase()] || tickerSymbol;
  };

  const getDirectionIcon = (direction) => {
    return direction === "UP" ? "📈" : "📉";
  };

  const getDirectionColor = (direction) => {
    return direction === "UP" ? "green" : "red";
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 80) return "high";
    if (confidence >= 60) return "medium";
    return "low";
  };

  const formatTimeframeName = (timeframe) => {
    const names = {
      "1d": "1 Day",
      "1w": "1 Week",
      "1mo": "1 Month",
      "1y": "1 Year",
    };
    return names[timeframe] || timeframe;
  };

  // Filter predictions if selectedTimeframes provided
  const timeframeOrder = ["1d", "1w", "1mo", "1y"];
  const allPredictions = Object.entries(predictions);
  const filteredPredictions =
    selectedTimeframes && selectedTimeframes.length > 0
      ? allPredictions.filter(([tf]) => selectedTimeframes.includes(tf))
      : allPredictions;

  const sortedPredictions = filteredPredictions.sort(
    ([a], [b]) => timeframeOrder.indexOf(a) - timeframeOrder.indexOf(b)
  );

  const currencySymbol = getCurrencySymbol(ticker);

  return (
    <div className="prediction-result">
      <div className="prediction-header">
        <h2>🎯 Predictions for {getCompanyName(ticker)}</h2>
        <div className="prediction-summary">
          <span>
            {sortedPredictions.length} timeframe
            {sortedPredictions.length !== 1 ? "s" : ""} analyzed
          </span>
          {marketInfo?.sector && (
            <span className="sector-badge"> • {marketInfo.sector}</span>
          )}
        </div>
      </div>

      <div className="predictions-grid">
        {sortedPredictions.map(([timeframe, prediction]) => (
          <div key={timeframe} className="prediction-card">
            <div className="card-header">
              <h3 className="timeframe-title">
                {formatTimeframeName(timeframe)}
              </h3>
              <div className={`model-badge ${prediction.model_type}`}>
                {prediction.model_type}
              </div>
            </div>

            <div className="prediction-main">
              <div
                className={`direction-indicator ${prediction.direction.toLowerCase()}`}
              >
                <span className="direction-icon">
                  {getDirectionIcon(prediction.direction)}
                </span>
                <span className="direction-text">{prediction.direction}</span>
              </div>

              <div className="confidence-section">
                <div className="confidence-bar-container">
                  <div className="confidence-label">
                    Confidence: {prediction.confidence}%
                  </div>
                  <div className="confidence-bar">
                    <div
                      className={`confidence-fill ${getConfidenceLevel(
                        prediction.confidence
                      )}`}
                      style={{ width: `${prediction.confidence}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="prediction-details">
              <div className="price-info">
                <div className="price-row">
                  <span className="label">Current Price:</span>
                  <span className="value">
                    {currencySymbol}
                    {prediction.current_price}
                  </span>
                </div>
                <div className="price-row">
                  <span className="label">Target Price:</span>
                  <span
                    className="value"
                    style={{ color: getDirectionColor(prediction.direction) }}
                  >
                    {currencySymbol}
                    {prediction.price_target}
                  </span>
                </div>
                <div className="price-row">
                  <span className="label">Expected Return:</span>
                  <span
                    className={`value ${
                      prediction.expected_return >= 0 ? "positive" : "negative"
                    }`}
                  >
                    {prediction.expected_return > 0 ? "+" : ""}
                    {prediction.expected_return}%
                  </span>
                </div>
              </div>

              <div className="model-info">
                <div className="accuracy-badge">
                  Model Accuracy: {prediction.model_accuracy}%
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Overall Summary */}
      {analysis?.recommendation && (
        <div className="overall-summary">
          <h3>📊 Overall Assessment</h3>
          <div className="summary-content">
            <div
              className={`overall-recommendation ${analysis.recommendation.overall.toLowerCase()}`}
            >
              <strong>{analysis.recommendation.overall}</strong>
              <span className="overall-confidence">
                ({analysis.recommendation.confidence}% confidence)
              </span>
            </div>
            <div className="summary-details">
              <span>Risk: {analysis.recommendation.risk_level}</span>
              <span>•</span>
              <span>
                Strategy: {analysis.recommendation.holding_period} term
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Consensus View */}
      <div className="consensus-view">
        <h4>🎭 Consensus Across Timeframes</h4>
        <div className="consensus-bars">
          {(() => {
            const upCount = sortedPredictions.filter(
              ([, p]) => p.direction === "UP"
            ).length;
            const totalCount = sortedPredictions.length;
            const bullishPercent = (upCount / totalCount) * 100;

            return (
              <div className="consensus-container">
                <div className="consensus-label">
                  {upCount}/{totalCount} timeframes bullish (
                  {bullishPercent.toFixed(0)}%)
                </div>
                <div className="consensus-bar">
                  <div
                    className="bullish-portion"
                    style={{ width: `${bullishPercent}%` }}
                  ></div>
                  <div
                    className="bearish-portion"
                    style={{ width: `${100 - bullishPercent}%` }}
                  ></div>
                </div>
                <div className="consensus-legend">
                  <span className="bullish-legend">📈 Bullish</span>
                  <span className="bearish-legend">📉 Bearish</span>
                </div>
              </div>
            );
          })()}
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
