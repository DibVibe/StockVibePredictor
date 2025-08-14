import React from "react";

const TechnicalAnalysisPanel = ({ analysis, expanded, onToggle }) => {
  if (!analysis) return null;

  // Extract recommendation text safely
  const getRecommendationText = () => {
    if (typeof analysis.recommendation === "string") {
      return analysis.recommendation;
    }
    if (analysis.recommendation?.overall) {
      return analysis.recommendation.overall;
    }
    return null;
  };

  const recommendationText = getRecommendationText();

  return (
    <div className="analysis-panel">
      <div className="panel-header" onClick={onToggle}>
        <h3>🔍 Technical Analysis</h3>
        <span className="toggle-icon">{expanded ? "−" : "+"}</span>
      </div>

      {expanded && (
        <>
          {/* Technical Indicators */}
          {(analysis.technical_indicators || analysis.technical) && (
            <div className="technical-analysis">
              <h4>Technical Indicators</h4>
              <div className="indicators">
                {/* RSI Indicator */}
                {(analysis.technical?.rsi ||
                  analysis.technical_indicators?.rsi !== undefined) && (
                  <div className="indicator">
                    <label>RSI:</label>
                    <span
                      className={`rsi-value ${
                        analysis.technical?.rsi_signal ||
                        (analysis.technical_indicators?.rsi > 70
                          ? "overbought"
                          : analysis.technical_indicators?.rsi < 30
                          ? "oversold"
                          : "neutral")
                      }`}
                    >
                      {analysis.technical?.rsi ||
                        analysis.technical_indicators?.rsi?.toFixed(2)}
                      {analysis.technical?.rsi_signal && (
                        <span className="indicator-hint">
                          {" "}
                          ({analysis.technical.rsi_signal})
                        </span>
                      )}
                    </span>
                  </div>
                )}

                {/* Trend */}
                {(analysis.technical?.trend || analysis.trend) && (
                  <div className="indicator">
                    <label>Trend:</label>
                    <span
                      className={`trend ${(
                        analysis.technical?.trend || analysis.trend
                      ).toLowerCase()}`}
                    >
                      {analysis.technical?.trend || analysis.trend}
                    </span>
                  </div>
                )}

                {/* Volume Trend */}
                {analysis.technical?.volume_trend && (
                  <div className="indicator">
                    <label>Volume:</label>
                    <span>{analysis.technical.volume_trend}</span>
                  </div>
                )}

                {/* Volatility */}
                {analysis.technical?.volatility_regime && (
                  <div className="indicator">
                    <label>Volatility:</label>
                    <span>{analysis.technical.volatility_regime}</span>
                  </div>
                )}

                {/* MACD */}
                {analysis.technical_indicators?.macd !== undefined && (
                  <div className="indicator">
                    <label>MACD:</label>
                    <span
                      className={`trend ${
                        analysis.technical_indicators.macd > 0
                          ? "bullish"
                          : "bearish"
                      }`}
                    >
                      {analysis.technical_indicators.macd.toFixed(4)}
                    </span>
                  </div>
                )}

                {/* SMA indicators */}
                {analysis.technical_indicators?.sma_20 && (
                  <div className="indicator">
                    <label>SMA (20):</label>
                    <span>
                      ${analysis.technical_indicators.sma_20.toFixed(2)}
                    </span>
                  </div>
                )}

                {analysis.technical_indicators?.sma_50 && (
                  <div className="indicator">
                    <label>SMA (50):</label>
                    <span>
                      ${analysis.technical_indicators.sma_50.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recommendation Section */}
          {(recommendationText || analysis.recommendation) && (
            <div className="recommendation">
              <h4>📋 Recommendation</h4>
              {recommendationText && (
                <div
                  className={`rec-badge ${recommendationText.toLowerCase()}`}
                >
                  {recommendationText}
                </div>
              )}

              {/* Handle recommendation details */}
              {analysis.recommendation?.confidence && (
                <div className="rec-details">
                  <p>Confidence: {analysis.recommendation.confidence}%</p>
                  {analysis.recommendation.risk_level && (
                    <p>Risk Level: {analysis.recommendation.risk_level}</p>
                  )}
                  {analysis.recommendation.holding_period && (
                    <p>
                      Suggested Holding:{" "}
                      {analysis.recommendation.holding_period}
                    </p>
                  )}
                </div>
              )}

              {/* Handle alternative format */}
              {analysis.recommendation_details && (
                <div className="rec-details">
                  {Object.entries(analysis.recommendation_details).map(
                    ([key, value]) => (
                      <p key={key}>
                        <strong>{key.replace(/_/g, " ").toUpperCase()}:</strong>{" "}
                        {value}
                      </p>
                    )
                  )}
                </div>
              )}
            </div>
          )}

          {/* Support and Resistance */}
          {analysis.support_resistance && (
            <div className="support-resistance">
              <h4>📊 Support & Resistance Levels</h4>
              <div className="levels-grid">
                {analysis.support_resistance.support && (
                  <div className="level-item support">
                    <label>Support:</label>
                    <span>
                      ${analysis.support_resistance.support.toFixed(2)}
                    </span>
                  </div>
                )}
                {analysis.support_resistance.resistance && (
                  <div className="level-item resistance">
                    <label>Resistance:</label>
                    <span>
                      ${analysis.support_resistance.resistance.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default TechnicalAnalysisPanel;
