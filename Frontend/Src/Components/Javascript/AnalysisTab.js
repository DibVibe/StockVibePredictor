import React, { useState } from "react";
import MarketInfoPanel from "./MarketInfoPanel";
import TechnicalAnalysisPanel from "./TechnicalAnalysisPanel";
import SentimentPanel from "./SentimentPanel";
import RiskAssessmentPanel from "./RiskAssessmentPanel";

const AnalysisTab = ({ analysis, marketInfo, companyData, ticker }) => {
  const [expandedSections, setExpandedSections] = useState({
    marketInfo: true,
    technical: true,
    sentiment: true,
    risk: true,
  });

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // Check if we have any data to display
  const hasData = analysis || marketInfo || companyData;

  if (!hasData) {
    return (
      <div className="no-data-message">
        <div className="no-data-icon">🔍</div>
        <h3>No Analysis Data Available</h3>
        <p>Analysis data will appear here after running a prediction.</p>
      </div>
    );
  }

  // Extract sentiment data from analysis object
  const sentimentData =
    analysis?.sentiment || analysis?.sentiment_analysis || null;

  return (
    <div className="analysis-tab-container">
      {/* Market Information */}
      {(marketInfo || companyData) && (
        <MarketInfoPanel
          marketInfo={marketInfo}
          companyData={companyData}
          expanded={expandedSections.marketInfo}
          onToggle={() => toggleSection("marketInfo")}
        />
      )}

      {/* Technical Analysis */}
      {analysis && (
        <TechnicalAnalysisPanel
          analysis={analysis}
          expanded={expandedSections.technical}
          onToggle={() => toggleSection("technical")}
        />
      )}

      {/* Sentiment Analysis */}
      {sentimentData && (
        <SentimentPanel
          sentiment={sentimentData}
          expanded={expandedSections.sentiment}
          onToggle={() => toggleSection("sentiment")}
        />
      )}

      {/* Risk Assessment */}
      {(analysis?.risk_level || analysis?.recommendation?.risk_level) && (
        <RiskAssessmentPanel
          riskData={{
            risk_level:
              analysis.risk_level || analysis.recommendation?.risk_level,
            risk_factors: analysis.risk_factors,
            risk_mitigation: analysis.risk_mitigation,
          }}
          expanded={expandedSections.risk}
          onToggle={() => toggleSection("risk")}
        />
      )}

      {/* Analysis Summary */}
      <div className="analysis-summary">
        <h3>📝 Analysis Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <label>Ticker:</label>
            <span>{ticker}</span>
          </div>
          <div className="summary-item">
            <label>Analysis Date:</label>
            <span>{new Date().toLocaleDateString()}</span>
          </div>
          {(analysis?.confidence_level ||
            analysis?.recommendation?.confidence) && (
            <div className="summary-item">
              <label>Confidence:</label>
              <span
                className={`confidence-badge ${
                  (analysis?.confidence_level ||
                    analysis?.recommendation?.confidence / 100) > 0.7
                    ? "high"
                    : (analysis?.confidence_level ||
                        analysis?.recommendation?.confidence / 100) > 0.4
                    ? "medium"
                    : "low"
                }`}
              >
                {analysis?.confidence_level
                  ? (analysis.confidence_level * 100).toFixed(1)
                  : analysis?.recommendation?.confidence}
                %
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisTab;
