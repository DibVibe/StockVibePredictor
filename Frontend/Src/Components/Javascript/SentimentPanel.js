import React from "react";

const SentimentPanel = ({ sentiment, expanded, onToggle }) => {
  if (!sentiment) return null;

  // Extract sentiment data safely
  const getSentimentData = () => {
    // If sentiment is a string
    if (typeof sentiment === "string") {
      return {
        label: sentiment,
        score: null,
        factors: null,
      };
    }

    // If sentiment is an object with sentiment_label
    if (sentiment.sentiment_label) {
      return {
        label: sentiment.sentiment_label,
        score: sentiment.sentiment_score,
        factors: sentiment.sentiment_factors,
      };
    }

    // If sentiment is an object with label property
    if (sentiment.label) {
      return {
        label: sentiment.label,
        score: sentiment.score,
        factors: sentiment.factors,
      };
    }

    // Default case
    return {
      label: "Neutral",
      score: null,
      factors: null,
    };
  };

  const sentimentData = getSentimentData();

  return (
    <div className="sentiment-panel">
      <div className="panel-header" onClick={onToggle}>
        <h3>💭 Market Sentiment</h3>
        <span className="toggle-icon">{expanded ? "−" : "+"}</span>
      </div>

      {expanded && (
        <div className="sentiment-analysis">
          <div className={`sentiment ${sentimentData.label.toLowerCase()}`}>
            {sentimentData.label}
            {sentimentData.score !== null &&
              sentimentData.score !== undefined && (
                <span className="sentiment-score">
                  ({(sentimentData.score * 100).toFixed(1)}%)
                </span>
              )}
          </div>

          {sentimentData.factors && Array.isArray(sentimentData.factors) && (
            <div className="sentiment-factors">
              <h5>Key Sentiment Drivers:</h5>
              <ul>
                {sentimentData.factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SentimentPanel;
