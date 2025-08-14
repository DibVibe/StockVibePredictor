import React from "react";
import PredictionResult from "./PredictionResult";

const PredictionsTab = ({ predictions, ticker, selectedTimeframes }) => {
  if (!predictions) {
    return (
      <div className="no-data-message">
        <div className="no-data-icon">📊</div>
        <h3>No Predictions Available</h3>
        <p>Please run a prediction to see results here.</p>
      </div>
    );
  }

  return (
    <div className="predictions-tab-container">
      <PredictionResult
        predictions={predictions}
        ticker={ticker}
        selectedTimeframes={selectedTimeframes}
      />
    </div>
  );
};

export default PredictionsTab;
