import React from "react";

const TimeframeSelector = ({
  timeframes,
  selectedTimeframes,
  onTimeframeChange,
  loading,
}) => {
  return (
    <div className="timeframe-selector">
      <h3>Select Prediction Timeframes:</h3>
      <div className="timeframe-buttons">
        {Object.entries(timeframes).map(([key, label]) => (
          <button
            key={key}
            className={`timeframe-btn ${
              selectedTimeframes.includes(key) ? "active" : ""
            }`}
            onClick={() => onTimeframeChange(key)}
            disabled={loading}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="selected-timeframes">
        Selected: {selectedTimeframes.map((tf) => timeframes[tf]).join(", ")}
      </p>
    </div>
  );
};

export default TimeframeSelector;
