import React from "react";
import "../CSS/LoadingSpinner.css";

const LoadingSpinner = () => {
  return (
    <div className="loading-spinner-container">
      <div className="loading-content">
        <div className="spinner">
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
          <div className="spinner-ring"></div>
        </div>

        <div className="loading-text">
          <h3>🤖 AI is Analyzing...</h3>
          <p>Fetching stock data and generating predictions</p>

          <div className="loading-steps">
            <div className="step active">
              <span className="step-icon">📊</span>
              <span className="step-text">Fetching Stock Data</span>
            </div>
            <div className="step active">
              <span className="step-icon">🔍</span>
              <span className="step-text">Analyzing Technical Indicators</span>
            </div>
            <div className="step active">
              <span className="step-icon">🧠</span>
              <span className="step-text">Running ML Prediction</span>
            </div>
            <div className="step">
              <span className="step-icon">✨</span>
              <span className="step-text">Generating Results</span>
            </div>
          </div>
        </div>
      </div>

      <div className="loading-progress">
        <div className="progress-bar">
          <div className="progress-fill"></div>
        </div>
        <p className="progress-text">
          Please wait while we process your request...
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
