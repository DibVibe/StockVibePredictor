import React from "react";

const RiskAssessmentPanel = ({ riskData, expanded, onToggle }) => {
  if (!riskData || !riskData.risk_level) return null;

  // Normalize risk level to string
  const getRiskLevel = () => {
    if (typeof riskData.risk_level === "string") {
      return riskData.risk_level;
    }
    if (typeof riskData.risk_level === "number") {
      if (riskData.risk_level < 0.33) return "Low";
      if (riskData.risk_level < 0.66) return "Medium";
      return "High";
    }
    return "Unknown";
  };

  const riskLevel = getRiskLevel();

  return (
    <div className="risk-panel">
      <div className="panel-header" onClick={onToggle}>
        <h3>⚠️ Risk Assessment</h3>
        <span className="toggle-icon">{expanded ? "−" : "+"}</span>
      </div>

      {expanded && (
        <div className="risk-assessment">
          <div className={`risk-badge ${riskLevel.toLowerCase()}`}>
            {riskLevel} Risk
          </div>

          {riskData.risk_factors && Array.isArray(riskData.risk_factors) && (
            <div className="risk-factors">
              <h5>Key Risk Factors:</h5>
              <ul>
                {riskData.risk_factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}

          {riskData.risk_mitigation &&
            Array.isArray(riskData.risk_mitigation) && (
              <div className="risk-mitigation">
                <h5>Risk Mitigation Strategies:</h5>
                <ul>
                  {riskData.risk_mitigation.map((strategy, index) => (
                    <li key={index}>{strategy}</li>
                  ))}
                </ul>
              </div>
            )}
        </div>
      )}
    </div>
  );
};

export default RiskAssessmentPanel;
