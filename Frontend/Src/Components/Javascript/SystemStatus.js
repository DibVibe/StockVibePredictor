import React from "react";

const SystemStatus = ({ systemHealth }) => {
  const getStatus = () => {
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

  const status = getStatus();

  return (
    <div className="system-status">
      <span
        className="status-indicator"
        style={{ color: status.color }}
        title={
          systemHealth?.services
            ? JSON.stringify(systemHealth.services, null, 2)
            : "System status"
        }
      >
        {status.text}
      </span>
      {systemHealth?.metrics && (
        <span className="system-metrics">
          | {systemHealth.metrics.model_cache_size || 0} models loaded
        </span>
      )}
    </div>
  );
};

export default SystemStatus;
