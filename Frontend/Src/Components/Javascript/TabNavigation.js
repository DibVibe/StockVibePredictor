import React from "react";

const TabNavigation = ({ tabs, activeTab, onTabChange }) => {
  return (
    <div className="tab-header">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          className={`tab-btn ${activeTab === tab.id ? "active" : ""} ${
            tab.disabled ? "disabled" : ""
          }`}
          onClick={() => !tab.disabled && onTabChange(tab.id)}
          data-tooltip={tab.tooltip}
          disabled={tab.disabled}
        >
          <span>{tab.icon}</span> {tab.label}
          {tab.badge && <span className="badge">{tab.badge}</span>}
        </button>
      ))}
    </div>
  );
};

export default TabNavigation;
