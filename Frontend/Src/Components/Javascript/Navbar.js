import React, { useState, useEffect } from "react";
import "../CSS/Navbar.css";

const Navbar = ({
  currentTicker,
  currentCompanyName,
  systemHealth,
  onQuickSearch,
}) => {
  // ==================== STATE MANAGEMENT ====================
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [quickSearchValue, setQuickSearchValue] = useState("");

  // ==================== EFFECTS ====================
  // Handle scroll effect for navbar styling
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      setScrolled(isScrolled);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close mobile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isMobileMenuOpen && !event.target.closest(".navbar")) {
        setIsMobileMenuOpen(false);
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, [isMobileMenuOpen]);

  // ==================== HANDLERS ====================
  const handleQuickSearch = (e) => {
    e.preventDefault();
    if (quickSearchValue.trim() && onQuickSearch) {
      onQuickSearch(quickSearchValue.trim().toUpperCase());
      setQuickSearchValue("");
      setIsMobileMenuOpen(false);
    }
  };

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleActionClick = (action) => {
    setIsMobileMenuOpen(false);
  };

  // ==================== HELPER FUNCTIONS ====================
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case "healthy":
        return "#4ade80";
      case "warning":
        return "#f59e0b";
      case "degraded":
        return "#f97316";
      case "unhealthy":
        return "#ef4444";
      default:
        return "#6b7280";
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case "healthy":
        return "✅";
      case "warning":
        return "⚠️";
      case "degraded":
        return "🟡";
      case "unhealthy":
        return "❌";
      default:
        return "❓";
    }
  };

  // Format company display name
  const getDisplayName = () => {
    if (!currentTicker) return "";

    if (currentCompanyName && currentCompanyName !== currentTicker) {
      return `${currentCompanyName} (${currentTicker})`;
    }

    return currentTicker;
  };

  // ==================== RENDER ====================
  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <div className="navbar-container">
        {/* ==================== BRAND SECTION - LEFTMOST ==================== */}
        <div className="navbar-brand">
          <div className="brand-logo">
            <span className="logo-text">StockVibePredictor</span>
          </div>
        </div>

        {/* ==================== CENTER SECTION - COMPANY INFO ==================== */}
        <div className="navbar-center">
          {currentTicker && (
            <div className="current-ticker">
              <div className="ticker-info">
                <div className="ticker-display">
                  <span
                    className="company-display-name"
                    title={getDisplayName()}
                  >
                    {getDisplayName()}
                  </span>
                  <div className="ticker-pulse"></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ==================== RIGHT SECTION - SEARCH & ACTIONS ==================== */}
        <div className="navbar-right">
          {/* Quick Search */}
          <div className="navbar-search">
            <form onSubmit={handleQuickSearch} className="search-form">
              <div className="search-input-group">
                <input
                  type="text"
                  placeholder="Search ticker..."
                  value={quickSearchValue}
                  onChange={(e) =>
                    setQuickSearchValue(e.target.value.toUpperCase())
                  }
                  className="search-input"
                  maxLength={10}
                />
                <button type="submit" className="search-btn" title="Search">
                  🔍
                </button>
              </div>
            </form>
          </div>

          {/* Actions Section */}
          <div className="navbar-actions">
            {/* System Status Indicator */}
            <div
              className="status-indicator"
              title={`System Status: ${systemHealth?.status || "Unknown"}`}
            >
              <div
                className="status-dot"
                style={{
                  backgroundColor: getStatusColor(systemHealth?.status),
                }}
              ></div>
              <span className="status-text">
                {systemHealth?.status || "Unknown"}
              </span>
              <span className="status-icon">
                {getStatusIcon(systemHealth?.status)}
              </span>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons">
              <button
                className="action-btn"
                title="Help & Documentation"
                onClick={() => handleActionClick("help")}
              >
                ❓
              </button>
              <button
                className="action-btn"
                title="Settings"
                onClick={() => handleActionClick("settings")}
              >
                ⚙️
              </button>
            </div>

            {/* Mobile Menu Toggle */}
            <button
              className={`mobile-menu-toggle ${
                isMobileMenuOpen ? "active" : ""
              }`}
              onClick={toggleMobileMenu}
              title="Menu"
            >
              <span></span>
              <span></span>
              <span></span>
            </button>
          </div>
        </div>
      </div>

      {/* ==================== MOBILE NAVIGATION MENU ==================== */}
      <div className={`mobile-nav ${isMobileMenuOpen ? "active" : ""}`}>
        <div className="mobile-nav-content">
          {/* Mobile Search */}
          <div className="mobile-search">
            <form onSubmit={handleQuickSearch} className="search-form">
              <div className="search-input-group">
                <input
                  type="text"
                  placeholder="Search ticker..."
                  value={quickSearchValue}
                  onChange={(e) =>
                    setQuickSearchValue(e.target.value.toUpperCase())
                  }
                  className="search-input"
                  maxLength={10}
                />
                <button
                  type="submit"
                  className={`search-btn ${
                    !quickSearchValue.trim() ? "disabled" : ""
                  }`}
                  disabled={!quickSearchValue.trim()}
                >
                  🔍
                </button>
              </div>
            </form>
          </div>

          {/* Mobile Current Stock */}
          {currentTicker && (
            <div className="mobile-current-stock">
              <div className="mobile-ticker-info">
                <div className="mobile-ticker-display">
                  <span className="mobile-company-display-name">
                    {getDisplayName()}
                  </span>
                  <div className="mobile-ticker-pulse"></div>
                </div>
              </div>
            </div>
          )}

          {/* Mobile Action Buttons */}
          <div className="mobile-actions">
            <button
              className="mobile-action-btn"
              onClick={() => handleActionClick("help")}
            >
              <span className="mobile-action-icon">❓</span>
              <span className="mobile-action-text">Help & Documentation</span>
            </button>

            <button
              className="mobile-action-btn"
              onClick={() => handleActionClick("settings")}
            >
              <span className="mobile-action-icon">⚙️</span>
              <span className="mobile-action-text">Settings</span>
            </button>
          </div>

          {/* Mobile Divider */}
          <div className="mobile-nav-divider"></div>

          {/* Mobile Footer with System Status */}
          <div className="mobile-nav-footer">
            <div className="mobile-status">
              <span className="mobile-status-label">System Status:</span>
              <div className="mobile-status-info">
                <span
                  className="mobile-status-badge"
                  style={{
                    backgroundColor: getStatusColor(systemHealth?.status),
                  }}
                >
                  {getStatusIcon(systemHealth?.status)}
                </span>
                <span className="mobile-status-text">
                  {systemHealth?.status || "Unknown"}
                </span>
              </div>
            </div>

            {/* Additional System Info */}
            {systemHealth?.metrics && (
              <div className="mobile-system-info">
                <span className="mobile-info-text">
                  Models: {systemHealth.metrics.model_cache_size || 0}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Overlay */}
      {isMobileMenuOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setIsMobileMenuOpen(false)}
        ></div>
      )}
    </nav>
  );
};

export default Navbar;
