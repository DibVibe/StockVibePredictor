import React, { useState, useEffect, useRef } from "react";
import "../CSS/StockInput.css";
import { searchStocks, getPopularStocks } from "../../Database/StockDatabase";

const StockInput = ({ onSubmit, loading }) => {
  const [ticker, setTicker] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [popularStocks] = useState(getPopularStocks());
  const inputRef = useRef(null);
  const dropdownRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (ticker.trim() && !loading) {
      onSubmit(ticker.trim().toUpperCase());
    }
  };

  const handleSuggestionClick = (suggestedTicker) => {
    setTicker(suggestedTicker);
    setShowDropdown(false);
    onSubmit(suggestedTicker);
  };

  // Search functionality
  useEffect(() => {
    if (ticker.length >= 1) {
      const results = searchStocks(ticker, 8);
      setSearchResults(results);
      setShowDropdown(results.length > 0);
      setSelectedIndex(-1);
    } else {
      setSearchResults([]);
      setShowDropdown(false);
      setSelectedIndex(-1);
    }
  }, [ticker]);

  // Handle keyboard navigation
  const handleKeyDown = (e) => {
    if (!showDropdown || searchResults.length === 0) return;

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < searchResults.length - 1 ? prev + 1 : 0
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : searchResults.length - 1
        );
        break;
      case "Enter":
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < searchResults.length) {
          const selectedStock = searchResults[selectedIndex];
          handleSuggestionClick(selectedStock.symbol);
        } else if (ticker.trim()) {
          handleSubmit(e);
        }
        break;
      case "Escape":
        setShowDropdown(false);
        setSelectedIndex(-1);
        inputRef.current?.blur();
        break;
      default:
        break;
    }
  };

  // Handle input changes
  const handleInputChange = (e) => {
    const value = e.target.value.toUpperCase();
    setTicker(value);
  };

  // Handle input focus
  const handleInputFocus = () => {
    if (ticker.length >= 1 && searchResults.length > 0) {
      setShowDropdown(true);
    }
  };

  // Handle clicking outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        inputRef.current &&
        !inputRef.current.contains(event.target)
      ) {
        setShowDropdown(false);
        setSelectedIndex(-1);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Handle dropdown item click
  const handleDropdownItemClick = (stock) => {
    setTicker(stock.symbol);
    setShowDropdown(false);
    setSelectedIndex(-1);
    onSubmit(stock.symbol);
  };

  return (
    <div className="stock-input-container">
      <form onSubmit={handleSubmit} className="stock-form">
        <div className="input-group">
          <div className="search-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={ticker}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              onFocus={handleInputFocus}
              placeholder="Search stocks by name or symbol (e.g., Apple, AAPL)"
              className="stock-input"
              disabled={loading}
              maxLength={10}
              autoComplete="off"
            />
            {showDropdown && searchResults.length > 0 && (
              <div ref={dropdownRef} className="search-dropdown">
                {searchResults.map((stock, index) => (
                  <div
                    key={`${stock.symbol}-${index}`}
                    className={`dropdown-item ${
                      selectedIndex === index ? "selected" : ""
                    }`}
                    onClick={() => handleDropdownItemClick(stock)}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <div className="stock-info">
                      <span className="stock-symbol">{stock.symbol}</span>
                      <span className="stock-name">{stock.name}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </form>

      <div className="suggestions">
        <p className="suggestions-label">Popular Stocks:</p>
        <div className="suggestions-grid">
          {popularStocks.map((stock) => (
            <button
              key={stock}
              onClick={() => handleSuggestionClick(stock)}
              className="suggestion-chip"
              disabled={loading}
            >
              {stock}
            </button>
          ))}
        </div>
      </div>

      <div className="info-text">
        <p>
          🔍 Search stocks by company name or symbol - we have thousands in our
          database!
        </p>
        <p>📈 Use ↑↓ arrow keys to navigate suggestions, Enter to select</p>
        <p>
          🤖 Get AI-powered predictions based on historical data and market
          trends
        </p>
      </div>
    </div>
  );
};

export default StockInput;
