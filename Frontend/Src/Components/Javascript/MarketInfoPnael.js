import React from "react";

const MarketInfoPanel = ({ marketInfo, companyData, expanded, onToggle }) => {
  const formatMarketCap = (value) => {
    if (!value) return "N/A";
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toLocaleString()}`;
  };

  const calculatePricePosition = () => {
    const low =
      companyData?.price_summary?.week_52_low || marketInfo?.week_52_low || 0;
    const high =
      companyData?.price_summary?.week_52_high || marketInfo?.week_52_high || 1;
    const current =
      companyData?.current_price || marketInfo?.current_price || 0;

    return ((current - low) / (high - low)) * 100;
  };

  return (
    <div className="market-info-panel">
      <div className="panel-header" onClick={onToggle}>
        <h3>📊 Market Information</h3>
        <span className="toggle-icon">{expanded ? "−" : "+"}</span>
      </div>

      {expanded && (
        <>
          <div className="market-stats">
            {(marketInfo?.sector || companyData?.company_info?.sector) && (
              <div className="stat">
                <label>Sector:</label>
                <span>
                  {marketInfo?.sector || companyData?.company_info?.sector}
                </span>
              </div>
            )}

            {(marketInfo?.industry || companyData?.company_info?.industry) && (
              <div className="stat">
                <label>Industry:</label>
                <span>
                  {marketInfo?.industry || companyData?.company_info?.industry}
                </span>
              </div>
            )}

            {(marketInfo?.market_cap ||
              companyData?.essentials?.market_cap?.value) && (
              <div className="stat">
                <label>Market Cap:</label>
                <span>
                  {companyData?.essentials?.market_cap?.formatted ||
                    formatMarketCap(marketInfo?.market_cap)}
                </span>
              </div>
            )}

            {(marketInfo?.pe_ratio ||
              companyData?.essentials?.pe_ratio?.value) && (
              <div className="stat">
                <label>P/E Ratio:</label>
                <span>
                  {companyData?.essentials?.pe_ratio?.formatted ||
                    marketInfo?.pe_ratio?.toFixed(2)}
                </span>
              </div>
            )}

            {marketInfo?.beta && (
              <div className="stat">
                <label>Beta:</label>
                <span>{marketInfo.beta.toFixed(2)}</span>
              </div>
            )}

            {(marketInfo?.dividend_yield ||
              companyData?.essentials?.dividend_yield?.value) && (
              <div className="stat">
                <label>Dividend Yield:</label>
                <span>
                  {companyData?.essentials?.dividend_yield?.formatted ||
                    `${(marketInfo.dividend_yield * 100).toFixed(2)}%`}
                </span>
              </div>
            )}

            {companyData?.company_info?.exchange && (
              <div className="stat">
                <label>Exchange:</label>
                <span>{companyData.company_info.exchange}</span>
              </div>
            )}

            {marketInfo?.volume && (
              <div className="stat">
                <label>Volume:</label>
                <span>{marketInfo.volume.toLocaleString()}</span>
              </div>
            )}
          </div>

          {(companyData?.price_summary || marketInfo?.week_52_high) && (
            <div className="price-range">
              <h4>52-Week Range</h4>
              <div className="range-info">
                <div className="range-values">
                  <span className="range-low">
                    Low: {companyData?.currency?.symbol || "$"}
                    {(
                      companyData?.price_summary?.week_52_low ||
                      marketInfo?.week_52_low
                    )?.toFixed(2)}
                  </span>
                  <span className="range-high">
                    High: {companyData?.currency?.symbol || "$"}
                    {(
                      companyData?.price_summary?.week_52_high ||
                      marketInfo?.week_52_high
                    )?.toFixed(2)}
                  </span>
                </div>
                <div className="range-bar-container">
                  <div className="range-bar">
                    <div
                      className="current-position"
                      style={{ left: `${calculatePricePosition()}%` }}
                    />
                  </div>
                </div>
                <div className="current-price">
                  Current: {companyData?.currency?.symbol || "$"}
                  {(
                    companyData?.current_price || marketInfo?.current_price
                  )?.toFixed(2)}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default MarketInfoPanel;
