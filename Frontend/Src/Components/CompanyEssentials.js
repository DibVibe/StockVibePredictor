import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './CompanyEssentials.css';

const CompanyEssentials = ({ ticker, onCompanyDataReceived = null }) => {
  const [companyData, setCompanyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://localhost:8000/api';

  useEffect(() => {
    if (ticker) {
      fetchCompanyEssentials(ticker);
    }
  }, [ticker]);

  const fetchCompanyEssentials = async (stockTicker) => {
    if (!stockTicker || !stockTicker.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_BASE_URL}/company-essentials/${stockTicker.toUpperCase()}/`);
      const data = response.data;
      
      setCompanyData(data);
      
      // Pass data to parent component if callback provided
      if (onCompanyDataReceived) {
        onCompanyDataReceived(data);
      }
      
      console.log('Company essentials loaded:', data.ticker);
    } catch (err) {
      console.error('Error fetching company essentials:', err);
      
      let errorMessage = 'Unable to fetch company essentials.';
      
      if (err.response?.status === 404) {
        errorMessage = `Company data not found for "${stockTicker}".`;
      } else if (err.response?.status === 400) {
        errorMessage = 'Invalid ticker format.';
      } else if (err.response?.data?.error) {
        errorMessage = err.response.data.error;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="company-essentials">
        <div className="essentials-header">
          <h3>📊 Company Essentials</h3>
        </div>
        <div className="essentials-loading">
          <div className="loading-spinner"></div>
          <p>Loading company data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="company-essentials">
        <div className="essentials-header">
          <h3>📊 Company Essentials</h3>
        </div>
        <div className="essentials-error">
          <span className="error-icon">⚠️</span>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!companyData) {
    return null;
  }

  const { essentials, company_name, current_price, price_summary, currency } = companyData;
  const currencySymbol = currency?.symbol || '₹';
  const currencyCode = currency?.code || 'INR';

  return (
    <div className="company-essentials">
      {/* Header Section */}
      <div className="essentials-header">
        <div className="company-title">
          <h3>📊 Company Essentials</h3>
          <h2>{company_name || ticker}</h2>
          <div className="current-price">
            <span className="price">{currencySymbol}{current_price?.toFixed(2) || 'N/A'}</span>
            {price_summary?.price_change && (
              <span className={`price-change ${price_summary.price_change >= 0 ? 'positive' : 'negative'}`}>
                {price_summary.price_change >= 0 ? '▲' : '▼'} 
                {currencySymbol}{Math.abs(price_summary.price_change).toFixed(2)} 
                ({price_summary.price_change_percent?.toFixed(2) || '0.00'}%)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Essentials Grid - 3 Column Layout */}
      <div className="essentials-grid">
        {/* Left Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>MARKET CAP</label>
            <span className="value">{essentials.market_cap?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>P/E</label>
            <span className="value">{essentials.pe_ratio?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>DIV. YIELD</label>
            <span className="value">{essentials.dividend_yield?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>DEBT</label>
            <span className="value">{essentials.debt?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>SALES GROWTH</label>
            <span className={`value ${essentials.sales_growth?.value >= 0 ? 'positive' : 'negative'}`}>
              {essentials.sales_growth?.formatted || 'N/A'}
            </span>
          </div>
          
          <div className="essential-item">
            <label>PROFIT GROWTH</label>
            <span className={`value ${essentials.profit_growth?.value >= 0 ? 'positive' : 'negative'}`}>
              {essentials.profit_growth?.formatted || 'N/A'}
            </span>
          </div>
        </div>

        {/* Middle Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>ENTERPRISE VALUE</label>
            <span className="value">{essentials.enterprise_value?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>P/B</label>
            <span className="value">{essentials.pb_ratio?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>BOOK VALUE (TTM)</label>
            <span className="value">{essentials.book_value_ttm?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>PROMOTER HOLDING</label>
            <span className="value">{essentials.promoter_holding?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>ROE</label>
            <span className={`value ${essentials.roe?.value >= 0 ? 'positive' : 'negative'}`}>
              {essentials.roe?.formatted || 'N/A'}
            </span>
          </div>
          
          <div className="essential-item">
            <label>Add Your Ratio</label>
            <button className="add-ratio-btn">+</button>
          </div>
        </div>

        {/* Right Column */}
        <div className="essentials-column">
          <div className="essential-item">
            <label>NO. OF SHARES</label>
            <span className="value">{essentials.num_shares?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>FACE VALUE</label>
            <span className="value">{essentials.face_value?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>CASH</label>
            <span className="value">{essentials.cash?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>EPS (TTM)</label>
            <span className="value">{essentials.eps_ttm?.formatted || 'N/A'}</span>
          </div>
          
          <div className="essential-item">
            <label>ROCE</label>
            <span className={`value ${essentials.roce?.value >= 0 ? 'positive' : 'negative'}`}>
              {essentials.roce?.formatted || 'N/A'}
            </span>
          </div>
          
          <div className="essential-item empty-slot">
            <label></label>
            <span className="value"></span>
          </div>
        </div>
      </div>

      {/* Price Summary Section */}
      {price_summary && (
        <div className="price-summary">
          <h4>Price Summary</h4>
          <div className="price-stats">
            <div className="price-stat">
              <label>Today's High</label>
              <span>{currencySymbol}{price_summary.day_high?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="price-stat">
              <label>Today's Low</label>
              <span>{currencySymbol}{price_summary.day_low?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="price-stat">
              <label>52W High</label>
              <span>{currencySymbol}{price_summary.week_52_high?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="price-stat">
              <label>52W Low</label>
              <span>{currencySymbol}{price_summary.week_52_low?.toFixed(2) || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Company Information Footer */}
      {companyData.company_info && (
        <div className="company-info-summary">
          <div className="info-item">
            <strong>Sector:</strong> {companyData.company_info.sector || 'N/A'}
          </div>
          <div className="info-item">
            <strong>Industry:</strong> {companyData.company_info.industry || 'N/A'}
          </div>
          {companyData.company_info.full_time_employees > 0 && (
            <div className="info-item">
              <strong>Employees:</strong> {companyData.company_info.full_time_employees.toLocaleString()}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default CompanyEssentials;
