"""
Company Essentials API
Provides detailed company financial metrics and essential information
"""
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
import yfinance as yf
import logging
from django.core.cache import cache

from .views import normalize_ticker, validate_ticker

logger = logging.getLogger("Apps.StockPredict")


@api_view(["GET"])
@permission_classes([AllowAny])
def company_essentials(request, ticker):
    """Get comprehensive company essentials and financial metrics"""
    try:
        ticker = normalize_ticker(ticker.upper())
        
        if not validate_ticker(ticker):
            return Response(
                {"error": "Invalid ticker format"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        # Check cache first
        cache_key = f"company_essentials_{ticker}"
        cached_data = cache.get(cache_key)
        if cached_data:
            return Response(cached_data, status=status.HTTP_200_OK)
        
        # Fetch comprehensive company data
        ticker_obj = yf.Ticker(ticker)
        
        try:
            info = ticker_obj.info
            if not info:
                raise ValueError("No company information available")
        except Exception as e:
            return Response(
                {"error": f"Unable to fetch company data for {ticker}: {str(e)}"},
                status=status.HTTP_404_NOT_FOUND,
            )
        
        # Get historical data for additional calculations
        hist_data = ticker_obj.history(period="1y", interval="1d")
        current_price = hist_data['Close'].iloc[-1] if not hist_data.empty else info.get('currentPrice', 0)
        
        # Determine currency based on ticker and market
        currency_symbol = "₹"  # Default to INR
        currency_code = "INR"
        
        # Detect market/exchange from ticker or info
        country = info.get('country', '').lower()
        exchange = info.get('exchange', '').upper()
        ticker_upper = ticker.upper()
        
        if ('.NS' in ticker_upper or '.BO' in ticker_upper or 
            country == 'india' or exchange in ['NSE', 'BSE']):
            currency_symbol = "₹"
            currency_code = "INR"
        elif (country == 'united states' or exchange in ['NASDAQ', 'NYSE', 'NYQ'] or 
              ticker_upper in ['GOOGL', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META']):
            currency_symbol = "$"
            currency_code = "USD"
        elif country == 'united kingdom' or exchange == 'LSE':
            currency_symbol = "£"
            currency_code = "GBP"
        elif country == 'germany' or exchange == 'FRA':
            currency_symbol = "€"
            currency_code = "EUR"
        elif country == 'japan' or exchange == 'JPX':
            currency_symbol = "¥"
            currency_code = "JPY"
        else:
            # Try to detect from financialCurrency or currency field
            financial_currency = info.get('financialCurrency', info.get('currency', 'INR'))
            if financial_currency == 'USD':
                currency_symbol = "$"
                currency_code = "USD"
            elif financial_currency == 'EUR':
                currency_symbol = "€"
                currency_code = "EUR"
            elif financial_currency == 'GBP':
                currency_symbol = "£"
                currency_code = "GBP"
            elif financial_currency == 'JPY':
                currency_symbol = "¥"
                currency_code = "JPY"
        
        # Calculate additional metrics
        book_value = info.get('bookValue', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        enterprise_value = info.get('enterpriseValue', 0)
        total_debt = info.get('totalDebt', 0)
        total_cash = info.get('totalCash', 0)
        
        # Price ranges
        day_high = info.get('dayHigh', current_price)
        day_low = info.get('dayLow', current_price)
        week_52_high = info.get('fiftyTwoWeekHigh', current_price)
        week_52_low = info.get('fiftyTwoWeekLow', current_price)
        
        # Ratios and metrics
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        dividend_yield = info.get('dividendYield', 0)
        roe = info.get('returnOnEquity', 0)
        roce = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0
        profit_margins = info.get('profitMargins', 0)
        revenue_growth = info.get('revenueGrowth', 0)
        earnings_growth = info.get('earningsGrowth', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        
        # EPS calculation
        eps_ttm = info.get('trailingEps', 0)
        
        # Helper function to format large numbers
        def format_large_number(value, currency_symbol, currency_code):
            if not value or value == 0:
                return "N/A"
            
            if currency_code == "INR":
                return f"{currency_symbol} {value / 1e7:.2f} Cr."
            else:
                if value >= 1e12:  # Trillion
                    return f"{currency_symbol}{value / 1e12:.2f}T"
                elif value >= 1e9:  # Billion
                    return f"{currency_symbol}{value / 1e9:.2f}B"
                elif value >= 1e6:  # Million
                    return f"{currency_symbol}{value / 1e6:.2f}M"
                else:
                    return f"{currency_symbol}{value:,.0f}"
        
        def format_shares(value, currency_code):
            if not value or value == 0:
                return "N/A"
            
            if currency_code == "INR":
                return f"{value / 1e7:.2f} Cr."
            else:
                if value >= 1e9:
                    return f"{value / 1e9:.2f}B"
                elif value >= 1e6:
                    return f"{value / 1e6:.2f}M"
                else:
                    return f"{value:,.0f}"
        
        # Format the company essentials data
        company_essentials = {
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "current_price": current_price,
            "currency": {
                "symbol": currency_symbol,
                "code": currency_code
            },
            
            # Price Summary
            "price_summary": {
                "current_price": current_price,
                "day_high": day_high,
                "day_low": day_low,
                "week_52_high": week_52_high,
                "week_52_low": week_52_low,
                "price_change": info.get('regularMarketChange', 0),
                "price_change_percent": info.get('regularMarketChangePercent', 0),
            },
            
            # Company Essentials (matching the image layout)
            "essentials": {
                # Left Column
                "market_cap": {
                    "value": market_cap,
                    "formatted": format_large_number(market_cap, currency_symbol, currency_code),
                    "label": "MARKET CAP"
                },
                "pe_ratio": {
                    "value": pe_ratio,
                    "formatted": f"{pe_ratio:.2f}" if pe_ratio else "N/A",
                    "label": "P/E"
                },
                "dividend_yield": {
                    "value": dividend_yield * 100 if dividend_yield else 0,
                    "formatted": f"{dividend_yield * 100:.2f} %" if dividend_yield else "0 %",
                    "label": "DIV. YIELD"
                },
                "debt": {
                    "value": total_debt,
                    "formatted": format_large_number(total_debt, currency_symbol, currency_code),
                    "label": "DEBT"
                },
                "sales_growth": {
                    "value": revenue_growth * 100 if revenue_growth else 0,
                    "formatted": f"{revenue_growth * 100:.2f}%" if revenue_growth else "N/A",
                    "label": "SALES GROWTH"
                },
                "profit_growth": {
                    "value": earnings_growth * 100 if earnings_growth else 0,
                    "formatted": f"{earnings_growth * 100:.2f} %" if earnings_growth else "N/A",
                    "label": "PROFIT GROWTH"
                },
                
                # Middle Column  
                "enterprise_value": {
                    "value": enterprise_value,
                    "formatted": format_large_number(enterprise_value, currency_symbol, currency_code),
                    "label": "ENTERPRISE VALUE"
                },
                "pb_ratio": {
                    "value": pb_ratio,
                    "formatted": f"{pb_ratio:.2f}" if pb_ratio else "N/A",
                    "label": "P/B"
                },
                "book_value_ttm": {
                    "value": book_value,
                    "formatted": f"{currency_symbol} {book_value:.2f}" if book_value else "N/A",
                    "label": "BOOK VALUE (TTM)"
                },
                "promoter_holding": {
                    "value": info.get('heldPercentInsiders', 0) * 100,
                    "formatted": f"{info.get('heldPercentInsiders', 0) * 100:.2f} %" if info.get('heldPercentInsiders') else "N/A",
                    "label": "PROMOTER HOLDING"
                },
                "roe": {
                    "value": roe * 100 if roe else 0,
                    "formatted": f"{roe * 100:.2f} %" if roe else "N/A",
                    "label": "ROE"
                },
                
                # Right Column
                "num_shares": {
                    "value": shares_outstanding,
                    "formatted": format_shares(shares_outstanding, currency_code),
                    "label": "NO. OF SHARES"
                },
                "face_value": {
                    "value": 1,  # Default face value
                    "formatted": f"{currency_symbol} 1" if currency_code == "INR" else "N/A",
                    "label": "FACE VALUE"
                },
                "cash": {
                    "value": total_cash,
                    "formatted": format_large_number(total_cash, currency_symbol, currency_code),
                    "label": "CASH"
                },
                "eps_ttm": {
                    "value": eps_ttm,
                    "formatted": f"{currency_symbol} {eps_ttm:.2f}" if eps_ttm else "N/A",
                    "label": "EPS (TTM)"
                },
                "roce": {
                    "value": roce,
                    "formatted": f"{roce:.2f}%" if roce else "N/A",
                    "label": "ROCE"
                },
            },
            
            # Additional company info
            "company_info": {
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "country": info.get('country', 'N/A'),
                "website": info.get('website', 'N/A'),
                "business_summary": info.get('longBusinessSummary', 'N/A'),
                "full_time_employees": info.get('fullTimeEmployees', 0),
            },
            
            # Timestamp
            "last_updated": timezone.now().isoformat(),
        }
        
        # Cache the data for 1 hour
        cache.set(cache_key, company_essentials, 3600)
        
        return Response(company_essentials, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error fetching company essentials for {ticker}: {str(e)}")
        return Response(
            {"error": f"Failed to fetch company essentials: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
