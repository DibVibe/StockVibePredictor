"""
Company Essentials API - Apps/StockPredict/CompanyEssentials.py
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

# Import these from your existing views.py
# You may need to adjust these imports based on your actual views.py structure
def normalize_ticker(ticker):
    """Normalize ticker symbol for consistency"""
    # Handle special cases
    ticker = ticker.upper().strip()

    # Map common variations
    ticker_mapping = {
        'BERKSHIRE': 'BRK-B',
        'ALPHABET': 'GOOGL',
        'GOOGLE': 'GOOGL',
        'NIFTY': '^NSEI',
        'NIFTY50': '^NSEI',
        'SENSEX': '^BSESN',
    }

    return ticker_mapping.get(ticker, ticker)

def validate_ticker(ticker):
    """Validate ticker format"""
    if not ticker or len(ticker) > 20:
        return False

    # Allow alphanumeric, dots, hyphens, and carets
    import re
    pattern = r'^[\w\.\-\^]+$'
    return bool(re.match(pattern, ticker))

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

        # Check cache first (cache for 1 hour)
        cache_key = f"company_essentials_{ticker}"
        cached_data = cache.get(cache_key)
        if cached_data:
            logger.info(f"Returning cached data for {ticker}")
            return Response(cached_data, status=status.HTTP_200_OK)

        # Fetch comprehensive company data from yfinance
        logger.info(f"Fetching company essentials for {ticker}")
        ticker_obj = yf.Ticker(ticker)

        try:
            info = ticker_obj.info
            if not info or 'symbol' not in info:
                raise ValueError("No company information available")
        except Exception as e:
            logger.error(f"yfinance error for {ticker}: {str(e)}")
            return Response(
                {"error": f"Unable to fetch company data for {ticker}. Please verify the ticker symbol."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get historical data for additional calculations
        try:
            hist_data = ticker_obj.history(period="1y", interval="1d")
            current_price = hist_data['Close'].iloc[-1] if not hist_data.empty else info.get('currentPrice', info.get('regularMarketPrice', 0))
        except Exception as e:
            logger.warning(f"Failed to get historical data for {ticker}: {str(e)}")
            current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0)))

        # Currency detection with comprehensive mapping
        currency_mapping = {
            'USD': ('$', 'USD'),
            'INR': ('₹', 'INR'),
            'EUR': ('€', 'EUR'),
            'GBP': ('£', 'GBP'),
            'JPY': ('¥', 'JPY'),
            'CNY': ('¥', 'CNY'),
            'HKD': ('HK$', 'HKD'),
            'SGD': ('S$', 'SGD'),
            'AUD': ('A$', 'AUD'),
            'CAD': ('C$', 'CAD'),
            'CHF': ('CHF', 'CHF'),
            'SEK': ('SEK', 'SEK'),
            'NOK': ('NOK', 'NOK'),
            'KRW': ('₩', 'KRW'),
        }

        # Get currency from yfinance data
        financial_currency = info.get('financialCurrency', info.get('currency', 'USD'))

        # Special handling for Indian tickers
        if '.NS' in ticker.upper() or '.BO' in ticker.upper() or ticker.startswith('^NSE'):
            financial_currency = 'INR'

        currency_symbol, currency_code = currency_mapping.get(
            financial_currency,
            ('$', 'USD')  # Default fallback
        )

        # Extract financial metrics with proper null handling
        def safe_get(value, default=0):
            """Safely get value, return default if None or invalid"""
            return value if value is not None and value != 'N/A' else default

        # Basic metrics
        market_cap = safe_get(info.get('marketCap'), 0)
        enterprise_value = safe_get(info.get('enterpriseValue'), 0)
        shares_outstanding = safe_get(info.get('sharesOutstanding'), 0)
        book_value = safe_get(info.get('bookValue'), 0)
        total_debt = safe_get(info.get('totalDebt'), 0)
        total_cash = safe_get(info.get('totalCash'), 0)

        # Price ranges
        day_high = safe_get(info.get('dayHigh', info.get('regularMarketDayHigh')), current_price)
        day_low = safe_get(info.get('dayLow', info.get('regularMarketDayLow')), current_price)
        week_52_high = safe_get(info.get('fiftyTwoWeekHigh'), current_price)
        week_52_low = safe_get(info.get('fiftyTwoWeekLow'), current_price)

        # Ratios and metrics - handle both decimal and percentage formats
        pe_ratio = safe_get(info.get('trailingPE', info.get('forwardPE')), 0)
        pb_ratio = safe_get(info.get('priceToBook'), 0)

        # Handle dividend yield - yfinance returns as decimal
        dividend_yield = safe_get(info.get('dividendYield'), 0)
        dividend_yield_percent = dividend_yield * 100 if dividend_yield and dividend_yield < 1 else dividend_yield

        # ROE - yfinance returns as decimal (e.g., 0.15 for 15%)
        roe = safe_get(info.get('returnOnEquity'), 0)
        roe_percent = roe * 100 if roe and roe < 1 else roe

        # ROA (Return on Assets) - we'll use this instead of ROCE
        roa = safe_get(info.get('returnOnAssets'), 0)
        roa_percent = roa * 100 if roa and roa < 1 else roa

        # Growth metrics
        revenue_growth = safe_get(info.get('revenueGrowth'), 0)
        revenue_growth_percent = revenue_growth * 100 if revenue_growth and abs(revenue_growth) < 1 else revenue_growth

        earnings_growth = safe_get(info.get('earningsGrowth'), 0)
        earnings_growth_percent = earnings_growth * 100 if earnings_growth and abs(earnings_growth) < 1 else earnings_growth

        # Other metrics
        debt_to_equity = safe_get(info.get('debtToEquity'), 0)
        profit_margins = safe_get(info.get('profitMargins'), 0)
        profit_margins_percent = profit_margins * 100 if profit_margins and profit_margins < 1 else profit_margins

        # EPS
        eps_ttm = safe_get(info.get('trailingEps'), 0)

        # Insider holdings
        insider_holding = safe_get(info.get('heldPercentInsiders'), 0)
        insider_holding_percent = insider_holding * 100 if insider_holding and insider_holding < 1 else insider_holding

        # Helper function to format large numbers
        def format_large_number(value, currency_symbol, currency_code):
            if not value or value == 0:
                return "N/A"

            abs_value = abs(value)
            sign = "-" if value < 0 else ""

            if currency_code == "INR":
                if abs_value >= 1e7:  # Crore
                    return f"{sign}{currency_symbol}{abs_value / 1e7:.2f} Cr"
                elif abs_value >= 1e5:  # Lakh
                    return f"{sign}{currency_symbol}{abs_value / 1e5:.2f} L"
                else:
                    return f"{sign}{currency_symbol}{abs_value:,.0f}"
            else:
                if abs_value >= 1e12:  # Trillion
                    return f"{sign}{currency_symbol}{abs_value / 1e12:.2f}T"
                elif abs_value >= 1e9:  # Billion
                    return f"{sign}{currency_symbol}{abs_value / 1e9:.2f}B"
                elif abs_value >= 1e6:  # Million
                    return f"{sign}{currency_symbol}{abs_value / 1e6:.2f}M"
                elif abs_value >= 1e3:  # Thousand
                    return f"{sign}{currency_symbol}{abs_value / 1e3:.2f}K"
                else:
                    return f"{sign}{currency_symbol}{abs_value:,.0f}"

        def format_shares(value, currency_code):
            if not value or value == 0:
                return "N/A"

            if currency_code == "INR":
                if value >= 1e7:
                    return f"{value / 1e7:.2f} Cr"
                elif value >= 1e5:
                    return f"{value / 1e5:.2f} L"
                else:
                    return f"{value:,.0f}"
            else:
                if value >= 1e9:
                    return f"{value / 1e9:.2f}B"
                elif value >= 1e6:
                    return f"{value / 1e6:.2f}M"
                elif value >= 1e3:
                    return f"{value / 1e3:.2f}K"
                else:
                    return f"{value:,.0f}"

        def format_percentage(value, default="N/A"):
            """Format percentage values consistently"""
            if value is None or value == 0:
                return default
            return f"{value:.2f}%"

        def format_currency_value(value, currency_symbol, precision=2):
            """Format currency values with proper symbol placement"""
            if value is None or value == 0:
                return "N/A"
            return f"{currency_symbol}{value:.{precision}f}"

        # Format the company essentials data
        company_essentials = {
            "ticker": ticker,
            "company_name": info.get('longName', info.get('shortName', ticker)),
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

            # Company Essentials (matching the UI layout)
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
                    "value": dividend_yield_percent,
                    "formatted": format_percentage(dividend_yield_percent, "0.00%"),
                    "label": "DIV. YIELD"
                },
                "debt": {
                    "value": total_debt,
                    "formatted": format_large_number(total_debt, currency_symbol, currency_code),
                    "label": "DEBT"
                },
                "sales_growth": {
                    "value": revenue_growth_percent,
                    "formatted": format_percentage(revenue_growth_percent),
                    "label": "SALES GROWTH"
                },
                "profit_growth": {
                    "value": earnings_growth_percent,
                    "formatted": format_percentage(earnings_growth_percent),
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
                    "formatted": format_currency_value(book_value, currency_symbol),
                    "label": "BOOK VALUE (TTM)"
                },
                "promoter_holding": {
                    "value": insider_holding_percent,
                    "formatted": format_percentage(insider_holding_percent),
                    "label": "PROMOTER HOLDING"
                },
                "roe": {
                    "value": roe_percent,
                    "formatted": format_percentage(roe_percent),
                    "label": "ROE"
                },

                # Right Column
                "num_shares": {
                    "value": shares_outstanding,
                    "formatted": format_shares(shares_outstanding, currency_code),
                    "label": "NO. OF SHARES"
                },
                "face_value": {
                    "value": None,  # Not available from yfinance
                    "formatted": "N/A",  # Would need separate data source for Indian stocks
                    "label": "FACE VALUE"
                },
                "cash": {
                    "value": total_cash,
                    "formatted": format_large_number(total_cash, currency_symbol, currency_code),
                    "label": "CASH"
                },
                "eps_ttm": {
                    "value": eps_ttm,
                    "formatted": format_currency_value(eps_ttm, currency_symbol),
                    "label": "EPS (TTM)"
                },
                "roa": {  # Changed from ROCE to ROA for accuracy
                    "value": roa_percent,
                    "formatted": format_percentage(roa_percent),
                    "label": "ROA"  # More accurate label
                },
            },

            # Additional metrics
            "additional_metrics": {
                "debt_to_equity": {
                    "value": debt_to_equity,
                    "formatted": f"{debt_to_equity:.2f}" if debt_to_equity else "N/A",
                    "label": "DEBT/EQUITY"
                },
                "profit_margins": {
                    "value": profit_margins_percent,
                    "formatted": format_percentage(profit_margins_percent),
                    "label": "PROFIT MARGIN"
                },
                "forward_pe": {
                    "value": info.get('forwardPE', 0),
                    "formatted": f"{info.get('forwardPE', 0):.2f}" if info.get('forwardPE') else "N/A",
                    "label": "FORWARD P/E"
                },
                "peg_ratio": {
                    "value": info.get('pegRatio', 0),
                    "formatted": f"{info.get('pegRatio', 0):.2f}" if info.get('pegRatio') else "N/A",
                    "label": "PEG RATIO"
                },
                "beta": {
                    "value": info.get('beta', 0),
                    "formatted": f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A",
                    "label": "BETA"
                },
                "52_week_change": {
                    "value": info.get('52WeekChange', 0) * 100 if info.get('52WeekChange') else 0,
                    "formatted": format_percentage(info.get('52WeekChange', 0) * 100 if info.get('52WeekChange') else 0),
                    "label": "52 WEEK CHANGE"
                },
            },

            # Company info
            "company_info": {
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "country": info.get('country', 'N/A'),
                "website": info.get('website', 'N/A'),
                "business_summary": info.get('longBusinessSummary', 'N/A'),
                "full_time_employees": info.get('fullTimeEmployees', 0),
                "exchange": info.get('exchange', 'N/A'),
                "quote_type": info.get('quoteType', 'N/A'),
                "market": info.get('market', 'N/A'),
            },

            # Timestamp
            "last_updated": timezone.now().isoformat(),
            "data_source": "yfinance",
        }

        # Cache the data for 1 hour
        cache.set(cache_key, company_essentials, 3600)
        logger.info(f"Successfully fetched and cached data for {ticker}")

        return Response(company_essentials, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error fetching company essentials for {ticker}: {str(e)}", exc_info=True)
        return Response(
            {"error": f"Failed to fetch company essentials: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
