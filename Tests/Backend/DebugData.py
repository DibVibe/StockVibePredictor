import yfinance as yf
import pandas as pd

def test_basic_fetch():
    print("Testing basic yfinance functionality...")

    # Test 1: Basic fetch
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1mo", interval="1d")
        print(f"✓ Basic AAPL fetch: {len(data)} rows")
        print(f"Columns: {list(data.columns)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"✗ Basic fetch failed: {e}")

    # Test 2: Different periods
    periods = ["1mo", "3mo", "6mo", "1y"]
    for period in periods:
        try:
            data = yf.download("AAPL", period=period, interval="1d")
            print(f"✓ AAPL {period}: {len(data)} rows")
        except Exception as e:
            print(f"✗ AAPL {period} failed: {e}")

if __name__ == "__main__":
    test_basic_fetch()
