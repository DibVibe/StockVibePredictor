import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

def simple_train_test():
    """Test with minimal, working code"""

    # Test single stock first
    ticker = "AAPL"
    print(f"Testing {ticker}...")

    try:
        # Simple data fetch
        data = yf.download(ticker, period="6mo", interval="1d")
        print(f"Downloaded {len(data)} rows")

        if data.empty:
            print("No data received!")
            return

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        print(f"Columns: {list(data.columns)}")
        print(f"First few rows:\n{data.head()}")

        # Basic features
        data['Return'] = data['Close'].pct_change()
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Clean data
        data = data.dropna()
        print(f"After cleaning: {len(data)} rows")

        if len(data) < 50:
            print("Not enough data after cleaning")
            return

        # Prepare features
        features = ['Return', 'MA5', 'MA20']
        X = data[features].values
        y = data['Target'].values

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        print(f"✓ Model trained successfully! Accuracy: {accuracy:.2%}")

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_train_test()
