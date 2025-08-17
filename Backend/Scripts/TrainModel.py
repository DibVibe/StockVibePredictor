"""
Enhanced Model Training System for StockVibePredictor - WORKING VERSION
Organization: Dibakar
Created: 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import os
import json
import re
from pathlib import Path
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

def setup_logging():
    """Setup logging system"""
    log_dir = Path("../Logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "ModelTraining.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "Scripts" / "Models"
MODELS_DIR.mkdir(exist_ok=True)

PERFORMANCE_DIR = BASE_DIR / "Scripts" / "Performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)

# SIMPLIFIED TIMEFRAMES
TIMEFRAMES = {
    "1d": {"period": "1mo", "interval": "1h"},
    "5d": {"period": "2mo", "interval": "1d"},
    "1w": {"period": "3mo", "interval": "1d"},
    "1mo": {"period": "6mo", "interval": "1d"},
    "3mo": {"period": "1y", "interval": "1d"},
    "6mo": {"period": "2y", "interval": "1wk"},
    "1y": {"period": "3y", "interval": "1wk"},
    "2y": {"period": "5y", "interval": "1mo"},
    "5y": {"period": "max", "interval": "1mo"},
}

# STOCK DATABASE
STOCK_DATABASE = {
    "mega_cap_tech": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "mega_cap_other": ["JPM", "JNJ", "V", "WMT"],
    "financial": ["BAC", "WFC", "GS"],
    "etfs": ["SPY", "QQQ", "ARKK"],
    "indian_market": ["^NSEI", "^BSESN"],
}

# ============================================================================
# SIMPLIFIED DATA PROCESSING
# ============================================================================

def normalize_ticker(ticker):
    """Normalize ticker"""
    ticker = ticker.upper().strip()
    ticker_mapping = {
        "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "SENSEX": "^BSESN",
        "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
        "FACEBOOK": "META", "TESLA": "TSLA",
    }
    return ticker_mapping.get(ticker, ticker)

def fetch_stock_data_sync(ticker, timeframe="1d"):
    """Simplified and reliable stock data fetching"""
    try:
        ticker = normalize_ticker(ticker)
        config = TIMEFRAMES[timeframe]

        logger.info(f"Fetching {ticker} with period={config['period']}, interval={config['interval']}")

        # Use yf.download (more reliable than ticker.history)
        data = yf.download(
            ticker,
            period=config["period"],
            interval=config["interval"],
            auto_adjust=True,
            prepost=False,
            progress=False,  # Disable progress bar
        )

        if data.empty:
            logger.error(f"No data returned for {ticker}")
            return None

        logger.info(f"✓ Fetched {len(data)} rows for {ticker}")
        return {"price_data": data, "market_info": {}}

    except Exception as e:
        logger.error(f"✗ Error fetching {ticker}: {str(e)}")
        return None

def compute_basic_features(data, timeframe="1d"):
    """Compute basic technical features"""
    try:
        # Handle MultiIndex columns from yf.download
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

        # Ensure required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing column: {col}")

        # Remove rows with missing data
        data = data.dropna(subset=required_cols)

        if len(data) < 30:
            raise ValueError(f"Insufficient data: {len(data)} rows")

        # Basic features - Fix MultiIndex issue
        close_series = data["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]  # Take first column if DataFrame
        data["Return"] = close_series.pct_change().fillna(0)


        # Moving averages
        data["MA5"] = data["Close"].rolling(5).mean()
        data["MA10"] = data["Close"].rolling(10).mean()
        data["MA20"] = data["Close"].rolling(20).mean()

        # Volatility
        data["Volatility"] = data["Return"].rolling(10).std().fillna(0)

        # Volume features
        data["Volume_MA"] = data["Volume"].rolling(10).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_MA"]
        data["Volume_Ratio"] = data["Volume_Ratio"].fillna(1)

        # RSI (simple version)
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        loss = loss.replace(0, 1e-10)  # Prevent division by zero
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))
        data["RSI"] = data["RSI"].fillna(50)  # Neutral RSI for NaN

        # Price position features
        data["High_Low_Ratio"] = (data["High"] - data["Low"]) / data["Close"]
        data["Close_Position"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
        data["Close_Position"] = data["Close_Position"].fillna(0.5)

        # Trend features
        data["Price_Above_MA5"] = (data["Close"] > data["MA5"]).astype(int)
        data["Price_Above_MA20"] = (data["Close"] > data["MA20"]).astype(int)

        # Fill remaining NaN values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'Target':  # Don't fill target
                data[col] = data[col].fillna(data[col].median())

        return data

    except Exception as e:
        logger.error(f"Error computing features: {str(e)}")
        raise

def prepare_training_data(ticker, timeframe="1d"):
    """Prepare training data"""
    try:
        logger.info(f"Preparing training data for {ticker} ({timeframe})")

        # Fetch data
        data_result = fetch_stock_data_sync(ticker, timeframe)
        if not data_result:
            raise ValueError(f"Failed to fetch data for {ticker}")

        # Compute features
        data = compute_basic_features(data_result["price_data"], timeframe)

        # Create target
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        # Remove last row and clean
        data = data[:-1].dropna()

        if len(data) < 30:
            raise ValueError(f"Insufficient data for {ticker}: {len(data)} rows")

        logger.info(f"✓ Prepared {len(data)} samples for {ticker}")
        return data

    except Exception as e:
        logger.error(f"✗ Error preparing {ticker}: {str(e)}")
        raise

# ============================================================================
# MODEL TRAINING
# ============================================================================

def get_feature_columns():
    """Get feature columns for training"""
    return [
        "Return", "MA5", "MA10", "MA20", "Volatility",
        "Volume_Ratio", "RSI", "High_Low_Ratio", "Close_Position",
        "Price_Above_MA5", "Price_Above_MA20"
    ]

def train_model_for_ticker(ticker, timeframe="1d", model_type="ensemble"):
    """Train model for specific ticker"""
    try:
        logger.info(f"Training {model_type} model for {ticker} ({timeframe})")

        # Prepare data
        data = prepare_training_data(ticker, timeframe)

        # Get features
        feature_cols = get_feature_columns()
        available_features = [col for col in feature_cols if col in data.columns]

        if len(available_features) < 5:
            return {"error": f"Insufficient features: {len(available_features)}"}

        # Prepare X and y
        X = data[available_features].values
        y = data["Target"].values

        # Check for infinite values
        if not np.all(np.isfinite(X)):
            logger.warning(f"Non-finite values found for {ticker}, cleaning...")
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
            y = y[mask]

        if len(X) < 30:
            return {"error": f"Insufficient clean data: {len(X)} samples"}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        if model_type == "ensemble":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            model = VotingClassifier(
                estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
                voting="soft"
            )
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            metrics["cv_mean"] = cv_scores.mean()
            metrics["cv_std"] = cv_scores.std()
        except:
            metrics["cv_mean"] = metrics["accuracy"]
            metrics["cv_std"] = 0.0

        # Save model
        model_filename = f"{ticker}_model_{timeframe}.pkl"
        model_path = MODELS_DIR / model_filename

        model_data = {
            "model": model,
            "scaler": scaler,
            "features": available_features,
            "accuracy": metrics["accuracy"],
            "metrics": metrics,
            "ticker": ticker,
            "timeframe": timeframe,
            "trained_at": datetime.now().isoformat(),
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"✓ Model saved: {model_filename} (Accuracy: {metrics['accuracy']:.2%})")

        return {
            "success": True,
            "ticker": ticker,
            "timeframe": timeframe,
            "metrics": metrics,
            "model_path": str(model_path),
        }

    except Exception as e:
        logger.error(f"✗ Training failed for {ticker} ({timeframe}): {str(e)}")
        return {"error": str(e)}

def train_universal_model(timeframe="1d", model_type="ensemble"):
    """Train universal model"""
    try:
        logger.info(f"Training universal {model_type} model for {timeframe}")

        # Reliable stocks for universal model
        universal_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "JPM"]

        all_data = []
        successful_stocks = []

        for ticker in universal_stocks:
            try:
                data = prepare_training_data(ticker, timeframe)
                if len(data) >= 50:  # Minimum threshold
                    all_data.append(data.tail(200))  # Use recent 200 samples max
                    successful_stocks.append(ticker)
                    logger.info(f"✓ Added {ticker}: {len(data)} samples")
            except Exception as e:
                logger.warning(f"Skipped {ticker}: {str(e)}")
                continue

        if len(successful_stocks) < 3:
            raise ValueError(f"Insufficient stocks for universal model: {len(successful_stocks)}")

        # Combine data
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined: {len(combined_data)} samples from {len(successful_stocks)} stocks")

        # Get features
        feature_cols = get_feature_columns()
        available_features = [col for col in feature_cols if col in combined_data.columns]

        # Prepare training data
        X = combined_data[available_features].values
        y = combined_data["Target"].values

        # Clean data
        if not np.all(np.isfinite(X)):
            mask = np.isfinite(X).all(axis=1)
            X = X[mask]
            y = y[mask]

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_type == "ensemble":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            model = VotingClassifier(
                estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
                voting="soft"
            )
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42)

        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)

        # Save model
        model_filename = f"universal_model_{timeframe}.pkl"
        model_path = MODELS_DIR / model_filename

        model_data = {
            "model": model,
            "scaler": scaler,
            "features": available_features,
            "accuracy": accuracy,
            "timeframe": timeframe,
            "training_tickers": successful_stocks,
            "trained_at": datetime.now().isoformat(),
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"✓ Universal model saved: {model_filename} (Accuracy: {accuracy:.2%})")

        return {
            "success": True,
            "timeframe": timeframe,
            "accuracy": accuracy,
            "samples": len(X_train),
            "path": str(model_path),
        }

    except Exception as e:
        logger.error(f"✗ Universal model failed for {timeframe}: {str(e)}")
        return {"error": str(e)}

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def comprehensive_training_pipeline(training_mode="essential", model_type="ensemble", max_workers=2):
    """Complete training pipeline"""

    logger.info("=" * 60)
    logger.info("🚀 STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    start_time = time.time()

    # Timeframe modes
    timeframe_modes = {
        "minimal": ["1d", "1w", "1mo"],
        "essential": ["1d", "1w", "1mo", "1y"],
        "extended": ["1d", "5d", "1w", "1mo", "3mo", "1y"],
        "complete": ["1d", "5d", "1w", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
    }

    timeframes = timeframe_modes.get(training_mode, timeframe_modes["essential"])

    logger.info(f"🎯 Mode: {training_mode}")
    logger.info(f"⏱️ Timeframes: {timeframes}")

    training_summary = {
        "start_time": datetime.now().isoformat(),
        "training_mode": training_mode,
        "timeframes": timeframes,
        "model_type": model_type,
        "universal_models": {},
        "category_models": {},
        "total_models": 0,
        "successful_models": 0,
    }

    # Phase 1: Universal Models
    logger.info("\n📊 Phase 1: Training Universal Models")
    for timeframe in timeframes:
        result = train_universal_model(timeframe, model_type)
        training_summary["universal_models"][timeframe] = result
        training_summary["total_models"] += 1
        if result.get("success"):
            training_summary["successful_models"] += 1

    # Phase 2: Category Models
    logger.info("\n🏢 Phase 2: Training Category Models")

    # Train only essential timeframes for individual stocks
    essential_timeframes = ["1d", "1w", "1mo", "6mo", "1y"]  # Add longer timeframes

    def train_stock_models(ticker):
        results = {}
        for tf in essential_timeframes:
            result = train_model_for_ticker(ticker, tf, model_type)
            results[f"{ticker}_{tf}"] = result
        return results

    for category, stocks in STOCK_DATABASE.items():
        logger.info(f"🏷️ Training {category}: {stocks}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_stock = {
                executor.submit(train_stock_models, stock): stock
                for stock in stocks
            }

            category_results = {}
            for future in future_to_stock:
                stock = future_to_stock[future]
                try:
                    result = future.result(timeout=180)  # 3 min timeout
                    category_results.update(result)
                except Exception as e:
                    logger.error(f"Stock {stock} failed: {str(e)}")
                    for tf in essential_timeframes:
                        category_results[f"{stock}_{tf}"] = {"error": str(e)}

            training_summary["category_models"][category] = category_results

            # Update totals
            successful = sum(1 for r in category_results.values() if r.get("success"))
            total = len(category_results)
            training_summary["total_models"] += total
            training_summary["successful_models"] += successful

            logger.info(f"✅ {category}: {successful}/{total} successful")

    # Summary
    end_time = time.time()
    duration = end_time - start_time
    training_summary["end_time"] = datetime.now().isoformat()
    training_summary["duration_seconds"] = duration
    training_summary["failed_models"] = training_summary["total_models"] - training_summary["successful_models"]

    success_rate = (training_summary["successful_models"] / training_summary["total_models"]) * 100

    logger.info("=" * 60)
    logger.info("🎉 TRAINING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"📈 Total Models: {training_summary['total_models']}")
    logger.info(f"✅ Successful: {training_summary['successful_models']}")
    logger.info(f"❌ Failed: {training_summary['failed_models']}")
    logger.info(f"🎯 Success Rate: {success_rate:.1f}%")
    logger.info(f"⏰ Duration: {duration//60:.0f}m {duration%60:.0f}s")

    # Save summary
    summary_file = PERFORMANCE_DIR / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, "w") as f:
        json.dump(training_summary, f, indent=2, default=str)

    logger.info(f"📄 Summary saved: {summary_file}")
    return training_summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "essential"

    logger.info("🚀 StockVibePredictor Model Training System v2.1 (WORKING)")

    if mode in ["minimal", "essential", "extended", "complete"]:
        result = comprehensive_training_pipeline(training_mode=mode)
    elif mode == "single":
        ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
        timeframe = sys.argv[3] if len(sys.argv) > 3 else "1d"
        result = train_model_for_ticker(ticker, timeframe)
        print(f"Result: {result}")
    else:
        print("Available modes: minimal, essential, extended, complete, single")
