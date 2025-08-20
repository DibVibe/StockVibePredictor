"""
StockVibePredictor Model Training System v5.0 (ENHANCED)
Organization: Dibakar
Created: 2025

Enhanced version with comprehensive validation, error handling, and monitoring
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
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Central configuration management"""

    # Version info
    VERSION = "5.0"
    BUILD_DATE = datetime.now().strftime("%Y-%m-%d")

    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "Scripts" / "Models"
    PERFORMANCE_DIR = BASE_DIR / "Scripts" / "Performance"
    LOGS_DIR = BASE_DIR / "Logs"
    VALIDATION_DIR = BASE_DIR / "Scripts" / "Validation"

    # Training parameters
    MIN_DATA_POINTS = 30
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 3

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    # Performance thresholds
    MIN_ACCEPTABLE_ACCURACY = 0.35
    TARGET_ACCURACY = 0.55

    # API rate limiting
    API_DELAY = 0.5  # seconds between API calls

def setup_logging():
    """Enhanced logging setup with multiple handlers"""
    Config.LOGS_DIR.mkdir(exist_ok=True)

    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s [%(name)s:%(lineno)d] %(message)s",
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / f"ModelTraining_{timestamp}.log"),
            logging.FileHandler(Config.LOGS_DIR / "ModelTraining_latest.log", mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Ensure all directories exist
for dir_path in [Config.MODELS_DIR, Config.PERFORMANCE_DIR, Config.VALIDATION_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# TIMEFRAMES & STOCK DATABASE CONFIGURATION
# ============================================================================

# TIMEFRAMES Configuration - Validated and optimized
TIMEFRAMES = {
    "1d": {"period": "1mo", "interval": "1h", "min_samples": 100},
    "5d": {"period": "5d", "interval": "15m", "min_samples": 80},
    "1w": {"period": "3mo", "interval": "1d", "min_samples": 40},
    "1mo": {"period": "6mo", "interval": "1d", "min_samples": 90},
    "3mo": {"period": "1y", "interval": "1d", "min_samples": 180},
    "6mo": {"period": "2y", "interval": "1wk", "min_samples": 70},
    "1y": {"period": "3y", "interval": "1wk", "min_samples": 100},
    "2y": {"period": "5y", "interval": "1mo", "min_samples": 40},
    "5y": {"period": "max", "interval": "1mo", "min_samples": 60},
}

# Fallback configurations for problematic timeframes
TIMEFRAME_FALLBACKS = {
    "5d": [
        {"period": "10d", "interval": "30m"},
        {"period": "1mo", "interval": "1d"},
        {"period": "2mo", "interval": "1d"}
    ],
    "1d": [
        {"period": "5d", "interval": "30m"},
        {"period": "1mo", "interval": "2h"}
    ]
}

# STOCK DATABASE - Categorized and validated
STOCK_DATABASE = {
    "mega_cap_tech": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "mega_cap_other": ["JPM", "JNJ", "V", "WMT"],
    "financial": ["BAC", "WFC", "GS"],
    "etfs": ["SPY", "QQQ", "ARKK"],
    "indian_market": ["^NSEI", "^BSESN"],
}

# Ticker normalization mapping
TICKER_MAPPING = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "FACEBOOK": "META",
    "FB": "META",
}

# ============================================================================
# VALIDATION & HEALTH CHECK FUNCTIONS
# ============================================================================

class ValidationManager:
    """Comprehensive validation and health checks"""

    @staticmethod
    def validate_timeframes() -> Dict[str, Dict]:
        """Validate all timeframe configurations"""
        logger.info("=" * 60)
        logger.info("🔍 VALIDATING TIMEFRAME CONFIGURATIONS")
        logger.info("=" * 60)

        results = {}
        test_ticker = "AAPL"  # Use reliable ticker for testing

        for tf_name, tf_config in TIMEFRAMES.items():
            logger.info(f"Testing {tf_name}: period={tf_config['period']}, interval={tf_config['interval']}")

            try:
                data = yf.download(
                    test_ticker,
                    period=tf_config["period"],
                    interval=tf_config["interval"],
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if not data.empty:
                    results[tf_name] = {
                        "valid": True,
                        "samples": len(data),
                        "min_required": tf_config.get("min_samples", 30),
                        "status": "✅ PASS" if len(data) >= tf_config.get("min_samples", 30) else "⚠️ LOW DATA"
                    }
                    logger.info(f"  {results[tf_name]['status']}: {len(data)} samples")
                else:
                    results[tf_name] = {
                        "valid": False,
                        "samples": 0,
                        "status": "❌ FAIL"
                    }
                    logger.error(f"  ❌ FAIL: No data returned")

            except Exception as e:
                results[tf_name] = {
                    "valid": False,
                    "error": str(e),
                    "status": "❌ ERROR"
                }
                logger.error(f"  ❌ ERROR: {str(e)}")

            time.sleep(0.5)  # Rate limiting

        # Summary
        valid_count = sum(1 for r in results.values() if r.get("valid", False))
        logger.info("-" * 60)
        logger.info(f"📊 Validation Summary: {valid_count}/{len(results)} timeframes valid")

        # Save validation results
        validation_file = Config.VALIDATION_DIR / f"timeframe_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Validation results saved: {validation_file.name}")

        return results

    @staticmethod
    def validate_stock_availability(stocks: List[str]) -> Dict[str, bool]:
        """Check if stocks are available for download"""
        logger.info("🔍 Validating stock availability...")

        availability = {}
        for ticker in stocks:
            try:
                info = yf.Ticker(ticker).info
                availability[ticker] = bool(info.get("symbol"))
                status = "✅" if availability[ticker] else "❌"
                logger.info(f"  {status} {ticker}")
            except:
                availability[ticker] = False
                logger.warning(f"  ❌ {ticker}: Not available")

            time.sleep(0.2)  # Rate limiting

        return availability

    @staticmethod
    def check_data_quality(data: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Comprehensive data quality checks"""
        quality_report = {
            "ticker": ticker,
            "timeframe": timeframe,
            "total_rows": len(data),
            "issues": [],
            "quality_score": 100
        }

        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            quality_report["issues"].append(f"High missing values: {missing_pct:.1f}%")
            quality_report["quality_score"] -= 20

        # Check for duplicate timestamps
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            quality_report["issues"].append(f"Duplicate timestamps: {dup_count}")
            quality_report["quality_score"] -= 10

        # Check for data gaps
        if len(data) > 1:
            time_diffs = pd.Series(data.index).diff()
            expected_freq = pd.Timedelta(time_diffs.mode()[0])
            gaps = time_diffs[time_diffs > expected_freq * 2]
            if len(gaps) > 0:
                quality_report["issues"].append(f"Data gaps detected: {len(gaps)}")
                quality_report["quality_score"] -= 5

        # Check for outliers in price movements
        if "Close" in data.columns:
            returns = data["Close"].pct_change().dropna()
            extreme_moves = returns[abs(returns) > 0.2]  # 20% moves
            if len(extreme_moves) > 0:
                quality_report["issues"].append(f"Extreme price moves: {len(extreme_moves)}")
                quality_report["quality_score"] -= 5

        # Check volume consistency
        if "Volume" in data.columns:
            zero_volume = (data["Volume"] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                quality_report["issues"].append(f"High zero volume days: {zero_volume}")
                quality_report["quality_score"] -= 10

        # Log quality report
        if quality_report["issues"]:
            logger.warning(f"Data quality issues for {ticker} ({timeframe}): {quality_report['issues']}")
        else:
            logger.info(f"✅ Data quality good for {ticker} ({timeframe})")

        return quality_report

# ============================================================================
# ENHANCED DATA FETCHING & PROCESSING
# ============================================================================

class DataFetcher:
    """Enhanced data fetching with retry logic and fallbacks"""

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """Normalize ticker symbol"""
        ticker = ticker.upper().strip()
        return TICKER_MAPPING.get(ticker, ticker)

    @staticmethod
    def fetch_with_retry(ticker: str, config: Dict, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch data with retry logic"""
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    ticker,
                    period=config["period"],
                    interval=config["interval"],
                    auto_adjust=True,
                    prepost=False,
                    progress=False,
                    threads=False
                )

                if not data.empty:
                    return data

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(Config.RETRY_DELAY * (attempt + 1))

        return None

    @staticmethod
    def fetch_stock_data(ticker: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Enhanced stock data fetching with comprehensive fallbacks"""
        ticker = DataFetcher.normalize_ticker(ticker)
        config = TIMEFRAMES[timeframe]

        logger.info(f"📊 Fetching {ticker} ({timeframe}): period={config['period']}, interval={config['interval']}")

        # Primary attempt
        data = DataFetcher.fetch_with_retry(ticker, config)

        # Try fallback configurations if primary fails
        if data is None or data.empty:
            if timeframe in TIMEFRAME_FALLBACKS:
                logger.warning(f"⚠️ Primary fetch failed for {ticker} ({timeframe}), trying fallbacks...")

                for fallback_config in TIMEFRAME_FALLBACKS[timeframe]:
                    logger.info(f"  Trying fallback: {fallback_config}")
                    data = DataFetcher.fetch_with_retry(ticker, fallback_config, max_retries=2)
                    if data is not None and not data.empty:
                        logger.info(f"  ✅ Fallback successful: {len(data)} rows")
                        break

        # Final validation
        if data is None or data.empty:
            logger.error(f"❌ Failed to fetch data for {ticker} ({timeframe})")
            return None

        # Check minimum data requirements
        min_required = config.get("min_samples", Config.MIN_DATA_POINTS)
        if len(data) < min_required:
            logger.error(f"❌ Insufficient data for {ticker}: {len(data)} < {min_required}")
            return None

        logger.info(f"✅ Successfully fetched {len(data)} rows for {ticker} ({timeframe})")
        return data

class FeatureEngineer:
    """Enhanced feature engineering with additional indicators"""

    @staticmethod
    def compute_features(data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute comprehensive technical features"""
        try:
            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

            # Ensure required columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return None

            # Remove rows with missing data
            data = data.dropna(subset=required_cols)

            if len(data) < Config.MIN_DATA_POINTS:
                logger.error(f"Insufficient data after cleaning: {len(data)} rows")
                return None

            # Basic returns
            data["Return"] = data["Close"].pct_change().fillna(0)
            data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1)).fillna(0)

            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(data) >= period:
                    data[f"MA{period}"] = data["Close"].rolling(window=period, min_periods=1).mean()

            # Volatility measures
            data["Volatility"] = data["Return"].rolling(window=10, min_periods=1).std().fillna(0)
            data["Volatility_20"] = data["Return"].rolling(window=20, min_periods=1).std().fillna(0)

            # Volume features
            data["Volume_MA"] = data["Volume"].rolling(window=10, min_periods=1).mean()
            data["Volume_Ratio"] = (data["Volume"] / data["Volume_MA"]).fillna(1)
            data["Volume_Change"] = data["Volume"].pct_change().fillna(0)

            # RSI calculation
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)
            data["RSI"] = 100 - (100 / (1 + rs))
            data["RSI"] = data["RSI"].fillna(50)

            # Bollinger Bands
            if len(data) >= 20:
                bb_period = 20
                bb_std = 2
                data["BB_Middle"] = data["Close"].rolling(window=bb_period, min_periods=1).mean()
                bb_std_dev = data["Close"].rolling(window=bb_period, min_periods=1).std()
                data["BB_Upper"] = data["BB_Middle"] + (bb_std * bb_std_dev)
                data["BB_Lower"] = data["BB_Middle"] - (bb_std * bb_std_dev)
                data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"])
                data["BB_Position"] = data["BB_Position"].fillna(0.5)

            # MACD
            if len(data) >= 26:
                exp1 = data["Close"].ewm(span=12, adjust=False).mean()
                exp2 = data["Close"].ewm(span=26, adjust=False).mean()
                data["MACD"] = exp1 - exp2
                data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
                data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

            # Price position features
            high_low_range = (data["High"] - data["Low"]).replace(0, 1e-10)
            data["High_Low_Ratio"] = high_low_range / data["Close"]
            data["Close_Position"] = ((data["Close"] - data["Low"]) / high_low_range).fillna(0.5)

            # Trend features
            if "MA5" in data.columns:
                data["Price_Above_MA5"] = (data["Close"] > data["MA5"]).astype(int)
            if "MA20" in data.columns:
                data["Price_Above_MA20"] = (data["Close"] > data["MA20"]).astype(int)

            # Market microstructure
            data["Spread"] = ((data["High"] - data["Low"]) / data["Close"]).fillna(0)
            data["Gap"] = ((data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)).fillna(0)

            # Clean data
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                data[col] = data[col].fillna(data[col].median())

            # Replace infinite values
            data = data.replace([np.inf, -np.inf], 0)

            return data

        except Exception as e:
            logger.error(f"Error computing features: {str(e)}")
            return None

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of feature columns for training"""
        return [
            "Return", "Log_Return", "Volatility", "Volatility_20",
            "Volume_Ratio", "Volume_Change", "RSI",
            "High_Low_Ratio", "Close_Position", "Spread", "Gap",
            "Price_Above_MA5", "Price_Above_MA20",
            "BB_Position", "MACD_Histogram"
        ]

# ============================================================================
# ENHANCED MODEL TRAINING
# ============================================================================

class ModelTrainer:
    """Enhanced model training with versioning and tracking"""

    @staticmethod
    def prepare_training_data(ticker: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Prepare data for training with quality checks"""
        try:
            # Fetch data
            data = DataFetcher.fetch_stock_data(ticker, timeframe)
            if data is None:
                return None

            # Quality check
            quality_report = ValidationManager.check_data_quality(data, ticker, timeframe)
            if quality_report["quality_score"] < 50:
                logger.warning(f"⚠️ Low quality data for {ticker} ({timeframe}): score={quality_report['quality_score']}")

            # Compute features
            data = FeatureEngineer.compute_features(data)
            if data is None:
                return None

            # Create target variable
            data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

            # Remove last row (no target) and rows with NaN
            data = data[:-1].dropna()

            if len(data) < Config.MIN_DATA_POINTS:
                logger.error(f"Insufficient data for {ticker}: {len(data)} rows")
                return None

            logger.info(f"✅ Prepared {len(data)} samples for {ticker} ({timeframe})")
            return data

        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {str(e)}")
            return None

    @staticmethod
    def create_model(model_type: str = "ensemble") -> Any:
        """Create model based on type"""
        if model_type == "ensemble":
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=Config.RANDOM_STATE,
                n_jobs=1
            )
            gb = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=Config.RANDOM_STATE
            )
            lr = LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                solver='lbfgs'
            )
            return VotingClassifier(
                estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
                voting="soft",
                n_jobs=1
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=Config.RANDOM_STATE,
                n_jobs=1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def train_model_for_ticker(ticker: str, timeframe: str = "1d",
                              model_type: str = "ensemble") -> Dict[str, Any]:
        """Train model with comprehensive tracking"""
        try:
            start_time = time.time()
            logger.info(f"🤖 Training {model_type} model for {ticker} ({timeframe})")

            # Prepare data
            data = ModelTrainer.prepare_training_data(ticker, timeframe)
            if data is None:
                return {"error": f"Failed to prepare data for {ticker}"}

            # Get features
            feature_cols = FeatureEngineer.get_feature_columns()
            available_features = [col for col in feature_cols if col in data.columns]

            if len(available_features) < 5:
                return {"error": f"Insufficient features: {len(available_features)}"}

            # Prepare X and y
            X = data[available_features].values
            y = data["Target"].values

            # Check for and remove non-finite values
            finite_mask = np.isfinite(X).all(axis=1)
            X = X[finite_mask]
            y = y[finite_mask]

            if len(X) < Config.MIN_DATA_POINTS:
                return {"error": f"Insufficient clean data: {len(X)} samples"}

            # Check class balance
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if len(unique_classes) < 2:
                return {"error": "Only one class present in target variable"}

            class_balance = min(class_counts) / max(class_counts)
            if class_balance < 0.2:
                logger.warning(f"⚠️ Class imbalance detected: {class_balance:.2%}")

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=Config.TEST_SIZE,
                    random_state=Config.RANDOM_STATE,
                    stratify=y
                )
            except ValueError:
                # If stratify fails, do without it
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=Config.TEST_SIZE,
                    random_state=Config.RANDOM_STATE
                )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create and train model
            model = ModelTrainer.create_model(model_type)
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "class_balance": class_balance,
                "training_time": time.time() - start_time
            }

            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train,
                    cv=Config.CV_FOLDS,
                    scoring='accuracy'
                )
                metrics["cv_mean"] = cv_scores.mean()
                metrics["cv_std"] = cv_scores.std()
            except:
                metrics["cv_mean"] = metrics["accuracy"]
                metrics["cv_std"] = 0.0

            # Generate model hash for versioning
            model_hash = hashlib.md5(
                f"{ticker}_{timeframe}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]

            # Save model
            model_filename = f"{ticker}_model_{timeframe}_v{model_hash}.pkl"
            model_path = Config.MODELS_DIR / model_filename

            model_data = {
                "model": model,
                "scaler": scaler,
                "features": available_features,
                "accuracy": metrics["accuracy"],
                "metrics": metrics,
                "ticker": ticker,
                "timeframe": timeframe,
                "model_type": model_type,
                "version": model_hash,
                "trained_at": datetime.now().isoformat(),
                "training_config": {
                    "test_size": Config.TEST_SIZE,
                    "random_state": Config.RANDOM_STATE,
                    "cv_folds": Config.CV_FOLDS
                }
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            # Log performance
            performance_status = "🎯" if metrics["accuracy"] >= Config.TARGET_ACCURACY else "✅"
            logger.info(f"{performance_status} Model saved: {model_filename} (Accuracy: {metrics['accuracy']:.2%})")

            # Check if model meets minimum standards
            if metrics["accuracy"] < Config.MIN_ACCEPTABLE_ACCURACY:
                logger.warning(f"⚠️ Low accuracy model: {metrics['accuracy']:.2%}")

            return {
                "success": True,
                "ticker": ticker,
                "timeframe": timeframe,
                "metrics": metrics,
                "model_path": str(model_path),
                "version": model_hash
            }

        except Exception as e:
            logger.error(f"❌ Training failed for {ticker} ({timeframe}): {str(e)}")
            return {"error": str(e)}

    @staticmethod
    def train_universal_model(timeframe: str = "1d",
                            model_type: str = "ensemble") -> Dict[str, Any]:
        """Train universal model with enhanced data collection"""
        try:
            logger.info(f"🌍 Training universal {model_type} model for {timeframe}")

            # Select reliable stocks
            universal_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "QQQ", "JPM"]

            all_data = []
            successful_stocks = []
            data_quality_scores = []

            # Collect data from multiple stocks
            for ticker in universal_stocks:
                try:
                    data = ModelTrainer.prepare_training_data(ticker, timeframe)
                    if data is not None and len(data) >= 50:
                        # Quality check
                        quality = ValidationManager.check_data_quality(data, ticker, timeframe)
                        data_quality_scores.append(quality["quality_score"])

                        # Limit to recent samples to balance dataset
                        max_samples = 200
                        if len(data) > max_samples:
                            data = data.tail(max_samples)

                        all_data.append(data)
                        successful_stocks.append(ticker)
                        logger.info(f"  ✅ Added {ticker}: {len(data)} samples (quality: {quality['quality_score']})")

                except Exception as e:
                    logger.warning(f"  ⚠️ Skipped {ticker}: {str(e)}")
                    continue

                time.sleep(Config.API_DELAY)  # Rate limiting

            if len(successful_stocks) < 3:
                raise ValueError(f"Insufficient stocks for universal model: {len(successful_stocks)}")

            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            avg_quality = np.mean(data_quality_scores) if data_quality_scores else 0

            logger.info(f"📊 Combined: {len(combined_data)} samples from {len(successful_stocks)} stocks")
            logger.info(f"📈 Average data quality: {avg_quality:.1f}")

            # Get features
            feature_cols = FeatureEngineer.get_feature_columns()
            available_features = [col for col in feature_cols if col in combined_data.columns]

            # Prepare training data
            X = combined_data[available_features].values
            y = combined_data["Target"].values

            # Clean data
            finite_mask = np.isfinite(X).all(axis=1)
            X = X[finite_mask]
            y = y[finite_mask]

            if len(X) < 100:
                raise ValueError(f"Insufficient clean data: {len(X)} samples")

            # Split and scale
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create and train model
            model = ModelTrainer.create_model(model_type)
            model.fit(X_train_scaled, y_train)

            accuracy = model.score(X_test_scaled, y_test)

            # Generate version hash
            model_hash = hashlib.md5(
                f"universal_{timeframe}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]

            # Save model
            model_filename = f"universal_model_{timeframe}_v{model_hash}.pkl"
            model_path = Config.MODELS_DIR / model_filename

            model_data = {
                "model": model,
                "scaler": scaler,
                "features": available_features,
                "accuracy": accuracy,
                "timeframe": timeframe,
                "training_tickers": successful_stocks,
                "model_type": model_type,
                "version": model_hash,
                "trained_at": datetime.now().isoformat(),
                "data_quality": avg_quality,
                "total_samples": len(X)
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)

            performance_status = "🎯" if accuracy >= Config.TARGET_ACCURACY else "✅"
            logger.info(f"{performance_status} Universal model saved: {model_filename} (Accuracy: {accuracy:.2%})")

            return {
                "success": True,
                "timeframe": timeframe,
                "accuracy": accuracy,
                "samples": len(X_train),
                "path": str(model_path),
                "version": model_hash,
                "data_quality": avg_quality
            }

        except Exception as e:
            logger.error(f"❌ Universal model failed for {timeframe}: {str(e)}")
            return {"error": str(e)}

# ============================================================================
# ENHANCED TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Enhanced training pipeline with monitoring and recovery"""

    @staticmethod
    def run_pipeline(training_mode: str = "essential",
                    model_type: str = "ensemble",
                    validate_first: bool = True) -> Dict[str, Any]:
        """Main training pipeline with enhanced features"""

        logger.info("=" * 80)
        logger.info(f"🚀 STOCKVIBEPREDICTOR MODEL TRAINING SYSTEM v{Config.VERSION}")
        logger.info(f"📅 Build Date: {Config.BUILD_DATE}")
        logger.info("=" * 80)

        start_time = time.time()

        # Step 1: Validation (optional)
        if validate_first:
            logger.info("📋 Running pre-training validation...")
            validation_results = ValidationManager.validate_timeframes()

            # Filter out invalid timeframes
            invalid_timeframes = [tf for tf, result in validation_results.items()
                                 if not result.get("valid", False)]
            if invalid_timeframes:
                logger.warning(f"⚠️ Invalid timeframes will be skipped: {invalid_timeframes}")

        # Define timeframe modes
        timeframe_modes = {
            "minimal": ["1d", "1w", "1mo"],
            "essential": ["1d", "1w", "1mo", "1y"],
            "extended": ["1d", "5d", "1w", "1mo", "3mo", "1y"],
            "complete": ["1d", "5d", "1w", "1mo", "3mo", "6mo", "1y", "2y", "5y"]
        }

        timeframes = timeframe_modes.get(training_mode, timeframe_modes["essential"])

        # Filter out invalid timeframes if validation was run
        if validate_first and 'validation_results' in locals():
            timeframes = [tf for tf in timeframes
                         if validation_results.get(tf, {}).get("valid", True)]

        logger.info(f"🎯 Training Mode: {training_mode}")
        logger.info(f"⏱️ Timeframes: {timeframes}")
        logger.info(f"🤖 Model Type: {model_type}")

        # Initialize tracking
        training_summary = {
            "version": Config.VERSION,
            "start_time": datetime.now().isoformat(),
            "training_mode": training_mode,
            "model_type": model_type,
            "timeframes": timeframes,
            "universal_models": {},
            "category_models": {},
            "total_models": 0,
            "successful_models": 0,
            "failed_models": [],
            "performance_summary": {
                "excellent": 0,  # >= 0.6
                "good": 0,       # >= 0.55
                "acceptable": 0, # >= 0.45
                "poor": 0        # < 0.45
            }
        }

        # Phase 1: Train Universal Models
        logger.info("\n" + "=" * 60)
        logger.info("📊 PHASE 1: TRAINING UNIVERSAL MODELS")
        logger.info("=" * 60)

        for timeframe in timeframes:
            result = ModelTrainer.train_universal_model(timeframe, model_type)
            training_summary["universal_models"][timeframe] = result
            training_summary["total_models"] += 1

            if result.get("success"):
                training_summary["successful_models"] += 1
                accuracy = result.get("accuracy", 0)

                # Categorize performance
                if accuracy >= 0.6:
                    training_summary["performance_summary"]["excellent"] += 1
                elif accuracy >= 0.55:
                    training_summary["performance_summary"]["good"] += 1
                elif accuracy >= 0.45:
                    training_summary["performance_summary"]["acceptable"] += 1
                else:
                    training_summary["performance_summary"]["poor"] += 1

                logger.info(f"✅ Universal {timeframe}: Success (Accuracy: {accuracy:.2%})")
            else:
                training_summary["failed_models"].append(f"universal_{timeframe}")
                logger.error(f"❌ Universal {timeframe}: Failed")

        # Phase 2: Train Individual Stock Models
        logger.info("\n" + "=" * 60)
        logger.info("🏢 PHASE 2: TRAINING INDIVIDUAL STOCK MODELS")
        logger.info("=" * 60)

        # For individual stocks, use subset of timeframes
        stock_timeframes = ["1d", "1w", "1mo"]
        if training_mode in ["extended", "complete"]:
            stock_timeframes.extend(["6mo", "1y"])

        for category, stocks in STOCK_DATABASE.items():
            logger.info(f"\n{'─' * 50}")
            logger.info(f"🏷️ Category: {category}")
            logger.info(f"📊 Stocks: {stocks}")
            logger.info('─' * 50)

            category_results = {}

            for stock in stocks:
                for tf in stock_timeframes:
                    model_key = f"{stock}_{tf}"
                    logger.info(f"  Training {model_key}...")

                    # Train model
                    result = ModelTrainer.train_model_for_ticker(stock, tf, model_type)
                    category_results[model_key] = result

                    training_summary["total_models"] += 1
                    if result.get("success"):
                        training_summary["successful_models"] += 1
                        accuracy = result["metrics"]["accuracy"]

                        # Categorize performance
                        if accuracy >= 0.6:
                            training_summary["performance_summary"]["excellent"] += 1
                        elif accuracy >= 0.55:
                            training_summary["performance_summary"]["good"] += 1
                        elif accuracy >= 0.45:
                            training_summary["performance_summary"]["acceptable"] += 1
                        else:
                            training_summary["performance_summary"]["poor"] += 1

                        logger.info(f"    ✅ Success (Accuracy: {accuracy:.2%})")
                    else:
                        training_summary["failed_models"].append(model_key)
                        logger.info(f"    ❌ Failed: {result.get('error', 'Unknown error')}")

                    # Rate limiting
                    time.sleep(Config.API_DELAY)

            training_summary["category_models"][category] = category_results

            # Category summary
            successful = sum(1 for r in category_results.values() if r.get("success"))
            total = len(category_results)
            success_rate = (successful / total * 100) if total > 0 else 0
            logger.info(f"📊 {category} Summary: {successful}/{total} successful ({success_rate:.1f}%)")

        # Final Summary
        end_time = time.time()
        duration = end_time - start_time
        training_summary["end_time"] = datetime.now().isoformat()
        training_summary["duration_seconds"] = duration
        training_summary["duration_formatted"] = f"{duration//60:.0f}m {duration%60:.0f}s"

        success_rate = (training_summary["successful_models"] / training_summary["total_models"] * 100
                       if training_summary["total_models"] > 0 else 0)

        # Display final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"📈 Total Models: {training_summary['total_models']}")
        logger.info(f"✅ Successful: {training_summary['successful_models']}")
        logger.info(f"❌ Failed: {len(training_summary['failed_models'])}")
        logger.info(f"🎯 Success Rate: {success_rate:.1f}%")
        logger.info(f"⏰ Duration: {training_summary['duration_formatted']}")

        logger.info("\n📊 PERFORMANCE DISTRIBUTION:")
        logger.info(f"  🌟 Excellent (≥60%): {training_summary['performance_summary']['excellent']}")
        logger.info(f"  ✅ Good (≥55%): {training_summary['performance_summary']['good']}")
        logger.info(f"  ⚠️ Acceptable (≥45%): {training_summary['performance_summary']['acceptable']}")
        logger.info(f"  ❌ Poor (<45%): {training_summary['performance_summary']['poor']}")

        # Save comprehensive summary
        summary_file = Config.PERFORMANCE_DIR / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w") as f:
            json.dump(training_summary, f, indent=2, default=str)

        logger.info(f"\n💾 Summary saved: {summary_file.name}")

        # Save a copy as latest
        latest_file = Config.PERFORMANCE_DIR / "training_summary_latest.json"
        with open(latest_file, "w") as f:
            json.dump(training_summary, f, indent=2, default=str)

        return training_summary

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class ModelManager:
    """Model management utilities"""

    @staticmethod
    def validate_models() -> Tuple[int, int]:
        """Validate all models for compatibility"""
        logger.info("🔍 Validating models for compatibility...")

        compatible = 0
        incompatible = 0

        for model_file in Config.MODELS_DIR.glob("*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                # Check required keys
                required_keys = ["model", "features", "accuracy", "scaler"]
                missing_keys = [key for key in required_keys if key not in model_data]

                if missing_keys:
                    logger.error(f"  ❌ {model_file.name} missing keys: {missing_keys}")
                    incompatible += 1
                elif not hasattr(model_data["model"], "predict"):
                    logger.error(f"  ❌ {model_file.name} model has no predict method")
                    incompatible += 1
                else:
                    version = model_data.get("version", "unknown")
                    accuracy = model_data.get("accuracy", 0)
                    logger.info(f"  ✅ {model_file.name} (v{version}, acc: {accuracy:.2%})")
                    compatible += 1

            except Exception as e:
                logger.error(f"  ❌ {model_file.name} validation error: {str(e)}")
                incompatible += 1

        logger.info(f"\n🎯 Validation complete: {compatible} compatible, {incompatible} incompatible")
        return compatible, incompatible

    @staticmethod
    def get_model_summary() -> List[Dict]:
        """Get comprehensive summary of all trained models"""
        models = []

        for model_file in Config.MODELS_DIR.glob("*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)

                models.append({
                    "file": model_file.name,
                    "ticker": model_data.get("ticker", "universal"),
                    "timeframe": model_data.get("timeframe", "unknown"),
                    "accuracy": model_data.get("accuracy", 0),
                    "version": model_data.get("version", "unknown"),
                    "model_type": model_data.get("model_type", "unknown"),
                    "trained_at": model_data.get("trained_at", "unknown"),
                    "file_size_mb": model_file.stat().st_size / (1024 * 1024)
                })
            except:
                continue

        if models:
            df = pd.DataFrame(models)

            logger.info("\n" + "=" * 80)
            logger.info("📊 MODEL PORTFOLIO SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total Models: {len(models)}")
            logger.info(f"Average Accuracy: {df['accuracy'].mean():.2%}")
            logger.info(f"Accuracy Range: {df['accuracy'].min():.2%} - {df['accuracy'].max():.2%}")
            logger.info(f"Total Size: {df['file_size_mb'].sum():.1f} MB")

            # Group by timeframe
            logger.info("\n📈 Accuracy by Timeframe:")
            timeframe_stats = df.groupby('timeframe')['accuracy'].agg(['mean', 'count', 'std'])
            for tf, row in timeframe_stats.iterrows():
                logger.info(f"  {tf:8s}: {row['mean']:.2%} ± {row['std']:.2%} ({int(row['count'])} models)")

            # Top performers
            logger.info("\n🏆 Top 10 Models:")
            for idx, row in df.nlargest(10, 'accuracy').iterrows():
                logger.info(f"  {row['ticker']:8s} {row['timeframe']:5s}: {row['accuracy']:.2%} (v{row['version']})")

            # Poor performers
            poor_models = df[df['accuracy'] < Config.MIN_ACCEPTABLE_ACCURACY]
            if len(poor_models) > 0:
                logger.warning(f"\n⚠️ Models below minimum accuracy ({Config.MIN_ACCEPTABLE_ACCURACY:.0%}):")
                for idx, row in poor_models.iterrows():
                    logger.warning(f"  {row['ticker']:8s} {row['timeframe']:5s}: {row['accuracy']:.2%}")

        return models

    @staticmethod
    def cleanup_old_models(keep_latest: int = 3):
        """Clean up old model versions, keeping only the latest N versions"""
        logger.info(f"🧹 Cleaning up old models (keeping latest {keep_latest} versions)...")

        # Group models by ticker and timeframe
        model_groups = {}
        for model_file in Config.MODELS_DIR.glob("*.pkl"):
            # Parse filename to extract ticker and timeframe
            parts = model_file.stem.split("_")
            if len(parts) >= 3:
                key = f"{parts[0]}_{parts[2]}"  # ticker_timeframe
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append(model_file)

        total_deleted = 0
        space_freed = 0

        for key, files in model_groups.items():
            if len(files) > keep_latest:
                # Sort by modification time
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # Delete old versions
                for old_file in files[keep_latest:]:
                    file_size = old_file.stat().st_size
                    old_file.unlink()
                    total_deleted += 1
                    space_freed += file_size
                    logger.info(f"  Deleted: {old_file.name}")

        logger.info(f"✅ Cleanup complete: {total_deleted} files deleted, {space_freed/(1024*1024):.1f} MB freed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "help"

    logger.info(f"🚀 StockVibePredictor Model Training System v{Config.VERSION}")
    logger.info("=" * 80)

    if mode == "help":
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║     StockVibePredictor Model Training System v5.0 (ENHANCED)              ║
╚════════════════════════════════════════════════════════════════════════════╝

TRAINING MODES:
  minimal      - Train 1d, 1w, 1mo (fastest, ~10-15 min)
  essential    - Train 1d, 1w, 1mo, 1y (recommended, ~20-30 min)
  extended     - Train 1d, 5d, 1w, 1mo, 3mo, 1y (thorough, ~45-60 min)
  complete     - Train all 9 timeframes (comprehensive, ~90-120 min)

UTILITIES:
  validate     - Validate timeframe configurations
  check        - Check models compatibility
  summary      - Display comprehensive model statistics
  cleanup      - Remove old model versions
  test         - Test specific timeframe: test [timeframe]
  single       - Train single model: single [ticker] [timeframe]

ENHANCED FEATURES:
  benchmark    - Run performance benchmarking
  health       - System health check
  export       - Export model metadata

EXAMPLES:
  python TrainModel.py essential       # Recommended training
  python TrainModel.py validate        # Pre-training validation
  python TrainModel.py summary         # View model portfolio
  python TrainModel.py cleanup         # Clean old models
  python TrainModel.py test 5d         # Test 5d timeframe
  python TrainModel.py single AAPL 1mo # Train single model

OPTIONS:
  --no-validate    Skip pre-training validation
  --model-type     Specify model type (ensemble/random_forest)
  --keep-models    Number of model versions to keep (default: 3)
        """)

    elif mode in ["minimal", "essential", "extended", "complete"]:
        # Check for options
        validate_first = "--no-validate" not in sys.argv
        model_type = "ensemble"  # Default

        for i, arg in enumerate(sys.argv):
            if arg == "--model-type" and i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]

        result = TrainingPipeline.run_pipeline(
            training_mode=mode,
            model_type=model_type,
            validate_first=validate_first
        )

    elif mode == "validate":
        ValidationManager.validate_timeframes()

    elif mode == "check":
        ModelManager.validate_models()

    elif mode == "summary":
        ModelManager.get_model_summary()

    elif mode == "cleanup":
        keep = 3
        for i, arg in enumerate(sys.argv):
            if arg == "--keep-models" and i + 1 < len(sys.argv):
                keep = int(sys.argv[i + 1])
        ModelManager.cleanup_old_models(keep_latest=keep)

    elif mode == "test":
        if len(sys.argv) < 3:
            print("Usage: python TrainModel.py test [timeframe]")
            print("Example: python TrainModel.py test 5d")
        else:
            timeframe = sys.argv[2]
            test_ticker = "AAPL"
            logger.info(f"Testing {timeframe} timeframe with {test_ticker}...")

            data = DataFetcher.fetch_stock_data(test_ticker, timeframe)
            if data is not None:
                quality = ValidationManager.check_data_quality(data, test_ticker, timeframe)
                logger.info(f"✅ Test successful!")
                logger.info(f"  Data points: {len(data)}")
                logger.info(f"  Quality score: {quality['quality_score']}")
            else:
                logger.error(f"❌ Test failed for {timeframe}")

    elif mode == "single":
        if len(sys.argv) < 4:
            print("Usage: python TrainModel.py single [ticker] [timeframe]")
            print("Example: python TrainModel.py single AAPL 1mo")
        else:
            ticker = sys.argv[2]
            timeframe = sys.argv[3]
            model_type = "ensemble"

            for i, arg in enumerate(sys.argv):
                if arg == "--model-type" and i + 1 < len(sys.argv):
                    model_type = sys.argv[i + 1]

            result = ModelTrainer.train_model_for_ticker(ticker, timeframe, model_type)
            if result.get("success"):
                logger.info(f"✅ Successfully trained {ticker} ({timeframe})")
                logger.info(f"   Accuracy: {result['metrics']['accuracy']:.2%}")
                logger.info(f"   Version: {result['version']}")
            else:
                logger.error(f"❌ Failed to train {ticker} ({timeframe}): {result.get('error')}")

    elif mode == "benchmark":
        logger.info("🏃 Running performance benchmark...")

        # Quick benchmark on key models
        benchmark_configs = [
            ("AAPL", "1d"),
            ("SPY", "1w"),
            ("GOOGL", "1mo")
        ]

        results = []
        for ticker, timeframe in benchmark_configs:
            start = time.time()
            result = ModelTrainer.train_model_for_ticker(ticker, timeframe)
            duration = time.time() - start

            if result.get("success"):
                results.append({
                    "model": f"{ticker}_{timeframe}",
                    "accuracy": result["metrics"]["accuracy"],
                    "time": duration
                })

        if results:
            logger.info("\n📊 Benchmark Results:")
            for r in results:
                logger.info(f"  {r['model']:12s}: {r['accuracy']:.2%} (trained in {r['time']:.1f}s)")

    elif mode == "health":
        logger.info("🏥 Running system health check...")

        # Check directories
        logger.info("\n📁 Directory Check:")
        for name, path in [("Models", Config.MODELS_DIR),
                          ("Logs", Config.LOGS_DIR),
                          ("Performance", Config.PERFORMANCE_DIR)]:
            exists = path.exists()
            status = "✅" if exists else "❌"
            logger.info(f"  {status} {name}: {path}")

        # Check dependencies
        logger.info("\n📦 Dependencies Check:")
        try:
            import yfinance
            import sklearn
            import pandas
            import numpy
            logger.info("  ✅ All required packages installed")
        except ImportError as e:
            logger.error(f"  ❌ Missing dependency: {e}")

        # Check API connectivity
        logger.info("\n🌐 API Connectivity Check:")
        try:
            test_data = yf.download("AAPL", period="1d", progress=False)
            if not test_data.empty:
                logger.info("  ✅ Yahoo Finance API accessible")
            else:
                logger.error("  ❌ Yahoo Finance API returned no data")
        except Exception as e:
            logger.error(f"  ❌ Yahoo Finance API error: {e}")

        # Check existing models
        logger.info("\n📊 Model Inventory:")
        model_count = len(list(Config.MODELS_DIR.glob("*.pkl")))
        logger.info(f"  Total models: {model_count}")

        if model_count > 0:
            compatible, incompatible = ModelManager.validate_models()

    elif mode == "export":
        logger.info("📤 Exporting model metadata...")

        models = ModelManager.get_model_summary()
        if models:
            export_file = Config.PERFORMANCE_DIR / f"model_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_file, "w") as f:
                json.dump(models, f, indent=2, default=str)
            logger.info(f"✅ Metadata exported to: {export_file.name}")

    else:
        logger.error(f"Unknown mode: {mode}")
        logger.info("Run 'python TrainModel.py help' for usage information")

    logger.info("\n✅ Execution completed successfully")

if __name__ == "__main__":
    main()
