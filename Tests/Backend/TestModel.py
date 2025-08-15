#!/usr/bin/env python3
"""
ComprehensiveTestModel.py - Complete ML Model Testing Suite
Tests ALL StockVibePredictor ML models in the Models directory
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback
from collections import defaultdict
import logging

warnings.filterwarnings("ignore")

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Backend"))

class ComprehensiveModelTestSuite:
    """Complete test suite for all ML models"""

    def __init__(self):
        """Initialize test suite"""
        self.base_dir = PROJECT_ROOT
        self.model_dir = self.base_dir / "Backend" / "Scripts" / "Models"
        self.results_dir = self.base_dir / "Backend" / "Scripts" / "TestResults"
        self.results_dir.mkdir(exist_ok=True)

        # Initialize results storage
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "total_models": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "models": {},
            "summary": {},
            "performance_rankings": {},
            "errors": []
        }

        # Setup logging
        self.setup_logging()

        print(f"🎯 Comprehensive Model Testing Suite")
        print(f"📁 Model Directory: {self.model_dir}")
        print(f"📊 Results Directory: {self.results_dir}")

    def setup_logging(self):
        """Setup logging system"""
        log_file = self.results_dir / f"model_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def discover_models(self):
        """Discover all model files"""
        model_files = list(self.model_dir.glob("*.pkl"))
        self.logger.info(f"📋 Discovered {len(model_files)} model files")

        if not model_files:
            self.logger.error("❌ No model files found!")
            return []

        # Categorize models
        categories = {
            "universal": [],
            "individual_stocks": [],
            "etfs": [],
            "indices": [],
            "crypto": [],
            "other": []
        }

        for model_file in model_files:
            name = model_file.stem.lower()
            if "universal" in name:
                categories["universal"].append(model_file)
            elif any(x in name for x in ["spy", "qqq", "arkk", "iwm", "vti"]):
                categories["etfs"].append(model_file)
            elif any(x in name for x in ["index_", "nsei"]):
                categories["indices"].append(model_file)
            elif any(x in name for x in ["coin", "sq", "pltr", "hood"]):
                categories["crypto"].append(model_file)
            else:
                categories["individual_stocks"].append(model_file)

        # Print categorization
        for category, files in categories.items():
            if files:
                self.logger.info(f"  📂 {category.replace('_', ' ').title()}: {len(files)} models")

        return model_files

    def generate_test_features(self, model_data):
        """Generate realistic test features for model"""
        try:
            required_features = model_data.get("features", [])

            # Create realistic dummy data
            test_data = {}

            for feature in required_features:
                feature_lower = feature.lower()

                # Price-based features
                if any(x in feature_lower for x in ["close", "open", "high", "low", "price"]):
                    base_price = np.random.uniform(100, 400)
                    if "high" in feature_lower:
                        test_data[feature] = [base_price + np.random.uniform(0, 20)]
                    elif "low" in feature_lower:
                        test_data[feature] = [base_price - np.random.uniform(0, 20)]
                    else:
                        test_data[feature] = [base_price + np.random.uniform(-10, 10)]

                # Volume features
                elif "volume" in feature_lower:
                    if "change" in feature_lower or "ratio" in feature_lower:
                        test_data[feature] = [np.random.uniform(-0.5, 2.0)]
                    elif "spike" in feature_lower:
                        test_data[feature] = [np.random.choice([0, 1])]
                    else:
                        test_data[feature] = [np.random.uniform(1000000, 200000000)]

                # Technical indicators
                elif "rsi" in feature_lower:
                    test_data[feature] = [np.random.uniform(20, 80)]
                elif "macd" in feature_lower:
                    if "histogram" in feature_lower:
                        test_data[feature] = [np.random.uniform(-5, 5)]
                    elif "signal" in feature_lower:
                        test_data[feature] = [np.random.uniform(-3, 3)]
                    elif "bullish" in feature_lower:
                        test_data[feature] = [np.random.choice([0, 1])]
                    else:
                        test_data[feature] = [np.random.uniform(-10, 10)]

                # Moving averages
                elif "ma" in feature_lower and any(char.isdigit() for char in feature):
                    base_price = np.random.uniform(100, 400)
                    test_data[feature] = [base_price + np.random.uniform(-50, 50)]

                # Bollinger Bands
                elif "bb" in feature_lower:
                    if "width" in feature_lower:
                        test_data[feature] = [np.random.uniform(0.01, 0.2)]
                    elif "position" in feature_lower:
                        test_data[feature] = [np.random.uniform(0, 1)]
                    elif "squeeze" in feature_lower:
                        test_data[feature] = [np.random.choice([0, 1])]
                    else:
                        test_data[feature] = [np.random.uniform(100, 500)]

                # Stochastic
                elif "stoch" in feature_lower:
                    if any(x in feature_lower for x in ["overbought", "oversold"]):
                        test_data[feature] = [np.random.choice([0, 1])]
                    else:
                        test_data[feature] = [np.random.uniform(0, 100)]

                # Binary indicators
                elif any(x in feature_lower for x in ["above", "bullish", "spike", "trend", "squeeze"]):
                    test_data[feature] = [np.random.choice([0, 1])]

                # Percentage features
                elif "pct" in feature_lower or "return" in feature_lower:
                    test_data[feature] = [np.random.uniform(-0.1, 0.1)]

                # Volatility
                elif "volatility" in feature_lower:
                    test_data[feature] = [np.random.uniform(0.01, 0.1)]

                # ATR
                elif "atr" in feature_lower:
                    test_data[feature] = [np.random.uniform(1, 20)]

                # Williams %R
                elif "williams" in feature_lower:
                    test_data[feature] = [np.random.uniform(-100, 0)]

                # OBV, VWAP
                elif "obv" in feature_lower:
                    test_data[feature] = [np.random.uniform(-1000000, 1000000)]
                elif "vwap" in feature_lower:
                    test_data[feature] = [np.random.uniform(100, 400)]

                # Ticker hash
                elif "ticker" in feature_lower and "hash" in feature_lower:
                    test_data[feature] = [np.random.randint(0, 1000)]

                # Default fallback
                else:
                    test_data[feature] = [np.random.uniform(0, 100)]

            return pd.DataFrame(test_data)

        except Exception as e:
            self.logger.error(f"Error generating test features: {str(e)}")
            return None

    def test_single_model(self, model_file):
        """Test a single model comprehensively"""
        start_time = time.time()
        model_name = model_file.stem

        test_result = {
            "model_name": model_name,
            "file_path": str(model_file),
            "file_size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
            "tests": {},
            "overall_status": "UNKNOWN",
            "test_duration": 0,
            "error_messages": []
        }

        try:
            self.logger.info(f"🧪 Testing model: {model_name}")

            # Test 1: Model Loading
            try:
                with open(model_file, "rb") as f:
                    model_data = pickle.load(f)

                test_result["tests"]["loading"] = {
                    "status": "PASS",
                    "details": "Model loaded successfully"
                }

                # Extract model information
                model = model_data.get("model")
                scaler = model_data.get("scaler")
                features = model_data.get("features", [])
                ticker = model_data.get("ticker", "Unknown")
                timeframe = model_data.get("timeframe", "Unknown")
                accuracy = model_data.get("accuracy", 0)

                test_result["model_info"] = {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "accuracy": accuracy,
                    "features_count": len(features),
                    "has_scaler": scaler is not None,
                    "model_type": type(model).__name__ if model else "Unknown"
                }

            except Exception as e:
                test_result["tests"]["loading"] = {
                    "status": "FAIL",
                    "details": f"Failed to load model: {str(e)}"
                }
                test_result["error_messages"].append(f"Loading error: {str(e)}")
                test_result["overall_status"] = "FAILED"
                return test_result

            # Test 2: Model Structure
            try:
                has_predict = hasattr(model, 'predict')
                has_predict_proba = hasattr(model, 'predict_proba')

                test_result["tests"]["structure"] = {
                    "status": "PASS" if has_predict else "FAIL",
                    "details": {
                        "has_predict": has_predict,
                        "has_predict_proba": has_predict_proba,
                        "features_available": len(features) > 0
                    }
                }

                if not has_predict:
                    test_result["error_messages"].append("Model missing predict method")

            except Exception as e:
                test_result["tests"]["structure"] = {
                    "status": "FAIL",
                    "details": f"Structure test error: {str(e)}"
                }

            # Test 3: Feature Generation and Prediction
            try:
                if features and model:
                    # Generate test features
                    test_features = self.generate_test_features(model_data)

                    if test_features is not None and not test_features.empty:
                        # Scale features if scaler is available
                        if scaler:
                            test_features_scaled = pd.DataFrame(
                                scaler.transform(test_features),
                                columns=test_features.columns
                            )
                        else:
                            test_features_scaled = test_features

                        # Make prediction
                        prediction = model.predict(test_features_scaled)
                        prediction_value = prediction[0]
                        prediction_readable = "UP" if prediction_value == 1 else "DOWN"

                        # Get probabilities if available
                        probabilities = None
                        confidence = None
                        if hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(test_features_scaled)
                            if len(probabilities[0]) > 1:
                                confidence = max(probabilities[0])

                        test_result["tests"]["prediction"] = {
                            "status": "PASS",
                            "details": {
                                "prediction": prediction_readable,
                                "raw_prediction": int(prediction_value),
                                "confidence": float(confidence) if confidence else None,
                                "features_used": len(test_features.columns)
                            }
                        }

                    else:
                        test_result["tests"]["prediction"] = {
                            "status": "FAIL",
                            "details": "Failed to generate test features"
                        }

                else:
                    test_result["tests"]["prediction"] = {
                        "status": "SKIP",
                        "details": "No features or model available"
                    }

            except Exception as e:
                test_result["tests"]["prediction"] = {
                    "status": "FAIL",
                    "details": f"Prediction error: {str(e)}"
                }
                test_result["error_messages"].append(f"Prediction error: {str(e)}")

            # Test 4: Batch Prediction Performance
            try:
                if features and model:
                    batch_size = 10
                    batch_predictions = []
                    batch_start = time.time()

                    for i in range(batch_size):
                        test_features = self.generate_test_features(model_data)
                        if test_features is not None:
                            if scaler:
                                test_features_scaled = pd.DataFrame(
                                    scaler.transform(test_features),
                                    columns=test_features.columns
                                )
                            else:
                                test_features_scaled = test_features

                            pred = model.predict(test_features_scaled)[0]
                            batch_predictions.append(pred)

                    batch_time = time.time() - batch_start
                    avg_prediction_time = batch_time / batch_size

                    up_count = sum(1 for p in batch_predictions if p == 1)
                    down_count = len(batch_predictions) - up_count

                    test_result["tests"]["performance"] = {
                        "status": "PASS",
                        "details": {
                            "batch_size": batch_size,
                            "avg_prediction_time_ms": round(avg_prediction_time * 1000, 2),
                            "up_predictions": up_count,
                            "down_predictions": down_count,
                            "prediction_distribution": f"{up_count}UP/{down_count}DOWN"
                        }
                    }

            except Exception as e:
                test_result["tests"]["performance"] = {
                    "status": "FAIL",
                    "details": f"Performance test error: {str(e)}"
                }

            # Overall status determination
            passed_tests = sum(1 for test in test_result["tests"].values() if test["status"] == "PASS")
            total_tests = len(test_result["tests"])

            if passed_tests == total_tests:
                test_result["overall_status"] = "PASSED"
            elif passed_tests > 0:
                test_result["overall_status"] = "PARTIAL"
            else:
                test_result["overall_status"] = "FAILED"

        except Exception as e:
            test_result["overall_status"] = "ERROR"
            test_result["error_messages"].append(f"Critical error: {str(e)}")
            self.logger.error(f"❌ Critical error testing {model_name}: {str(e)}")

        test_result["test_duration"] = round(time.time() - start_time, 2)
        return test_result

    def run_all_tests(self, max_workers=4):
        """Run tests on all models"""
        print(f"\n🚀 Starting Comprehensive Model Testing...")
        print(f"⚡ Using {max_workers} parallel workers")

        # Discover models
        model_files = self.discover_models()
        if not model_files:
            print("❌ No models found to test!")
            return

        self.test_results["total_models"] = len(model_files)

        # Run tests in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_model = {
                executor.submit(self.test_single_model, model_file): model_file
                for model_file in model_files
            }

            # Collect results
            completed = 0
            for future in as_completed(future_to_model):
                model_file = future_to_model[future]
                completed += 1

                try:
                    result = future.result(timeout=120)  # 2 minute timeout per model
                    model_name = result["model_name"]
                    self.test_results["models"][model_name] = result

                    # Update counters
                    if result["overall_status"] in ["PASSED", "PARTIAL"]:
                        self.test_results["successful_tests"] += 1
                        status_icon = "✅" if result["overall_status"] == "PASSED" else "⚠️"
                    else:
                        self.test_results["failed_tests"] += 1
                        status_icon = "❌"

                    self.logger.info(f"{status_icon} [{completed}/{len(model_files)}] {model_name} - {result['overall_status']}")

                except Exception as e:
                    model_name = model_file.stem
                    error_msg = f"Test execution failed: {str(e)}"
                    self.test_results["models"][model_name] = {
                        "model_name": model_name,
                        "overall_status": "ERROR",
                        "error_messages": [error_msg]
                    }
                    self.test_results["failed_tests"] += 1
                    self.test_results["errors"].append(f"{model_name}: {error_msg}")
                    self.logger.error(f"❌ [{completed}/{len(model_files)}] {model_name} - ERROR: {str(e)}")

    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print(f"\n📊 Generating Comprehensive Report...")

        # Calculate summary statistics
        total = self.test_results["total_models"]
        successful = self.test_results["successful_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (successful / total * 100) if total > 0 else 0

        self.test_results["summary"] = {
            "total_models": total,
            "successful_models": successful,
            "failed_models": failed,
            "success_rate": round(success_rate, 1),
            "test_completion_time": datetime.now().isoformat()
        }

        # Performance rankings
        rankings = {
            "best_accuracy": [],
            "fastest_prediction": [],
            "largest_models": [],
            "most_features": []
        }

        for model_name, model_result in self.test_results["models"].items():
            if model_result["overall_status"] in ["PASSED", "PARTIAL"]:
                model_info = model_result.get("model_info", {})

                # Accuracy ranking
                accuracy = model_info.get("accuracy", 0)
                rankings["best_accuracy"].append((model_name, accuracy))

                # Speed ranking
                perf_test = model_result.get("tests", {}).get("performance", {})
                if perf_test.get("status") == "PASS":
                    avg_time = perf_test.get("details", {}).get("avg_prediction_time_ms", float('inf'))
                    rankings["fastest_prediction"].append((model_name, avg_time))

                # Size ranking
                file_size = model_result.get("file_size_mb", 0)
                rankings["largest_models"].append((model_name, file_size))

                # Feature count ranking
                features_count = model_info.get("features_count", 0)
                rankings["most_features"].append((model_name, features_count))

        # Sort rankings
        rankings["best_accuracy"].sort(key=lambda x: x[1], reverse=True)
        rankings["fastest_prediction"].sort(key=lambda x: x[1])
        rankings["largest_models"].sort(key=lambda x: x[1], reverse=True)
        rankings["most_features"].sort(key=lambda x: x[1], reverse=True)

        # Take top 10 for each category
        for category in rankings:
            rankings[category] = rankings[category][:10]

        self.test_results["performance_rankings"] = rankings

        # Save detailed JSON report
        json_report_file = self.results_dir / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_report_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)

        # Generate HTML report
        html_report = self.generate_html_report()
        html_report_file = self.results_dir / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_report_file, "w") as f:
            f.write(html_report)

        # Print summary
        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE MODEL TEST RESULTS")
        print(f"{'='*80}")
        print(f"📊 Total Models Tested: {total}")
        print(f"✅ Successful Tests: {successful}")
        print(f"❌ Failed Tests: {failed}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"\n🏆 TOP PERFORMING MODELS:")

        if rankings["best_accuracy"]:
            print(f"\n📊 Best Accuracy:")
            for i, (model, accuracy) in enumerate(rankings["best_accuracy"][:5], 1):
                print(f"  {i}. {model}: {accuracy:.1%}")

        if rankings["fastest_prediction"]:
            print(f"\n⚡ Fastest Prediction:")
            for i, (model, time_ms) in enumerate(rankings["fastest_prediction"][:5], 1):
                print(f"  {i}. {model}: {time_ms:.2f}ms")

        print(f"\n📄 Detailed Reports:")
        print(f"  JSON: {json_report_file}")
        print(f"  HTML: {html_report_file}")

        # Print failed models if any
        if failed > 0:
            print(f"\n❌ FAILED MODELS:")
            failed_models = [name for name, result in self.test_results["models"].items()
                           if result["overall_status"] in ["FAILED", "ERROR"]]
            for model in failed_models[:10]:  # Show first 10 failed models
                print(f"  • {model}")
            if len(failed_models) > 10:
                print(f"  ... and {len(failed_models) - 10} more")

        return json_report_file, html_report_file

    def generate_html_report(self):
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>StockVibePredictor Model Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .success {{ color: #27ae60; }} .failed {{ color: #e74c3c; }} .partial {{ color: #f39c12; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .model-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; background: white; }}
        .model-card.passed {{ border-left: 5px solid #27ae60; }}
        .model-card.failed {{ border-left: 5px solid #e74c3c; }}
        .model-card.partial {{ border-left: 5px solid #f39c12; }}
        .test-item {{ margin: 5px 0; padding: 5px; background: #f8f9fa; border-radius: 3px; }}
        .rankings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .ranking-card {{ background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 StockVibePredictor Model Test Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary">
            <h2>📊 Test Summary</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; text-align: center;">
                <div>
                    <h3>{self.test_results['summary']['total_models']}</h3>
                    <p>Total Models</p>
                </div>
                <div class="success">
                    <h3>{self.test_results['summary']['successful_models']}</h3>
                    <p>Successful</p>
                </div>
                <div class="failed">
                    <h3>{self.test_results['summary']['failed_models']}</h3>
                    <p>Failed</p>
                </div>
                <div>
                    <h3>{self.test_results['summary']['success_rate']:.1f}%</h3>
                    <p>Success Rate</p>
                </div>
            </div>
        </div>
"""

        # Add rankings section
        html += """
        <div class="rankings">
            <div class="ranking-card">
                <h3>🏆 Best Accuracy Models</h3>
                <table>
                    <tr><th>Rank</th><th>Model</th><th>Accuracy</th></tr>
"""
        for i, (model, accuracy) in enumerate(self.test_results["performance_rankings"]["best_accuracy"][:10], 1):
            html += f"<tr><td>{i}</td><td>{model}</td><td>{accuracy:.1%}</td></tr>"

        html += """
                </table>
            </div>

            <div class="ranking-card">
                <h3>⚡ Fastest Prediction Models</h3>
                <table>
                    <tr><th>Rank</th><th>Model</th><th>Time (ms)</th></tr>
"""
        for i, (model, time_ms) in enumerate(self.test_results["performance_rankings"]["fastest_prediction"][:10], 1):
            html += f"<tr><td>{i}</td><td>{model}</td><td>{time_ms:.2f}</td></tr>"

        html += """
                </table>
            </div>
        </div>

        <h2>🔍 Detailed Model Results</h2>
        <div class="model-grid">
"""

        # Add individual model results
        for model_name, result in self.test_results["models"].items():
            status_class = result["overall_status"].lower()
            status_icon = {"passed": "✅", "partial": "⚠️", "failed": "❌", "error": "💥"}.get(status_class, "❓")

            html += f"""
            <div class="model-card {status_class}">
                <h3>{status_icon} {model_name}</h3>
                <p><strong>Status:</strong> {result['overall_status']}</p>
                <p><strong>File Size:</strong> {result.get('file_size_mb', 'N/A')} MB</p>
                <p><strong>Test Duration:</strong> {result.get('test_duration', 'N/A')}s</p>
"""

            # Add model info if available
            if "model_info" in result:
                info = result["model_info"]
                html += f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 3px;">
                    <strong>Model Info:</strong><br>
                    Ticker: {info.get('ticker', 'N/A')}<br>
                    Timeframe: {info.get('timeframe', 'N/A')}<br>
                    Accuracy: {info.get('accuracy', 0):.1%}<br>
                    Features: {info.get('features_count', 0)}<br>
                    Type: {info.get('model_type', 'N/A')}
                </div>
"""

            # Add test results
            if "tests" in result:
                html += "<div style='margin-top: 10px;'><strong>Test Results:</strong>"
                for test_name, test_result in result["tests"].items():
                    status_icon = "✅" if test_result["status"] == "PASS" else ("⚠️" if test_result["status"] == "SKIP" else "❌")
                    html += f"<div class='test-item'>{status_icon} {test_name.title()}: {test_result['status']}</div>"
                html += "</div>"

            # Add errors if any
            if result.get("error_messages"):
                html += "<div style='color: #e74c3c; margin-top: 10px;'><strong>Errors:</strong><ul>"
                for error in result["error_messages"][:3]:  # Show first 3 errors
                    html += f"<li>{error}</li>"
                html += "</ul></div>"

            html += "</div>"

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Model Testing Suite...")

    # Create test suite
    test_suite = ComprehensiveModelTestSuite()

    # Run all tests
    test_suite.run_all_tests(max_workers=6)

    # Generate comprehensive report
    json_report, html_report = test_suite.generate_comprehensive_report()

    print(f"\n🎉 Testing Complete!")
    print(f"📊 View your results:")
    print(f"  🌐 HTML Report: {html_report}")
    print(f"  📄 JSON Report: {json_report}")

    return test_suite.test_results

if __name__ == "__main__":
    results = main()
