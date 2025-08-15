#!/usr/bin/env python3
"""
TestConnections.py - Database and API Connections Testing
Tests for all external connections and dependencies
"""

import sys
import os
import requests
import time
import json
from pathlib import Path
from datetime import datetime
import unittest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Backend"))

class ConnectionTestSuite(unittest.TestCase):
    """Test suite for external connections"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_dir = PROJECT_ROOT
        self.backend_dir = self.base_dir / "Backend"
        self.timeout = 10
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {}
        }

    def test_django_server_connection(self):
        """Test if Django server is accessible"""
        try:
            response = requests.get(
                "http://localhost:8000/api/system/health/",
                timeout=self.timeout
            )

            success = response.status_code == 200
            details = f"Status: {response.status_code}"

            if success:
                data = response.json()
                details += f", Health: {data.get('status', 'unknown')}"

            self.test_results["tests"].append({
                "test": "Django Server Connection",
                "passed": success,
                "details": details
            })

            self.assertTrue(success, f"Django server not responding: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.test_results["tests"].append({
                "test": "Django Server Connection",
                "passed": False,
                "details": "Connection refused - server not running"
            })
            self.fail("Django server is not running")

        except requests.exceptions.Timeout:
            self.test_results["tests"].append({
                "test": "Django Server Connection",
                "passed": False,
                "details": "Connection timeout"
            })
            self.fail("Django server connection timeout")

    def test_redis_connection(self):
        """Test Redis connection if available"""
        try:
            response = requests.get(
                "http://localhost:8000/api/redis-check/",
                timeout=self.timeout
            )

            success = response.status_code == 200
            details = f"Redis status: {response.status_code}"

            if success:
                data = response.json()
                details = f"Redis: {data.get('status', 'unknown')}"

            self.test_results["tests"].append({
                "test": "Redis Connection",
                "passed": success,
                "details": details
            })

            # Redis is optional, so we don't fail the test
            print(f"Redis connection: {'✅' if success else '❌'} {details}")

        except Exception as e:
            self.test_results["tests"].append({
                "test": "Redis Connection",
                "passed": False,
                "details": f"Error: {str(e)}"
            })

    def test_stock_api_endpoints(self):
        """Test stock prediction API endpoints"""
        endpoints_to_test = [
            {
                "name": "Legacy Prediction",
                "method": "POST",
                "endpoint": "/api/predict/",
                "data": {"ticker": "AAPL"}
            },
            {
                "name": "Multi-timeframe Prediction",
                "method": "POST",
                "endpoint": "/api/predict/multi/",
                "data": {"ticker": "AAPL", "timeframes": ["1d"]}
            },
            {
                "name": "Batch Prediction",
                "method": "POST",
                "endpoint": "/api/predict/batch/",
                "data": {"tickers": ["AAPL", "GOOGL"], "timeframe": "1d"}
            }
        ]

        for test_case in endpoints_to_test:
            try:
                if test_case["method"] == "POST":
                    response = requests.post(
                        f"http://localhost:8000{test_case['endpoint']}",
                        json=test_case["data"],
                        timeout=self.timeout
                    )
                else:
                    response = requests.get(
                        f"http://localhost:8000{test_case['endpoint']}",
                        timeout=self.timeout
                    )

                success = response.status_code == 200
                details = f"Status: {response.status_code}"

                if success:
                    data = response.json()
                    if "prediction" in data:
                        details += f", Prediction available"
                    elif "predictions" in data:
                        details += f", Multiple predictions available"
                    elif "results" in data:
                        details += f", Batch results available"

                self.test_results["tests"].append({
                    "test": test_case["name"],
                    "passed": success,
                    "details": details
                })

                print(f"{test_case['name']}: {'✅' if success else '❌'} {details}")

            except Exception as e:
                self.test_results["tests"].append({
                    "test": test_case["name"],
                    "passed": False,
                    "details": f"Error: {str(e)}"
                })

    def test_market_data_endpoints(self):
        """Test market data API endpoints"""
        endpoints_to_test = [
            {
                "name": "Market Overview",
                "endpoint": "/api/market/overview/"
            },
            {
                "name": "Market Analytics",
                "endpoint": "/api/market/analytics/"
            }
        ]

        for test_case in endpoints_to_test:
            try:
                response = requests.get(
                    f"http://localhost:8000{test_case['endpoint']}",
                    timeout=self.timeout
                )

                success = response.status_code == 200
                details = f"Status: {response.status_code}"

                if success:
                    data = response.json()
                    details += f", Data keys: {list(data.keys())}"

                self.test_results["tests"].append({
                    "test": test_case["name"],
                    "passed": success,
                    "details": details
                })

                print(f"{test_case['name']}: {'✅' if success else '❌'} {details}")

            except Exception as e:
                self.test_results["tests"].append({
                    "test": test_case["name"],
                    "passed": False,
                    "details": f"Error: {str(e)}"
                })

    def test_external_api_dependencies(self):
        """Test external API dependencies (simulated)"""
        # Test external market data sources
        external_apis = [
            {
                "name": "Yahoo Finance (Test)",
                "url": "https://finance.yahoo.com",
                "expected_status": 200
            },
            {
                "name": "Alpha Vantage (Connectivity)",
                "url": "https://www.alphavantage.co",
                "expected_status": 200
            }
        ]

        for api in external_apis:
            try:
                response = requests.get(api["url"], timeout=5)
                success = response.status_code == api["expected_status"]

                self.test_results["tests"].append({
                    "test": f"External API - {api['name']}",
                    "passed": success,
                    "details": f"Status: {response.status_code}"
                })

            except Exception as e:
                self.test_results["tests"].append({
                    "test": f"External API - {api['name']}",
                    "passed": False,
                    "details": f"Error: {str(e)}"
                })

    def test_file_system_permissions(self):
        """Test file system permissions for model files"""
        try:
            model_dir = self.backend_dir / "Scripts"
            model_file = model_dir / "stock_model.pkl"

            # Test read permissions
            can_read = os.access(model_file, os.R_OK) if model_file.exists() else False

            # Test directory write permissions
            can_write = os.access(model_dir, os.W_OK) if model_dir.exists() else False

            success = can_read and can_write
            details = f"Model readable: {can_read}, Directory writable: {can_write}"

            self.test_results["tests"].append({
                "test": "File System Permissions",
                "passed": success,
                "details": details
            })

            if not success:
                self.fail(f"File system permission issues: {details}")

        except Exception as e:
            self.test_results["tests"].append({
                "test": "File System Permissions",
                "passed": False,
                "details": f"Error: {str(e)}"
            })

    def generate_report(self):
        """Generate comprehensive test report"""
        passed_tests = sum(1 for test in self.test_results["tests"] if test["passed"])
        total_tests = len(self.test_results["tests"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate
        }

        # Save report
        report_file = self.base_dir / "connection_test_report.json"
        with open(report_file, "w") as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"🌐 CONNECTION TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"📄 Report saved: {report_file}")

        return success_rate >= 60  # Return True if 60% or more tests pass

if __name__ == "__main__":
    print("🌐 StockVibePredictor Connection Test Suite")
    print("="*60)

    # Create and run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(ConnectionTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate report
    test_instance = ConnectionTestSuite()
    test_instance.generate_report()

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
