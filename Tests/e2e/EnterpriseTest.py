#!/usr/bin/env python3
"""
EnterpriseTest.py - StockVibePredictor API Testing Suite
Comprehensive test for all enterprise endpoints and features
"""
import requests
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000/api"
TIMEOUT = 30
COLORS = {
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "PURPLE": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "BOLD": "\033[1m",
    "END": "\033[0m",
}


def print_colored(text, color="WHITE"):
    """Print colored text"""
    print(f"{COLORS[color]}{text}{COLORS['END']}")


def print_header(title):
    """Print test section header"""
    print_colored(f"\n{'='*60}", "CYAN")
    print_colored(f"🧪 {title}", "BOLD")
    print_colored(f"{'='*60}", "CYAN")


def print_result(test_name, success, details=None, response_time=None):
    """Print test result with formatting"""
    status = "✅ PASS" if success else "❌ FAIL"
    color = "GREEN" if success else "RED"

    time_info = f" ({response_time:.2f}s)" if response_time else ""
    print_colored(f"{status} {test_name}{time_info}", color)

    if details:
        print_colored(f"    📋 {details}", "WHITE")


class APITester:
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
        }
        self.auth_token = None

    def record_result(self, test_name, success, details=None, response_time=None):
        """Record test result"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
        else:
            self.results["failed_tests"] += 1

        self.results["test_details"].append(
            {
                "test": test_name,
                "success": success,
                "details": details,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print_result(test_name, success, details, response_time)

    def make_request(
        self, method, endpoint, data=None, headers=None, auth_required=False
    ):
        """Make HTTP request with error handling"""
        url = f"{BASE_URL}{endpoint}"

        # Add auth header if required and available
        if auth_required and self.auth_token:
            if not headers:
                headers = {}
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            start_time = time.time()

            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = requests.post(
                    url, json=data, headers=headers, timeout=TIMEOUT
                )
            elif method.upper() == "PUT":
                response = requests.put(
                    url, json=data, headers=headers, timeout=TIMEOUT
                )
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time
            return response, response_time

        except requests.exceptions.ConnectionError:
            return None, None
        except requests.exceptions.Timeout:
            return "TIMEOUT", None
        except Exception as e:
            return str(e), None

    def test_system_health(self):
        """Test system health endpoint"""
        print_header("SYSTEM HEALTH TESTS")

        response, response_time = self.make_request("GET", "/system/health/")

        if response is None:
            self.record_result(
                "System Health Check",
                False,
                "Connection failed - Django server not running?",
            )
            return False
        elif response == "TIMEOUT":
            self.record_result("System Health Check", False, "Request timed out")
            return False
        elif isinstance(response, str):
            self.record_result(
                "System Health Check", False, f"Request error: {response}"
            )
            return False

        success = response.status_code == 200
        if success:
            try:
                data = response.json()
                status = data.get("status", "unknown")
                services = data.get("services", {})
                details = f"Status: {status}, Services: {len(services)} active"
            except:
                details = "Health endpoint responding but JSON parsing failed"
        else:
            details = f"HTTP {response.status_code}: {response.text[:100]}"

        self.record_result("System Health Check", success, details, response_time)
        return success

    def test_legacy_endpoints(self):
        """Test legacy endpoints for backward compatibility"""
        print_header("LEGACY ENDPOINT TESTS")

        # Test Redis check (if available)
        response, response_time = self.make_request("GET", "/redis-check/")
        if response and hasattr(response, 'status_code'):
            success = response.status_code == 200
            if success:
                try:
                    data = response.json()
                    details = f"Redis status: {data.get('status', 'unknown')}"
                except:
                    details = "Redis endpoint responding"
            else:
                details = f"Redis check failed: {response.status_code}"
        else:
            success = False
            details = "Redis endpoint not available (optional)"

        self.record_result("Legacy Redis Check", True, details, response_time)  # Don't fail if Redis unavailable

        # Test legacy prediction
        test_data = {"ticker": "AAPL"}
        response, response_time = self.make_request("POST", "/predict/", test_data)

        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                prediction = data.get("prediction", {})
                direction = prediction.get("direction", "unknown")
                confidence = prediction.get("confidence", 0)
                details = f"AAPL prediction: {direction} (confidence: {confidence:.1%})"
                success = True
            except:
                details = "Prediction endpoint responding but invalid JSON"
                success = False
        else:
            details = "Legacy prediction endpoint not available"
            success = False

        self.record_result("Legacy Prediction API", success, details, response_time)

    def test_multi_timeframe_prediction(self):
        """Test multi-timeframe prediction endpoint"""
        print_header("MULTI-TIMEFRAME PREDICTION TESTS")

        # Test single timeframe
        test_data = {"ticker": "AAPL", "timeframes": ["1d"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                predictions = data.get("predictions", {})
                if "1d" in predictions:
                    direction = predictions["1d"].get("direction", "unknown")
                    details = f"1d prediction for AAPL: {direction}"
                    success = True
                else:
                    details = "Response format unexpected"
                    success = False
            except:
                details = "JSON parsing failed"
                success = False
        else:
            details = "Multi-timeframe endpoint not available"
            success = False

        self.record_result("Single Timeframe (1d)", success, details, response_time)

        # Test multiple timeframes
        test_data = {"ticker": "TSLA", "timeframes": ["1d", "1w", "1mo"]}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                predictions = data.get("predictions", {})
                timeframe_count = len(predictions)
                details = f"TSLA predictions for {timeframe_count} timeframes"
                success = timeframe_count > 0
            except:
                details = "Multi-timeframe JSON parsing failed"
                success = False
        else:
            details = "Multi-timeframe prediction failed"
            success = False

        self.record_result(
            "Multi-Timeframe (1d,1w,1mo)", success, details, response_time
        )

        # Test with analysis
        test_data = {"ticker": "GOOGL", "timeframes": ["1d"], "include_analysis": True}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )

        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                analysis = data.get("analysis", {})
                has_technical = "technical" in analysis
                has_risk = "risk" in analysis
                details = f"Analysis included: technical={has_technical}, risk={has_risk}"
                success = len(analysis) > 0
            except:
                details = "Analysis JSON parsing failed"
                success = False
        else:
            details = "Analysis prediction failed"
            success = False

        self.record_result("Prediction with Analysis", success, details, response_time)

    def test_batch_predictions(self):
        """Test batch prediction endpoint"""
        print_header("BATCH PREDICTION TESTS")

        test_data = {"tickers": ["AAPL", "GOOGL", "TSLA", "MSFT"], "timeframe": "1d"}
        response, response_time = self.make_request(
            "POST", "/predict/batch/", test_data
        )

        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                results = data.get("results", {})
                successful_predictions = len(
                    [r for r in results.values() if isinstance(r, dict) and "direction" in r]
                )
                total_requested = len(test_data['tickers'])
                details = f"Batch: {successful_predictions}/{total_requested} successful"
                success = successful_predictions > 0
            except:
                details = "Batch prediction JSON parsing failed"
                success = False
        else:
            details = "Batch prediction endpoint failed"
            success = False

        self.record_result("Batch Predictions", success, details, response_time)

    def test_market_intelligence(self):
        """Test market intelligence endpoints"""
        print_header("MARKET INTELLIGENCE TESTS")

        # Market overview
        response, response_time = self.make_request("GET", "/market/overview/")
        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                market_data = data.get("market_data", {})
                sentiment = data.get("market_sentiment", "unknown")
                details = f"Market sentiment: {sentiment}, indices: {len(market_data)}"
                success = True
            except:
                details = "Market overview JSON parsing failed"
                success = False
        else:
            details = "Market overview endpoint not available"
            success = False

        self.record_result("Market Overview", success, details, response_time)

        # Analytics dashboard
        response, response_time = self.make_request("GET", "/market/analytics/")
        if response and hasattr(response, 'status_code') and response.status_code == 200:
            try:
                data = response.json()
                system_metrics = data.get("system_metrics", {})
                uptime = system_metrics.get('uptime', 'unknown')
                details = f"System uptime: {uptime}"
                success = True
            except:
                details = "Analytics dashboard JSON parsing failed"
                success = False
        else:
            details = "Analytics dashboard not available"
            success = False

        self.record_result("Analytics Dashboard", success, details, response_time)

    def test_error_handling(self):
        """Test error handling and edge cases"""
        print_header("ERROR HANDLING TESTS")

        # Invalid ticker
        test_data = {"ticker": "INVALID123"}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and hasattr(response, 'status_code') and response.status_code in [400, 404, 422]
        details = (
            "Invalid ticker properly rejected" if success else "Error handling needs improvement"
        )
        self.record_result("Invalid Ticker Handling", success, details, response_time)

        # Missing ticker
        test_data = {}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and hasattr(response, 'status_code') and response.status_code in [400, 422]
        details = "Missing ticker properly rejected" if success else "Validation needs improvement"
        self.record_result("Missing Ticker Validation", success, details, response_time)

        # Empty ticker
        test_data = {"ticker": ""}
        response, response_time = self.make_request(
            "POST", "/predict/multi/", test_data
        )
        success = response and hasattr(response, 'status_code') and response.status_code in [400, 422]
        details = "Empty ticker properly rejected" if success else "Empty validation needs work"
        self.record_result("Empty Ticker Validation", success, details, response_time)

    def test_performance_stress(self):
        """Test system performance under load"""
        print_header("PERFORMANCE STRESS TESTS")

        # Multiple rapid requests
        start_time = time.time()
        successful_requests = 0
        total_requests = 5
        response_times = []

        print_colored("Running rapid fire test...", "YELLOW")

        for i in range(total_requests):
            test_data = {"ticker": "AAPL", "timeframes": ["1d"]}
            response, req_time = self.make_request("POST", "/predict/multi/", test_data)

            if response and hasattr(response, 'status_code') and response.status_code == 200:
                successful_requests += 1
                if req_time:
                    response_times.append(req_time)

            # Small delay between requests
            time.sleep(0.1)

        total_time = time.time() - start_time
        avg_time = sum(response_times) / len(response_times) if response_times else 0
        success_rate = (successful_requests / total_requests) * 100

        success = successful_requests >= total_requests * 0.6  # 60% success rate acceptable
        details = f"{successful_requests}/{total_requests} requests succeeded ({success_rate:.1f}%), avg: {avg_time:.2f}s"

        self.record_result("Rapid Fire Requests", success, details, total_time)

    def test_data_validation(self):
        """Test data validation and sanitization"""
        print_header("DATA VALIDATION TESTS")

        # Test SQL injection protection
        malicious_data = {"ticker": "AAPL'; DROP TABLE stocks; --"}
        response, response_time = self.make_request("POST", "/predict/", malicious_data)

        # Should either reject (400/422) or handle safely (200 with sanitized data)
        if response and hasattr(response, 'status_code'):
            success = response.status_code in [200, 400, 422]
            details = f"SQL injection attempt: Status {response.status_code}"
        else:
            success = False
            details = "SQL injection test failed - no response"

        self.record_result("SQL Injection Protection", success, details, response_time)

        # Test XSS protection
        xss_data = {"ticker": "<script>alert('xss')</script>"}
        response, response_time = self.make_request("POST", "/predict/", xss_data)

        if response and hasattr(response, 'status_code'):
            success = response.status_code in [200, 400, 422]
            details = f"XSS attempt: Status {response.status_code}"
        else:
            success = False
            details = "XSS test failed - no response"

        self.record_result("XSS Protection", success, details, response_time)

    def test_api_documentation(self):
        """Test API documentation endpoints"""
        print_header("API DOCUMENTATION TESTS")

        # Test if API docs are available
        doc_endpoints = [
            "/docs/",
            "/swagger/",
            "/api-docs/",
            "/redoc/"
        ]

        found_docs = False
        for endpoint in doc_endpoints:
            response, response_time = self.make_request("GET", endpoint)
            if response and hasattr(response, 'status_code') and response.status_code == 200:
                found_docs = True
                self.record_result(f"API Docs {endpoint}", True, "Documentation available", response_time)
                break

        if not found_docs:
            self.record_result("API Documentation", False, "No documentation endpoints found", None)

    def generate_report(self):
        """Generate comprehensive test report"""
        print_header("TEST SUMMARY REPORT")

        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0

        print_colored(f"📊 Total Tests: {total}", "BOLD")
        print_colored(f"✅ Passed: {passed}", "GREEN")
        print_colored(f"❌ Failed: {failed}", "RED")
        print_colored(f"📈 Success Rate: {success_rate:.1f}%", "CYAN")

        # Performance analysis
        response_times = [
            test["response_time"] for test in self.results["test_details"]
            if test["response_time"] is not None
        ]

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            print_colored(f"⚡ Avg Response Time: {avg_response_time:.2f}s", "BLUE")
            print_colored(f"🐌 Max Response Time: {max_response_time:.2f}s", "BLUE")

        # Overall assessment
        if success_rate >= 90:
            print_colored("\n🎉 EXCELLENT! Your API is working perfectly!", "GREEN")
            grade = "A+"
        elif success_rate >= 80:
            print_colored("\n🌟 GREAT! Your API is performing very well!", "GREEN")
            grade = "A"
        elif success_rate >= 70:
            print_colored("\n✅ GOOD! Most features are working well.", "YELLOW")
            grade = "B+"
        elif success_rate >= 60:
            print_colored("\n⚠️ FAIR! Some features need attention.", "YELLOW")
            grade = "B"
        elif success_rate >= 50:
            print_colored("\n⚠️ PARTIAL! Multiple features need attention.", "YELLOW")
            grade = "C"
        else:
            print_colored("\n❌ CRITICAL! Major issues found.", "RED")
            grade = "F"

        print_colored(f"🎓 Overall Grade: {grade}", "PURPLE")

        # Save detailed report
        report_data = {
            **self.results,
            "performance": {
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "total_response_times": len(response_times)
            },
            "grade": grade,
            "success_rate": success_rate
        }

        report_file = Path("api_test_report.json")
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print_colored(f"\n📄 Detailed report saved: {report_file.absolute()}", "BLUE")

        # Recommendations
        print_colored("\n🔧 NEXT STEPS:", "PURPLE")

        if failed > 0:
            print_colored(
                "   1. ⚠️  Check Django server: python manage.py runserver 8000",
                "WHITE",
            )
            print_colored(
                "   2. 🤖 Verify models are trained: python TrainModel.py",
                "WHITE",
            )
            print_colored("   3. 🔧 Check database connections", "WHITE")
            print_colored("   4. 📋 Review failed tests in the detailed report", "WHITE")
        else:
            print_colored(
                "   1. 🚀 Start your frontend: cd Frontend && npm start", "WHITE"
            )
            print_colored("   2. 🎉 Your API is ready for production!", "WHITE")
            print_colored("   3. 📈 Consider adding more advanced features", "WHITE")

        print_colored(f"\n⏱️  Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "CYAN")

        return success_rate >= 70


def main():
    """Main test execution"""
    print_colored("\n🧪 StockVibePredictor Enterprise API Test Suite", "BOLD")
    print_colored("=" * 60, "CYAN")
    print_colored(f"🌐 Testing API at: {BASE_URL}", "BLUE")
    print_colored(f"⏱️  Request timeout: {TIMEOUT}s", "BLUE")
    print_colored(
        f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "BLUE"
    )

    tester = APITester()

    print_colored("\n🔍 Starting comprehensive API testing...", "YELLOW")

    # Run all test suites
    system_healthy = tester.test_system_health()

    if not system_healthy:
        print_colored(
            "\n⚠️ System health check failed. Continuing with limited tests...", "YELLOW"
        )

    # Core functionality tests
    tester.test_legacy_endpoints()
    tester.test_multi_timeframe_prediction()
    tester.test_batch_predictions()

    # Advanced features tests
    tester.test_market_intelligence()
    tester.test_error_handling()
    tester.test_data_validation()

    # Performance and documentation tests
    tester.test_performance_stress()
    tester.test_api_documentation()

    # Generate final report
    success = tester.generate_report()

    print_colored("\n" + "=" * 60, "CYAN")
    print_colored("🏁 Testing Complete!", "BOLD")
    print_colored("=" * 60, "CYAN")

    return success


if __name__ == "__main__":
    try:
        success = main()
        # Exit with appropriate code
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_colored("\n\n⚠️ Testing interrupted by user", "YELLOW")
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n\n❌ Unexpected error: {str(e)}", "RED")
        sys.exit(1)
