# Postman Setup Guide

## Quick Start

### 1. Import Collection

- Open Postman
- Click "Import" → Select `collections/StockVibePredictor_API_v2.postman_collection.json`

### 2. Import Environments

Import all environment files from `environments/` folder:

- Development.postman_environment.json
- Staging.postman_environment.json
- Production.postman_environment.json

### 3. Select Environment

- Use the environment dropdown (top-right) to select your environment
- Development: For local testing
- Staging: For staging server
- Production: For production API

### 4. Configure Credentials

For each environment, update:

- `username`: Your API username
- `password`: Your API password
- `apiKey`: Your API key (if applicable)

### 5. Run Tests

- Select a request
- Click "Send"
- Check "Tests" tab for results

---

## 📁 Directory Structure

```bash
Postman/
├── Collections/
│   └── StockVibePredictor_API_v2.postman_collection.json (Main collection)
├── Environments/
│   ├── Development.postman_environment.json
│   ├── Staging.postman_environment.json
│   └── Production.postman_environment.json
├── Tests/
│   ├── integration_tests.json (Test scenarios)
│   ├── newman_config.json (CLI test config)
│   └── results/ (Test outputs)
├── Examples/
│   └── sample_responses.json (Response examples)
├── Scripts/
│   ├── run_tests.sh (Test runner)
│   └── setup.py (Setup validator)
└── README.md
```

---

## 🚀 Quick Setup Script

Run the setup script to validate your configuration:

```bash
cd Postman/Scripts
python setup.py
```

---

## 🧪 Running Tests

### Using Newman (CLI) :

```bash
cd Scripts
./run_tests.sh
```

### Using Postman Runner :

1. Open Postman
2. Select Collection → Run
3. Choose environment
4. Configure iterations
5. Run tests

---

## 📊 Test Reports

### Test results are saved in :

- ⁠Tests/results/dev-test-results.html - HTML report
- ⁠Tests/results/test-results.json - JSON report

---

## Environment Variables

| Variable   | Description                | Example                      |
| ---------- | -------------------------- | ---------------------------- |
| protocol   | HTTP or HTTPS              | http                         |
| host       | API host                   | localhost                    |
| port       | API port                   | 8000                         |
| baseUrl    | Complete base URL          | http://localhost:8000/api/v1 |
| authToken  | JWT token (auto-populated) | eyJhbGc...                   |
| testTicker | Default ticker for testing | AAPL                         |

---

## Authentication Flow

1. Run "Get Auth Token" request first.
2. Token will be automatically saved to environment.
3. All subsequent requests will use this token.

---

## Tips

- Use "Runner" for automated testing.
- Enable "Postman Console" for debugging.
- Check "Pre-request Script" tab for automatic setup.

---

```

```
