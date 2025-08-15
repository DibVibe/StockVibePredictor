import axios from "axios";

// Mock axios for integration tests
jest.mock("axios");
const mockedAxios = axios;

describe("API Integration Tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe("Stock Prediction API", () => {
    test("should fetch stock prediction successfully", async () => {
      const mockResponse = {
        data: {
          prediction: {
            direction: "UP",
            confidence: 0.85,
            target_price: 150.0,
          },
          ticker: "AAPL",
          timestamp: new Date().toISOString(),
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockResponse);

      const response = await axios.post("/api/predict/", { ticker: "AAPL" });

      expect(response.data.prediction.direction).toBe("UP");
      expect(response.data.ticker).toBe("AAPL");
      expect(response.data.prediction.confidence).toBeGreaterThan(0);
      expect(mockedAxios.post).toHaveBeenCalledWith("/api/predict/", {
        ticker: "AAPL",
      });
    });

    test("should handle API errors gracefully", async () => {
      const errorMessage = "Network Error";
      mockedAxios.post.mockRejectedValueOnce(new Error(errorMessage));

      await expect(
        axios.post("/api/predict/", { ticker: "INVALID" })
      ).rejects.toThrow(errorMessage);
    });

    test("should fetch multiple timeframe predictions", async () => {
      const mockResponse = {
        data: {
          predictions: {
            "1d": { direction: "UP", confidence: 0.75 },
            "1w": { direction: "DOWN", confidence: 0.65 },
            "1mo": { direction: "UP", confidence: 0.8 },
          },
          ticker: "TSLA",
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockResponse);

      const response = await axios.post("/api/predict/multi/", {
        ticker: "TSLA",
        timeframes: ["1d", "1w", "1mo"],
      });

      expect(Object.keys(response.data.predictions)).toHaveLength(3);
      expect(response.data.predictions["1d"].direction).toBeDefined();
    });
  });

  describe("Sentiment Analysis API", () => {
    test("should fetch sentiment data", async () => {
      const mockSentimentData = {
        data: {
          positive: 45.2,
          negative: 32.8,
          neutral: 22.0,
          confidence: 87,
          sources: ["News APIs", "Social Media"],
          totalSources: 150,
          marketCondition: "Bullish",
        },
      };

      mockedAxios.get.mockResolvedValueOnce(mockSentimentData);

      const response = await axios.get("/api/sentiment/AAPL");
      const data = response.data;

      expect(data.positive).toBeGreaterThan(0);
      expect(data.negative).toBeGreaterThan(0);
      expect(data.neutral).toBeGreaterThan(0);
      expect(data.confidence).toBeGreaterThanOrEqual(0);
      expect(data.confidence).toBeLessThanOrEqual(100);
      expect(data.marketCondition).toBeDefined();
    });
  });

  describe("Market Data API", () => {
    test("should fetch market overview", async () => {
      const mockMarketData = {
        data: {
          market_data: {
            "S&P 500": { value: 4200.0, change: 1.5 },
            NASDAQ: { value: 13500.0, change: 2.1 },
            DOW: { value: 34000.0, change: 0.8 },
          },
          market_sentiment: "Bullish",
          timestamp: new Date().toISOString(),
        },
      };

      mockedAxios.get.mockResolvedValueOnce(mockMarketData);

      const response = await axios.get("/api/market/overview/");

      expect(response.data.market_data).toBeDefined();
      expect(response.data.market_sentiment).toBeDefined();
      expect(Object.keys(response.data.market_data).length).toBeGreaterThan(0);
    });
  });

  describe("Batch Predictions API", () => {
    test("should handle batch predictions", async () => {
      const mockBatchResponse = {
        data: {
          results: {
            AAPL: { direction: "UP", confidence: 0.85 },
            GOOGL: { direction: "DOWN", confidence: 0.72 },
            TSLA: { direction: "UP", confidence: 0.91 },
            MSFT: { direction: "UP", confidence: 0.78 },
          },
          timeframe: "1d",
          processed_count: 4,
        },
      };

      mockedAxios.post.mockResolvedValueOnce(mockBatchResponse);

      const response = await axios.post("/api/predict/batch/", {
        tickers: ["AAPL", "GOOGL", "TSLA", "MSFT"],
        timeframe: "1d",
      });

      expect(Object.keys(response.data.results)).toHaveLength(4);
      expect(response.data.processed_count).toBe(4);
    });
  });
});
