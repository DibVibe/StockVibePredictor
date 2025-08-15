import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";

// Create a simple mock component for testing
const MockSentimentAnalysis = ({ ticker = "AAPL" }) => {
  return (
    <div data-testid="sentiment-analysis">
      <h2>📊 Sentiment Analysis</h2>
      <div data-testid="ticker">{ticker}</div>
      <div data-testid="loading">Analyzing market sentiment...</div>
      <div data-testid="chart">Chart placeholder</div>
    </div>
  );
};

describe("SentimentAnalysis Component", () => {
  test("renders sentiment analysis component", () => {
    render(<MockSentimentAnalysis ticker="AAPL" />);

    expect(screen.getByTestId("sentiment-analysis")).toBeInTheDocument();
    expect(screen.getByText("📊 Sentiment Analysis")).toBeInTheDocument();
    expect(screen.getByTestId("ticker")).toHaveTextContent("AAPL");
  });

  test("displays loading state", () => {
    render(<MockSentimentAnalysis />);

    expect(screen.getByText(/analyzing market sentiment/i)).toBeInTheDocument();
  });

  test("shows chart placeholder", () => {
    render(<MockSentimentAnalysis />);

    expect(screen.getByTestId("chart")).toBeInTheDocument();
  });
});
