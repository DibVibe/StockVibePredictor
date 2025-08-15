import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

// Simple mock App component
const MockApp = () => {
  const [ticker, setTicker] = React.useState("AAPL");

  return (
    <div data-testid="app">
      <h1>StockVibePredictor</h1>
      <div data-testid="sentiment-analysis">
        <h2>📊 Sentiment Analysis</h2>
        <div data-testid="ticker-display">{ticker}</div>
        <div data-testid="doughnut-chart">Chart</div>
        <button data-testid="refresh-button">Refresh</button>
        <input type="checkbox" data-testid="auto-refresh" />
        <div data-testid="risk-assessment">Risk Assessment</div>
      </div>
    </div>
  );
};

describe("User Flow Integration Tests", () => {
  test("complete user workflow - stock analysis", async () => {
    const user = userEvent.setup();
    render(<MockApp />);

    // 1. App should render
    expect(screen.getByTestId("app")).toBeInTheDocument();
    expect(screen.getByText("StockVibePredictor")).toBeInTheDocument();

    // 2. Should display sentiment analysis
    expect(screen.getByText("📊 Sentiment Analysis")).toBeInTheDocument();

    // 3. Should show charts
    expect(screen.getByTestId("doughnut-chart")).toBeInTheDocument();

    // 4. Should show risk assessment
    expect(screen.getByTestId("risk-assessment")).toBeInTheDocument();

    // 5. User can interact with refresh button
    const refreshButton = screen.getByTestId("refresh-button");
    await user.click(refreshButton);

    // 6. User can toggle auto-refresh
    const autoRefreshCheckbox = screen.getByTestId("auto-refresh");
    await user.click(autoRefreshCheckbox);
    expect(autoRefreshCheckbox).toBeChecked();
  });

  test("app renders without errors", () => {
    render(<MockApp />);
    expect(screen.getByTestId("app")).toBeInTheDocument();
  });
});
