import React, { useRef } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { Line, Bar } from "react-chartjs-2";
import "../CSS/StockChart.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  zoomPlugin
);

const StockChart = ({ data, ticker }) => {
  const priceChartRef = useRef(null);
  const volumeChartRef = useRef(null);

  const preparePriceData = () => {
    if (!data || data.length === 0) {
      return { labels: [], datasets: [] };
    }

    return {
      labels: data.map((record) => record.Date),
      datasets: [
        {
          label: `${ticker} Price`,
          data: data.map((record) => record.Close),
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.05)",
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointBackgroundColor: "#22c55e",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
        },
      ],
    };
  };

  const prepareVolumeData = () => {
    if (!data || data.length === 0) {
      return { labels: [], datasets: [] };
    }

    return {
      labels: data.map((record) => record.Date),
      datasets: [
        {
          label: "Volume",
          data: data.map((record) => record.Volume),
          backgroundColor: "#ef4444",
          borderColor: "#dc2626",
          borderWidth: 1,
        },
      ],
    };
  };

  const priceChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12,
            weight: "600",
          },
        },
      },
      title: {
        display: true,
        text: `${ticker} - Price Chart 📈`,
        font: {
          size: 16,
          weight: "bold",
        },
        padding: 15,
        color: "#1f2937",
      },
      tooltip: {
        backgroundColor: "rgba(17, 24, 39, 0.95)",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        borderColor: "#22c55e",
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            return `Price: $${value.toFixed(2)}`;
          },
        },
      },
      zoom: {
        wheel: {
          enabled: true,
          speed: 0.1,
          modifierKey: null,
        },
        pinch: {
          enabled: true,
        },
        mode: "x",
        limits: {
          x: { min: "original", max: "original" },
        },
        onZoom: (chart) => {
          // Prevent page zoom when zooming chart
          chart.canvas.style.touchAction = "none";
        },
      },
      pan: {
        enabled: true,
        mode: "x",
        threshold: 10,
        onPan: (chart) => {
          // Prevent page pan when panning chart
          chart.canvas.style.touchAction = "none";
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Date",
          font: {
            weight: "600",
            size: 12,
          },
          color: "#4b5563",
        },
        grid: {
          display: true,
          color: "rgba(107, 114, 128, 0.2)",
        },
        ticks: {
          maxTicksLimit: 8,
          color: "#6b7280",
          font: {
            size: 11,
          },
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Price ($)",
          font: {
            weight: "600",
            size: 12,
          },
          color: "#4b5563",
        },
        grid: {
          display: true,
          color: "rgba(107, 114, 128, 0.2)",
        },
        ticks: {
          color: "#6b7280",
          font: {
            size: 11,
          },
          callback: function (value) {
            return "$" + value.toFixed(2);
          },
        },
      },
    },
  };

  const volumeChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: "index",
      intersect: false,
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 12,
            weight: "600",
          },
        },
      },
      title: {
        display: true,
        text: `${ticker} - Volume Chart 📊`,
        font: {
          size: 16,
          weight: "bold",
        },
        padding: 15,
        color: "#1f2937",
      },
      tooltip: {
        backgroundColor: "rgba(17, 24, 39, 0.95)",
        titleColor: "#ffffff",
        bodyColor: "#ffffff",
        borderColor: "#ef4444",
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const value = context.parsed.y;
            return `Volume: ${(value / 1000000).toFixed(2)}M`;
          },
        },
      },
      zoom: {
        wheel: {
          enabled: true,
          speed: 0.1,
          modifierKey: null,
        },
        pinch: {
          enabled: true,
        },
        mode: "x",
        limits: {
          x: { min: "original", max: "original" },
        },
        onZoom: (chart) => {
          // Prevent page zoom when zooming chart
          chart.canvas.style.touchAction = "none";
        },
      },
      pan: {
        enabled: true,
        mode: "x",
        threshold: 10,
        onPan: (chart) => {
          // Prevent page pan when panning chart
          chart.canvas.style.touchAction = "none";
        },
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Date",
          font: {
            weight: "600",
            size: 12,
          },
          color: "#4b5563",
        },
        grid: {
          display: true,
          color: "rgba(107, 114, 128, 0.2)",
        },
        ticks: {
          maxTicksLimit: 8,
          color: "#6b7280",
          font: {
            size: 11,
          },
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: "Volume",
          font: {
            weight: "600",
            size: 12,
          },
          color: "#4b5563",
        },
        grid: {
          display: true,
          color: "rgba(107, 114, 128, 0.2)",
        },
        ticks: {
          color: "#6b7280",
          font: {
            size: 11,
          },
          callback: function (value) {
            return (value / 1000000).toFixed(1) + "M";
          },
        },
      },
    },
  };

  const calculateStats = () => {
    if (!data || data.length === 0) return null;

    const prices = data.map((record) => record.Close);
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2];
    const change = currentPrice - previousPrice;
    const changePercent = (change / previousPrice) * 100;

    const highPrice = Math.max(...prices);
    const lowPrice = Math.min(...prices);

    return {
      current: currentPrice,
      change: change,
      changePercent: changePercent,
      high: highPrice,
      low: lowPrice,
    };
  };

  const stats = calculateStats();

  return (
    <div className="stock-chart-container">
      <div className="chart-header">
        <h2 className="chart-title">📈 {ticker} Stock Analysis</h2>
        {stats && (
          <div className="price-stats">
            <div className="stat-item current-price">
              <span className="stat-label">Current Price</span>
              <span className="stat-value">${stats.current.toFixed(2)}</span>
            </div>
            <div
              className={`stat-item price-change ${
                stats.change >= 0 ? "positive" : "negative"
              }`}
            >
              <span className="stat-label">Change</span>
              <span className="stat-value">
                {stats.change >= 0 ? "+" : ""}${stats.change.toFixed(2)} (
                {stats.change >= 0 ? "+" : ""}
                {stats.changePercent.toFixed(2)}%)
              </span>
            </div>
            <div className="stat-item">
              <span className="stat-label">High</span>
              <span className="stat-value">${stats.high.toFixed(2)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Low</span>
              <span className="stat-value">${stats.low.toFixed(2)}</span>
            </div>
          </div>
        )}
      </div>

      <div className="charts-grid">
        <div className="chart-wrapper price-chart">
          <Line
            ref={priceChartRef}
            data={preparePriceData()}
            options={priceChartOptions}
          />
        </div>

        <div className="chart-wrapper volume-chart">
          <Bar
            ref={volumeChartRef}
            data={prepareVolumeData()}
            options={volumeChartOptions}
          />
        </div>
      </div>

      <div className="chart-info">
        <p>
          📊 Professional charts showing separate price and volume analysis. The
          AI model analyzes patterns, trends, and technical indicators from this
          historical data.
        </p>
        <p className="zoom-instructions">
          🖱️ <strong>Zoom:</strong> Use mouse wheel to zoom in/out on charts |{" "}
          <strong>Pan:</strong> Click and drag to move around
        </p>
      </div>
    </div>
  );
};

export default StockChart;
