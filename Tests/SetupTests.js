import "@testing-library/jest-dom";

// Mock Chart.js
jest.mock(
  "chart.js",
  () => ({
    Chart: {
      register: jest.fn(),
    },
    ArcElement: jest.fn(),
    Tooltip: jest.fn(),
    Legend: jest.fn(),
    Title: jest.fn(),
    CategoryScale: jest.fn(),
    LinearScale: jest.fn(),
    BarElement: jest.fn(),
    LineElement: jest.fn(),
    PointElement: jest.fn(),
  }),
  { virtual: true }
);

// Mock react-chartjs-2
jest.mock(
  "react-chartjs-2",
  () => ({
    Doughnut: ({ data, options }) => (
      <div
        data-testid="doughnut-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
    Bar: ({ data, options }) => (
      <div
        data-testid="bar-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
    Line: ({ data, options }) => (
      <div
        data-testid="line-chart"
        data-chart-data={JSON.stringify(data)}
        data-options={JSON.stringify(options)}
      />
    ),
  }),
  { virtual: true }
);

// DON'T mock CSS here - let moduleNameMapper handle it
// jest.mock("*.css", () => ({})); // REMOVED THIS LINE

// Mock axios
jest.mock(
  "axios",
  () => ({
    default: {
      get: jest.fn(() => Promise.resolve({ data: {} })),
      post: jest.fn(() => Promise.resolve({ data: {} })),
    },
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
  }),
  { virtual: true }
);

// Global mocks
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
  })
);

global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock matchMedia
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.localStorage = localStorageMock;

// Mock sessionStorage
const sessionStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
global.sessionStorage = sessionStorageMock;

// Clean up after each test
afterEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});
