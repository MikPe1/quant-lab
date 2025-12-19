# Quantitative Finance Portfolio Analysis Tool

A comprehensive web application for quantitative finance analysis, combining Dollar Cost Averaging (DCA) simulation with advanced portfolio optimization, risk analysis, and market benchmarking. Built with Streamlit, this tool provides both beginner-friendly DCA strategies and professional-grade portfolio analysis capabilities.

##  Features

### ** DCA Simulator**
- Interactive stock ticker input with real-time data fetching
- Dollar Cost Averaging (DCA) strategy simulation
- Advanced "buy on dip" strategy with configurable thresholds
- Portfolio performance visualization with Plotly charts
- Detailed transaction history and profit/loss tracking

### ** Financial Analysis**
- Comprehensive statistical analysis of stock returns
- Stationarity tests (ADF, KPSS) and autocorrelation analysis
- Volatility modeling with GARCH(1,1) and EGARCH
- Time series forecasting with ARIMA and GARCH models
- Risk metrics including Sharpe ratio, Sortino ratio, and maximum drawdown

### ** Advanced Portfolio Analysis**
- **6 Portfolio Optimization Methods:**
  - Mean-Variance Optimization (Markowitz)
  - Risk Parity
  - Hierarchical Risk Parity (HRP)
  - Minimum Variance
  - Equal Weight
  - Maximum Sharpe Ratio

- **Out-of-Sample Testing** with configurable train/test splits
- **S&P 500 Benchmarking** with outperformance analysis
- **Risk Analysis Tools:**
  - Value at Risk (VaR) at 95% and 99% confidence levels
  - Conditional VaR (CVaR) calculations
  - Portfolio skewness and kurtosis analysis
  - Stress testing with multiple risk metrics

- **Monte Carlo Simulations** (1-year forecasts with probability curves)
- **Interactive Visualizations:**
  - Cumulative returns comparison charts
  - Returns distribution histograms with VaR overlays
  - Monte Carlo simulation dashboards (4-panel layout)
  - Portfolio weights breakdowns and allocation charts

- **Method Comparison & Ranking** with composite scoring system
- **AI-Generated Investment Recommendations** based on analysis results

##  Technical Stack

- **Frontend:** Streamlit (multi-page application with sidebar navigation)
- **Data Fetching:** yfinance (Yahoo Finance API)
- **Data Processing:** pandas, numpy
- **Optimization:** scipy.optimize, scikit-learn (Ledoit-Wolf covariance shrinkage)
- **Risk Modeling:** arch (GARCH models)
- **Visualization:** plotly (interactive charts and dashboards)
- **Statistical Analysis:** statsmodels (time series analysis)

##  Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MikPe1/quant-lab.git
   cd quant-lab/dca_simulator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 How to Run

1. **Ensure virtual environment is activated**

2. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
dca_simulator/
├── app.py                  # Main Streamlit application (multi-page)
├── dca_engine.py           # DCA simulation logic and buy-on-dip strategies
├── data_fetcher.py         # Financial data fetching via yfinance
├── visualizer.py           # Plotly chart generation and visualization utilities
├── config.py               # Application configuration and default parameters
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
```

##  Usage Guide

### **DCA Simulator Page**
- Enter stock tickers (comma-separated for portfolios)
- Set investment parameters (initial capital, regular amounts, frequency)
- Configure "buy on dip" thresholds if desired
- View interactive charts and detailed transaction logs

### **Financial Analysis Page**
- Analyze individual stocks or portfolios
- Review statistical properties and stationarity tests
- Examine volatility patterns and GARCH model fits
- Generate time series forecasts

### **Advanced Portfolio Analysis Page**
- Select multiple assets for portfolio construction
- Choose optimization method and risk preferences
- Configure out-of-sample testing parameters
- Review comprehensive risk analysis and Monte Carlo simulations
- Compare methods and receive investment recommendations

##  Methodology

### **Portfolio Optimization**
- **Mean-Variance:** Classic Markowitz optimization with covariance shrinkage
- **Risk Parity:** Equal risk contribution across assets
- **HRP:** Machine learning approach using hierarchical clustering
- **Minimum Variance:** Lowest possible portfolio volatility
- **Equal Weight:** Simple 1/N allocation
- **Max Sharpe:** Highest risk-adjusted returns

### **Risk Analysis**
- **VaR/CVaR:** Historical simulation approach
- **Monte Carlo:** Geometric Brownian Motion with Cholesky decomposition
- **Stress Testing:** Comprehensive risk metric calculations

### **Benchmarking**
- S&P 500 index (^GSPC) as market benchmark
- Outperformance analysis and market timing metrics

##  Sample Analysis Output

The application provides:
- Performance metrics (returns, volatility, Sharpe ratios)
- Risk metrics (VaR, CVaR, drawdowns)
- Method rankings with composite scores
- Monte Carlo probability distributions
- Investment recommendations based on quantitative analysis

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

Originally created by MikPe1

## Disclaimer

This tool is for educational and research purposes only. Not financial advice.