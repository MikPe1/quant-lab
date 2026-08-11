# Quant Lab

Quant Lab is a Streamlit application for researching systematic investment strategies and portfolio risk. It combines dollar-cost averaging, single-asset analysis, portfolio construction, backtesting, and Monte Carlo forecasting in one interface.

## Scope

- DCA simulation with scheduled contributions and an optional buy-on-dip rule
- Return distributions, stationarity tests, technical indicators, and forecasts
- CAPM metrics and portfolio construction methods, including HRP, risk parity, minimum variance, inverse volatility, and mean-variance optimization
- Out-of-sample portfolio evaluation and benchmark comparison
- Historical and Monte Carlo risk metrics, including VaR and expected shortfall

## Run Locally

From the repository root:

```powershell
cd dca_simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app_new.py
```

The application uses Yahoo Finance for market data and requires an active network connection when an analysis is run.

## Project Layout

```text
dca_simulator/
├── app_new.py                   Streamlit entry point
├── config.py                    Default simulation parameters
├── data_fetcher.py              Market data access
├── dca_engine.py                DCA simulation logic
├── visualizer.py                DCA result charts and tables
├── pages/                       Streamlit page implementations
└── portfolio/                   Portfolio construction and simulation logic
```

## Research Notes

The application is intended for research and education. Backtests and forecasts are model-dependent estimates, not guarantees of future performance. Results do not include all real-world effects such as transaction costs, taxes, slippage, liquidity constraints, or execution timing unless explicitly stated by a page.
