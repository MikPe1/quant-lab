# Interactive DCA Simulation Tool

An interactive web application for simulating Dollar Cost Averaging (DCA) and 'buy on dip' investment strategies for stocks, visualizing performance with charts and data tables.

## Features

- User input for stock ticker, start date, initial capital, regular investment amount, and interval.
- Fetching of historical stock data using `yfinance`.
- Simulation of Dollar Cost Averaging (DCA) strategy.
- Optional 'buy on dip' advanced strategy with configurable threshold and multiplier.
- Interactive Plotly charts visualizing Portfolio Value vs. Total Invested and Profit/Loss over time.
- Detailed tabular summary of simulation results (Date, Price, Shares Bought, Total Shares, Total Invested, Portfolio Value, Profit/Loss).

## Setup and Installation

1. **Clone the repository (if applicable):**
   ```bash
   git clone <repository_url>
   cd dca_simulator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Ensure your virtual environment is activated.**

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your web browser** and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Project Structure

```
dca_simulator/
├── app.py                  # Main Streamlit application entry point
├── dca_engine.py           # Core DCA and 'buy on dip' simulation logic
├── data_fetcher.py         # Handles historical stock data fetching via yfinance
├── visualizer.py           # Generates interactive Plotly charts and data tables
├── config.py               # Configuration constants (e.g., default intervals)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```