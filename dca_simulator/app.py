# app.py

import streamlit as st
import datetime
from config import (
    DEFAULT_TICKER,
    DEFAULT_START_DATE,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_REGULAR_INVESTMENT_AMOUNT,
    DEFAULT_INVESTMENT_INTERVAL,
    DEFAULT_BUY_ON_DIP_THRESHOLD,
    DEFAULT_BUY_ON_DIP_MULTIPLIER
)
from data_fetcher import fetch_historical_data
from dca_engine import simulate_dca
from visualizer import plot_simulation_results, display_results_table

st.set_page_config(layout="wide", page_title="Interactive DCA Simulation Tool")

st.title("📈 Interactive DCA Simulation Tool")

st.sidebar.header("Investment Parameters")

with st.sidebar.form("investment_form"):
    ticker = st.text_input("Stock Ticker (e.g., AAPL)", DEFAULT_TICKER).upper()
    start_date = st.date_input("Start Date", datetime.datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date())
    initial_capital = st.number_input("Initial Capital ($)", min_value=0, value=DEFAULT_INITIAL_CAPITAL, step=100)
    regular_investment_amount = st.number_input("Regular Investment Amount ($)", min_value=0, value=DEFAULT_REGULAR_INVESTMENT_AMOUNT, step=10)
    investment_interval = st.selectbox("Investment Interval", ["Daily", "Weekly", "Bi-Weekly", "Monthly"], index=["Daily", "Weekly", "Bi-Weekly", "Monthly"].index(DEFAULT_INVESTMENT_INTERVAL))

    st.sidebar.header("Advanced Strategy: Buy on Dip")
    enable_buy_on_dip = st.checkbox("Enable 'Buy on Dip' Strategy", value=False)
    buy_on_dip_threshold = st.slider("Dip Threshold (%)", min_value=1, max_value=20, value=DEFAULT_BUY_ON_DIP_THRESHOLD, step=1, disabled=not enable_buy_on_dip)
    buy_on_dip_multiplier = st.slider("Investment Multiplier on Dip", min_value=1.0, max_value=5.0, value=DEFAULT_BUY_ON_DIP_MULTIPLIER, step=0.5, disabled=not enable_buy_on_dip)

    submitted = st.form_submit_button("Run Simulation")

if submitted:
    st.write(f"Running simulation for {ticker} from {start_date} with initial capital of ${initial_capital} and regular investment of ${regular_investment_amount} {investment_interval}.")
    if enable_buy_on_dip:
        st.write(f"'Buy on Dip' enabled with threshold: {buy_on_dip_threshold}% and multiplier: {buy_on_dip_multiplier}x.")
    else:
        st.write("'Buy on Dip' strategy is disabled.")

    # Fetch historical data
    start_date_str = start_date.strftime("%Y-%m-%d")
    historical_data = fetch_historical_data(ticker, start_date_str)
    
    if historical_data.empty:
        st.error(f"No historical data found for ticker '{ticker}' starting from {start_date_str}. Please check the ticker and date.")
    else:
        # Run simulation
        buy_on_dip_thresh = buy_on_dip_threshold if enable_buy_on_dip else 0
        buy_on_dip_mult = buy_on_dip_multiplier if enable_buy_on_dip else 1.0
        simulation_results = simulate_dca(
            historical_data,
            initial_capital,
            regular_investment_amount,
            investment_interval,
            buy_on_dip_thresh,
            buy_on_dip_mult
        )
        
        if simulation_results.empty:
            st.error("Simulation failed. Please check your parameters.")
        else:
            # Display results
            st.subheader("Simulation Results")
            
            # Plot
            fig = plot_simulation_results(simulation_results)
            st.plotly_chart(fig, width='stretch')
            
            # Table
            st.subheader("Detailed Results Table")
            results_table = display_results_table(simulation_results)
            st.dataframe(results_table, width='stretch')