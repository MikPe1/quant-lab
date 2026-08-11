"""
DCA Simulator Page - Dollar Cost Averaging simulation and visualization.
"""

import datetime

import streamlit as st
import config
from data_fetcher import fetch_historical_data
from dca_engine import simulate_dca
from visualizer import plot_simulation_results, display_results_table


def render_dca_page():
    """Render the DCA Simulator page."""
    
    st.title("DCA Simulator")
    with st.expander("Key Assumptions & Methodology", expanded=False):
        st.markdown("""
        **Dollar Cost Averaging (DCA)** is an investment strategy where a fixed amount is invested at regular intervals, regardless of asset price. This simulator allows you to compare standard DCA with an optional **"Buy on Dip"** enhancement, which increases investment after significant price drops.
        
        **Key Assumptions:**
        - All investments are made at the daily closing price (no intraday execution).
        - No transaction costs, taxes, or slippage are considered.
        - Sufficient liquidity is assumed for all trades.
        - Historical data is used as-is; past performance does not guarantee future results.
        - "Buy on Dip" triggers when the asset price drops by the specified percentage from the previous interval.
        - Buy & Hold benchmark assumes a single lump-sum investment at the start date.
        
        **Methodology:**
        - At each interval, the specified amount is invested.
        - If "Buy on Dip" is enabled and the price drop threshold is met, the investment amount is multiplied accordingly.
        - Portfolio value, total invested, and profit/loss are tracked over time.
        - Results are compared to a Buy & Hold strategy for the same period.
        """)
    
    st.sidebar.header("Investment Parameters")

    with st.sidebar.form("investment_form"):
        ticker = st.text_input(
            "Stock Ticker (e.g., AAPL)", 
            config.DEFAULT_TICKER
        ).upper()
        
        start_date = st.date_input(
            "Start Date", 
            datetime.datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date()
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)", 
            min_value=0, 
            value=config.DEFAULT_INITIAL_CAPITAL, 
            step=100
        )
        
        regular_investment_amount = st.number_input(
            "Regular Investment Amount ($)", 
            min_value=0, 
            value=config.DEFAULT_REGULAR_INVESTMENT_AMOUNT, 
            step=10
        )
        
        investment_interval = st.selectbox(
            "Investment Interval", 
            ["Daily", "Weekly", "Bi-Weekly", "Monthly"], 
            index=["Daily", "Weekly", "Bi-Weekly", "Monthly"].index(config.DEFAULT_INVESTMENT_INTERVAL)
        )

        st.sidebar.header("Advanced Strategy: Buy on Dip")
        enable_buy_on_dip = st.checkbox("Enable 'Buy on Dip' Strategy", value=False)
        
        buy_on_dip_threshold = st.slider(
            "Dip Threshold (%)", 
            min_value=1, 
            max_value=20, 
            value=config.DEFAULT_BUY_ON_DIP_THRESHOLD, 
            step=1, 
            disabled=not enable_buy_on_dip
        )
        
        buy_on_dip_multiplier = st.slider(
            "Investment Multiplier on Dip", 
            min_value=1.0, 
            max_value=5.0, 
            value=config.DEFAULT_BUY_ON_DIP_MULTIPLIER, 
            step=0.5, 
            disabled=not enable_buy_on_dip
        )

        submitted = st.form_submit_button("Run Simulation")
    
    if submitted:
        st.write(f"Running simulation for **{ticker}** from **{start_date}**")
        st.write(
            f"Initial capital: ${initial_capital:,.2f}  |  Regular investment: ${regular_investment_amount:,.2f} {investment_interval}"
        )
        
        if enable_buy_on_dip:
            st.info(f"'Buy on Dip' enabled with threshold: {buy_on_dip_threshold}% and multiplier: {buy_on_dip_multiplier}x")
        else:
            st.info("'Buy on Dip' strategy is disabled.")
    
        # Fetch historical data
        start_date_str = start_date.strftime("%Y-%m-%d")
        historical_data = fetch_historical_data(ticker, start_date_str)
        
        if historical_data.empty:
            st.error(f"No historical data found for ticker '{ticker}' starting from {start_date_str}. Please check the ticker and date.")
        else:
            # Run simulation
            buy_on_dip_thresh = buy_on_dip_threshold if enable_buy_on_dip else 0
            buy_on_dip_mult = buy_on_dip_multiplier if enable_buy_on_dip else 1.0
            
            with st.spinner("Running DCA simulation..."):
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
                st.success("Simulation complete.")
                st.subheader("Simulation Results")
                
                # Summary metrics
                final_row = simulation_results.iloc[-1]
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Final Portfolio Value", 
                        f"${final_row['Portfolio Value']:,.2f}"
                    )
                with col2:
                    st.metric(
                        "Total Invested", 
                        f"${final_row['Total Invested']:,.2f}"
                    )
                with col3:
                    profit_loss = final_row['Profit/Loss']
                    st.metric(
                        "Profit/Loss", 
                        f"${profit_loss:,.2f}",
                        delta=f"{(profit_loss / final_row['Total Invested'] * 100):.1f}%"
                    )
                with col4:
                    bh_profit = final_row['Buy&Hold Profit/Loss']
                    st.metric(
                        "Buy&Hold Profit/Loss", 
                        f"${bh_profit:,.2f}",
                        delta=f"{(bh_profit / final_row['Total Invested'] * 100):.1f}%"
                    )
                
                # Plot
                fig = plot_simulation_results(simulation_results)
                st.plotly_chart(fig, width='stretch')
                
                # Table
                with st.expander("View detailed results table"):
                    results_table = display_results_table(simulation_results)
                    st.dataframe(results_table, width='stretch')
                    
                    # Download button
                    csv = simulation_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"dca_simulation_{ticker}_{start_date_str}.csv",
                        mime="text/csv"
                    )
