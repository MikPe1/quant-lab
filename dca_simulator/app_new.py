"""Streamlit entry point for Quant Lab."""

import streamlit as st
from pages import (
    render_dca_page,
    render_financial_analysis_page,
    render_portfolio_backtest_page,
    render_portfolio_models_page,
)
from pages.arima_garch_page import render_arima_garch_page

st.set_page_config(
    layout="wide",
    page_title="Quant Lab",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Render the application and route to the selected analysis."""
    st.sidebar.title("Quant Lab")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Select tool",
        ["DCA Simulator", "Financial Analysis", "Portfolio Models", "Portfolio Backtest"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Quantitative portfolio analysis")

    if page == "DCA Simulator":
        render_dca_page()
    elif page == "Financial Analysis":
        analysis_type = st.sidebar.radio(
            "Analysis type",
            ["Standard Analysis", "ARIMA-GARCH Monte Carlo"],
            index=0
        )
        if analysis_type == "Standard Analysis":
            render_financial_analysis_page()
        else:
            render_arima_garch_page()
    elif page == "Portfolio Models":
        render_portfolio_models_page()
    elif page == "Portfolio Backtest":
        render_portfolio_backtest_page()


if __name__ == "__main__":
    main()
