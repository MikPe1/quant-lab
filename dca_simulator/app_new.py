"""
DCA Simulator - Main Application Entry Point

A comprehensive Dollar Cost Averaging simulator with advanced portfolio analysis,
Monte Carlo forecasting, and financial analysis tools.

Refactored for maintainability - main logic is in separate page modules.
"""

import streamlit as st
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Import page modules
try:
    from pages.dca_page import render_dca_page
except ImportError:
    render_dca_page = None

try:
    from pages.financial_analysis_page import render_financial_analysis_page
except ImportError:
    render_financial_analysis_page = None

try:
    from pages.portfolio_models_page import render_portfolio_models_page
except ImportError:
    render_portfolio_models_page = None

try:
    from pages.portfolio_backtest_page import render_portfolio_backtest_page
except ImportError:
    render_portfolio_backtest_page = None


# ============================================================================
# APP CONFIGURATION
# ============================================================================

st.set_page_config(
    layout="wide", 
    page_title="Quant Lab",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Hide module names at top
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("📊 Quant Lab")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Select Tool", 
        ["DCA Simulator", "Financial Analysis", "Portfolio Models", "Portfolio Backtest"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quant Lab**
    
    Professional tools for quantitative portfolio analysis and optimization.
    """)
    
    # Route to appropriate page
    if page == "DCA Simulator":
        if render_dca_page:
            render_dca_page()
        else:
            st.error("DCA page module not found. Please check installation.")
            
    elif page == "Financial Analysis":
        if render_financial_analysis_page:
            # Import ARIMA-GARCH page
            from pages.arima_garch_page import render_arima_garch_page
            
            # Add submenu for Financial Analysis
            analysis_type = st.sidebar.radio(
                "Analysis Type",
                ["Standard Analysis", "ARIMA-GARCH Monte Carlo"],
                index=0
            )
            
            if analysis_type == "Standard Analysis":
                render_financial_analysis_page()
            else:  # ARIMA-GARCH Monte Carlo
                render_arima_garch_page()
        else:
            st.warning("Financial Analysis page coming soon...")
            st.info("This page is being refactored. Please use the original app.py temporarily.")
            
    elif page == "Portfolio Models":
        if render_portfolio_models_page:
            render_portfolio_models_page()
        else:
            st.warning("Portfolio Models page coming soon...")
            st.info("This page is being refactored. Please use the original app.py temporarily.")
    
    elif page == "Portfolio Backtest":
        if render_portfolio_backtest_page:
            render_portfolio_backtest_page()
        else:
            st.warning("Portfolio Backtest page not available.")
            st.info("Please check that portfolio_backtest_page.py is installed correctly.")


if __name__ == "__main__":
    main()
