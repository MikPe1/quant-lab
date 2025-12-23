"""
Quant Lab - Główny Punkt Wejścia Aplikacji

Kompleksowy symulator uśredniania kosztu zakupu (DCA) z zaawansowaną analizą portfela,
prognozowaniem Monte Carlo i narzędziami analizy instrumentów finansowych.

Zrefaktoryzowany dla łatwiejszej konserwacji - główna logika w oddzielnych modułach stron.
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
        "Wybierz Narzędzie", 
        ["Symulator DCA", "Analiza Instrumentu Finansowego", "Modele Portfelowe", "Backtest Portfela"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quant Lab**
    
    Profesjonalne narzędzia do ilościowej analizy i optymalizacji portfeli inwestycyjnych.
    """)
    
    # Routing do odpowiedniej strony
    if page == "Symulator DCA":
        if render_dca_page:
            render_dca_page()
        else:
            st.error("Moduł symulatora DCA nie został znaleziony. Sprawdź instalację.")
            
    elif page == "Analiza Instrumentu Finansowego":
        if render_financial_analysis_page:
            # Import ARIMA-GARCH page
            from pages.arima_garch_page import render_arima_garch_page
            
            # Podmenu dla Analizy Instrumentu Finansowego
            analysis_type = st.sidebar.radio(
                "Typ Analizy",
                ["Analiza Standardowa", "ARIMA-GARCH Monte Carlo"],
                index=0
            )
            
            if analysis_type == "Analiza Standardowa":
                render_financial_analysis_page()
            else:  # ARIMA-GARCH Monte Carlo
                render_arima_garch_page()
        else:
            st.warning("Strona Analizy Instrumentu Finansowego w przygotowaniu...")
            st.info("Ta strona jest refaktoryzowana. Tymczasowo użyj oryginalnego app.py.")
            
    elif page == "Modele Portfelowe":
        if render_portfolio_models_page:
            render_portfolio_models_page()
        else:
            st.warning("Strona Modeli Portfelowych w przygotowaniu...")
            st.info("Ta strona jest refaktoryzowana. Tymczasowo użyj oryginalnego app.py.")
    
    elif page == "Backtest Portfela":
        if render_portfolio_backtest_page:
            render_portfolio_backtest_page()
        else:
            st.warning("Strona Backtestu Portfela niedostępna.")
            st.info("Sprawdź, czy plik portfolio_backtest_page.py jest poprawnie zainstalowany.")


if __name__ == "__main__":
    main()
