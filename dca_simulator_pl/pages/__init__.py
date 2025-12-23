"""Pages package - Streamlit page components."""

from .dca_page import render_dca_page
from .financial_analysis_page import render_financial_analysis_page
from .portfolio_models_page import render_portfolio_models_page
from .portfolio_backtest_page import render_portfolio_backtest_page
from .arima_garch_page import render_arima_garch_page

__all__ = [
    'render_dca_page',
    'render_financial_analysis_page',
    'render_portfolio_models_page',
    'render_portfolio_backtest_page',
    'render_arima_garch_page'
]
