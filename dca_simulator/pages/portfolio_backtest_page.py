"""Portfolio backtest page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

from data_fetcher import fetch_historical_data


def render_portfolio_backtest_page():
    """Main function to render Portfolio Backtest page"""
    
    st.header("Portfolio Backtest Tool")
    
    # Key Assumptions
    with st.expander("Key Assumptions & Methodology", expanded=False):
        st.markdown("""
        **Testing Assumptions:**
        - No transaction costs or slippage included
        - Rebalancing is cost-free and instantaneous
        - All positions are liquid (can buy/sell any amount)
        - Fractional shares allowed
        - No bid-ask spread
        
        **Performance Metrics:**
        - Returns: Arithmetic and annualized returns
        - Sharpe Ratio: Risk-adjusted return (assumes 4% risk-free rate)
        - Maximum Drawdown: Largest peak-to-trough decline
        - Win Rate: Percentage of positive return days
        
        **Benchmark Comparison:**
        - Default benchmark: S&P 500 (^GSPC)
        - Benchmark uses same time period as portfolio
        - Performance metrics calculated identically for fair comparison
        
        **Important Notes:**
        - Past performance does not guarantee future results
        - Backtest results are theoretical and may not reflect real trading
        - Consider transaction costs in real implementation
        """)
    
    st.markdown("""
    Test any portfolio with custom weights on a selected time period.
    Compare performance against a benchmark index.
    
    **Workflow:**
    1. Copy weights from Portfolio Models page or enter manually
    2. Select backtest period and benchmark
    3. View performance metrics and charts
    """)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Portfolio Weights")
        weights_input = st.text_area(
            "Paste weights (ticker,weight format):",
            height=200,
            placeholder="ticker,weight\nAAPL,0.15\nMSFT,0.20\nGOOGL,0.18\n...",
            help="Format: ticker,weight (one per line)"
        )
        
    with col2:
        st.markdown("### Backtest Settings")
        start_date = st.date_input(
            "Start Date",
            value=pd.to_datetime("2020-01-01")
        )
        end_date = st.date_input(
            "End Date",
            value=pd.to_datetime("today")
        )
        
        benchmark_ticker = st.text_input(
            "Benchmark",
            value="^GSPC",
            help="Ticker symbol for benchmark comparison"
        )
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            value=100000,
            min_value=1000,
            step=10000,
            help="Starting portfolio value"
        )
    
    if st.button("Run Backtest", type="primary"):
        if not weights_input.strip():
            st.error("Please enter portfolio weights!")
            return
        
        # Parse weights
        try:
            weights_dict = _parse_weights(weights_input)
            
            if not weights_dict:
                st.error("No valid weights found. Check format: ticker,weight")
                return
            
            # Validate weights sum to ~1
            total_weight = sum(weights_dict.values())
            if not (0.95 <= total_weight <= 1.05):
                st.warning(f"Weights sum to {total_weight:.2%}. Normalizing to 100%...")
                # Normalize
                weights_dict = {k: v/total_weight for k, v in weights_dict.items()}
            
            # Display parsed weights
            st.success(f"Parsed {len(weights_dict)} positions")
            
            with st.expander("Parsed Weights"):
                weights_df = pd.DataFrame({
                    'Ticker': list(weights_dict.keys()),
                    'Weight': list(weights_dict.values()),
                    'Weight %': [v*100 for v in weights_dict.values()]
                })
                st.dataframe(weights_df.style.format({'Weight %': '{:.2f}%', 'Weight': '{:.4f}'}))
            
            # Run backtest
            _run_backtest(
                weights_dict,
                start_date,
                end_date,
                benchmark_ticker,
                initial_capital
            )
            
        except Exception as e:
            st.error(f"Error parsing weights: {str(e)}")
            st.info("Expected format:\nticker,weight\nAAPL,0.15\nMSFT,0.20")


def _parse_weights(weights_input):
    """Parse weights from text input"""
    weights_dict = {}
    
    # Try to read as CSV
    try:
        # Clean input
        lines = weights_input.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith('ticker'):
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                ticker = parts[0].strip().upper()
                try:
                    weight = float(parts[1].strip())
                    if 0 <= weight <= 1:
                        weights_dict[ticker] = weight
                except ValueError:
                    continue
    except Exception as e:
        st.error(f"Parse error: {e}")
    
    return weights_dict


def _run_backtest(weights_dict, start_date, end_date, benchmark_ticker, initial_capital):
    """Run portfolio backtest and display results"""
    
    st.markdown("---")
    st.subheader("Backtest Results")
    
    with st.spinner("Fetching data and running backtest..."):
        # Fetch portfolio data
        portfolio_data = {}
        failed_tickers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers = list(weights_dict.keys())
        for i, ticker in enumerate(tickers):
            status_text.text(f"Fetching {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            try:
                data = fetch_historical_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if not data.empty and len(data) > 10:
                    portfolio_data[ticker] = data
                else:
                    failed_tickers.append(ticker)
            except:
                failed_tickers.append(ticker)
        
        progress_bar.empty()
        status_text.empty()
        
        if failed_tickers:
            st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
            # Remove failed tickers and renormalize
            for ticker in failed_tickers:
                del weights_dict[ticker]
            
            if not weights_dict:
                st.error("No valid tickers with data!")
                return
            
            total = sum(weights_dict.values())
            weights_dict = {k: v/total for k, v in weights_dict.items()}
            st.info(f"Continuing with {len(weights_dict)} tickers (weights renormalized)")
        
        # Fetch benchmark
        benchmark_data = fetch_historical_data(benchmark_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if benchmark_data.empty:
            st.warning("Could not fetch benchmark data")
            benchmark_data = None
        
        # Calculate portfolio performance
        _calculate_and_display_performance(
            portfolio_data,
            weights_dict,
            benchmark_data,
            benchmark_ticker,
            initial_capital
        )


def _calculate_and_display_performance(portfolio_data, weights_dict, benchmark_data, benchmark_ticker, initial_capital):
    """Calculate and display portfolio performance metrics"""
    
    # Combine portfolio data
    price_df = pd.DataFrame(portfolio_data)
    
    # Calculate returns
    returns_df = price_df.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(0.0, index=returns_df.index)
    for ticker, weight in weights_dict.items():
        if ticker in returns_df.columns:
            portfolio_returns += returns_df[ticker] * weight
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_value = portfolio_cumulative * initial_capital
    
    # Calculate metrics
    total_return = portfolio_cumulative.iloc[-1] - 1
    num_years = len(portfolio_returns) / 252
    annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    cumulative_max = portfolio_cumulative.expanding().max()
    drawdown = (portfolio_cumulative - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)
    
    # Benchmark metrics
    benchmark_metrics = None
    if benchmark_data is not None:
        # Align benchmark with portfolio dates
        common_dates = benchmark_data.index.intersection(portfolio_returns.index)
        if len(common_dates) > 0:
            benchmark_returns = benchmark_data.loc[common_dates].pct_change().dropna()
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_value = benchmark_cumulative * initial_capital
            
            bench_total_return = benchmark_cumulative.iloc[-1] - 1
            bench_annual_return = (1 + bench_total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
            bench_annual_vol = benchmark_returns.std() * np.sqrt(252)
            bench_sharpe = (bench_annual_return - 0.04) / bench_annual_vol if bench_annual_vol > 0 else 0
            
            bench_cumulative_max = benchmark_cumulative.expanding().max()
            bench_drawdown = (benchmark_cumulative - bench_cumulative_max) / bench_cumulative_max
            bench_max_drawdown = bench_drawdown.min()
            
            benchmark_metrics = {
                'returns': benchmark_returns,
                'cumulative': benchmark_cumulative,
                'value': benchmark_value,
                'total_return': bench_total_return,
                'annual_return': bench_annual_return,
                'annual_vol': bench_annual_vol,
                'sharpe': bench_sharpe,
                'max_drawdown': bench_max_drawdown,
                'drawdown': bench_drawdown
            }
    
    # Display metrics
    st.markdown("### Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{total_return:.1%}")
        st.metric("Annual Return", f"{annual_return:.1%}")
        st.metric("Volatility (Annual)", f"{annual_vol:.1%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.metric("Max Drawdown", f"{max_drawdown:.1%}")
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    with col3:
        st.metric("Final Value", f"${portfolio_value.iloc[-1]:,.0f}")
        st.metric("Initial Capital", f"${initial_capital:,.0f}")
        st.metric("Profit/Loss", f"${portfolio_value.iloc[-1] - initial_capital:,.0f}")
    
    # Benchmark comparison
    if benchmark_metrics:
        st.markdown(f"### 🏁 vs {benchmark_ticker} Benchmark")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_return = annual_return - benchmark_metrics['annual_return']
            st.metric(
                "Return vs Benchmark",
                f"{annual_return:.1%}",
                delta=f"{delta_return:+.1%}"
            )
        
        with col2:
            delta_sharpe = sharpe - benchmark_metrics['sharpe']
            st.metric(
                "Sharpe vs Benchmark",
                f"{sharpe:.3f}",
                delta=f"{delta_sharpe:+.3f}"
            )
        
        with col3:
            st.metric(
                "Benchmark Return",
                f"{benchmark_metrics['annual_return']:.1%}"
            )
        
        with col4:
            st.metric(
                "Benchmark Sharpe",
                f"{benchmark_metrics['sharpe']:.3f}"
            )
    
    # Visualization
    _plot_backtest_results(
        portfolio_value,
        portfolio_cumulative,
        drawdown,
        benchmark_metrics,
        benchmark_ticker
    )
    
    # Download results
    _create_backtest_download(
        portfolio_returns,
        portfolio_value,
        weights_dict,
        start_date,
        end_date
    )


def _plot_backtest_results(portfolio_value, portfolio_cumulative, drawdown, benchmark_metrics, benchmark_ticker):
    """Plot backtest results"""
    
    st.markdown("### Performance Charts")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Cumulative Returns (%)',
            'Drawdown',
            'Rolling 30-Day Return'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Portfolio Value
    fig.add_trace(
        go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            name='Portfolio',
            line=dict(color='#2E86AB', width=2)
        ),
        row=1, col=1
    )
    
    if benchmark_metrics:
        fig.add_trace(
            go.Scatter(
                x=benchmark_metrics['value'].index,
                y=benchmark_metrics['value'].values,
                name=benchmark_ticker,
                line=dict(color='lightgray', width=2, dash='dash')
            ),
            row=1, col=1
        )
    
    # 2. Cumulative Returns (%)
    fig.add_trace(
        go.Scatter(
            x=portfolio_cumulative.index,
            y=(portfolio_cumulative - 1) * 100,
            name='Portfolio',
            line=dict(color='#2E86AB', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    if benchmark_metrics:
        fig.add_trace(
            go.Scatter(
                x=benchmark_metrics['cumulative'].index,
                y=(benchmark_metrics['cumulative'] - 1) * 100,
                name=benchmark_ticker,
                line=dict(color='lightgray', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name='Portfolio DD',
            fill='tozeroy',
            line=dict(color='#C73E1D', width=1),
            showlegend=False
        ),
        row=2, col=1
    )
    
    if benchmark_metrics:
        fig.add_trace(
            go.Scatter(
                x=benchmark_metrics['drawdown'].index,
                y=benchmark_metrics['drawdown'].values * 100,
                name='Benchmark DD',
                line=dict(color='lightgray', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Rolling 30-day return
    portfolio_returns = portfolio_cumulative.pct_change()
    rolling_30d = portfolio_returns.rolling(window=30).sum() * 100
    
    fig.add_trace(
        go.Scatter(
            x=rolling_30d.index,
            y=rolling_30d.values,
            name='30-Day Return',
            line=dict(color='#2E86AB', width=1),
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, width='stretch')


def _create_backtest_download(portfolio_returns, portfolio_value, weights_dict, start_date, end_date):
    """Create downloadable CSV with backtest results"""
    
    st.markdown("### Download Results")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Date': portfolio_returns.index,
        'Daily_Return': portfolio_returns.values,
        'Portfolio_Value': portfolio_value.values,
        'Cumulative_Return': (portfolio_value.values / portfolio_value.iloc[0] - 1)
    })
    
    # Add weights info as header comment
    weights_str = ', '.join([f"{k}:{v:.4f}" for k, v in weights_dict.items()])
    
    csv_string = results_df.to_csv(index=False)
    csv_with_header = f"# Portfolio Backtest Results\n# Period: {start_date} to {end_date}\n# Weights: {weights_str}\n{csv_string}"
    
    st.download_button(
        label="Download Backtest Results",
        data=csv_with_header,
        file_name=f"backtest_results_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
