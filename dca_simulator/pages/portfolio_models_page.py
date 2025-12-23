"""
Portfolio Models Page
CAPM analysis, portfolio optimization, and comprehensive advanced portfolio analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_fetcher import fetch_historical_data
from portfolio.optimization import extract_optimal_weights_with_oos, stress_test_analysis
from portfolio.monte_carlo import monte_carlo_forecast_streamlit


def render_portfolio_models_page():
    """Main function to render the Portfolio Models page"""

    # Basic CAPM and Portfolio Optimization Section
    _render_basic_portfolio_section()

    # Separator
    st.markdown("---")

    # Extended Advanced Analysis Section
    _render_extended_analysis_section()


def _render_basic_portfolio_section():
    """Render basic CAPM and portfolio optimization"""
    # Inputs
    tickers_input = st.text_area("Enter tickers separated by commas", "AAPL, MSFT, GOOGL")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    market_ticker = st.text_input("Market Ticker", "^GSPC")
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.5, min_value=0.0, max_value=20.0) / 100
    num_portfolios = st.number_input("Number of Portfolios for Monte Carlo", value=10000, min_value=100, max_value=50000, step=1000)

    if st.button("Calculate CAPM and Portfolio Metrics"):
        if not tickers:
            st.error("Please enter at least one ticker.")
        else:
            # Fetch market data
            market_data = fetch_historical_data(market_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if market_data.empty:
                st.error(f"No data for market ticker {market_ticker}")
            else:
                market_returns = market_data.pct_change().dropna()
                
                stock_data = {}
                stock_returns = {}
                capm_results = []
                
                for ticker in tickers:
                    data = fetch_historical_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    if data.empty:
                        capm_results.append({"Ticker": ticker, "Beta": None, "Expected Return": None, "Sharpe Ratio": None, "Error": "No data"})
                        continue
                    
                    returns = data.pct_change().dropna()
                    stock_data[ticker] = data
                    stock_returns[ticker] = returns
                    
                    # Align with market
                    common_index = market_returns.index.intersection(returns.index)
                    if len(common_index) < 30:
                        capm_results.append({"Ticker": ticker, "Beta": None, "Expected Return": None, "Sharpe Ratio": None, "Error": "Insufficient data"})
                        continue
                    
                    market_ret_aligned = market_returns.loc[common_index]
                    stock_ret_aligned = returns.loc[common_index]
                    
                    # Beta
                    covariance = np.cov(stock_ret_aligned, market_ret_aligned)[0, 1]
                    market_variance = np.var(market_ret_aligned)
                    beta = covariance / market_variance if market_variance > 0 else np.nan
                    
                    # Expected market return
                    expected_market_return = market_ret_aligned.mean() * 252
                    
                    # CAPM expected return
                    expected_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)
                    
                    # Calculate both actual and theoretical metrics
                    actual_annual_return = stock_ret_aligned.mean() * 252
                    annual_volatility = stock_ret_aligned.std() * np.sqrt(252)
                    
                    # CAPM Sharpe (theoretical - ex-ante)
                    capm_sharpe = (expected_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else np.nan
                    
                    # Actual Sharpe (empirical - ex-post)
                    actual_sharpe = (actual_annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else np.nan
                    
                    capm_results.append({
                        "Ticker": ticker, 
                        "Beta": beta, 
                        "CAPM Expected Return": expected_return,
                        "Actual Annual Return": actual_annual_return,
                        "Annual Volatility": annual_volatility,
                        "CAPM Sharpe Ratio": capm_sharpe,
                        "Actual Sharpe Ratio": actual_sharpe,
                        "Error": None
                    })
                
                # Display CAPM results
                df_capm = pd.DataFrame(capm_results)
                st.subheader("CAPM Results")
                
                st.info("""
                📊 **Understanding the Sharpe Ratios:**
                - **CAPM Sharpe** = Theoretical (ex-ante) using CAPM expected return
                - **Actual Sharpe** = Historical (ex-post) using realized return
                - If they're similar, CAPM predicts well for that asset
                """)
                
                # Format display
                display_df = df_capm.copy()
                for col in ['Beta', 'CAPM Expected Return', 'Actual Annual Return', 'Annual Volatility']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                for col in ['CAPM Sharpe Ratio', 'Actual Sharpe Ratio']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Plots
                valid_capm = df_capm.dropna(subset=['Beta'])
                if not valid_capm.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_beta = px.bar(valid_capm, x='Ticker', y='Beta', title="Betas (Market Sensitivity)")
                        fig_beta.update_layout(height=400)
                        st.plotly_chart(fig_beta, use_container_width=True)
                    
                    with col2:
                        # Sharpe Ratios comparison
                        fig_sharpe = go.Figure()
                        fig_sharpe.add_trace(go.Bar(
                            x=valid_capm['Ticker'],
                            y=valid_capm['CAPM Sharpe Ratio'],
                            name='CAPM Sharpe (Theoretical)',
                            marker_color='lightcoral'
                        ))
                        fig_sharpe.add_trace(go.Bar(
                            x=valid_capm['Ticker'],
                            y=valid_capm['Actual Sharpe Ratio'],
                            name='Actual Sharpe (Historical)',
                            marker_color='darkred'
                        ))
                        fig_sharpe.update_layout(
                            title="Sharpe Ratios: CAPM vs Actual",
                            xaxis_title="Ticker",
                            yaxis_title="Sharpe Ratio",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig_sharpe, use_container_width=True)
                    
                    # Returns comparison
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Bar(
                        x=valid_capm['Ticker'],
                        y=valid_capm['CAPM Expected Return'],
                        name='CAPM Expected Return',
                        marker_color='lightblue'
                    ))
                    fig_returns.add_trace(go.Bar(
                        x=valid_capm['Ticker'],
                        y=valid_capm['Actual Annual Return'],
                        name='Actual Historical Return',
                        marker_color='darkblue'
                    ))
                    fig_returns.update_layout(
                        title="Expected vs Actual Returns",
                        xaxis_title="Ticker",
                        yaxis_title="Annual Return",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
                
                # Portfolio optimization with Monte Carlo
                if len(stock_returns) > 1:
                    st.subheader("Portfolio Optimization (Monte Carlo)")
                    
                    # Combine returns
                    returns_df = pd.DataFrame(stock_returns).dropna()
                    if returns_df.shape[1] < 2:
                        st.error("Need at least 2 stocks with data for portfolio optimization.")
                    else:
                        mean_returns = returns_df.mean()
                        cov_matrix = returns_df.cov()
                        
                        # Monte Carlo simulation
                        results = np.zeros((3, num_portfolios))
                        weights_record = []
                        
                        for i in range(num_portfolios):
                            weights = np.random.random(len(tickers))
                            weights /= np.sum(weights)
                            weights_record.append(weights)
                            
                            portfolio_return = np.sum(mean_returns * weights) * 252
                            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
                            
                            results[0, i] = portfolio_return
                            results[1, i] = portfolio_std
                            results[2, i] = sharpe_ratio
                        
                        # Find optimal
                        max_sharpe_idx = np.argmax(results[2])
                        min_vol_idx = np.argmin(results[1])
                        
                        st.write(f"Maximum Sharpe Ratio Portfolio: Return {results[0, max_sharpe_idx]:.2%}, Vol {results[1, max_sharpe_idx]:.2%}, Sharpe {results[2, max_sharpe_idx]:.2f}")
                        st.write(f"Minimum Volatility Portfolio: Return {results[0, min_vol_idx]:.2%}, Vol {results[1, min_vol_idx]:.2%}, Sharpe {results[2, min_vol_idx]:.2f}")
                        
                        # Plot
                        fig_port = go.Figure()
                        fig_port.add_trace(go.Scatter(x=results[1,:], y=results[0,:], mode='markers', marker=dict(color=results[2,:], colorscale='Viridis', showscale=True), name='Portfolios'))
                        fig_port.add_trace(go.Scatter(x=[results[1, max_sharpe_idx]], y=[results[0, max_sharpe_idx]], mode='markers', marker=dict(size=15, color='red'), name='Max Sharpe'))
                        fig_port.add_trace(go.Scatter(x=[results[1, min_vol_idx]], y=[results[0, min_vol_idx]], mode='markers', marker=dict(size=15, color='blue'), name='Min Vol'))
                        fig_port.update_layout(title="Portfolio Optimization", xaxis_title="Volatility", yaxis_title="Return")
                        st.plotly_chart(fig_port)
                        
                        # Weights for max Sharpe
                        max_sharpe_weights_array = weights_record[max_sharpe_idx]
                        max_sharpe_weights = pd.Series(max_sharpe_weights_array, index=tickers)
                        
                        st.subheader("🎯 Max Sharpe Portfolio Weights")
                        
                        # Display table
                        weights_df = pd.DataFrame({
                            'Ticker': max_sharpe_weights.index,
                            'Weight': max_sharpe_weights.values,
                            'Weight %': max_sharpe_weights.values * 100
                        })
                        st.dataframe(
                            weights_df.style.format({'Weight': '{:.4f}', 'Weight %': '{:.2f}%'}),
                            use_container_width=True
                        )
                        
                        # CSV Download with Copy-Paste Preview
                        st.markdown("### 📥 Download Weights")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Simple CSV format
                            simple_csv = pd.DataFrame({
                                'ticker': max_sharpe_weights.index,
                                'weight': max_sharpe_weights.values
                            })
                            csv_string = simple_csv.to_csv(index=False)
                            
                            import datetime
                            filename = f"max_sharpe_weights_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
                            
                            st.download_button(
                                label="📊 Download Weights (Simple)",
                                data=csv_string,
                                file_name=filename,
                                mime="text/csv",
                                help="Simple format: ticker,weight - Ready for copy-paste"
                            )
                        
                        with col2:
                            st.metric(
                                "Total Positions",
                                len(max_sharpe_weights),
                                help="Number of assets in portfolio"
                            )
                        
                        # Copy-Paste Preview
                        with st.expander("📋 Copy-Paste Format Preview", expanded=False):
                            st.caption("Copy this format for Portfolio Backtest page")
                            st.code(csv_string, language="csv")
                else:
                    st.info("Add more tickers for portfolio optimization.")


def _render_extended_analysis_section():
    """Render extended advanced portfolio analysis"""
    st.header("🚀 Extended Advanced Portfolio Analysis")
    st.markdown("""
    **Comprehensive analysis including:**
    - Multiple portfolio construction methods (HRP, Risk Parity, Mean-Variance, etc.)
    - Out-of-sample testing to avoid overfitting
    - Stress testing with VaR and CVaR
    - Monte Carlo forecasting
    - Hierarchical Risk Parity implementation
    """)
    
    # Separate ticker input for extended analysis
    extended_tickers_input = st.text_area(
        "Tickers for Extended Analysis (separate with commas)", 
        "AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, JNJ, WMT, PG",
        key="extended_tickers"
    )
    extended_tickers = [t.strip().upper() for t in extended_tickers_input.split(',') if t.strip()]
    
    # Date inputs for extended analysis
    col1, col2 = st.columns(2)
    with col1:
        extended_start_date = st.date_input(
            "Extended Analysis Start Date", 
            value=pd.to_datetime("2020-01-01"),
            key="extended_start"
        )
    with col2:
        extended_end_date = st.date_input(
            "Extended Analysis End Date", 
            value=pd.to_datetime("today"),
            key="extended_end"
        )
    
    # Mode selection
    st.markdown("### Analysis Mode")
    analysis_mode = st.radio(
        "Select Mode",
        ["Backtesting (Validate Strategy)", "Production (Optimize for Future)"],
        help="Backtesting: Split data to validate. Production: Use all data to optimize real portfolio."
    )
    
    col3, col4 = st.columns([1, 1])
    with col3:
        if analysis_mode == "Backtesting (Validate Strategy)":
            train_split = st.slider("Training Data Split (%)", 30, 90, 80, 5)
        else:
            train_split = 100  # Use all data in production mode
            st.info("🎯 **Production Mode**: Using 100% of selected date range")
        mc_simulations = st.number_input("Monte Carlo Simulations", 1000, 10000, 5000, 1000)
    with col4:
        mc_period_years = st.slider("Monte Carlo Period (years)", 1, 5, 1, 1)
        mc_distribution = st.selectbox("Monte Carlo Distribution", ["normal", "t-student"], index=0, key="mc_dist")
    
    # Portfolio method selection
    st.markdown("### Portfolio Method Selection")
    
    if analysis_mode == "Production (Optimize for Future)":
        portfolio_method = st.selectbox(
            "Preferred Portfolio Method (or Auto-select best)",
            ["Auto-select (Best by Sharpe)", "HRP", "Equal Weight", "Inverse Volatility", 
             "Mean-Variance (Markowitz)", "Minimum Variance", "Risk Parity"],
            index=0
        )
    else:
        portfolio_method = st.selectbox(
            "Preferred Portfolio Method (or Auto-select best)",
            ["Auto-select (Best OOS Performance)", "HRP", "Equal Weight", "Inverse Volatility", 
             "Mean-Variance (Markowitz)", "Minimum Variance", "Risk Parity"],
            index=0
        )
    
    # Info based on mode
    if analysis_mode == "Backtesting (Validate Strategy)":
        st.info(f"""
        📊 **Backtesting Mode**: {train_split}% / {100-train_split}%
        
        Portfolio weights are calculated **only on training data** ({train_split}%), 
        then tested out-of-sample on test data ({100-train_split}%) to avoid overfitting.
        
        ✅ Distribution parameters fitted on training data only
        ✅ All optimization uses only training period
        ✅ Performance metrics show out-of-sample results
        
        **Purpose**: Validate that the strategy would have worked historically.
        """)
    else:
        date_range_str = f"{extended_start_date.strftime('%Y-%m-%d')} to {extended_end_date.strftime('%Y-%m-%d')}"
        st.success(f"""
        🎯 **Production Mode**: Real Portfolio Optimization
        
        Portfolio weights are calculated using **ALL data in date range** ({100}%).
        
        **Selected Period**: {date_range_str}
        
        ✅ Uses most recent market conditions (up to end date)
        ✅ Maximum information for optimization
        ✅ Monte Carlo forecasts future performance
        
        **Purpose**: Build optimal portfolio for deployment with real capital.
        
        ⚠️ Note: No out-of-sample test (we're optimizing for unknown future).
        Backtesting mode validates the methodology works historically.
        """)
    
    button_text = "🔬 Run Backtest Analysis" if analysis_mode == "Backtesting (Validate Strategy)" else "🎯 Optimize Production Portfolio"
    
    if st.button(button_text):
        if not extended_tickers or len(extended_tickers) < 3:
            st.error("Please enter at least 3 tickers for meaningful portfolio analysis.")
        elif len(extended_tickers) > 15:
            st.error("Please use no more than 15 tickers to avoid computational issues.")
        else:
            is_production = (analysis_mode == "Production (Optimize for Future)")
            with st.spinner("Running comprehensive portfolio analysis... This may take a few minutes."):
                try:
                    # Fetch data with better error handling
                    data = {}
                    failed_tickers = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, ticker in enumerate(extended_tickers):
                        status_text.text(f"Fetching data for {ticker}...")
                        progress_bar.progress((i + 1) / len(extended_tickers))
                        
                        try:
                            ticker_data = fetch_historical_data(ticker, extended_start_date.strftime("%Y-%m-%d"), extended_end_date.strftime("%Y-%m-%d"))
                            if not ticker_data.empty and len(ticker_data) > 100:
                                data[ticker] = ticker_data
                            else:
                                failed_tickers.append(ticker)
                        except Exception as e:
                            failed_tickers.append(ticker)
                            continue
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if failed_tickers:
                        st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")
                    
                    if len(data) < 3:
                        st.error("Need at least 3 tickers with valid data.")
                    else:
                        st.info(f"Successfully loaded data for {len(data)} tickers: {', '.join(data.keys())}")
                        
                        # Create combined DataFrame
                        combined_data = pd.DataFrame(data)
                        combined_data = combined_data.dropna(how='all')
                        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
                        combined_data = combined_data.dropna()
                        
                        if len(combined_data) < 504:
                            st.error(f"Need at least 2 years of data. Only {len(combined_data)} days available.")
                        else:
                            # Calculate returns
                            returns = combined_data.pct_change().dropna()
                            
                            if len(returns) < 500:
                                st.error("Insufficient return data after processing.")
                            else:
                                st.success(f"Data prepared: {len(returns)} days of returns for {len(returns.columns)} assets")
                                
                                # Fetch S&P 500 benchmark data (full period for visualization)
                                st.info("Fetching S&P 500 benchmark data...")
                                sp500_data = fetch_historical_data('^GSPC', extended_start_date.strftime("%Y-%m-%d"), extended_end_date.strftime("%Y-%m-%d"))
                                sp500_returns = None
                                if not sp500_data.empty and len(sp500_data) > 100:
                                    sp500_returns = sp500_data.pct_change().dropna()
                                    st.success("✅ S&P 500 benchmark data loaded")
                                else:
                                    st.warning("⚠️ Could not load S&P 500 benchmark data")
                                
                                # Run optimization
                                oos_results = extract_optimal_weights_with_oos(
                                    returns, 
                                    train_test_split=train_split/100
                                )
                                
                                # Calculate benchmark performance
                                benchmark_metrics = None
                                if sp500_returns is not None:
                                    if is_production:
                                        # Production: use all SP500 data for info
                                        sp500_cumulative = (1 + sp500_returns).cumprod()
                                        sp500_total_return = sp500_cumulative.iloc[-1] - 1
                                        sp500_annual_return = (1 + sp500_total_return) ** (252 / len(sp500_returns)) - 1
                                        sp500_volatility = sp500_returns.std() * np.sqrt(252)
                                        sp500_sharpe = (sp500_annual_return - 0.04) / sp500_volatility if sp500_volatility > 0 else 0
                                        sp500_max_dd = ((sp500_cumulative - sp500_cumulative.expanding().max()) / sp500_cumulative.expanding().max()).min()
                                        
                                        benchmark_metrics = {
                                            'Annual_Return': sp500_annual_return,
                                            'Annual_Vol': sp500_volatility,
                                            'Sharpe_Ratio': sp500_sharpe,
                                            'Max_Drawdown': sp500_max_dd,
                                            'Total_Return': sp500_total_return,
                                            'Period': 'Full History'
                                        }
                                    else:
                                        # Backtesting: use only test period
                                        common_dates = sp500_returns.index.intersection(oos_results['test_returns'].index)
                                        if len(common_dates) > 0:
                                            sp500_test_returns = sp500_returns.loc[common_dates]
                                            sp500_cumulative = (1 + sp500_test_returns).cumprod()
                                            
                                            sp500_total_return = sp500_cumulative.iloc[-1] - 1
                                            sp500_annual_return = (1 + sp500_total_return) ** (252 / len(sp500_test_returns)) - 1
                                            sp500_volatility = sp500_test_returns.std() * np.sqrt(252)
                                            sp500_sharpe = (sp500_annual_return - 0.04) / sp500_volatility if sp500_volatility > 0 else 0
                                            sp500_max_dd = ((sp500_cumulative - sp500_cumulative.expanding().max()) / sp500_cumulative.expanding().max()).min()
                                            
                                            benchmark_metrics = {
                                                'Annual_Return': sp500_annual_return,
                                                'Annual_Vol': sp500_volatility,
                                                'Sharpe_Ratio': sp500_sharpe,
                                                'Max_Drawdown': sp500_max_dd,
                                                'Total_Return': sp500_total_return,
                                                'Period': 'Test Period'
                                            }
                                
                                # Select method based on user preference and mode
                                if "Auto-select" in portfolio_method:
                                    if is_production:
                                        # Production: select best by in-sample Sharpe (most recent data)
                                        best_sharpe = oos_results['oos_performance']['Sharpe_Ratio'].idxmax()
                                        selected_method = best_sharpe
                                        st.info(f"🎯 Auto-selected method for production: **{selected_method}** (Highest Sharpe on full data)")
                                    else:
                                        # Backtesting: use OOS composite score
                                        selected_method = oos_results['best_method']
                                        st.success(f"✅ Auto-selected method: **{selected_method}** (Best OOS composite score)")
                                else:
                                    selected_method = portfolio_method
                                    if selected_method not in oos_results['all_weights']:
                                        st.error(f"{selected_method} optimization failed. Using best method: {oos_results['best_method']}")
                                        selected_method = oos_results['best_method']
                                    else:
                                        st.info(f"💼 User selected method: **{selected_method}**")
                                
                                selected_weights = oos_results['all_weights'][selected_method]
                                
                                # Display results based on mode
                                if is_production:
                                    st.success("🎯 **Production Portfolio Optimized!**")
                                    st.markdown(f"""
                                    ### Portfolio Ready for Deployment
                                    - **Method**: {selected_method}
                                    - **Period**: {extended_start_date.strftime('%Y-%m-%d')} to {extended_end_date.strftime('%Y-%m-%d')}
                                    - **Data Used**: All {len(returns)} days in selected range
                                    - **Last Data Point**: {returns.index[-1].strftime('%Y-%m-%d')}
                                    - **Assets**: {len(selected_weights)} positions
                                    
                                    This portfolio is optimized using all available data in the selected period.
                                    Monte Carlo simulation shows expected future performance.
                                    """)
                                else:
                                    if selected_method == oos_results['best_method']:
                                        st.success(f"✅ Analysis completed! Using best method: **{selected_method}**")
                                    else:
                                        st.info(f"✅ Analysis completed! Using selected method: **{selected_method}** (Best was: {oos_results['best_method']})")
                                
                                # Performance Summary - ALL METHODS + BENCHMARK
                                _render_performance_summary_all(oos_results, selected_method, benchmark_metrics, is_production)
                                
                                # Portfolio Weights with CSV download
                                _render_portfolio_weights(selected_method, selected_weights, oos_results, returns, is_production)
                                
                                # Visualization: Cumulative Returns
                                _render_cumulative_returns(oos_results, selected_method, sp500_returns)
                                
                                # Stress Testing
                                _render_stress_testing(oos_results, selected_weights)
                                
                                # Returns Distribution with VaR
                                _render_returns_distribution(oos_results, selected_weights)
                                
                                # Monte Carlo Forecast (consistent with recommendations)
                                mc_results = _render_monte_carlo_forecast(
                                    oos_results['test_returns'], selected_weights, 
                                    mc_simulations, mc_period_years * 252, mc_distribution
                                )
                                
                                # Download Complete Results
                                _render_complete_results_download(
                                    oos_results, selected_method, selected_weights, 
                                    mc_results, benchmark_metrics, returns, is_production
                                )
                                
                                # Recommendations
                                _render_recommendations(selected_method, oos_results, selected_weights, mc_results, is_production)
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Try with fewer tickers or different date range.")


def _render_performance_summary_all(oos_results, selected_method, benchmark_metrics, is_production=False):
    """Display comprehensive performance summary with all methods and benchmark"""
    
    if is_production:
        st.subheader("🎯 Production Portfolio Metrics (Full Historical Data)")
        st.caption("These metrics show historical performance using all available data. Monte Carlo shows expected future.")
    else:
        st.subheader("📊 Performance Summary (Out-of-Sample) - All Methods")
        st.caption("These metrics show true out-of-sample performance to validate strategy.")
    
    # Get all results and sort by composite score (best to worst)
    perf_df = oos_results['oos_performance'].copy()
    ranking = oos_results['ranking']
    
    # Add composite score and rank
    perf_df['Composite_Score'] = ranking
    perf_df['Rank'] = range(1, len(perf_df) + 1)
    
    # Add benchmark if available
    if benchmark_metrics:
        period_label = f"S&P 500 ({benchmark_metrics.get('Period', 'Benchmark')})"
        benchmark_row = pd.DataFrame({
            'Total_Return': [benchmark_metrics['Total_Return']],
            'Annual_Return': [benchmark_metrics['Annual_Return']],
            'Annual_Vol': [benchmark_metrics['Annual_Vol']],
            'Sharpe_Ratio': [benchmark_metrics['Sharpe_Ratio']],
            'Sortino_Ratio': [np.nan],
            'Max_Drawdown': [benchmark_metrics['Max_Drawdown']],
            'Calmar_Ratio': [benchmark_metrics['Annual_Return'] / abs(benchmark_metrics['Max_Drawdown']) if benchmark_metrics['Max_Drawdown'] != 0 else 0],
            'Win_Rate': [np.nan],
            'Effective_N_Assets': [1.0],
            'Concentration': [1.0],
            'Composite_Score': [np.nan],
            'Rank': [np.nan]
        }, index=[period_label])
        
        perf_df = pd.concat([perf_df, benchmark_row])
    
    # Highlight selected method
    def highlight_selected(row):
        if row.name == selected_method:
            return ['background-color: #90EE90'] * len(row)
        elif row.name == 'S&P 500 Benchmark':
            return ['background-color: #FFE4B5'] * len(row)
        return [''] * len(row)
    
    # Display with formatting
    styled_df = perf_df.style.apply(highlight_selected, axis=1).format({
        'Total_Return': '{:.1%}',
        'Annual_Return': '{:.1%}', 
        'Annual_Vol': '{:.1%}',
        'Sharpe_Ratio': '{:.3f}',
        'Sortino_Ratio': '{:.3f}',
        'Max_Drawdown': '{:.1%}',
        'Calmar_Ratio': '{:.3f}',
        'Win_Rate': '{:.1%}',
        'Effective_N_Assets': '{:.1f}',
        'Concentration': '{:.3f}',
        'Composite_Score': '{:.3f}',
        'Rank': '{:.0f}'
    }, na_rep='-')
    
    st.dataframe(styled_df)
    
    st.caption(f"🟢 **Selected Method**: {selected_method} | 🟡 **Benchmark**: S&P 500 | Ranked by Composite Score (higher is better)")


def _render_portfolio_weights(selected_method, selected_weights, oos_results, returns, is_production=False):
    """Display portfolio weights with CSV download"""
    st.subheader(f"📋 {selected_method} Portfolio Weights")
    
    weights_df = pd.DataFrame({
        'Ticker': selected_weights.index,
        'Weight': selected_weights.values,
        'Weight %': selected_weights.values * 100
    })
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(weights_df.style.format({'Weight %': '{:.2f}%', 'Weight': '{:.4f}'}), use_container_width=True)
    
    with col2:
        # Prepare simple CSV for copy-paste (ticker,weight format)
        simple_csv = pd.DataFrame({
            'ticker': selected_weights.index,
            'weight': selected_weights.values
        })
        
        # Convert to CSV without index
        csv_string = simple_csv.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Weights (Simple)",
            data=csv_string,
            file_name=f"weights_{selected_method.replace(' ', '_')}_{returns.index[-1].strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Simple format: ticker,weight - Ready for copy-paste"
        )
        
        # Show preview
        with st.expander("📋 Copy-Paste Format Preview"):
            st.code(csv_string, language="csv")
            st.caption("Copy this format for Portfolio Backtest page")


def _render_complete_results_download(oos_results, selected_method, selected_weights, 
                                       mc_results, benchmark_metrics, returns, is_production):
    """Generate comprehensive CSV with all portfolio data"""
    st.subheader("📦 Complete Results Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV 1: All Methods Performance
        st.markdown("**All Methods Performance**")
        perf_df = oos_results['oos_performance'].copy()
        perf_df['Rank'] = range(1, len(perf_df) + 1)
        perf_df['Selected'] = perf_df.index == selected_method
        perf_df['Date'] = returns.index[-1].strftime('%Y-%m-%d')
        perf_df['Mode'] = 'Production' if is_production else 'Backtest'
        
        # Add benchmark
        if benchmark_metrics:
            period_label = f"S&P_500_{benchmark_metrics.get('Period', 'Benchmark').replace(' ', '_')}"
            benchmark_row = pd.DataFrame({
                'Total_Return': [benchmark_metrics['Total_Return']],
                'Annual_Return': [benchmark_metrics['Annual_Return']],
                'Annual_Vol': [benchmark_metrics['Annual_Vol']],
                'Sharpe_Ratio': [benchmark_metrics['Sharpe_Ratio']],
                'Sortino_Ratio': [np.nan],
                'Max_Drawdown': [benchmark_metrics['Max_Drawdown']],
                'Calmar_Ratio': [benchmark_metrics['Annual_Return'] / abs(benchmark_metrics['Max_Drawdown']) if benchmark_metrics['Max_Drawdown'] != 0 else 0],
                'Win_Rate': [np.nan],
                'Effective_N_Assets': [1.0],
                'Concentration': [1.0],
                'Rank': [np.nan],
                'Selected': [False],
                'Date': [returns.index[-1].strftime('%Y-%m-%d')],
                'Mode': ['Benchmark']
            }, index=[period_label])
            
            perf_df = pd.concat([perf_df, benchmark_row])
        
        # Reset index to include method names
        perf_df = perf_df.reset_index().rename(columns={'index': 'Method'})
        
        csv_all_methods = perf_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download All Methods Performance",
            data=csv_all_methods,
            file_name=f"all_methods_performance_{returns.index[-1].strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Performance metrics for all optimization methods + benchmark"
        )
    
    with col2:
        # CSV 2: Selected Portfolio Complete Data
        st.markdown("**Selected Portfolio - Complete Data**")
        
        # Get performance for selected method
        perf = oos_results['oos_performance'].loc[selected_method]
        stress_results = stress_test_analysis(oos_results['test_returns'], selected_weights)
        
        # Create detailed portfolio data
        portfolio_data = []
        for ticker, weight in selected_weights.items():
            portfolio_data.append({
                'Date': returns.index[-1].strftime('%Y-%m-%d'),
                'Mode': 'Production' if is_production else 'Backtest',
                'Method': selected_method,
                'Ticker': ticker,
                'Weight': weight,
                'Weight_Percent': weight * 100,
                # Performance metrics (same for all rows but useful for filtering)
                'Annual_Return': perf['Annual_Return'],
                'Annual_Volatility': perf['Annual_Vol'],
                'Sharpe_Ratio': perf['Sharpe_Ratio'],
                'Sortino_Ratio': perf['Sortino_Ratio'],
                'Max_Drawdown': perf['Max_Drawdown'],
                'Calmar_Ratio': perf['Calmar_Ratio'],
                'Win_Rate': perf['Win_Rate'],
                'Effective_N_Assets': perf['Effective_N_Assets'],
                # Risk metrics
                'VaR_95_Daily': stress_results['var_95'],
                'VaR_99_Daily': stress_results['var_99'],
                'CVaR_95_Daily': stress_results['cvar_95'],
                'CVaR_99_Daily': stress_results['cvar_99'],
                'Skewness': stress_results['skewness'],
                'Kurtosis': stress_results['kurtosis'],
                # Monte Carlo results
                'MC_Expected_Value': mc_results['mean_final'],
                'MC_Median_Value': mc_results['median_final'],
                'MC_Probability_Profit': mc_results['prob_profit'],
                'MC_VaR_95': mc_results['var_95'],
                'MC_Distribution': mc_results.get('distribution_params', {}).get('df', 'Normal') if 'distribution_params' in mc_results and 'df' in mc_results.get('distribution_params', {}) else 'Normal'
            })
        
        complete_df = pd.DataFrame(portfolio_data)
        csv_complete = complete_df.to_csv(index=False)
        
        st.download_button(
            label="📦 Download Complete Portfolio Data",
            data=csv_complete,
            file_name=f"portfolio_complete_{selected_method.replace(' ', '_')}_{returns.index[-1].strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Complete data: weights, performance, risk metrics, and Monte Carlo results"
        )


def _render_cumulative_returns(oos_results, best_method, sp500_returns):
    """Render cumulative returns comparison chart"""
    st.subheader("📈 Cumulative Returns Comparison")
    portfolio_series = oos_results['portfolio_series']
    
    fig_cum = go.Figure()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    
    # Add S&P 500 benchmark if available
    if sp500_returns is not None:
        common_dates = sp500_returns.index.intersection(portfolio_series[best_method].index)
        if len(common_dates) > 0:
            sp500_cum = (1 + sp500_returns.loc[common_dates]).cumprod()
            fig_cum.add_trace(go.Scatter(
                x=sp500_cum.index,
                y=(sp500_cum - 1) * 100,
                name='S&P 500',
                line=dict(color='lightgray', width=3, dash='dash')
            ))
    
    for i, (method_name, cum_returns) in enumerate(portfolio_series.items()):
        color = colors[i % len(colors)]
        width = 4 if method_name == best_method else 2
        fig_cum.add_trace(go.Scatter(
            x=cum_returns.index,
            y=(cum_returns - 1) * 100,
            name=method_name,
            line=dict(color=color, width=width)
        ))
    
    fig_cum.update_layout(
        title="Out-of-Sample Cumulative Returns (with S&P 500 Benchmark)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500
    )
    st.plotly_chart(fig_cum)


def _render_stress_testing(oos_results, best_weights):
    """Display stress testing results"""
    st.subheader("⚠️ Stress Testing Results")
    stress_results = stress_test_analysis(oos_results['test_returns'], best_weights)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VaR 95% (Daily)", f"{stress_results['var_95']:.2%}")
        st.metric("VaR 99% (Daily)", f"{stress_results['var_99']:.2%}")
    with col2:
        st.metric("CVaR 95% (Daily)", f"{stress_results['cvar_95']:.2%}")
        st.metric("CVaR 99% (Daily)", f"{stress_results['cvar_99']:.2%}")
    with col3:
        st.metric("Skewness", f"{stress_results['skewness']:.3f}")
        st.metric("Kurtosis", f"{stress_results['kurtosis']:.3f}")


def _render_returns_distribution(oos_results, selected_weights):
    """Display portfolio returns distribution"""
    st.subheader("📊 Portfolio Returns Distribution (Out-of-Sample)")
    
    test_portfolio_returns = (oos_results['test_returns'] * selected_weights).sum(axis=1)
    stress_results = stress_test_analysis(oos_results['test_returns'], selected_weights)
    
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Histogram(
        x=test_portfolio_returns * 100,
        nbinsx=50,
        name='Daily Returns',
        marker_color='#2E86AB',
        opacity=0.7
    ))
    
    # Add VaR lines
    var_95_pct = stress_results['var_95'] * 100
    var_99_pct = stress_results['var_99'] * 100
    
    fig_returns.add_vline(
        x=var_95_pct, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"VaR 95%: {var_95_pct:.2f}%",
        annotation_position="top left"
    )
    
    fig_returns.add_vline(
        x=var_99_pct, 
        line_dash="dash", 
        line_color="darkred",
        annotation_text=f"VaR 99%: {var_99_pct:.2f}%",
        annotation_position="top left"
    )
    
    fig_returns.update_layout(
        title="Portfolio Daily Returns Distribution (Out-of-Sample)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig_returns)


def _render_monte_carlo_forecast(returns, best_weights, mc_simulations, mc_period, mc_distribution):
    """Display Monte Carlo forecast and return results"""
    st.subheader(f"🎲 Monte Carlo Forecast ({mc_period // 252} Year{'s' if mc_period > 252 else ''})")
    mc_results = monte_carlo_forecast_streamlit(
        returns,  # Already test_returns from caller
        best_weights, 
        n_simulations=mc_simulations,
        n_days=mc_period,
        distribution=mc_distribution
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Value ($100k)", f"${mc_results['mean_final']:,.0f}")
    with col2:
        st.metric("Median Value ($100k)", f"${mc_results['median_final']:,.0f}")
    with col3:
        st.metric("Probability of Profit", f"{mc_results['prob_profit']:.1f}%")
    with col4:
        st.metric("VaR 95% Loss", f"${mc_results['var_95']:,.0f}")
    
    # Display distribution info if available
    if 'distribution_params' in mc_results and mc_results['distribution_params']:
        with st.expander("📊 Distribution Parameters"):
            params = mc_results['distribution_params']
            if 'df' in params:
                st.write(f"**T-Student Distribution:**")
                st.write(f"- Degrees of Freedom: {params['df']:.2f}")
                st.write(f"- Location: {params['loc']:.6f}")
                st.write(f"- Scale: {params['scale']:.6f}")
            else:
                st.write(f"**Normal Distribution:**")
                st.write(f"- Mean: {params.get('mean', 'N/A'):.6f}")
                st.write(f"- Std Dev: {params.get('std', 'N/A'):.6f}")
    
    # Monte Carlo Visualization
    fig_mc = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Final Value Distribution (with VaR)', 'Sample Paths', 'Probability of Positive Return Over Time'),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, None]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    # Histogram with VaR line
    fig_mc.add_trace(
        go.Histogram(
            x=mc_results['final_values'],
            nbinsx=50,
            name='Final Values',
            marker_color='#2E86AB',
            showlegend=False
        ),
        row=1, col=1
    )
    
    var_95_value = np.percentile(mc_results['final_values'], 5)
    fig_mc.add_vline(
        x=var_95_value, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"VaR 95%: ${var_95_value:,.0f}",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Sample paths
    n_paths_plot = min(50, mc_simulations)
    sample_indices = np.random.choice(mc_simulations, n_paths_plot, replace=False)
    
    for idx in sample_indices:
        fig_mc.add_trace(
            go.Scatter(
                x=list(range(mc_period)),
                y=mc_results['simulation_results'][idx],
                mode='lines',
                line=dict(width=0.5, color='lightblue'),
                opacity=0.3,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Mean path
    mean_path = mc_results['simulation_results'].mean(axis=0)
    fig_mc.add_trace(
        go.Scatter(
            x=list(range(mc_period)),
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='red'),
            name='Mean Path'
        ),
        row=1, col=2
    )
    
    # Probability of positive return over time
    prob_profit_over_time = (mc_results['simulation_results'] > 100000).mean(axis=0) * 100
    
    fig_mc.add_trace(
        go.Scatter(
            x=list(range(mc_period)),
            y=prob_profit_over_time,
            mode='lines',
            fill='tozeroy',
            line=dict(width=2, color='green'),
            name='Probability of Profit',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add reference line at 50%
    fig_mc.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="50% threshold",
        row=2, col=1
    )
    
    fig_mc.update_layout(
        height=800,
        title_text="<b>Monte Carlo Simulation Results</b><br><sup>Final distribution with VaR 95%, sample paths, and probability of positive return over time</sup>"
    )
    fig_mc.update_xaxes(title_text="Final Value ($)", row=1, col=1)
    fig_mc.update_xaxes(title_text="Days", row=1, col=2)
    fig_mc.update_yaxes(title_text="Portfolio Value ($)", row=1, col=2)
    fig_mc.update_xaxes(title_text="Days", row=2, col=1)
    fig_mc.update_yaxes(title_text="Probability of Profit (%)", row=2, col=1)
    
    st.plotly_chart(fig_mc)
    
    return mc_results  # Return for use in recommendations


def _render_recommendations(selected_method, oos_results, selected_weights, mc_results, is_production=False):
    """Display investment recommendations"""
    st.subheader("💡 Investment Recommendations")
    
    perf_df = oos_results['oos_performance'].loc[[selected_method]]
    stress_results = stress_test_analysis(oos_results['test_returns'], selected_weights)
    
    if is_production:
        st.markdown(f"""
        **🎯 Production Portfolio: {selected_method}**
        
        **Historical Performance (Full Data):**
        - Annual return: {perf_df['Annual_Return'].iloc[0]:.1%}
        - Sharpe ratio: {perf_df['Sharpe_Ratio'].iloc[0]:.3f}
        - Maximum drawdown: {perf_df['Max_Drawdown'].iloc[0]:.1%}
        - Diversification (effective assets): {perf_df['Effective_N_Assets'].iloc[0]:.1f}
        
        **Forward-Looking Risk (Monte Carlo):**
        - Expected return probability: {mc_results['prob_profit']:.1f}%
        - Estimated daily VaR (95%): {stress_results['var_95']:.2%}
        
        **Recommended Actions:**
        1. ✅ **Deploy capital** according to weights below
        2. 📅 **Rebalance** monthly or when drift >5%
        3. 🔄 **Re-optimize** quarterly with fresh data
        4. ⚠️ **Monitor drawdown** - review if exceeds {abs(perf_df['Max_Drawdown'].iloc[0])*1.5:.1%}
        5. 📊 **Update data** - re-run weekly for latest market conditions
        
        **Important Notes:**
        - These weights are optimized on historical data through {oos_results['test_returns'].index[-1].strftime('%Y-%m-%d')}
        - Past performance doesn't guarantee future results
        - Monitor risk metrics continuously
        - Consider transaction costs and tax implications
        """)
    else:
        st.markdown(f"""
        **🔬 Backtest Results: {selected_method}**
        
        **Out-of-Sample Performance:**
        - Annual return: {perf_df['Annual_Return'].iloc[0]:.1%}
        - Sharpe ratio: {perf_df['Sharpe_Ratio'].iloc[0]:.3f}
        - Maximum drawdown: {perf_df['Max_Drawdown'].iloc[0]:.1%}
        - Diversification (effective assets): {perf_df['Effective_N_Assets'].iloc[0]:.1f}
        
        **Risk Considerations:**
        - Daily VaR (95%): {stress_results['var_95']:.2%}
        - Monte Carlo probability of profit: {mc_results['prob_profit']:.1f}%
        
        **Next Steps:**
        1. Strategy validated on out-of-sample data ✅
        2. Switch to **Production Mode** to optimize for deployment
        3. Consider quarterly rebalancing
        4. Monitor drawdown limits
        5. Review strategy when market conditions change significantly
        
        **To Deploy:**
        Run analysis again in **Production Mode** to get weights optimized on ALL data.
        """)
