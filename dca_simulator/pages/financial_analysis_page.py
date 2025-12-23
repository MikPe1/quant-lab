"""
Financial Analysis Page
Comprehensive financial analysis including returns, distributions, technical indicators, and forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from arch import arch_model
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from data_fetcher import fetch_historical_data
from portfolio.monte_carlo import monte_carlo_forecast_streamlit


def render_financial_analysis_page():
    """Main function to render the Financial Analysis page"""

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL").upper()
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))
        forecast_days = st.number_input("Forecast Days", value=30, min_value=1, max_value=365)
    with col2:
        mc_simulations = st.number_input("Monte Carlo Simulations", value=1000, min_value=100, max_value=10000)
        mc_period = st.number_input("Monte Carlo Period (days)", value=30, min_value=1, max_value=365)
        mc_distribution = st.selectbox("Monte Carlo Distribution", ["normal", "t-student"], index=0)
        var_confidence = st.slider("VaR Confidence Level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

    if st.button("Run Analysis"):
        # Fetch data
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        data = fetch_historical_data(ticker, start_date_str, end_date_str)
        
        if data.empty:
            st.error(f"No data found for {ticker} from {start_date_str} to {end_date_str}")
        else:
            # Compute returns
            returns = data.pct_change().dropna()
            
            # Display all analysis sections
            _render_returns_analysis(ticker, data, returns)
            _render_price_distribution(ticker, data)
            _render_returns_distribution(ticker, returns)
            _render_unit_root_tests(data, returns)
            _render_var_analysis(returns, var_confidence)
            _render_monte_carlo(ticker, returns, mc_simulations, mc_period, mc_distribution)
            _render_technical_analysis(ticker, data)
            _render_forecasting(ticker, data, forecast_days)


def _render_returns_analysis(ticker, data, returns):
    """Display returns time series and descriptive statistics"""
    st.subheader("📈 Daily Returns Time Series")
    fig_returns = px.line(
        returns,
        title=f"{ticker} Daily Returns",
        labels={'value': 'Daily Return', 'index': 'Date'}
    )
    fig_returns.update_traces(line_color='#2E86AB')
    fig_returns.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig_returns)
    
    # Descriptive statistics
    st.subheader("📊 Descriptive Statistics")
    stats_data = {
        "Statistic": ["Mean Daily Return", "Median Daily Return", "Daily Volatility", "Q1 Daily Return", "Q3 Daily Return"],
        "Value": [f"{returns.mean():.2%}", f"{returns.median():.2%}", f"{returns.std():.2%}", 
                  f"{returns.quantile(0.25):.2%}", f"{returns.quantile(0.75):.2%}"]
    }
    st.table(pd.DataFrame(stats_data))


def _render_price_distribution(ticker, data):
    """Analyze and display historical price distribution"""
    st.subheader("📈 Historical Price Distribution")

    # Calculate price percentiles
    current_price = data.iloc[-1]
    price_percentile = stats.percentileofscore(data, current_price)

    # Create histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=data,
        nbinsx=50,
        name='Price Distribution',
        marker_color='#2E86AB',
        opacity=0.7
    ))

    # Add vertical line for current price
    fig_hist.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="top right"
    )

    fig_hist.update_layout(
        title=f"{ticker} Price Distribution (Historical)",
        xaxis_title="Price ($)",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig_hist)

    # Price distribution analysis
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
        st.metric("Percentile", f"{price_percentile:.1f}%")

    with col2:
        if price_percentile > 50:
            st.metric("Position", "↗️ ABOVE MEDIAN")
            position_desc = "Above the middle of distribution"
        else:
            st.metric("Position", "↘️ BELOW MEDIAN")
            position_desc = "Below the middle of distribution"

    # Detailed percentile table
    percentiles = [10, 25, 50, 75, 90]
    percentile_values = [data.quantile(p/100) for p in percentiles]

    percentile_data = {
        "Percentile": [f"{p}th" for p in percentiles],
        "Price Level": [f"${val:.2f}" for val in percentile_values],
        "Status": ["Below Current" if val < current_price else "Above Current" for val in percentile_values]
    }

    st.markdown("### Percentile Analysis")
    st.markdown(f"""
    **Interpretation:**
    Over the last {len(data)} trading days,
    price was **LOWER** than current price for **{price_percentile:.1f}%** of the time
    price was **HIGHER** than current price for **{100-price_percentile:.1f}%** of the time

    **{position_desc}**
    """)

    st.table(pd.DataFrame(percentile_data))


def _render_returns_distribution(ticker, returns):
    """Display returns distribution analysis with theoretical fits"""
    st.subheader("📊 Returns Distribution Analysis")

    # Fit normal distribution
    mu, sigma = stats.norm.fit(returns)

    # Fit t-distribution
    df, loc, scale = stats.t.fit(returns)

    # Create histogram with theoretical distributions
    fig_returns = go.Figure()

    # Histogram of actual returns
    fig_returns.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Actual Returns',
        marker_color='rgba(46, 134, 171, 0.7)',
        opacity=0.7,
        histnorm='probability density'
    ))

    # Normal distribution curve
    x_normal = np.linspace(returns.min(), returns.max(), 100)
    y_normal = stats.norm.pdf(x_normal, mu, sigma)
    fig_returns.add_trace(go.Scatter(
        x=x_normal,
        y=y_normal,
        mode='lines',
        name=f'Normal (μ={mu:.3f}, σ={sigma:.3f})',
        line=dict(color='red', width=2, dash='dash')
    ))

    # T-distribution curve
    x_t = np.linspace(returns.min(), returns.max(), 100)
    y_t = stats.t.pdf(x_t, df, loc, scale)
    fig_returns.add_trace(go.Scatter(
        x=x_t,
        y=y_t,
        mode='lines',
        name=f'T-Student (df={df:.1f})',
        line=dict(color='green', width=2, dash='dot')
    ))

    fig_returns.update_layout(
        title=f"{ticker} Daily Returns Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Density",
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    st.plotly_chart(fig_returns)

    # Returns distribution statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Return", f"{returns.mean():.3f}")
        st.metric("Skewness", f"{stats.skew(returns):.3f}")

    with col2:
        st.metric("Volatility", f"{returns.std():.3f}")
        st.metric("Kurtosis", f"{stats.kurtosis(returns):.3f}")

    with col3:
        st.metric("Normal Fit μ", f"{mu:.3f}")
        st.metric("T-Dist df", f"{df:.1f}")

    # Distribution comparison
    st.markdown("### Distribution Fit Analysis")
    st.markdown(f"""
    **Normal Distribution:** μ = {mu:.3f}, σ = {sigma:.3f}
    **T-Distribution:** degrees of freedom = {df:.1f}

    **Interpretation:**
    - **Skewness**: {stats.skew(returns):.3f} ({'Right-skewed' if stats.skew(returns) > 0 else 'Left-skewed' if stats.skew(returns) < 0 else 'Symmetric'})
    - **Kurtosis**: {stats.kurtosis(returns):.3f} ({'Heavy tails' if stats.kurtosis(returns) > 0 else 'Light tails'})
    - **T-distribution** typically fits financial returns better due to fat tails
    """)


def _render_unit_root_tests(data, returns):
    """Perform and display unit root tests for stationarity analysis"""
    st.subheader("🔍 Unit Root Tests (Stationarity Analysis)")

    # ADF on prices
    adf_price_c = adfuller(data, regression='c')
    adf_price_ct = adfuller(data, regression='ct')

    # ADF on returns
    adf_returns_c = adfuller(returns, regression='c')
    adf_returns_ct = adfuller(returns, regression='ct')

    # KPSS on prices
    kpss_price_c = kpss(data, regression='c')
    kpss_price_ct = kpss(data, regression='ct')

    # KPSS on returns
    kpss_returns_c = kpss(returns, regression='c')
    kpss_returns_ct = kpss(returns, regression='ct')

    tests = [
        ("ADF Prices (constant)", adf_price_c[0], adf_price_c[1], "Stationary" if adf_price_c[1] < 0.05 else "Non-stationary"),
        ("ADF Prices (constant + trend)", adf_price_ct[0], adf_price_ct[1], "Stationary" if adf_price_ct[1] < 0.05 else "Non-stationary"),
        ("ADF Returns (constant)", adf_returns_c[0], adf_returns_c[1], "Stationary" if adf_returns_c[1] < 0.05 else "Non-stationary"),
        ("ADF Returns (constant + trend)", adf_returns_ct[0], adf_returns_ct[1], "Stationary" if adf_returns_ct[1] < 0.05 else "Non-stationary"),
        ("KPSS Prices (constant)", kpss_price_c[0], kpss_price_c[1], "Non-stationary" if kpss_price_c[1] < 0.05 else "Stationary"),
        ("KPSS Prices (constant + trend)", kpss_price_ct[0], kpss_price_ct[1], "Non-stationary" if kpss_price_ct[1] < 0.05 else "Stationary"),
        ("KPSS Returns (constant)", kpss_returns_c[0], kpss_returns_c[1], "Non-stationary" if kpss_returns_c[1] < 0.05 else "Stationary"),
        ("KPSS Returns (constant + trend)", kpss_returns_ct[0], kpss_returns_ct[1], "Non-stationary" if kpss_returns_ct[1] < 0.05 else "Stationary"),
    ]

    test_df = pd.DataFrame(tests, columns=["Test", "Statistic", "p-value", "Conclusion"])
    test_df["p-value"] = test_df["p-value"].round(4)
    test_df["Statistic"] = test_df["Statistic"].round(4)

    # Style the dataframe
    def style_tests(val):
        if val == "Stationary":
            return 'background-color: #d4edda; color: #155724'
        elif val == "Non-stationary":
            return 'background-color: #f8d7da; color: #721c24'
        return ''

    st.dataframe(test_df.style.map(style_tests, subset=['Conclusion']))


def _render_var_analysis(returns, var_confidence):
    """Calculate and display Value at Risk using GARCH"""
    st.subheader("⚠️ Value at Risk (VaR) using GARCH")
    try:
        model = arch_model(returns, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        forecast_1 = res.forecast(horizon=1)
        z_score = stats.norm.ppf(1 - var_confidence)
        var_1 = -z_score * np.sqrt(forecast_1.variance.iloc[-1, 0])
        # Approximate for longer horizons using sqrt(h) scaling
        var_2 = var_1 * np.sqrt(2)
        var_5 = var_1 * np.sqrt(5)
        var_30 = var_1 * np.sqrt(30)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("1-day VaR", f"{var_1:.2%}")
        with col2:
            st.metric("2-day VaR", f"{var_2:.2%}")
        with col3:
            st.metric("5-day VaR", f"{var_5:.2%}")
        with col4:
            st.metric("30-day VaR", f"{var_30:.2%}")
    except Exception as e:
        st.error(f"GARCH fitting failed: {e}")


def _render_monte_carlo(ticker, returns, mc_simulations, mc_period, mc_distribution):
    """Run and display Monte Carlo simulation"""
    st.subheader("🎲 Monte Carlo Forecast")
    mc_results = monte_carlo_forecast_streamlit(
        returns.to_frame(),  # Convert to DataFrame for compatibility
        np.array([1.0]),     # Single stock, 100% weight
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

    # Visualizations
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
        title_text=f"<b>Monte Carlo Simulation Results ({mc_period} days)</b><br><sup>Final distribution with VaR 95%, sample paths, and probability of positive return over time</sup>"
    )
    fig_mc.update_xaxes(title_text="Final Value ($)", row=1, col=1)
    fig_mc.update_xaxes(title_text="Days", row=1, col=2)
    fig_mc.update_yaxes(title_text="Portfolio Value ($)", row=1, col=2)
    fig_mc.update_xaxes(title_text="Days", row=2, col=1)
    fig_mc.update_yaxes(title_text="Probability of Profit (%)", row=2, col=1)

    st.plotly_chart(fig_mc)


def _render_technical_analysis(ticker, data):
    """Display technical indicators: Bollinger Bands, MACD, RSI"""
    st.subheader("📈 Technical Analysis: Bollinger Bands, MACD & RSI")

    # Calculate Bollinger Bands (20-day)
    bb_window = 20
    sma_bb = data.rolling(window=bb_window).mean()
    std_bb = data.rolling(window=bb_window).std()
    upper_band = sma_bb + 2 * std_bb
    lower_band = sma_bb - 2 * std_bb

    # Calculate MACD (12, 26, 9)
    ema12 = data.ewm(span=12, adjust=False).mean()
    ema26 = data.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    # Calculate RSI (14-day)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Create subplots
    fig_tech = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f'{ticker} Price with Bollinger Bands (20 days)',
                       'MACD (12,26,9)',
                       'RSI (14-day)'),
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.1
    )

    # Panel 1: Price with Bollinger Bands
    fig_tech.add_trace(go.Scatter(
        x=data.index[-200:], y=data.values[-200:],
        mode='lines', name='Price', line=dict(color='#2E86AB', width=2)
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=sma_bb.index[-200:], y=sma_bb.values[-200:],
        mode='lines', name='SMA-20', line=dict(color='#A23B72', width=1, dash='dash')
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=upper_band.index[-200:], y=upper_band.values[-200:],
        mode='lines', name='Upper Band', line=dict(color='#F18F01', width=1),
        fill=None
    ), row=1, col=1)

    fig_tech.add_trace(go.Scatter(
        x=lower_band.index[-200:], y=lower_band.values[-200:],
        mode='lines', name='Lower Band', line=dict(color='#F18F01', width=1),
        fill='tonexty', fillcolor='rgba(241, 143, 1, 0.1)'
    ), row=1, col=1)

    # Panel 2: MACD
    fig_tech.add_trace(go.Scatter(
        x=macd.index[-200:], y=macd.values[-200:],
        mode='lines', name='MACD', line=dict(color='#2E86AB', width=1)
    ), row=2, col=1)

    fig_tech.add_trace(go.Scatter(
        x=signal.index[-200:], y=signal.values[-200:],
        mode='lines', name='Signal', line=dict(color='#A23B72', width=1)
    ), row=2, col=1)

    fig_tech.add_trace(go.Bar(
        x=histogram.index[-200:], y=histogram.values[-200:],
        name='Histogram', marker_color='rgba(161, 195, 185, 0.5)'
    ), row=2, col=1)

    # Panel 3: RSI
    fig_tech.add_trace(go.Scatter(
        x=rsi.index[-200:], y=rsi.values[-200:],
        mode='lines', name='RSI', line=dict(color='#2E86AB', width=2)
    ), row=3, col=1)

    # Add RSI levels
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig_tech.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    fig_tech.update_layout(height=800, showlegend=False)
    fig_tech.update_xaxes(title_text="Date", row=3, col=1)
    fig_tech.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig_tech.update_yaxes(title_text="MACD", row=2, col=1)
    fig_tech.update_yaxes(title_text="RSI", row=3, col=1)

    st.plotly_chart(fig_tech)

    # Technical Analysis Summary
    _render_technical_summary(data, sma_bb, upper_band, lower_band, macd, signal, histogram, rsi)


def _render_technical_summary(data, sma_bb, upper_band, lower_band, macd, signal, histogram, rsi):
    """Display summary of technical indicators"""
    st.subheader("📊 Technical Analysis Summary")

    # Bollinger Bands Analysis
    current_price = data.iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]

    bb_position = (current_price - current_lower) / (current_upper - current_lower) * 100

    # MACD Analysis
    current_macd = macd.iloc[-1]
    current_signal = signal.iloc[-1]
    current_hist = histogram.iloc[-1]

    # RSI Analysis
    current_rsi = rsi.iloc[-1]

    # Determine trend signals
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Bollinger Bands (20-day)**")
        st.metric("Position in Band", f"{bb_position:.1f}%")

        if bb_position > 80:
            st.error("🔴 NEAR UPPER BAND - Overbought")
            bb_signal = "Overbought"
        elif bb_position < 20:
            st.success("🟢 NEAR LOWER BAND - Oversold")
            bb_signal = "Oversold"
        else:
            st.info("✅ WITHIN BANDS - Normal")
            bb_signal = "Normal"

    with col2:
        st.markdown("**MACD (12,26,9)**")
        st.metric("MACD Value", f"{current_macd:.3f}")
        st.metric("Signal", f"{current_signal:.3f}")
        st.metric("Histogram", f"{current_hist:.3f}")

        if current_macd > current_signal and current_hist > 0:
            st.success("🟢 BULLISH - MACD above Signal")
            macd_signal = "Bullish"
        elif current_macd < current_signal and current_hist < 0:
            st.error("🔴 BEARISH - MACD below Signal")
            macd_signal = "Bearish"
        else:
            st.info("⚪ NEUTRAL - Lines crossing")
            macd_signal = "Neutral"

    with col3:
        st.markdown("**RSI (14-day)**")
        st.metric("RSI Value", f"{current_rsi:.1f}")

        if current_rsi > 70:
            st.error("🔴 OVERBOUGHT (>70)")
            rsi_signal = "Overbought"
        elif current_rsi < 30:
            st.success("🟢 OVERSOLD (<30)")
            rsi_signal = "Oversold"
        else:
            st.info("✅ NEUTRAL (30-70)")
            rsi_signal = "Neutral"

    # Overall Trend Analysis
    st.subheader("🎯 Overall Trend Analysis")

    bullish_signals = sum([bb_signal == "Oversold", macd_signal == "Bullish", rsi_signal == "Oversold"])
    bearish_signals = sum([bb_signal == "Overbought", macd_signal == "Bearish", rsi_signal == "Overbought"])

    if bullish_signals >= 2:
        st.success("🚀 **BULLISH TREND** - Multiple indicators suggest upward momentum")
        trend = "Bullish"
    elif bearish_signals >= 2:
        st.error("📉 **BEARISH TREND** - Multiple indicators suggest downward pressure")
        trend = "Bearish"
    else:
        st.info("⚖️ **NEUTRAL/MIXED** - Indicators are not aligned")
        trend = "Neutral"

    st.markdown(f"""
    **Current Trend: {trend}**

    **Summary:**
    - **Bollinger Bands**: {bb_signal} ({bb_position:.1f}% position)
    - **MACD**: {macd_signal} (MACD: {current_macd:.3f}, Signal: {current_signal:.3f})
    - **RSI**: {rsi_signal} ({current_rsi:.1f})

    **Interpretation:**
    - Position in Bollinger Bands shows where price is relative to recent volatility
    - MACD indicates momentum direction (above signal = bullish)
    - RSI shows overbought (>70) or oversold (<30) conditions
    """)


def _render_forecasting(ticker, data, forecast_days):
    """Display Prophet, ARIMA, and Holt-Winters forecasts"""
    # Prophet Model
    st.subheader("🔮 Prophet Forecast")
    df_prophet = pd.DataFrame({'ds': data.index, 'y': data.values})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(df_prophet)

    future = model_prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet = model_prophet.predict(future)

    # Prophet Forecast visualization
    fig_forecast = go.Figure()

    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name='Historical',
        line=dict(color='#2E86AB', width=2)
    ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_prophet['ds'],
        y=forecast_prophet['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#A23B72', width=2)
    ))

    # Confidence intervals
    fig_forecast.add_trace(go.Scatter(
        x=forecast_prophet['ds'],
        y=forecast_prophet['yhat_lower'],
        fill=None,
        mode='lines',
        line_color='rgba(162, 59, 114, 0.3)',
        name='Lower Bound',
        showlegend=False
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_prophet['ds'],
        y=forecast_prophet['yhat_upper'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(162, 59, 114, 0.3)',
        name='Upper Bound (95%)'
    ))

    fig_forecast.update_layout(
        title=f"{ticker} Price Forecast ({forecast_days} days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    st.plotly_chart(fig_forecast)

    # Prophet Components
    _render_prophet_components(forecast_prophet)

    # ARIMA Model
    st.subheader("📊 ARIMA Forecast")
    try:
        model_arima = ARIMA(data, order=(1, 1, 1))
        res_arima = model_arima.fit()
        forecast_arima = res_arima.forecast(steps=forecast_days)

        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Historical',
            line=dict(color='#2E86AB', width=2)
        ))

        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        fig_arima.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_arima,
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='#F18F01', width=2)
        ))

        fig_arima.update_layout(
            title=f"{ticker} ARIMA Forecast ({forecast_days} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_arima)
    except Exception as e:
        st.error(f"ARIMA fitting failed: {e}")

    # Holt-Winters Model
    st.subheader("🌊 Holt-Winters Forecast")
    try:
        model_hw = ExponentialSmoothing(data, seasonal='add', seasonal_periods=252)
        res_hw = model_hw.fit()
        forecast_hw = res_hw.forecast(forecast_days)

        fig_hw = go.Figure()
        fig_hw.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='Historical',
            line=dict(color='#2E86AB', width=2)
        ))

        future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        fig_hw.add_trace(go.Scatter(
            x=future_dates,
            y=forecast_hw,
            mode='lines',
            name='Holt-Winters Forecast',
            line=dict(color='#C73E1D', width=2)
        ))

        fig_hw.update_layout(
            title=f"{ticker} Holt-Winters Forecast ({forecast_days} days)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_hw)
    except Exception as e:
        st.error(f"Holt-Winters fitting failed: {e}")


def _render_prophet_components(forecast_prophet):
    """Display Prophet time series decomposition"""
    st.subheader("📈 Prophet Components (Trend & Seasonality)")

    # Extract components data
    components = forecast_prophet[['ds', 'trend', 'weekly', 'yearly']].copy()
    components = components.set_index('ds')

    # Create subplots
    fig_components = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend Component', 'Weekly Seasonality (Mon-Fri)', 'Monthly Seasonality'),
        vertical_spacing=0.12
    )

    # Trend component
    fig_components.add_trace(
        go.Scatter(
            x=components.index[-252:],  # Last year of data
            y=components['trend'].values[-252:],
            mode='lines',
            line=dict(color='#2E86AB', width=2),
            name='Trend'
        ),
        row=1, col=1
    )

    # Weekly seasonality (only weekdays Mon-Fri)
    weekly_pattern = components['weekly'].tail(7)
    weekdays_only = weekly_pattern[weekly_pattern.index.dayofweek < 5]

    fig_components.add_trace(
        go.Scatter(
            x=weekdays_only.index.day_name(),
            y=weekdays_only.values,
            mode='lines+markers',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            name='Weekly Pattern (Mon-Fri)'
        ),
        row=2, col=1
    )

    # Monthly seasonality
    monthly_pattern = components['yearly'].groupby(components.index.month).mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig_components.add_trace(
        go.Scatter(
            x=month_names,
            y=monthly_pattern.values,
            mode='lines+markers',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            name='Monthly Pattern'
        ),
        row=3, col=1
    )

    fig_components.update_layout(
        height=900,
        title_text="<b>Prophet Time Series Decomposition</b><br><sup>Trend, weekly (weekdays only) and monthly seasonal patterns</sup>",
        showlegend=False
    )

    fig_components.update_xaxes(title_text="Date", row=1, col=1)
    fig_components.update_yaxes(title_text="Trend Value", row=1, col=1)
    fig_components.update_xaxes(title_text="Day of Week", row=2, col=1)
    fig_components.update_yaxes(title_text="Seasonal Effect", row=2, col=1)
    fig_components.update_xaxes(title_text="Month", row=3, col=1)
    fig_components.update_yaxes(title_text="Seasonal Effect", row=3, col=1)

    st.plotly_chart(fig_components)
