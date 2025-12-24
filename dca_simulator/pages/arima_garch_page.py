"""
ARIMA-GARCH Monte Carlo Analysis Page
Advanced time series modeling with volatility forecasting and Monte Carlo simulations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import yfinance as yf

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch import arch_model
from scipy import stats

warnings.filterwarnings('ignore')


def fetch_data_with_today(ticker, start_date):
    """Fetch historical data including today's intraday data if available.
    
    Returns:
        data: DataFrame with OHLCV data
        returns: Pure logarithmic returns (natural log, not scaled)
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=pd.to_datetime("today").strftime('%Y-%m-%d'),
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Try to add today's data
    ticker_obj = yf.Ticker(ticker)
    today_data = ticker_obj.history(period="1d")
    
    if not today_data.empty:
        today_data.index = today_data.index.tz_localize(None)
        if today_data.index[0].date() not in data.index.date:
            data = pd.concat([data, today_data])
    
    # Validate data
    if data.empty:
        raise ValueError(f"Nie udało się pobrać danych dla {ticker}. Sprawdź symbol tickera.")
    
    if 'Close' not in data.columns:
        raise ValueError(f"Brak kolumny 'Close' w danych dla {ticker}.")
    
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    returns = data['log_returns'].dropna()
    
    if len(returns) == 0:
        raise ValueError(f"Brak wystarczających danych do obliczenia zwrotów dla {ticker}. Spróbuj wcześniejszej daty początkowej.")
    
    return data, returns


def run_stationarity_tests(returns, alpha=0.05):
    """Run ADF and KPSS tests for stationarity."""
    results = {}
    
    # ADF Test
    adf_specs = {'Bez trendu i stałej': 'n', 'Ze stałą': 'c', 'Ze stałą i trendem': 'ct'}
    adf_results = {}
    for spec_name, spec in adf_specs.items():
        adf_result = adfuller(returns, regression=spec, autolag='AIC')
        adf_results[spec_name] = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'stationary': adf_result[1] < alpha
        }
    results['adf'] = adf_results
    
    # KPSS Test
    kpss_specs = {'Stacjonarność wokół stałej': 'c', 'Stacjonarność wokół trendu': 'ct'}
    kpss_results = {}
    for spec_name, spec in kpss_specs.items():
        kpss_result = kpss(returns, regression=spec, nlags='auto')
        kpss_results[spec_name] = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'stationary': kpss_result[1] >= alpha
        }
    results['kpss'] = kpss_results
    
    return results


def calculate_technical_indicators(data, lag=1):
    """Calculate technical indicators with log transformation: MACD (all components), Bollinger Bands, RSI.
    
    Args:
        data: DataFrame with price data
        lag: Number of periods to lag indicators (default=1 to avoid look-ahead bias)
    
    Note: 
        - Lagging prevents data leakage - we use indicator values from t-1 to predict t.
        - Log transformation stabilizes variance and makes distributions more symmetric
        - Interpretation after log: changes are relative (multiplicative) rather than absolute
    """
    close = data['Close'].copy()
    
    # MACD - all components (line, signal, histogram)
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    # Log transform MACD components (shift to positive values first)
    # Add abs() + 1 to handle negative values, then apply sign
    macd_line_log = np.sign(macd_line) * np.log1p(np.abs(macd_line))
    macd_signal_log = np.sign(macd_signal) * np.log1p(np.abs(macd_signal))
    macd_hist_log = np.sign(macd_hist) * np.log1p(np.abs(macd_hist))
    
    # Bollinger Bands (20-day, 2 std)
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    
    # Bollinger position: -1 (at lower band) to +1 (at upper band)
    bb_range = bb_upper - bb_lower
    bb_position = (close - bb_lower) / bb_range
    bb_position = (bb_position - 0.5) * 2  # Scale to -1 to +1
    # BB position already bounded, log transform less useful but we apply log1p to compress extremes
    bb_position_log = np.sign(bb_position) * np.log1p(np.abs(bb_position))
    
    # RSI (14-day)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize RSI to -1 to +1 range, then log transform
    rsi_normalized = (rsi - 50) / 50
    rsi_log = np.sign(rsi_normalized) * np.log1p(np.abs(rsi_normalized))
    
    # Create DataFrame with indicators and LAG them to avoid look-ahead bias
    indicators = pd.DataFrame({
        'macd_line_log_L1': macd_line_log.shift(lag),
        'macd_signal_log_L1': macd_signal_log.shift(lag),
        'macd_hist_log_L1': macd_hist_log.shift(lag),
        'bb_position_log_L1': bb_position_log.shift(lag),
        'rsi_log_L1': rsi_log.shift(lag)
    }, index=data.index)
    
    return indicators.dropna()


def plot_acf_pacf(returns, nlags=40):
    """Create ACF and PACF plots using Plotly."""
    n = len(returns)
    alpha = 0.05
    z_critical = 1.96
    ci_value = z_critical / np.sqrt(n)
    
    # Calculate ACF and PACF
    acf_values = acf(returns, nlags=nlags, alpha=alpha)[0]
    pacf_values = pacf(returns, nlags=nlags, method='ywm', alpha=alpha)[0]
    
    lags = np.arange(1, nlags + 1)
    
    # ACF Plot
    fig_acf = go.Figure()
    
    # Confidence intervals
    fig_acf.add_hline(y=ci_value, line_dash="dash", line_color="red", line_width=2)
    fig_acf.add_hline(y=-ci_value, line_dash="dash", line_color="red", line_width=2)
    
    # ACF bars
    colors_acf = ['red' if abs(val) > ci_value else 'blue' for val in acf_values[1:]]
    for i, (lag, val) in enumerate(zip(lags, acf_values[1:])):
        fig_acf.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode='lines', line=dict(color=colors_acf[i], width=8),
            showlegend=False, hovertemplate=f'Lag {lag}: {val:.4f}<extra></extra>'
        ))
    
    fig_acf.update_layout(
        title=dict(text=f"Funkcja Autokorelacji (ACF) - n={n}", font=dict(size=18)),
        xaxis_title="Opóźnienie (Lag)", yaxis_title="ACF",
        template="plotly_dark", height=400,
        yaxis=dict(range=[-0.3, 0.3], zeroline=True),
        font=dict(size=14),
        hoverlabel=dict(font_size=14)
    )
    
    # PACF Plot
    fig_pacf = go.Figure()
    
    # Confidence intervals
    fig_pacf.add_hline(y=ci_value, line_dash="dash", line_color="red", line_width=2)
    fig_pacf.add_hline(y=-ci_value, line_dash="dash", line_color="red", line_width=2)
    
    # PACF bars
    colors_pacf = ['red' if abs(val) > ci_value else 'blue' for val in pacf_values[1:]]
    for i, (lag, val) in enumerate(zip(lags, pacf_values[1:])):
        fig_pacf.add_trace(go.Scatter(
            x=[lag, lag], y=[0, val],
            mode='lines', line=dict(color=colors_pacf[i], width=8),
            showlegend=False, hovertemplate=f'Lag {lag}: {val:.4f}<extra></extra>'
        ))
    
    fig_pacf.update_layout(
        title=dict(text=f"Częściowa Funkcja Autokorelacji (PACF) - n={n}", font=dict(size=18)),
        xaxis_title="Opóźnienie (Lag)", yaxis_title="PACF",
        template="plotly_dark", height=400,
        yaxis=dict(range=[-0.3, 0.3], zeroline=True),
        font=dict(size=14),
        hoverlabel=dict(font_size=14)
    )
    
    # Identify significant lags
    significant_acf = [(i, acf_values[i]) for i in range(1, len(acf_values)) if abs(acf_values[i]) > ci_value]
    significant_pacf = [(i, pacf_values[i]) for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > ci_value]
    
    return fig_acf, fig_pacf, significant_acf, significant_pacf, ci_value


def build_sparse_arma_model(returns, ar_lags, ma_lags, tech_indicators=None):
    """Build sparse ARMA model with selected lags and optional technical indicators."""
    # Stage 1: Build AR model to get residuals
    df_ar = pd.DataFrame({'returns': returns})
    for lag in ar_lags:
        df_ar[f'ar_L{lag}'] = returns.shift(lag)
    
    # Add technical indicators if provided
    if tech_indicators is not None:
        # Align indicators with returns
        for col in tech_indicators.columns:
            df_ar[col] = tech_indicators[col]
    
    df_ar = df_ar.dropna()
    
    y_ar = df_ar['returns']
    X_ar = df_ar.drop('returns', axis=1)
    X_ar = sm.add_constant(X_ar)
    
    model_ar_only = sm.tsa.ARIMA(y_ar, exog=X_ar, order=(0, 0, 0), trend='n').fit()
    residuals = model_ar_only.resid
    
    # Stage 2: Build final ARMA model with MA lags
    df_ma_regressors = pd.DataFrame(index=residuals.index)
    for lag in ma_lags:
        df_ma_regressors[f'ma_L{lag}'] = residuals.shift(lag)
    
    X_final = X_ar.join(df_ma_regressors)
    final_df = y_ar.to_frame('returns').join(X_final)
    final_df = final_df.dropna()
    
    y_final = final_df['returns']
    X_final = final_df.drop('returns', axis=1)
    
    model_sparse_arma = sm.tsa.ARIMA(y_final, exog=X_final, order=(0, 0, 0), trend='n').fit()
    
    return model_sparse_arma, X_final, residuals, returns


def fit_garch_model(residuals, p=1, q=1, dist='t'):
    """Fit GARCH model on ARMA residuals."""
    residuals_clean = residuals.dropna()
    garch = arch_model(residuals_clean, vol='Garch', p=p, q=q, dist=dist)
    garch_res = garch.fit(disp="off")
    return garch_res


def monte_carlo_multistep(model_sparse_arma, garch_res, returns, residuals, X_final, 
                          ar_lags, ma_lags, forecast_horizon, n_simulations, distribution='t', 
                          tech_indicators=None):
    """
    Run multi-step Monte Carlo simulation with normal or t-student distributions.
    
    Args:
        distribution: 't' for t-Student or 'normal' for Gaussian
        
    Note: 
        - Technical indicators are NOT used in forecasts to avoid data leakage.
        - For t-distribution: standard_t(df) has variance df/(df-2), so we normalize
          it to variance=1 by multiplying by sqrt((df-2)/df), then scale by sigma_t
        - This ensures epsilon_t has the correct variance structure from GARCH
    """
    np.random.seed(None)
    
    # Prepare GARCH forecast
    garch_forecast = garch_res.forecast(horizon=forecast_horizon, reindex=False)
    sigma_array = np.sqrt(garch_forecast.variance.values[-1])
    
    if distribution == 't':
        df_t = garch_res.params['nu']
        # Standard t-distribution has variance df/(df-2) for df > 2
        # We need to normalize it to variance=1, then scale by sigma_t
        # Normalization factor: sqrt((df-2)/df)
        norm_factor = np.sqrt((df_t - 2) / df_t) if df_t > 2 else 1.0
    else:
        df_t = None
        norm_factor = 1.0
    
    # Technical indicators are NOT used in Monte Carlo forecasts
    # (we don't know their future values - this would be data leakage)
    
    # Monte Carlo simulation
    all_paths = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_simulations):
        returns_extended = returns.copy()
        residuals_extended = residuals.copy()
        single_path = []
        
        for step in range(forecast_horizon):
            # Prepare exogenous variables (only AR and MA lags, no technical indicators)
            future_exog = {'const': 1.0}
            for lag in ar_lags:
                future_exog[f'ar_L{lag}'] = returns_extended.iloc[-lag]
            for lag in ma_lags:
                future_exog[f'ma_L{lag}'] = residuals_extended.iloc[-lag]
            
            # Technical indicators are set to 0 (neutral) or their column average from training
            # since we cannot know their future values
            future_exog_df = pd.DataFrame([future_exog]).reindex(columns=X_final.columns, fill_value=0.0)
            
            # Forecast mean
            mu = model_sparse_arma.forecast(steps=1, exog=future_exog_df).iloc[0]
            sigma_t = sigma_array[step]
            
            # Generate shock
            if distribution == 't':
                # Generate from standard t-distribution and normalize to variance=1
                raw_shock = np.random.standard_t(df_t)
                normalized_shock = raw_shock * norm_factor
                shock = normalized_shock * sigma_t
            else:
                shock = np.random.normal(0, sigma_t)
            
            simulated_return = mu + shock
            single_path.append(simulated_return)
            
            # Update series
            returns_extended = pd.concat([returns_extended, pd.Series([simulated_return])], ignore_index=True)
            residuals_extended = pd.concat([residuals_extended, pd.Series([shock])], ignore_index=True)
        
        all_paths.append(single_path)
        
        # Update progress
        if (i + 1) % max(1, n_simulations // 100) == 0:
            progress = (i + 1) / n_simulations
            progress_bar.progress(progress)
            status_text.text(f"Obliczanie ścieżek Monte Carlo: {i+1}/{n_simulations} ({progress*100:.0f}%)")
    
    progress_bar.empty()
    status_text.empty()
    
    # Process results
    last_date = returns.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
    paths_df = pd.DataFrame(np.array(all_paths).T, index=forecast_index)
    cumulative_paths_df = paths_df.cumsum()
    
    return paths_df, cumulative_paths_df


def create_fan_chart(paths_df, cumulative_paths_df, stock_name, n_simulations, dist_type):
    """Create interactive fan chart with toggle between daily and cumulative returns."""
    # Calculate quantiles for daily returns
    q05_daily = paths_df.quantile(0.05, axis=1)
    q25_daily = paths_df.quantile(0.25, axis=1)
    q50_daily = paths_df.quantile(0.50, axis=1)
    q75_daily = paths_df.quantile(0.75, axis=1)
    q95_daily = paths_df.quantile(0.95, axis=1)
    
    # Calculate quantiles for cumulative returns
    q05_cum = cumulative_paths_df.quantile(0.05, axis=1)
    q25_cum = cumulative_paths_df.quantile(0.25, axis=1)
    q50_cum = cumulative_paths_df.quantile(0.50, axis=1)
    q75_cum = cumulative_paths_df.quantile(0.75, axis=1)
    q95_cum = cumulative_paths_df.quantile(0.95, axis=1)
    
    # Create figure
    fig = go.Figure()
    
    # Daily returns traces
    fig.add_trace(go.Scatter(
        x=np.concatenate([paths_df.index, paths_df.index[::-1]]),
        y=np.concatenate([q95_daily, q05_daily[::-1]]),
        fill='toself', fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Przedział Ufności',
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([paths_df.index, paths_df.index[::-1]]),
        y=np.concatenate([q75_daily, q25_daily[::-1]]),
        fill='toself', fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% Przedział Ufności',
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=paths_df.index, y=q50_daily,
        line=dict(color='rgb(0,176,246)', width=3),
        name='Mediana',
        visible=True
    ))
    
    # Cumulative returns traces
    fig.add_trace(go.Scatter(
        x=np.concatenate([cumulative_paths_df.index, cumulative_paths_df.index[::-1]]),
        y=np.concatenate([q95_cum, q05_cum[::-1]]),
        fill='toself', fillcolor='rgba(255,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Przedział Ufności (Skum.)',
        visible=False
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([cumulative_paths_df.index, cumulative_paths_df.index[::-1]]),
        y=np.concatenate([q75_cum, q25_cum[::-1]]),
        fill='toself', fillcolor='rgba(255,150,50,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='50% Przedział Ufności (Skum.)',
        visible=False
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_paths_df.index, y=q50_cum,
        line=dict(color='rgb(255, 100, 80)', width=3),
        name='Mediana (Skum.)',
        visible=False
    ))
    
    # Update layout with buttons
    fig.update_layout(
        title=dict(text=f'Prognoza Monte Carlo dla {stock_name} ({n_simulations} ścieżek, rozkład: {dist_type})', font=dict(size=18)),
        xaxis_title='Data',
        yaxis_title='Prognozowany zwrot logarytmiczny (dzienny)',
        yaxis_tickformat='.4f',
        template='plotly_dark',
        font=dict(size=14),
        hoverlabel=dict(font_size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57, y=1.15,
                xanchor="left", yanchor="top",
                buttons=[
                    dict(label="Zwroty Dzienne",
                         method="update",
                         args=[{"visible": [True, True, True, False, False, False]},
                               {"title": f"Prognoza Monte Carlo dla {stock_name} - Zwroty dzienne",
                                "yaxis.title": "Prognozowany zwrot logarytmiczny (dzienny)"}]),
                    dict(label="Zwroty Skumulowane",
                         method="update",
                         args=[{"visible": [False, False, False, True, True, True]},
                               {"title": f"Prognoza Monte Carlo dla {stock_name} - Zwroty skumulowane",
                                "yaxis.title": "Prognozowany zwrot logarytmiczny (skumulowany)"}])
                ]
            )
        ]
    )
    
    return fig


def plot_residuals_diagnostics(residuals, model_name="ARIMA"):
    """
    Comprehensive residuals diagnostics for ARIMA model.
    Shows: time series, histogram with normal curve, Q-Q plot, ACF of residuals and squared residuals.
    """
    from scipy.stats import probplot
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'Reszty {model_name} w czasie',
            f'Histogram reszt + rozkład normalny',
            f'Q-Q Plot (normalność)',
            f'ACF Reszt',
            f'Reszty kwadratowe (efekty ARCH)',
            f'ACF Reszt Kwadratowych'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.12
    )
    
    residuals_clean = residuals.dropna()
    n = len(residuals_clean)
    
    # 1. Residuals over time
    fig.add_trace(
        go.Scatter(x=residuals_clean.index, y=residuals_clean.values,
                   mode='lines', line=dict(color='cyan', width=1),
                   name='Reszty'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # 2. Histogram with normal curve
    fig.add_trace(
        go.Histogram(x=residuals_clean.values, nbinsx=50,
                     name='Histogram', marker_color='lightblue',
                     histnorm='probability density'),
        row=1, col=2
    )
    
    # Add normal distribution curve
    x_range = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    normal_curve = stats.norm.pdf(x_range, residuals_clean.mean(), residuals_clean.std())
    fig.add_trace(
        go.Scatter(x=x_range, y=normal_curve,
                   mode='lines', line=dict(color='red', width=2),
                   name='Rozkład normalny'),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    qq = probplot(residuals_clean, dist="norm")
    fig.add_trace(
        go.Scatter(x=qq[0][0], y=qq[0][1],
                   mode='markers', marker=dict(color='cyan', size=4),
                   name='Q-Q'),
        row=2, col=1
    )
    # Add reference line
    qq_min, qq_max = qq[0][0].min(), qq[0][0].max()
    fig.add_trace(
        go.Scatter(x=[qq_min, qq_max], y=[qq_min * qq[1][0] + qq[1][1], qq_max * qq[1][0] + qq[1][1]],
                   mode='lines', line=dict(color='red', dash='dash', width=2),
                   name='Linia referencyjna'),
        row=2, col=1
    )
    
    # 4. ACF of residuals
    nlags = min(40, len(residuals_clean) // 4)
    acf_values = acf(residuals_clean, nlags=nlags, alpha=0.05)[0]
    ci_value = 1.96 / np.sqrt(n)
    
    lags = np.arange(1, nlags + 1)
    colors_acf = ['red' if abs(val) > ci_value else 'lightblue' for val in acf_values[1:]]
    
    for i, (lag, val) in enumerate(zip(lags, acf_values[1:])):
        fig.add_trace(
            go.Scatter(x=[lag, lag], y=[0, val],
                       mode='lines', line=dict(color=colors_acf[i], width=6),
                       showlegend=False),
            row=2, col=2
        )
    
    fig.add_hline(y=ci_value, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-ci_value, line_dash="dash", line_color="red", row=2, col=2)
    
    # 5. Squared residuals (ARCH effects)
    squared_residuals = residuals_clean ** 2
    fig.add_trace(
        go.Scatter(x=squared_residuals.index, y=squared_residuals.values,
                   mode='lines', line=dict(color='orange', width=1),
                   name='Reszty²'),
        row=3, col=1
    )
    
    # 6. ACF of squared residuals
    acf_squared = acf(squared_residuals, nlags=nlags, alpha=0.05)[0]
    colors_acf_sq = ['red' if abs(val) > ci_value else 'orange' for val in acf_squared[1:]]
    
    for i, (lag, val) in enumerate(zip(lags, acf_squared[1:])):
        fig.add_trace(
            go.Scatter(x=[lag, lag], y=[0, val],
                       mode='lines', line=dict(color=colors_acf_sq[i], width=6),
                       showlegend=False),
            row=3, col=2
        )
    
    fig.add_hline(y=ci_value, line_dash="dash", line_color="red", row=3, col=2)
    fig.add_hline(y=-ci_value, line_dash="dash", line_color="red", row=3, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Data", row=1, col=1)
    fig.update_xaxes(title_text="Wartość", row=1, col=2)
    fig.update_xaxes(title_text="Kwantyle teoretyczne", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    fig.update_xaxes(title_text="Data", row=3, col=1)
    fig.update_xaxes(title_text="Lag", row=3, col=2)
    
    fig.update_yaxes(title_text="Reszty", row=1, col=1)
    fig.update_yaxes(title_text="Gęstość", row=1, col=2)
    fig.update_yaxes(title_text="Kwantyle próbki", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=2)
    fig.update_yaxes(title_text="Reszty²", row=3, col=1)
    fig.update_yaxes(title_text="ACF", row=3, col=2)
    
    fig.update_layout(
        height=1000,
        template='plotly_dark',
        showlegend=True,
        title_text=f"Diagnostyka Reszt Modelu {model_name}",
        title_x=0.5,
        title_font_size=18,
        font=dict(size=13),
        hoverlabel=dict(font_size=13)
    )
    
    return fig


def plot_garch_standardized_residuals(garch_res, model_name="GARCH"):
    """
    Diagnostics for standardized residuals from GARCH model.
    Standardized residuals should be i.i.d. with mean 0 and variance 1.
    """
    from scipy.stats import probplot
    
    # Get standardized residuals
    std_resid = garch_res.std_resid
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Standaryzowane reszty {model_name}',
            f'Histogram + rozkład t-Student',
            f'Q-Q Plot',
            f'ACF Standaryzowanych Reszt'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    n = len(std_resid)
    
    # 1. Standardized residuals over time
    fig.add_trace(
        go.Scatter(x=np.arange(len(std_resid)), y=std_resid,
                   mode='lines', line=dict(color='lightgreen', width=1),
                   name='Std. reszty'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=2, line_dash="dot", line_color="orange", row=1, col=1)
    fig.add_hline(y=-2, line_dash="dot", line_color="orange", row=1, col=1)
    
    # 2. Histogram with t-distribution
    fig.add_trace(
        go.Histogram(x=std_resid, nbinsx=50,
                     name='Histogram', marker_color='lightgreen',
                     histnorm='probability density'),
        row=1, col=2
    )
    
    # Add fitted distribution curve (t-student if available)
    x_range = np.linspace(std_resid.min(), std_resid.max(), 100)
    if 'nu' in garch_res.params:
        df_t = garch_res.params['nu']
        fitted_curve = stats.t.pdf(x_range, df_t, loc=std_resid.mean(), scale=std_resid.std())
        curve_name = f't-Student (df={df_t:.2f})'
    else:
        fitted_curve = stats.norm.pdf(x_range, std_resid.mean(), std_resid.std())
        curve_name = 'Normalny'
    
    fig.add_trace(
        go.Scatter(x=x_range, y=fitted_curve,
                   mode='lines', line=dict(color='red', width=2),
                   name=curve_name),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    if 'nu' in garch_res.params:
        qq = probplot(std_resid, dist=stats.t, sparams=(garch_res.params['nu'],))
    else:
        qq = probplot(std_resid, dist="norm")
    
    fig.add_trace(
        go.Scatter(x=qq[0][0], y=qq[0][1],
                   mode='markers', marker=dict(color='lightgreen', size=4),
                   name='Q-Q'),
        row=2, col=1
    )
    
    qq_min, qq_max = qq[0][0].min(), qq[0][0].max()
    fig.add_trace(
        go.Scatter(x=[qq_min, qq_max], y=[qq_min * qq[1][0] + qq[1][1], qq_max * qq[1][0] + qq[1][1]],
                   mode='lines', line=dict(color='red', dash='dash', width=2),
                   name='Linia referencyjna'),
        row=2, col=1
    )
    
    # 4. ACF of standardized residuals
    nlags = min(40, len(std_resid) // 4)
    acf_values = acf(std_resid, nlags=nlags, alpha=0.05)[0]
    ci_value = 1.96 / np.sqrt(n)
    
    lags = np.arange(1, nlags + 1)
    colors_acf = ['red' if abs(val) > ci_value else 'lightgreen' for val in acf_values[1:]]
    
    for i, (lag, val) in enumerate(zip(lags, acf_values[1:])):
        fig.add_trace(
            go.Scatter(x=[lag, lag], y=[0, val],
                       mode='lines', line=dict(color=colors_acf[i], width=6),
                       showlegend=False),
            row=2, col=2
        )
    
    fig.add_hline(y=ci_value, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-ci_value, line_dash="dash", line_color="red", row=2, col=2)
    
    # Update layout
    fig.update_xaxes(title_text="Obserwacja", row=1, col=1)
    fig.update_xaxes(title_text="Wartość", row=1, col=2)
    fig.update_xaxes(title_text="Kwantyle teoretyczne", row=2, col=1)
    fig.update_xaxes(title_text="Lag", row=2, col=2)
    
    fig.update_yaxes(title_text="Std. reszty", row=1, col=1)
    fig.update_yaxes(title_text="Gęstość", row=1, col=2)
    fig.update_yaxes(title_text="Kwantyle próbki", row=2, col=1)
    fig.update_yaxes(title_text="ACF", row=2, col=2)
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=True,
        title_text=f"Diagnostyka Standaryzowanych Reszt Modelu {model_name}",
        title_x=0.5,
        title_font_size=18,
        font=dict(size=13),
        hoverlabel=dict(font_size=13)
    )
    
    # Calculate statistics
    stats_dict = {
        'mean': std_resid.mean(),
        'std': std_resid.std(),
        'skewness': stats.skew(std_resid),
        'kurtosis': stats.kurtosis(std_resid, fisher=True),  # Excess kurtosis
        'jb_stat': stats.jarque_bera(std_resid)[0],
        'jb_pvalue': stats.jarque_bera(std_resid)[1]
    }
    
    return fig, stats_dict


def create_histogram(data, median_val, var_95, var_99, title, xlabel):
    """Create histogram with VaR lines (values as pure log returns)."""
    fig = px.histogram(
        data, nbins=max(50, len(data) // 20), marginal="box",
        title=title
    )
    
    fig.add_vline(
        x=median_val, line_width=2, line_dash="dash", line_color="cyan",
        annotation_text=f"Mediana: {median_val:.4f}",
        annotation_position="top left",
        annotation_font_size=13
    )
    fig.add_vline(
        x=var_95, line_width=2, line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.4f}",
        annotation_position="bottom left",
        annotation_font_size=13
    )
    fig.add_vline(
        x=var_99, line_width=2, line_color="red",
        annotation_text=f"VaR 99%: {var_99:.4f}",
        annotation_position="bottom right",
        annotation_font_size=13
    )
    
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title='Częstość (liczba symulacji)',
        xaxis_tickformat='.4f',
        template='plotly_dark',
        showlegend=False,
        title_font_size=16,
        font=dict(size=13),
        hoverlabel=dict(font_size=13)
    )
    
    return fig


def create_model_flow_diagram():
    """Create a visual diagram showing data flow between ARIMA, GARCH, and Monte Carlo."""
    fig = go.Figure()
    
    # Define boxes
    boxes = [
        {"name": "Dane historyczne\n(zwroty)", "x": 0.5, "y": 4, "color": "lightblue"},
        {"name": "Model ARIMA\n(średnia)", "x": 0.5, "y": 3, "color": "cyan"},
        {"name": "Reszty ARIMA\n(ε_t)", "x": 0.5, "y": 2, "color": "orange"},
        {"name": "Model GARCH\n(zmienność)", "x": 0.5, "y": 1, "color": "lightgreen"},
        {"name": "Prognozy\nμ_t (ARIMA)", "x": 0.2, "y": 0, "color": "cyan"},
        {"name": "Prognozy\nσ_t² (GARCH)", "x": 0.8, "y": 0, "color": "lightgreen"},
        {"name": "Monte Carlo\nr_t = μ_t + ε_t·σ_t", "x": 0.5, "y": -1, "color": "gold"},
    ]
    
    # Add boxes
    for box in boxes:
        fig.add_shape(
            type="rect",
            x0=box["x"]-0.15, y0=box["y"]-0.3,
            x1=box["x"]+0.15, y1=box["y"]+0.3,
            line=dict(color="white", width=2),
            fillcolor=box["color"]
        )
        fig.add_annotation(
            x=box["x"], y=box["y"],
            text=box["name"],
            showarrow=False,
            font=dict(size=11, color="black", family="Arial Black"),
            align="center"
        )
    
    # Add arrows
    arrows = [
        {"x0": 0.5, "y0": 3.7, "x1": 0.5, "y1": 3.3, "label": "zwroty"},
        {"x0": 0.5, "y0": 2.7, "x1": 0.5, "y1": 2.3, "label": "reszty"},
        {"x0": 0.5, "y0": 1.7, "x1": 0.5, "y1": 1.3, "label": "warunkowa\nwariancja"},
        {"x0": 0.35, "y0": 0.7, "x1": 0.28, "y1": 0.3, "label": ""},
        {"x0": 0.65, "y0": 0.7, "x1": 0.72, "y1": 0.3, "label": ""},
        {"x0": 0.2, "y0": -0.3, "x1": 0.35, "y1": -0.7, "label": ""},
        {"x0": 0.8, "y0": -0.3, "x1": 0.65, "y1": -0.7, "label": ""},
    ]
    
    for arrow in arrows:
        fig.add_annotation(
            x=arrow["x1"], y=arrow["y1"],
            ax=arrow["x0"], ay=arrow["y0"],
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="white"
        )
        if arrow["label"]:
            mid_x = (arrow["x0"] + arrow["x1"]) / 2
            mid_y = (arrow["y0"] + arrow["y1"]) / 2
            fig.add_annotation(
                x=mid_x + 0.15, y=mid_y,
                text=arrow["label"],
                showarrow=False,
                font=dict(size=9, color="lightgray"),
                align="left"
            )
    
    fig.update_layout(
        xaxis=dict(range=[-0.1, 1.1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-1.5, 4.5], showgrid=False, zeroline=False, visible=False),
        template="plotly_dark",
        height=600,
        title="Przepływ danych między modelami",
        title_x=0.5,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def render_arima_garch_page():
    """Main function to render ARIMA-GARCH Monte Carlo Analysis page."""
    
    st.header("📈 ARIMA-GARCH + Monte Carlo")
    st.markdown("""
    Zaawansowana analiza szeregów czasowych łącząca modelowanie średniej (ARIMA), 
    zmienności (GARCH) i symulacje Monte Carlo dla wieloetapowych prognoz.
    """)
    
    # Sidebar configuration
    st.sidebar.subheader("⚙️ Konfiguracja Analizy")
    
    ticker = st.sidebar.text_input("Ticker akcji", value="AAPL", help="Wprowadź symbol giełdowy")
    start_date = st.sidebar.date_input("Data początkowa", value=pd.to_datetime("2023-01-01"))
    
    with st.sidebar.expander("🔧 Parametry Modelu"):
        model_type = st.radio(
            "Typ modelu ARIMA",
            ["Auto (ACF/PACF)", "Prosty ARIMA(2,2)"],
            index=0,
            help="Auto: automatyczna identyfikacja lagów na podstawie ACF/PACF. Prosty: stały model ARIMA(2,2)"
        )
        use_tech_indicators = st.checkbox(
            "Użyj wskaźników technicznych (log-transform, tylko in-sample)",
            value=False,
            help="Dodaj opóźnione (L1) i zlogarytmowane wskaźniki: MACD (line+signal+hist), Bollinger, RSI. UWAGA: Używane tylko do dopasowania modelu, NIE w prognozach Monte Carlo (unikamy data leakage). Log transform stabilizuje wariancję."
        )
        nlags_acf = st.number_input("Liczba lagów ACF/PACF", min_value=10, max_value=60, value=40)
        garch_p = st.number_input("GARCH p", min_value=1, max_value=5, value=1)
        garch_q = st.number_input("GARCH q", min_value=1, max_value=5, value=1)
    
    with st.sidebar.expander("🎲 Parametry Monte Carlo"):
        forecast_horizon = st.number_input("Horyzont prognozy (dni)", min_value=1, max_value=30, value=5)
        n_simulations = st.number_input("Liczba symulacji", min_value=100, max_value=10000, value=1000, step=100)
        distribution_type = st.selectbox("Typ rozkładu", ["t-Student", "Normalny"], index=0)
    
    run_analysis = st.sidebar.button("🚀 Uruchom Analizę", type="primary")
    
    # Key Assumptions
    with st.expander("📚 Kluczowe Założenia i Metodologia", expanded=False):
        st.markdown("""
        ### **Jak działają te modele razem?**
        
        **ARIMA, GARCH i Monte Carlo to trzy współpracujące elementy:**
        """)
        
        # Show flow diagram
        st.plotly_chart(create_model_flow_diagram(), use_container_width=True)
        
        st.markdown("""
        ### **Przepływ danych krok po kroku:**
        
        **1️⃣ Model ARIMA (średnia warunkowa)**
        ```
        r_t = μ + φ₁·r_{t-1} + φ₂·r_{t-2} + ... + θ₁·ε_{t-1} + θ₂·ε_{t-2} + ... + ε_t
        ```
        - **Input**: Historyczne zwroty (r_t)
        - **Output**: Prognoza średniej (μ_t) + **reszty (ε_t)**
        - **Co to są reszty?** To różnica między rzeczywistym zwrotem a prognozą: ε_t = r_t - μ_t
        
        **2️⃣ Model GARCH (zmienność warunkowa)**
        ```
        σ²_t = ω + α₁·ε²_{t-1} + β₁·σ²_{t-1}
        ```
        - **Input**: **Reszty z ARIMA (ε_t)** ← tutaj następuje połączenie!
        - **Output**: Prognoza wariancji (σ²_t)
        - **Dlaczego reszty?** Bo chcemy modelować zmienność "niespodzianek", nie samych zwrotów
        
        **3️⃣ Symulacja Monte Carlo**
        ```
        r_t^sim = μ_t + ε_t·σ_t,  gdzie ε_t ~ t(ν) lub N(0,1)
        ```
        - **Input**: Prognoza średniej z ARIMA (μ_t) + Prognoza odch. std. z GARCH (σ_t)
        - **Process**: Losujemy szok (ε_t) ze standardowego rozkładu i skalujemy przez σ_t
        - **Output**: Wiele scenariuszy przyszłych zwrotów
        
        ---
        
        ### **Proces modelowania:**
        1. Identyfikacja lagów AR i MA na podstawie ACF/PACF
        2. Budowa "rzadkiego" modelu ARMA z wybranymi lagami
        3. **Diagnostyka reszt ARMA** - sprawdzenie czy model średniej jest dobry
        4. Dopasowanie modelu GARCH **na resztach ARMA** ← kluczowe!
        5. **Diagnostyka standaryzowanych reszt GARCH** - sprawdzenie czy model zmienności jest dobry
        6. Wieloetapowa symulacja Monte Carlo używająca **obu modeli**
        
        ### **Dlaczego diagnostyka jest ważna:**
        - Reszty ARMA powinny być białym szumem (brak autokorelacji w poziomach)
        - Reszty kwadratowe powinny pokazywać autokorelację (potrzeba GARCH)
        - Standaryzowane reszty GARCH powinny być i.i.d. (niezależne, identycznie rozłożone)
        - Jeśli diagnostyka pokazuje problemy, model może być źle specyfikowany
        
        ### **O optymalizacji jednoczesnej:**
        - **Dwuetapowe (używane tutaj)**: ARIMA → reszty → GARCH
          - ✅ Prostsze w implementacji
          - ✅ Bardziej elastyczne (możemy użyć custom ARIMA)
          - ✅ Łatwiejsza diagnostyka każdego etapu
        - **Jednoczesne**: Maksymalizacja łącznej funkcji wiarygodności
          - ✅ Teoretycznie bardziej efektywne (mniejsze błędy standardowe)
          - ❌ Trudniejsze w implementacji
          - ❌ W praktyce różnice są minimalne
        
        ### **Rozkłady:**
        - **Rozkład normalny**: GARCH z rozkładem normalnym innowacji, zakłada symetryczne ogony
        - **Rozkład t-Studenta**: GARCH-t z rozkładem t-Studenta innowacji (grube ogony, leptokurtoza)
        - **Spójność**: Model GARCH jest dopasowywany z wybranym rozkładem, Monte Carlo używa tego samego
        - **Standaryzacja**: Dla t-Studenta normalizujemy wariancję do 1 przed skalowaniem przez σ_t
        
        ### **Ostrzeżenia:**
        - Wyniki to symulacje probabilistyczne, nie gwarancje
        - VaR 95% to wartość, poniżej której znajduje się 5% najgorszych scenariuszy
        - Model zakłada stałą strukturę zależności (może nie sprawdzić się w kryzysach)
        - Monte Carlo zakłada, że przyszłość będzie podobna do przeszłości
        
        ### **⚠️ Data Leakage i Wskaźniki Techniczne:**
        - **Problem**: Wskaźniki (MACD, Bollinger, RSI) są obliczane z cen, których jeszcze nie znamy
        - **Rozwiązanie**: 
          1. **In-sample**: Używamy opóźnionych wskaźników (lag=1) - wartość z t-1 do predykcji na t
          2. **Out-of-sample (Monte Carlo)**: Wskaźniki NIE są używane w prognozach
        - **Dlaczego?** W rzeczywistym tradingu nie znamy przyszłych wartości wskaźników!
        - Wskaźniki mogą poprawić dopasowanie modelu (in-sample), ale nie są dostępne w prognozach
        """)
    
    if run_analysis:
        try:
            with st.spinner(f"Pobieranie danych dla {ticker}..."):
                data, returns = fetch_data_with_today(ticker, start_date.strftime('%Y-%m-%d'))
            
            st.success(f"✅ Pobrano {len(data)} obserwacji dla {ticker}")
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Średnia zwrotów", f"{returns.mean():.4f}%")
            with col2:
                st.metric("Odch. std.", f"{returns.std():.4f}%")
            with col3:
                st.metric("Skośność", f"{returns.skew():.4f}")
            with col4:
                st.metric("Kurtoza", f"{returns.kurtosis():.4f}")
            
            # Stationarity tests
            st.subheader("1️⃣ Testy Stacjonarności")
            with st.spinner("Przeprowadzanie testów stacjonarności..."):
                stationarity_results = run_stationarity_tests(returns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Test ADF** (H0: szereg niestacjonarny)")
                for spec_name, result in stationarity_results['adf'].items():
                    status = "✅ Stacjonarny" if result['stationary'] else "⚠️ Niestacjonarny"
                    st.write(f"{spec_name}: {status} (p={result['pvalue']:.4f})")
            
            with col2:
                st.markdown("**Test KPSS** (H0: szereg stacjonarny)")
                for spec_name, result in stationarity_results['kpss'].items():
                    status = "✅ Stacjonarny" if result['stationary'] else "⚠️ Niestacjonarny"
                    st.write(f"{spec_name}: {status} (p={result['pvalue']:.4f})")
            
            # ACF and PACF (tylko dla trybu Auto)
            if model_type == "Auto (ACF/PACF)":
                st.subheader("2️⃣ Analiza ACF i PACF")
                with st.spinner("Obliczanie ACF i PACF..."):
                    fig_acf, fig_pacf, sig_acf, sig_pacf, ci_value = plot_acf_pacf(returns, nlags=nlags_acf)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_acf, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_pacf, use_container_width=True)
                
                # Model identification
                ar_lags = [lag for lag, _ in sig_pacf]
                ma_lags = [lag for lag, _ in sig_acf]
                
                st.info(f"""
                **Identyfikacja modelu:**
                - Istotne lagi AR (PACF): {ar_lags if ar_lags else 'Brak'}
                - Istotne lagi MA (ACF): {ma_lags if ma_lags else 'Brak'}
                """)
                
                if not ar_lags and not ma_lags:
                    st.warning("⚠️ Brak istotnych autokorelacji. Dane mogą przypominać biały szum. Spróbuj modelu ARIMA(2,2).")
                    return
            else:
                # Prosty model ARIMA(2,2)
                st.subheader("2️⃣ Model ARIMA(2,2)")
                ar_lags = [1, 2]
                ma_lags = [1, 2]
                st.info("""
                **Wybrany model:**
                - AR lagi: [1, 2]
                - MA lagi: [1, 2]
                - Model ARIMA(2,0,2) z wybranymi parametrami
                """)
            
            # Calculate technical indicators (opcjonalnie)
            tech_indicators = None
            if use_tech_indicators:
                st.subheader("3️⃣ Wskaźniki Techniczne (Transformacja Log)")
                st.warning("""⚠️ **Ważne:** 
                - Wskaźniki są **opóźnione o 1 dzień** (lag=1) aby uniknąć data leakage
                - Używamy wartości z t-1 do predykcji na t
                - **Transformacja logarytmiczna**: sign(x) * log(1 + |x|) stabilizuje wariancję i normalizuje rozkład
                - W prognozach Monte Carlo wskaźniki **NIE są używane** (nie znamy przyszłych wartości)
                """)
                with st.spinner("Obliczanie wskaźników technicznych..."):
                    tech_indicators = calculate_technical_indicators(data, lag=1)
                
                st.info("""💡 **Interpretacja po transformacji log:**
                - Wartości są względne (multiplikatywne) zamiast bezwzględnych
                - Wartości bliskie 0 = neutralne, dodatnie = bycze, ujemne = niedźwiedzie
                - Log kompresuje ekstremalne wartości, zmniejszając wpływ outlierów
                """)
                
                # Display last values of indicators (already lagged)
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("MACD Line (log, L1)", f"{tech_indicators['macd_line_log_L1'].iloc[-1]:.3f}",
                             help="Linia MACD po transformacji log")
                with col2:
                    st.metric("MACD Signal (log, L1)", f"{tech_indicators['macd_signal_log_L1'].iloc[-1]:.3f}",
                             help="Linia sygnału MACD po transformacji log")
                with col3:
                    st.metric("MACD Hist (log, L1)", f"{tech_indicators['macd_hist_log_L1'].iloc[-1]:.3f}",
                             help="Histogram MACD po transformacji log")
                with col4:
                    st.metric("BB Pos. (log, L1)", f"{tech_indicators['bb_position_log_L1'].iloc[-1]:.3f}", 
                             help="Pozycja Bollingera po log: -1 (dolny) do +1 (górny), lag 1 dzień")
                with col5:
                    st.metric("RSI (log, L1)", f"{tech_indicators['rsi_log_L1'].iloc[-1]:.3f}",
                             help="RSI po log: -1 (oversold) do +1 (overbought), lag 1 dzień")
            
            # Build ARMA model with or without technical indicators
            step_number = "4️⃣" if use_tech_indicators else "3️⃣"
            model_desc = "ARMA + Wskaźniki" if use_tech_indicators else "ARMA"
            st.subheader(f"{step_number} Budowa Modelu {model_desc}")
            st.info(f"💡 Model ARMA generuje **reszty** (ε_t = rzeczywisty zwrot - prognoza), które będą użyte w modelu GARCH")
            with st.spinner(f"Dopasowywanie modelu {model_desc}..."):
                model_sparse_arma, X_final, residuals, returns_full = build_sparse_arma_model(
                    returns, ar_lags, ma_lags, tech_indicators
                )
            
            with st.expander("📊 Podsumowanie modelu ARMA (kliknij aby rozwinąć)"):
                st.code(str(model_sparse_arma.summary()), language="text")
            
            # ARMA Residuals Diagnostics
            step_number = "5️⃣" if use_tech_indicators else "4️⃣"
            st.subheader(f"{step_number} Diagnostyka Reszt ARMA")
            st.markdown("""
            **Co sprawdzamy:**
            - **Reszty w czasie**: powinny wyglądać losowo, bez wzorców
            - **Histogram + Q-Q plot**: sprawdzamy normalność rozkładu
            - **ACF reszt**: brak autokorelacji = dobry model średniej
            - **ACF reszt kwadratowych**: istotne lagi = efekty ARCH/GARCH (oczekiwane przed dopasowaniem GARCH)
            """)
            
            with st.spinner("Generowanie diagnostyki reszt ARMA..."):
                fig_arma_diag = plot_residuals_diagnostics(residuals, model_name="ARMA")
            st.plotly_chart(fig_arma_diag, use_container_width=True)
            
            # Interpretacja ACF reszt kwadratowych
            squared_resid = residuals.dropna() ** 2
            acf_sq = acf(squared_resid, nlags=min(20, len(squared_resid)//4))[1:]
            n_significant_sq = np.sum(np.abs(acf_sq) > 1.96/np.sqrt(len(squared_resid)))
            
            if n_significant_sq > 0:
                st.info(f"✅ Wykryto {n_significant_sq} istotnych lagów w ACF reszt kwadratowych - to potwierdza potrzebę modelu GARCH")
            else:
                st.warning("⚠️ Brak istotnych lagów w ACF reszt kwadratowych - efekty GARCH mogą być słabe")
            
            # GARCH model
            step_number = "6️⃣" if use_tech_indicators else "5️⃣"
            st.subheader(f"{step_number} Model GARCH na Resztach")
            st.info(f"""💡 Model GARCH otrzymuje **reszty z ARIMA** (ε_t) i modeluje ich **warunkową wariancję** (σ²_t)
            
            **Rozkład: {distribution_type}**
            - Normal: GARCH(p,q) z rozkładem normalnym innowacji
            - t-Student: GARCH(p,q)-t z rozkładem t-Studenta (lepiej modeluje grube ogony)
            """)
            with st.spinner("Dopasowywanie modelu GARCH..."):
                dist_param = 't' if distribution_type == "t-Student" else 'normal'
                garch_res = fit_garch_model(residuals, p=garch_p, q=garch_q, dist=dist_param)
            
            with st.expander("📊 Podsumowanie modelu GARCH (kliknij aby rozwinąć)"):
                st.code(str(garch_res.summary()), language="text")
            
            # GARCH Standardized Residuals Diagnostics
            step_number = "7️⃣" if use_tech_indicators else "6️⃣"
            st.subheader(f"{step_number} Diagnostyka Standaryzowanych Reszt GARCH")
            st.markdown("""
            **Co sprawdzamy:**
            - **Standaryzowane reszty**: powinny mieć średnią ~0 i odch. std. ~1
            - **Histogram + Q-Q plot**: zgodność z rozkładem (t-Student lub normalny)
            - **ACF standaryzowanych reszt**: brak autokorelacji = dobry model GARCH
            - Standaryzowane reszty = reszty / √(warunkowa wariancja z GARCH)
            """)
            
            with st.spinner("Generowanie diagnostyki GARCH..."):
                fig_garch_diag, garch_stats = plot_garch_standardized_residuals(
                    garch_res, model_name=f"GARCH({garch_p},{garch_q})"
                )
            st.plotly_chart(fig_garch_diag, use_container_width=True)
            
            # Display standardized residuals statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                color = "normal" if abs(garch_stats['mean']) < 0.1 else "off"
                st.metric("Średnia std. reszt", f"{garch_stats['mean']:.4f}", 
                         help="Powinna być bliska 0", delta_color=color)
            with col2:
                color = "normal" if abs(garch_stats['std'] - 1.0) < 0.2 else "off"
                st.metric("Odch. std.", f"{garch_stats['std']:.4f}",
                         help="Powinna być bliska 1", delta_color=color)
            with col3:
                st.metric("Skośność", f"{garch_stats['skewness']:.4f}",
                         help="0 = symetryczny")
            with col4:
                st.metric("Kurtoza (excess)", f"{garch_stats['kurtosis']:.4f}",
                         help="0 = normalny, >0 = grube ogony")
            
            # Jarque-Bera test result
            jb_status = "✅ Normalny" if garch_stats['jb_pvalue'] > 0.05 else "⚠️ Nienormalny"
            st.info(f"**Test Jarque-Bera**: {jb_status} (statystyka={garch_stats['jb_stat']:.2f}, p={garch_stats['jb_pvalue']:.4f})")
            
            # Diagnostic tests (pozostałe testy)
            step_number = "8️⃣" if use_tech_indicators else "7️⃣"
            st.subheader(f"{step_number} Testy Statystyczne (ARMA)")
            residuals_final = model_sparse_arma.resid.dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Ljung-Box test
                lb_test = acorr_ljungbox(residuals_final, lags=[10], return_df=True)
                lb_pvalue = lb_test['lb_pvalue'].iloc[0]
                lb_status = "✅ Brak autokorelacji" if lb_pvalue > 0.05 else "⚠️ Autokorelacja wykryta"
                st.metric("Test Ljung-Box (lag 10)", lb_status, f"p={lb_pvalue:.4f}")
            
            with col2:
                # Jarque-Bera test
                jb_stat, jb_pvalue = stats.jarque_bera(residuals_final)
                jb_status = "✅ Normalny" if jb_pvalue > 0.05 else "⚠️ Nienormalny"
                st.metric("Test Jarque-Bera", jb_status, f"p={jb_pvalue:.4f}")
            
            with col3:
                # ARCH test
                arch_test = het_arch(residuals_final, nlags=10)
                arch_pvalue = arch_test[1]
                arch_status = "✅ Brak ARCH" if arch_pvalue > 0.05 else "⚠️ Efekty ARCH"
                st.metric("Test ARCH-LM", arch_status, f"p={arch_pvalue:.4f}")
            
            # Monte Carlo simulation
            step_number = "9️⃣" if use_tech_indicators else "8️⃣"
            st.subheader(f"{step_number} Symulacja Monte Carlo")
            st.info("💡 Monte Carlo używa **obu modeli**: prognozy średniej z ARIMA (μ_t) + prognozy zmienności z GARCH (σ_t)")
            
            dist_explanation = "standaryzowany rozkład t-Studenta (df={:.2f}) znormalizowany do wariancji=1".format(
                garch_res.params.get('nu', 0)) if distribution_type == "t-Student" else "rozkład normalny N(0,1)"
            
            st.markdown(f"""
            **Formuła symulacji:** r_t = μ_t + ε_t · σ_t, gdzie:
            - μ_t = prognoza z modelu ARMA
            - σ_t = √(prognoza wariancji z GARCH)
            - ε_t ~ {dist_explanation}
            - **Spójność**: GARCH dopasowany z rozkładem {distribution_type}, symulacje używają tego samego rozkładu
            """)
            st.write(f"Symulacja {n_simulations} ścieżek na {forecast_horizon} dni naprzód...")
            
            with st.spinner("Uruchamianie symulacji Monte Carlo..."):
                paths_df, cumulative_paths_df = monte_carlo_multistep(
                    model_sparse_arma, garch_res, returns_full, residuals,
                    X_final, ar_lags, ma_lags, forecast_horizon, n_simulations,
                    distribution='t' if distribution_type == "t-Student" else 'normal',
                    tech_indicators=tech_indicators
                )
            
            st.success(f"✅ Wygenerowano {n_simulations} ścieżek symulacyjnych")
            
            # Statistics for final period
            final_day_returns = paths_df.iloc[-1]
            final_cum_returns = cumulative_paths_df.iloc[-1]
            
            # Daily returns statistics
            st.markdown("### 📊 Statystyki dla Ostatniego Dnia Prognozy (zwroty logarytmiczne)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Średnia", f"{final_day_returns.mean():.4f}")
            with col2:
                st.metric("Mediana", f"{final_day_returns.median():.4f}")
            with col3:
                st.metric("VaR 95%", f"{final_day_returns.quantile(0.05):.4f}",
                         help="5% scenariuszy jest gorsze niż ta wartość")
            with col4:
                st.metric("VaR 99%", f"{final_day_returns.quantile(0.01):.4f}",
                         help="1% scenariuszy jest gorsze niż ta wartość")
            
            col1, col2 = st.columns(2)
            with col1:
                prob_gain = (final_day_returns > 0).mean()
                st.metric("Prawdopodobieństwo zysku", f"{prob_gain:.2%}")
            with col2:
                prob_loss = (final_day_returns <= 0).mean()
                st.metric("Prawdopodobieństwo straty", f"{prob_loss:.2%}")
            
            # Cumulative returns statistics
            st.markdown("### 📊 Statystyki dla Zwrotów Skumulowanych (zwroty logarytmiczne)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Średnia", f"{final_cum_returns.mean():.4f}")
            with col2:
                st.metric("Mediana", f"{final_cum_returns.median():.4f}")
            with col3:
                st.metric("VaR 95%", f"{final_cum_returns.quantile(0.05):.4f}",
                         help="5% scenariuszy ma skumulowany zwrot gorszy niż ta wartość")
            with col4:
                st.metric("VaR 99%", f"{final_cum_returns.quantile(0.01):.4f}",
                         help="1% scenariuszy ma skumulowany zwrot gorszy niż ta wartość")
            
            col1, col2 = st.columns(2)
            with col1:
                prob_gain_cum = (final_cum_returns > 0).mean()
                st.metric("Prawdopodobieństwo zysku (skum.)", f"{prob_gain_cum:.2%}")
            with col2:
                prob_loss_cum = (final_cum_returns <= 0).mean()
                st.metric("Prawdopodobieństwo straty (skum.)", f"{prob_loss_cum:.2%}")
            
            # Visualizations
            step_number = "🔟" if use_tech_indicators else "9️⃣"
            st.subheader(f"{step_number} Wizualizacje Prognozy")
            
            # Fan chart
            st.markdown("**Wykres Wachlarzowy (Fan Chart)**")
            fig_fan = create_fan_chart(
                paths_df, cumulative_paths_df, ticker, n_simulations, distribution_type
            )
            st.plotly_chart(fig_fan, use_container_width=True)
            
            # Histograms
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rozkład Zwrotów Dziennych**")
                fig_hist_daily = create_histogram(
                    final_day_returns,
                    final_day_returns.median(),
                    final_day_returns.quantile(0.05),
                    final_day_returns.quantile(0.01),
                    f"Rozkład zwrotów dziennych (t+{forecast_horizon})",
                    "Zwrot logarytmiczny"
                )
                st.plotly_chart(fig_hist_daily, use_container_width=True)
            
            with col2:
                st.markdown("**Rozkład Zwrotów Skumulowanych**")
                fig_hist_cum = create_histogram(
                    final_cum_returns,
                    final_cum_returns.median(),
                    final_cum_returns.quantile(0.05),
                    final_cum_returns.quantile(0.01),
                    f"Rozkład zwrotów skumulowanych (0 → t+{forecast_horizon})",
                    "Skumulowany zwrot logarytmiczny"
                )
                st.plotly_chart(fig_hist_cum, use_container_width=True)
            
            # Summary insights
            st.subheader("📝 Podsumowanie")
            st.info(f"""
            **Kluczowe wnioski z analizy {ticker}:**
            - Używamy **czystych zwrotów logarytmicznych** (log returns) zgodnie ze standardami ekonometrii
            - Model ARMA z {len(ar_lags)} lagami AR i {len(ma_lags)} lagami MA
            - GARCH({garch_p},{garch_q}) z rozkładem {distribution_type}
            - Horyzont prognozy: {forecast_horizon} dni
            - Liczba symulacji: {n_simulations}
            - Prawdopodobieństwo zysku (skumulowane): {prob_gain_cum:.1%}
            - VaR 95% (skumulowany): {final_cum_returns.quantile(0.05):.4f}
            """)
            
        except Exception as e:
            st.error(f"❌ Błąd podczas analizy: {str(e)}")
            st.exception(e)
