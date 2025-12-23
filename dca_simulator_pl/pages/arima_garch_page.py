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
    """Fetch historical data including today's intraday data if available."""
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
    
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
    returns = data['log_returns'].dropna() * 100
    
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
        title=f"Funkcja Autokorelacji (ACF) - n={n}",
        xaxis_title="Opóźnienie (Lag)", yaxis_title="ACF",
        template="plotly_dark", height=400,
        yaxis=dict(range=[-0.3, 0.3], zeroline=True)
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
        title=f"Częściowa Funkcja Autokorelacji (PACF) - n={n}",
        xaxis_title="Opóźnienie (Lag)", yaxis_title="PACF",
        template="plotly_dark", height=400,
        yaxis=dict(range=[-0.3, 0.3], zeroline=True)
    )
    
    # Identify significant lags
    significant_acf = [(i, acf_values[i]) for i in range(1, len(acf_values)) if abs(acf_values[i]) > ci_value]
    significant_pacf = [(i, pacf_values[i]) for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > ci_value]
    
    return fig_acf, fig_pacf, significant_acf, significant_pacf, ci_value


def build_sparse_arma_model(returns, ar_lags, ma_lags):
    """Build sparse ARMA model with selected lags."""
    # Stage 1: Build AR model to get residuals
    df_ar = pd.DataFrame({'returns': returns})
    for lag in ar_lags:
        df_ar[f'ar_L{lag}'] = returns.shift(lag)
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
                          ar_lags, ma_lags, forecast_horizon, n_simulations, distribution='t'):
    """
    Run multi-step Monte Carlo simulation with both normal and t-student distributions.
    """
    np.random.seed(None)
    
    # Prepare GARCH forecast
    garch_forecast = garch_res.forecast(horizon=forecast_horizon, reindex=False)
    sigma_array = np.sqrt(garch_forecast.variance.values[-1])
    
    if distribution == 't':
        df_t = garch_res.params['nu']
        std_factor = np.sqrt((df_t - 2) / df_t) if df_t > 2 else 1.0
    else:
        df_t = None
        std_factor = 1.0
    
    # Monte Carlo simulation
    all_paths = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(n_simulations):
        returns_extended = returns.copy()
        residuals_extended = residuals.copy()
        single_path = []
        
        for step in range(forecast_horizon):
            # Prepare exogenous variables
            future_exog = {'const': 1.0}
            for lag in ar_lags:
                future_exog[f'ar_L{lag}'] = returns_extended.iloc[-lag]
            for lag in ma_lags:
                future_exog[f'ma_L{lag}'] = residuals_extended.iloc[-lag]
            
            future_exog_df = pd.DataFrame([future_exog]).reindex(columns=X_final.columns, fill_value=0.0)
            
            # Forecast mean
            mu = model_sparse_arma.forecast(steps=1, exog=future_exog_df).iloc[0]
            sigma_t = sigma_array[step]
            
            # Generate shock
            if distribution == 't':
                shock = (np.random.standard_t(df_t) / std_factor) * sigma_t
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
        title=f'Prognoza Monte Carlo dla {stock_name} ({n_simulations} ścieżek, rozkład: {dist_type})',
        xaxis_title='Data',
        yaxis_title='Prognozowany zwrot logarytmiczny (dzienny)',
        yaxis_tickformat='.4f',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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


def create_histogram(data, median_val, var_95, var_99, title, xlabel):
    """Create histogram with VaR lines."""
    fig = px.histogram(
        data, nbins=max(50, len(data) // 20), marginal="box",
        title=title
    )
    
    fig.add_vline(
        x=median_val, line_width=2, line_dash="dash", line_color="cyan",
        annotation_text=f"Mediana: {median_val:.4f}",
        annotation_position="top left"
    )
    fig.add_vline(
        x=var_95, line_width=2, line_color="orange",
        annotation_text=f"VaR 95%: {var_95:.4f}",
        annotation_position="bottom left"
    )
    fig.add_vline(
        x=var_99, line_width=2, line_color="red",
        annotation_text=f"VaR 99%: {var_99:.4f}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title='Częstość (liczba symulacji)',
        xaxis_tickformat='.4f',
        template='plotly_dark',
        showlegend=False
    )
    
    return fig


def render_arima_garch_page():
    """Main function to render ARIMA-GARCH Monte Carlo Analysis page."""
    
    st.header("📈 Modelowanie ARIMA-GARCH z Symulacjami Monte Carlo")
    st.markdown("""
    Zaawansowana analiza szeregów czasowych łącząca ekonometryczne modelowanie warunkowej średniej (ARIMA), 
    warunkowej wariancji (GARCH) oraz symulacje stochastyczne Monte Carlo dla wieloetapowych prognoz.
    """)
    
    # Sidebar configuration
    st.sidebar.subheader("⚙️ Konfiguracja Parametrów")
    
    ticker = st.sidebar.text_input("Symbol giełdowy (ticker)", value="AAPL", help="Wprowadź symbol giełdowy waloru")
    start_date = st.sidebar.date_input("Data początkowa próby", value=pd.to_datetime("2023-01-01"))
    
    with st.sidebar.expander("🔧 Specyfikacja Modelu"):
        nlags_acf = st.number_input("Maksymalna liczba opóźnień (lags)", min_value=10, max_value=60, value=40, 
                                    help="Liczba opóźnień dla funkcji ACF i PACF")
        garch_p = st.number_input("GARCH: rząd p (ARCH)", min_value=1, max_value=5, value=1,
                                  help="Liczba opóźnień kwadratów reszt (część ARCH)")
        garch_q = st.number_input("GARCH: rząd q (GARCH)", min_value=1, max_value=5, value=1,
                                  help="Liczba opóźnień wariancji warunkowej")
    
    with st.sidebar.expander("🎲 Parametry Symulacji Monte Carlo"):
        forecast_horizon = st.number_input("Horyzont prognozy (dni sesyjne)", min_value=1, max_value=30, value=5,
                                          help="Liczba przyszłych okresów do zasymulowania")
        n_simulations = st.number_input("Liczba ścieżek symulacyjnych", min_value=100, max_value=10000, value=1000, step=100,
                                       help="Większa liczba symulacji zwiększa dokładność estymacji")
        distribution_type = st.selectbox("Rozkład innowacji", ["t-Studenta", "Normalny (Gaussa)"], index=0,
                                        help="Rozkład t-Studenta lepiej odzwierciedla grube ogony")
    
    run_analysis = st.sidebar.button("🚀 Przeprowadź Analizę Ekonometyczną", type="primary")
    
    # Kluczowe założenia i metodologia
    with st.expander("📚 Założenia Modelowe i Metodologia Ekonometryczna", expanded=False):
        st.markdown("""
        **Metodologia ekonometryczna ARIMA-GARCH:**
        
        **1. Model ARIMA (AutoRegressive Integrated Moving Average)**
        - Modeluje *warunkową wartość oczekiwaną* (pierwszą cechę rozkładu) zwrotów logarytmicznych
        - **AR (autoregresja)**: zależność od wcześniejszych realizacji zmiennej
        - **MA (moving average)**: zaleśność od wcześniejszych innowacji (błędów prognozy)
        - **I (integrated)**: bieżący model nie wymaga różnicowania (dane już stacjonarne)
        
        **2. Model GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)**
        - Modeluje *warunkową wariancję* (drugą cechę rozkładu) - zjawisko heteroskedastyczności
        - Uwzględnia **grupowanie zmienności** (*volatility clustering*) - typowe dla szeregów finansowych
        - GARCH(p,q): p = opóźnienia kwadratów reszt (ARCH), q = opóźnienia wariancji warunkowej
        
        **3. Symulacje Monte Carlo**
        - Metoda stochastyczna generująca wielokrotne ścieżki przyszłych zwrotów
        - Każda ścieżka wykorzystuje: prognozy z ARIMA (mu) + GARCH (sigma) + losowy szok (epsilon)
        - Pozwala estymować pełny rozkład prawdopodobieństwa przyszłych wyników
        
        **Proces modelowania (metoda Boxa-Jenkinsa):**
        1. **Identyfikacja**: analiza ACF/PACF w celu wybór istotnych opóźnień
        2. **Estymacja**: dopasowanie "rzadkiego" modelu ARMA tylko z istotnymi lagami (NMW/MLE)
        3. **Weryfikacja**: diagnostyka reszt (testy Ljung-Box, JB, ARCH-LM)
        4. **Modelowanie wariancji**: GARCH na resztach z modelu ARMA
        5. **Prognozowanie**: wieloetapowe symulacje Monte Carlo
        
        **Rozkłady innowacji:**
        - **Rozkład normalny (Gaussa)**: klasyczne założenie, symetryczne ogony, kurtoza = 3
        - **Rozkład t-Studenta**: *grube ogony* (leptokurtoza, kurtoza > 3), lepiej oddaje ekstrema w danych finansowych
        
        **Miary ryzyka:**
        - **VaR (Value at Risk)**: kwantyl rozkładu - maksymalna oczekiwana strata przy danym poziomie ufności
        - **VaR 95%** (kwantyl 5%): w 95% przypadków strata nie przekroczy tej wartości
        - **VaR 99%** (kwantyl 1%): bardziej konserwatywna miara ryzyka ekstremalnego
        
        **Kluczowe założenia i ograniczenia:**
        - ⚠️ Wyniki są estymacjami probabilistycznymi, nie deterministycznymi prognozami
        - ⚠️ Model zakłada *stabilność strukturalną* - parametry nie zmieniają się w czasie
        - ⚠️ Nie uwzględnia *przełamań strukturalnych* (kryzysy, black swans)
        - ⚠️ Stacjonarność szeregu jest warunkiem koniecznym poprawności modelu
        - ℹ️ Model nie uwzględnia kosztów transakcyjnych ani płynności
        """)
    
    if run_analysis:
        try:
            with st.spinner(f"Pobieranie danych dla {ticker}..."):
                data, returns = fetch_data_with_today(ticker, start_date.strftime('%Y-%m-%d'))
            
            st.success(f"✅ Pobrano {len(data)} obserwacji sesyjnych dla {ticker}")
            
            # Statystyki opisowe szeregu zwrotów
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Średnia arytmetyczna zwrotów", f"{returns.mean():.4f}%")
            with col2:
                st.metric("Odch. standardowe (σ)", f"{returns.std():.4f}%")
            with col3:
                st.metric("Skośność (skewness)", f"{returns.skew():.4f}")
            with col4:
                st.metric("Kurtoza (nadmiar)", f"{returns.kurtosis():.4f}")
            
            # Testy pierwiastka jednostkowego
            st.subheader("1️⃣ Testy Stacjonarności (Unit Root Tests)")
            with st.spinner("Testowanie własności stochastycznych szeregu..."):
                stationarity_results = run_stationarity_tests(returns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Test ADF (Augmented Dickey-Fuller)**")
                st.caption("H₀: szereg zawiera pierwiastek jednostkowy (niestacjonarny)")
                for spec_name, result in stationarity_results['adf'].items():
                    status = "✅ Stacjonarny" if result['stationary'] else "⚠️ Niestacjonarny"
                    st.write(f"{spec_name}: {status} (p={result['pvalue']:.4f})")
            
            with col2:
                st.markdown("**Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**")
                st.caption("H₀: szereg jest stacjonarny (brak pierwiastka jednostkowego)")
                for spec_name, result in stationarity_results['kpss'].items():
                    status = "✅ Stacjonarny" if result['stationary'] else "⚠️ Niestacjonarny"
                    st.write(f"{spec_name}: {status} (p={result['pvalue']:.4f})")
            
            # Analiza autokorelacji
            st.subheader("2️⃣ Funkcje Autokorelacji (Korelogram)")
            with st.spinner("Obliczanie funkcji ACF i PACF..."):
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
            **Identyfikacja modelu (metoda Boxa-Jenkinsa):**
            - Istotne statystycznie opóźnienia AR (z PACF): {ar_lags if ar_lags else 'Brak'}
            - Istotne statystycznie opóźnienia MA (z ACF): {ma_lags if ma_lags else 'Brak'}
            """)
            
            if not ar_lags and not ma_lags:
                st.warning("⚠️ Brak istotnych autokorelacji. Szereg może przypominać biały szum (white noise).")
                return
            
            # Estymacja modelu ARMA
            st.subheader("3️⃣ Estymacja Modelu ARMA (Rzadka Specyfikacja)")
            with st.spinner("Estymacja parametrów modelu ARMA metodą największej wiarygodności..."):
                model_sparse_arma, X_final, residuals, returns_full = build_sparse_arma_model(
                    returns, ar_lags, ma_lags
                )
            
            with st.expander("Wyniki estymacji modelu ARMA (parametry, błędy std., testy)"):
                st.text(str(model_sparse_arma.summary()))
            
            # Model GARCH
            st.subheader("4️⃣ Model GARCH - Warunkowa Heteroskedastyczność")
            with st.spinner("Estymacja modelu GARCH na resztach z ARMA..."):
                dist_param = 't' if distribution_type == "t-Studenta" else 'normal'
                garch_res = fit_garch_model(residuals, p=garch_p, q=garch_q, dist=dist_param)
            
            with st.expander("Wyniki estymacji modelu GARCH (parametry, testy)"):
                st.text(str(garch_res.summary()))
            
            # Diagnostyka reszt
            st.subheader("5️⃣ Diagnostyka Reszt (Weryfikacja Modelu)")
            residuals_final = model_sparse_arma.resid.dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Test Ljung-Box na autokorelację
                lb_test = acorr_ljungbox(residuals_final, lags=[10], return_df=True)
                lb_pvalue = lb_test['lb_pvalue'].iloc[0]
                lb_status = "✅ Brak autokorelacji" if lb_pvalue > 0.05 else "⚠️ Autokorelacja wykryta"
                st.metric("Test Ljung-Box (lag=10)", lb_status, f"p={lb_pvalue:.4f}",
                         help="Test niezależności reszt. H₀: brak autokorelacji")
            
            with col2:
                # Test Jarque-Bera na normalność
                jb_stat, jb_pvalue = stats.jarque_bera(residuals_final)
                jb_status = "✅ Normalny" if jb_pvalue > 0.05 else "⚠️ Nienormalny"
                st.metric("Test Jarque-Bera", jb_status, f"p={jb_pvalue:.4f}",
                         help="Test normalności rozkładu reszt. H₀: rozkład normalny")
            
            with col3:
                # Test ARCH-LM na efekty ARCH
                arch_test = het_arch(residuals_final, nlags=10)
                arch_pvalue = arch_test[1]
                arch_status = "✅ Brak efektów ARCH" if arch_pvalue > 0.05 else "⚠️ Efekty ARCH"
                st.metric("Test ARCH-LM", arch_status, f"p={arch_pvalue:.4f}",
                         help="Test heteroskedastyczności. H₀: brak efektów ARCH w resztach")
            
            # Symulacje Monte Carlo
            st.subheader("6️⃣ Symulacje Stochastyczne Monte Carlo")
            st.write(f"Generowanie {n_simulations} ścieżek trajektorii zwrotów na {forecast_horizon} sesji naprzód...")
            
            with st.spinner("Przeprowadzanie symulacji Monte Carlo..."):
                paths_df, cumulative_paths_df = monte_carlo_multistep(
                    model_sparse_arma, garch_res, returns_full, residuals,
                    X_final, ar_lags, ma_lags, forecast_horizon, n_simulations,
                    distribution='t' if distribution_type == "t-Student" else 'normal'
                )
            
            st.success(f"✅ Wygenerowano {n_simulations} ścieżek trajektorii zwrotów")
            
            # Statystyki dla końcowego okresu prognozy
            final_day_returns = paths_df.iloc[-1]
            final_cum_returns = cumulative_paths_df.iloc[-1]
            
            # Statystyki zwrotów dziennych
            st.markdown("### 📊 Statystyki Opisowe Zwrotów Dziennych (Ostatni Dzień Prognozy)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wartość oczekiwana (E[r])", f"{final_day_returns.mean():.4f}")
            with col2:
                st.metric("Mediana (Q₂)", f"{final_day_returns.median():.4f}")
            with col3:
                st.metric("VaR 95% (kwantyl 5%)", f"{final_day_returns.quantile(0.05):.4f}")
            with col4:
                st.metric("VaR 99% (kwantyl 1%)", f"{final_day_returns.quantile(0.01):.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                prob_gain = (final_day_returns > 0).mean()
                st.metric("P(zwrot > 0)", f"{prob_gain:.2%}", help="Prawdopodobieństwo dodatniego zwrotu")
            with col2:
                prob_loss = (final_day_returns <= 0).mean()
                st.metric("P(zwrot ≤ 0)", f"{prob_loss:.2%}", help="Prawdopodobieństwo ujemnego zwrotu lub zerowego")
            
            # Statystyki zwrotów skumulowanych
            st.markdown("### 📊 Statystyki Opisowe Zwrotów Skumulowanych (Pełen Horyzont)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Wartość oczekiwana (E[Σr])", f"{final_cum_returns.mean():.4f}")
            with col2:
                st.metric("Mediana (Q₂)", f"{final_cum_returns.median():.4f}")
            with col3:
                st.metric("VaR 95% (kwantyl 5%)", f"{final_cum_returns.quantile(0.05):.4f}")
            with col4:
                st.metric("VaR 99% (kwantyl 1%)", f"{final_cum_returns.quantile(0.01):.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                prob_gain_cum = (final_cum_returns > 0).mean()
                st.metric("P(zwrot skumulowany > 0)", f"{prob_gain_cum:.2%}")
            with col2:
                prob_loss_cum = (final_cum_returns <= 0).mean()
                st.metric("P(zwrot skumulowany ≤ 0)", f"{prob_loss_cum:.2%}")
            
            # Wizualizacje
            st.subheader("7️⃣ Wizualizacje Rozkładów Prognozowanych")
            
            # Wykres wachlarzowy
            st.markdown("**Wykres Wachlarzowy (Fan Chart) - Przedziały Ufności**")
            fig_fan = create_fan_chart(
                paths_df, cumulative_paths_df, ticker, n_simulations, distribution_type
            )
            st.plotly_chart(fig_fan, use_container_width=True)
            
            # Histogramy rozkładów
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rozkład Zwrotów Dziennych (Funkcja Gęstości)**")
                fig_hist_daily = create_histogram(
                    final_day_returns,
                    final_day_returns.median(),
                    final_day_returns.quantile(0.05),
                    final_day_returns.quantile(0.01),
                    f"Empiryczna funkcja gęstości zwrotów (t+{forecast_horizon})",
                    "Zwrot logarytmiczny"
                )
                st.plotly_chart(fig_hist_daily, use_container_width=True)
            
            with col2:
                st.markdown("**Rozkład Zwrotów Skumulowanych (Funkcja Gęstości)**")
                fig_hist_cum = create_histogram(
                    final_cum_returns,
                    final_cum_returns.median(),
                    final_cum_returns.quantile(0.05),
                    final_cum_returns.quantile(0.01),
                    f"Empiryczna funkcja gęstości zwrotów skumulowanych (0 → t+{forecast_horizon})",
                    "Skumulowany zwrot logarytmiczny"
                )
                st.plotly_chart(fig_hist_cum, use_container_width=True)
            
            # Podsumowanie analityczne
            st.subheader("📝 Podsumowanie Analityczne")
            st.info(f"""
            **Wyniki analizy ekonometrycznej dla {ticker}:**
            - **Specyfikacja modelu**: ARMA({len(ar_lags)},{len(ma_lags)}) + GARCH({garch_p},{garch_q})
            - **Rozkład innowacji**: {distribution_type}
            - **Horyzont prognozy**: {forecast_horizon} sesji giełdowych
            - **Liczba replikacji Monte Carlo**: {n_simulations}
            
            **Analiza ryzyka (zwroty skumulowane):**
            - Prawdopodobieństwo dodatniego wyniku: {prob_gain_cum:.1%}
            - VaR 95% (maksymalna strata w 95% scenariuszy): {final_cum_returns.quantile(0.05):.4f}
            - VaR 99% (maksymalna strata w 99% scenariuszy): {final_cum_returns.quantile(0.01):.4f}
            
            **Interpretacja:**
            - Model ARIMA przechwytuje zależności liniowe w szeregu zwrotów
            - Model GARCH uwzględnia grupowanie zmienności (volatility clustering)
            - Symulacje Monte Carlo generują pełen rozkład prawdopodobieństwa przyszłych wyników
            - VaR dostarcza ilościowej oceny ryzyka ekstremalnego (downside risk)
            """)
            
        except Exception as e:
            st.error(f"❌ Błąd podczas analizy: {str(e)}")
            st.exception(e)
