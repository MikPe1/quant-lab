# app.py

import streamlit as st
import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import config
from data_fetcher import fetch_historical_data
from dca_engine import simulate_dca
from visualizer import plot_simulation_results, display_results_table

# Ignore warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# ============================================================================
# ADVANCED PORTFOLIO ANALYSIS FUNCTIONS
# ============================================================================

class HierarchicalRiskParity:
    """
    Implementation of Hierarchical Risk Parity (HRP) algorithm
    following Marcos López de Prado's methodology.
    """
    def __init__(self, returns):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be in pd.DataFrame format.")
        self.returns = returns
        self.cov_matrix = self.returns.cov()
        self.corr_matrix = self.returns.corr()
        self.tickers = self.returns.columns.tolist()

    def get_linkage_matrix(self):
        """Calculate linkage matrix for hierarchical clustering."""
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        dist_condensed = squareform(dist)
        return linkage(dist_condensed, method='single')

    def get_quasi_diag_matrix(self, link):
        """Sort assets in clusters to obtain quasi-diagonal covariance matrix."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def get_recursive_bisection_weights(self, sort_ix):
        """Recursively bisect portfolio and distribute weights."""
        weights = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]

        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                
                cov_sub = self.cov_matrix.iloc[c_items0, c_items0]
                inv_diag = 1 / np.diag(cov_sub.values)
                w0 = inv_diag / inv_diag.sum()
                v0 = np.dot(w0, np.dot(cov_sub, w0))

                cov_sub = self.cov_matrix.iloc[c_items1, c_items1]
                inv_diag = 1 / np.diag(cov_sub.values)
                w1 = inv_diag / inv_diag.sum()
                v1 = np.dot(w1, np.dot(cov_sub, w1))

                alpha = 1 - v0 / (v0 + v1) if v0 + v1 != 0 else 0.5
                
                weights[c_items0] *= alpha
                weights[c_items1] *= 1 - alpha
        
        return weights

    def get_hrp_weights(self):
        """Main function to calculate HRP weights."""
        link = self.get_linkage_matrix()
        sort_ix = self.get_quasi_diag_matrix(link)
        
        sorted_tickers = [self.tickers[i] for i in sort_ix]
        
        hrp_weights = self.get_recursive_bisection_weights(sort_ix)
        hrp_weights.index = sorted_tickers
        
        # Ensure weights are sorted alphabetically like original data
        hrp_weights = hrp_weights.sort_index()
        
        return hrp_weights, link

def stress_test_analysis(returns, weights):
    """
    Perform basic stress-test analysis for portfolio.
    """
    portfolio_returns = (returns * weights).sum(axis=1)

    # Value at Risk (VaR) - historical simulation
    var_95 = portfolio_returns.quantile(0.05)
    var_99 = portfolio_returns.quantile(0.01)

    # Conditional Value at Risk (CVaR) / Expected Shortfall
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()

    # Skewness and Kurtosis
    skewness = portfolio_returns.skew()
    kurtosis = portfolio_returns.kurtosis() # Excess kurtosis (above 3)

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

def extract_optimal_weights_with_oos(returns, train_test_split=0.8):
    """
    Calculate optimal weights with true train/test split.
    Automatically selects best method based on OOS performance.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    train_test_split : float
        % of data for training (e.g., 0.8 = 80% train, 20% test)
    
    Returns:
    --------
    dict with results and weights
    """
    
    # Split data
    split_idx = int(len(returns) * train_test_split)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    
    # Calculate weights using different methods ON TRAINING DATA
    methods_weights = {}
    
    # 1. HRP
    hrp = HierarchicalRiskParity(train_returns)
    hrp_weights, _ = hrp.get_hrp_weights()
    methods_weights['HRP'] = hrp_weights
    
    # 2. Equal Weight
    ew_weights = pd.Series(1/len(train_returns.columns), index=train_returns.columns)
    methods_weights['Equal Weight'] = ew_weights
    
    # 3. Inverse Volatility
    train_vols = train_returns.std()
    inv_vol_weights = (1 / train_vols) / (1 / train_vols).sum()
    methods_weights['Inverse Volatility'] = inv_vol_weights
    
    # 4. Mean-Variance (Markowitz - Max Sharpe)
    mean_ret = train_returns.mean() * 252
    
    # Ledoit-Wolf shrinkage for more stable covariance matrix
    lw = LedoitWolf()
    cov_mat_lw = pd.DataFrame(
        lw.fit(train_returns).covariance_ * 252,
        index=train_returns.columns,
        columns=train_returns.columns
    )
    
    def neg_sharpe(weights, mean_returns, cov_matrix, rf=0.04):
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_vol == 0:
            return -np.inf
        return -(port_return - rf) / port_vol
    
    n = len(train_returns.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial = n * [1. / n]
    
    # Optimization with Ledoit-Wolf
    result_mv = minimize(
        neg_sharpe, 
        initial, 
        args=(mean_ret, cov_mat_lw),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000}
    )
    mv_weights = pd.Series(result_mv.x, index=train_returns.columns)
    methods_weights['Mean-Variance (Markowitz)'] = mv_weights
    
    # 5. Minimum Variance
    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    result_minvar = minimize(
        portfolio_variance,
        initial,
        args=(cov_mat_lw,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    minvar_weights = pd.Series(result_minvar.x, index=train_returns.columns)
    methods_weights['Min Variance'] = minvar_weights
    
    # 6. Risk Parity
    def risk_parity_objective(weights, cov_matrix):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_vol < 1e-10:
            return 1e10
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        target_contrib = portfolio_vol / len(weights)
        return np.sum((contrib - target_contrib) ** 2)
    
    result_rp = minimize(
        lambda w: risk_parity_objective(w, cov_mat_lw),
        initial,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    rp_weights = pd.Series(result_rp.x / result_rp.x.sum(), index=train_returns.columns)
    methods_weights['Risk Parity'] = rp_weights
    
    # ========================================================================
    # OUT-OF-SAMPLE TESTING
    # ========================================================================
    
    oos_results = []
    portfolio_series = {}
    
    for method_name, weights in methods_weights.items():
        # Portfolio returns on test set
        test_portfolio_returns = (test_returns * weights).sum(axis=1)
        cumulative_returns = (1 + test_portfolio_returns).cumprod()
        
        # Metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(test_returns)) - 1
        annual_vol = test_portfolio_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
        max_dd = ((cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max()).min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        downside_returns = test_portfolio_returns[test_portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - 0.04) / downside_std if downside_std > 0 else 0
        win_rate = (test_portfolio_returns > 0).sum() / len(test_portfolio_returns)
        herfindahl = (weights ** 2).sum()
        effective_n = 1 / herfindahl
        
        oos_results.append({
            'Method': method_name,
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Annual_Vol': annual_vol,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd,
            'Calmar_Ratio': calmar,
            'Win_Rate': win_rate,
            'Effective_N_Assets': effective_n,
            'Concentration': herfindahl
        })
        
        portfolio_series[method_name] = cumulative_returns
    
    oos_df = pd.DataFrame(oos_results).set_index('Method')
    
    # ========================================================================
    # SELECT BEST METHOD
    # ========================================================================
    
    # Multi-criteria scoring
    scoring_df = oos_df.copy()
    
    # Normalize metrics (higher = better)
    for col in ['Annual_Return', 'Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio', 
                'Win_Rate', 'Effective_N_Assets']:
        if scoring_df[col].max() - scoring_df[col].min() > 0:
            scoring_df[f'{col}_norm'] = (scoring_df[col] - scoring_df[col].min()) / \
                                        (scoring_df[col].max() - scoring_df[col].min())
        else:
            scoring_df[f'{col}_norm'] = 0.5
    
    # Normalize metrics (lower = better)
    for col in ['Annual_Vol', 'Max_Drawdown', 'Concentration']:
        col_series = scoring_df[col]
        if col == 'Max_Drawdown': col_series = abs(col_series)
        
        if col_series.max() - col_series.min() > 0:
            scoring_df[f'{col}_norm'] = (col_series.max() - col_series) / \
                                        (col_series.max() - col_series.min())
        else:
            scoring_df[f'{col}_norm'] = 0.5
    
    # Composite score (weighted)
    weights_scoring = {
        'Annual_Return_norm': 0.20,
        'Sharpe_Ratio_norm': 0.25,
        'Sortino_Ratio_norm': 0.15,
        'Calmar_Ratio_norm': 0.10,
        'Max_Drawdown_norm': 0.15,
        'Annual_Vol_norm': 0.05,
        'Win_Rate_norm': 0.05,
        'Effective_N_Assets_norm': 0.05
    }
    
    scoring_df['Composite_Score'] = sum(
        scoring_df[col] * weight for col, weight in weights_scoring.items()
    )
    
    ranking = scoring_df['Composite_Score'].sort_values(ascending=False)
    best_method = ranking.idxmax()
    best_weights = methods_weights[best_method]
    
    return {
        'best_method': best_method,
        'best_weights': best_weights,
        'all_weights': methods_weights,
        'oos_performance': oos_df,
        'ranking': ranking,
        'train_returns': train_returns,
        'test_returns': test_returns,
        'portfolio_series': portfolio_series,
        'composite_scores': scoring_df['Composite_Score']
    }

def monte_carlo_forecast_streamlit(returns, weights, n_simulations=5000, n_days=252):
    """
    Monte Carlo simulation for portfolio forecasting (Streamlit version).
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    simulation_results = np.zeros((n_simulations, n_days))
    final_values = []
    initial_capital = 100000
    
    # Progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for sim in range(n_simulations):
        if sim % 500 == 0:
            progress_bar.progress(sim / n_simulations)
            status_text.text(f"Running simulation {sim}/{n_simulations}")
        
        random_returns = np.random.normal(mean_return, std_return, n_days)
        capital_path = initial_capital * (1 + random_returns).cumprod()
        simulation_results[sim, :] = capital_path
        final_values.append(capital_path[-1])
    
    progress_bar.empty()
    status_text.empty()
    
    final_values = np.array(final_values)
    
    return {
        'simulation_results': simulation_results,
        'final_values': final_values,
        'mean_final': final_values.mean(),
        'median_final': np.median(final_values),
        'prob_profit': (final_values > initial_capital).sum() / n_simulations * 100,
        'var_95': initial_capital - np.percentile(final_values, 5),
        'expected_shortfall': initial_capital - final_values[final_values <= np.percentile(final_values, 5)].mean()
    }

st.set_page_config(layout="wide", page_title="Interactive DCA Simulation Tool")

page = st.sidebar.selectbox("Select Page", ["DCA Simulator", "Financial Analysis", "Portfolio Models"])

if page == "DCA Simulator":

    st.sidebar.header("Investment Parameters")

    with st.sidebar.form("investment_form"):
        ticker = st.text_input("Stock Ticker (e.g., AAPL)", config.DEFAULT_TICKER).upper()
        start_date = st.date_input("Start Date", datetime.datetime.strptime(config.DEFAULT_START_DATE, "%Y-%m-%d").date())
        initial_capital = st.number_input("Initial Capital ($)", min_value=0, value=config.DEFAULT_INITIAL_CAPITAL, step=100)
        regular_investment_amount = st.number_input("Regular Investment Amount ($)", min_value=0, value=config.DEFAULT_REGULAR_INVESTMENT_AMOUNT, step=10)
        investment_interval = st.selectbox("Investment Interval", ["Daily", "Weekly", "Bi-Weekly", "Monthly"], index=["Daily", "Weekly", "Bi-Weekly", "Monthly"].index(config.DEFAULT_INVESTMENT_INTERVAL))

        st.sidebar.header("Advanced Strategy: Buy on Dip")
        enable_buy_on_dip = st.checkbox("Enable 'Buy on Dip' Strategy", value=False)
        buy_on_dip_threshold = st.slider("Dip Threshold (%)", min_value=1, max_value=20, value=config.DEFAULT_BUY_ON_DIP_THRESHOLD, step=1, disabled=not enable_buy_on_dip)
        buy_on_dip_multiplier = st.slider("Investment Multiplier on Dip", min_value=1.0, max_value=5.0, value=config.DEFAULT_BUY_ON_DIP_MULTIPLIER, step=0.5, disabled=not enable_buy_on_dip)

        submitted = st.form_submit_button("Run Simulation")
    
    if submitted:
        st.write(f"Running simulation for {ticker} from {start_date} with initial capital of ${initial_capital} and regular investment of ${regular_investment_amount} {investment_interval}.")
        if enable_buy_on_dip:
            st.write(f"'Buy on Dip' enabled with threshold: {buy_on_dip_threshold}% and multiplier: {buy_on_dip_multiplier}x.")
        else:
            st.write("'Buy on Dip' strategy is disabled.")
    
        # Fetch historical data
        start_date_str = start_date.strftime("%Y-%m-%d")
        historical_data = fetch_historical_data(ticker, start_date_str)
        
        if historical_data.empty:
            st.error(f"No historical data found for ticker '{ticker}' starting from {start_date_str}. Please check the ticker and date.")
        else:
            # Run simulation
            buy_on_dip_thresh = buy_on_dip_threshold if enable_buy_on_dip else 0
            buy_on_dip_mult = buy_on_dip_multiplier if enable_buy_on_dip else 1.0
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
                st.subheader("Simulation Results")
                
                # Plot
                fig = plot_simulation_results(simulation_results)
                st.plotly_chart(fig, width='stretch')
                
                # Table
                st.subheader("Detailed Results Table")
                results_table = display_results_table(simulation_results)
                st.dataframe(results_table, width='stretch')

elif page == "Financial Analysis":
    st.title("📊 Financial Analysis")

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Stock Ticker", "AAPL").upper()
        start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("today"))
    with col2:
        forecast_days = st.number_input("Forecast Days", value=30, min_value=1, max_value=365)
        mc_simulations = st.number_input("Monte Carlo Simulations", value=1000, min_value=100, max_value=10000)
        mc_period = st.number_input("Monte Carlo Period (days)", value=30, min_value=1, max_value=365)
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
            
            # Returns plot
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
                "Value": [f"{returns.mean():.2%}", f"{returns.median():.2%}", f"{returns.std():.2%}", f"{returns.quantile(0.25):.2%}", f"{returns.quantile(0.75):.2%}"]
            }
            st.table(pd.DataFrame(stats_data))

            # Historical Price Distribution
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

            # Returns Distribution Analysis
            st.subheader("📊 Returns Distribution Analysis")

            # Calculate theoretical distributions
            from scipy import stats as scipy_stats

            # Fit normal distribution
            mu, sigma = scipy_stats.norm.fit(returns)

            # Fit t-distribution
            df, loc, scale = scipy_stats.t.fit(returns)

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
            y_normal = scipy_stats.norm.pdf(x_normal, mu, sigma)
            fig_returns.add_trace(go.Scatter(
                x=x_normal,
                y=y_normal,
                mode='lines',
                name=f'Normal (μ={mu:.3f}, σ={sigma:.3f})',
                line=dict(color='red', width=2, dash='dash')
            ))

            # T-distribution curve
            x_t = np.linspace(returns.min(), returns.max(), 100)
            y_t = scipy_stats.t.pdf(x_t, df, loc, scale)
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
                st.metric("Skewness", f"{scipy_stats.skew(returns):.3f}")

            with col2:
                st.metric("Volatility", f"{returns.std():.3f}")
                st.metric("Kurtosis", f"{scipy_stats.kurtosis(returns):.3f}")

            with col3:
                st.metric("Normal Fit μ", f"{mu:.3f}")
                st.metric("T-Dist df", f"{df:.1f}")

            # Distribution comparison
            st.markdown("### Distribution Fit Analysis")
            st.markdown(f"""
            **Normal Distribution:** μ = {mu:.3f}, σ = {sigma:.3f}
            **T-Distribution:** degrees of freedom = {df:.1f}

            **Interpretation:**
            - **Skewness**: {scipy_stats.skew(returns):.3f} ({'Right-skewed' if scipy_stats.skew(returns) > 0 else 'Left-skewed' if scipy_stats.skew(returns) < 0 else 'Symmetric'})
            - **Kurtosis**: {scipy_stats.kurtosis(returns):.3f} ({'Heavy tails' if scipy_stats.kurtosis(returns) > 0 else 'Light tails'})
            - **T-distribution** typically fits financial returns better due to fat tails
            """)

            # Unit root tests
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

            st.dataframe(test_df.style.applymap(style_tests, subset=['Conclusion']))
            
            # VaR with GARCH
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
            
            # Monte Carlo Simulation using GBM
            st.subheader("🎲 Monte Carlo Forecast")
            mc_results = monte_carlo_forecast_streamlit(
                returns.to_frame(),  # Convert to DataFrame for compatibility
                np.array([1.0]),     # Single stock, 100% weight
                n_simulations=mc_simulations,
                n_days=mc_period
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

            # Add VaR 95% line
            var_95_value = np.percentile(mc_results['final_values'], 5)  # 5th percentile = 95% VaR
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

            # Technical Analysis: Bollinger Bands, MACD & RSI
            st.subheader("📈 Technical Analysis: Bollinger Bands, MACD & RSI")

            # Calculate Bollinger Bands (20-day for short-term analysis)
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

            # Create subplots for technical analysis
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
            st.subheader("📊 Technical Analysis Summary")

            # Bollinger Bands Analysis
            current_price = data.iloc[-1]
            current_sma = sma_bb.iloc[-1]
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

            # Prophet Model
            df_prophet = pd.DataFrame({'ds': data.index, 'y': data.values})
            model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model_prophet.fit(df_prophet)

            future = model_prophet.make_future_dataframe(periods=forecast_days)
            forecast_prophet = model_prophet.predict(future)

            # Prophet Forecast
            st.subheader("🔮 Prophet Forecast")
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
            
            # Prophet Components (Plotly version - Trend, Weekly and Monthly seasonality)
            st.subheader("📈 Prophet Components (Trend & Seasonality)")

            # Extract components data
            components = forecast_prophet[['ds', 'trend', 'weekly', 'yearly']].copy()
            components = components.set_index('ds')

            # Create subplots for trend, weekly and monthly seasonality
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
            weekly_pattern = components['weekly'].tail(7)  # Last 7 days to show weekly pattern
            weekdays_only = weekly_pattern[weekly_pattern.index.dayofweek < 5]  # Monday=0, Friday=4

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

            # Monthly seasonality (average seasonal effect by month)
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
                model_hw = ExponentialSmoothing(data, seasonal='add', seasonal_periods=252)  # Approximate yearly seasonality
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

elif page == "Portfolio Models":
    st.title("📊 Portfolio Models")

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
                    
                    # Sharpe ratio
                    sharpe = (expected_return - risk_free_rate) / (stock_ret_aligned.std() * np.sqrt(252)) if stock_ret_aligned.std() > 0 else np.nan
                    
                    capm_results.append({"Ticker": ticker, "Beta": beta, "Expected Return": expected_return, "Sharpe Ratio": sharpe, "Error": None})
                
                # Display CAPM results
                df_capm = pd.DataFrame(capm_results)
                st.subheader("CAPM Results")
                st.dataframe(df_capm)
                
                # Plot
                valid_capm = df_capm.dropna(subset=['Beta'])
                if not valid_capm.empty:
                    fig_beta = px.bar(valid_capm, x='Ticker', y='Beta', title="Betas")
                    st.plotly_chart(fig_beta)
                    
                    fig_return = px.bar(valid_capm, x='Ticker', y='Expected Return', title="Expected Returns (CAPM)")
                    st.plotly_chart(fig_return)
                    
                    fig_sharpe = px.bar(valid_capm, x='Ticker', y='Sharpe Ratio', title="Sharpe Ratios")
                    st.plotly_chart(fig_sharpe)
                
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
                        max_sharpe_weights = pd.DataFrame({'Ticker': tickers, 'Weight': weights_record[max_sharpe_idx]})
                        st.subheader("Weights for Max Sharpe Portfolio")
                        st.dataframe(max_sharpe_weights)
                else:
                    st.info("Add more tickers for portfolio optimization.")
    
    # Extended Advanced Analysis Section
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
    
    col3, col4 = st.columns([1, 1])
    with col3:
        train_split = st.slider("Training Data Split (%)", 60, 90, 80, 5)
    with col4:
        mc_simulations = st.number_input("Monte Carlo Simulations", 1000, 10000, 5000, 1000)
    
    if st.button("🔬 Run Extended Analysis (Advanced)"):
        if not extended_tickers or len(extended_tickers) < 3:
            st.error("Please enter at least 3 tickers for meaningful portfolio analysis.")
        elif len(extended_tickers) > 15:
            st.error("Please use no more than 15 tickers to avoid computational issues.")
        else:
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
                            if not ticker_data.empty and len(ticker_data) > 100:  # Minimum 100 data points
                                data[ticker] = ticker_data  # Data is already a Series with prices
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
                        
                        # Create combined DataFrame more safely
                        combined_data = pd.DataFrame(data)
                        
                        # Remove any rows with all NaN values
                        combined_data = combined_data.dropna(how='all')
                        
                        # Forward fill and backward fill to handle missing values
                        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
                        
                        # Remove any remaining rows with NaN
                        combined_data = combined_data.dropna()
                        
                        if len(combined_data) < 504:  # Minimum 2 years of daily data
                            st.error(f"Need at least 2 years of data. Only {len(combined_data)} days available.")
                        else:
                            # Calculate returns
                            returns = combined_data.pct_change().dropna()
                            
                            if len(returns) < 500:
                                st.error("Insufficient return data after processing.")
                            else:
                                st.success(f"Data prepared: {len(returns)} days of returns for {len(returns.columns)} assets")
                                
                        # Fetch S&P 500 benchmark data
                        st.info("Fetching S&P 500 benchmark data...")
                        sp500_data = fetch_historical_data('^GSPC', extended_start_date.strftime("%Y-%m-%d"), extended_end_date.strftime("%Y-%m-%d"))
                        sp500_returns = None
                        if not sp500_data.empty and len(sp500_data) > 100:
                            sp500_returns = sp500_data.pct_change().dropna()
                            st.success("✅ S&P 500 benchmark data loaded")
                        else:
                            st.warning("⚠️ Could not load S&P 500 benchmark data")
                        
                        # Run comprehensive analysis
                        oos_results = extract_optimal_weights_with_oos(
                            returns, 
                            train_test_split=train_split/100
                        )
                        
                        best_method = oos_results['best_method']
                        best_weights = oos_results['best_weights']
                        
                        # Display results
                        st.success(f"✅ Analysis completed! Best method: **{best_method}**")
                        
                        # Performance Summary
                        st.subheader("📊 Performance Summary (Out-of-Sample)")
                        perf_df = oos_results['oos_performance'].loc[[best_method]]
                        st.dataframe(perf_df.style.format({
                            'Total_Return': '{:.1%}',
                            'Annual_Return': '{:.1%}', 
                            'Annual_Vol': '{:.1%}',
                            'Sharpe_Ratio': '{:.3f}',
                            'Max_Drawdown': '{:.1%}',
                            'Win_Rate': '{:.1%}'
                        }))
                        
                        # Method Comparison
                        st.subheader("🏆 Method Comparison")
                        ranking = oos_results['ranking']
                        comparison_df = pd.DataFrame({
                            'Rank': range(1, len(ranking) + 1),
                            'Method': ranking.index,
                            'Composite Score': ranking.values
                        }).set_index('Rank')
                        st.dataframe(comparison_df)
                        
                        # Portfolio Weights
                        st.subheader(f"📋 {best_method} Portfolio Weights")
                        weights_df = pd.DataFrame({
                            'Ticker': best_weights.index,
                            'Weight': best_weights.values,
                            'Weight %': best_weights.values * 100
                        })
                        st.dataframe(weights_df.style.format({'Weight %': '{:.2f}%'}))
                        
                        # Visualization: Cumulative Returns
                        st.subheader("📈 Cumulative Returns Comparison")
                        portfolio_series = oos_results['portfolio_series']
                        
                        fig_cum = go.Figure()
                        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
                        
                        # Add S&P 500 benchmark if available
                        if sp500_returns is not None:
                            # Align S&P 500 data with portfolio data
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
                        
                        # Stress Testing
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
                        
                        # S&P 500 Benchmark Comparison
                        if sp500_returns is not None:
                            st.subheader("🏁 S&P 500 Benchmark Comparison")
                            
                            # Calculate S&P 500 performance on test period
                            common_dates = sp500_returns.index.intersection(oos_results['test_returns'].index)
                            if len(common_dates) > 0:
                                sp500_test_returns = sp500_returns.loc[common_dates]
                                sp500_cumulative = (1 + sp500_test_returns).cumprod()
                                
                                sp500_total_return = sp500_cumulative.iloc[-1] - 1
                                sp500_annual_return = (1 + sp500_total_return) ** (252 / len(sp500_test_returns)) - 1
                                sp500_volatility = sp500_test_returns.std() * np.sqrt(252)
                                sp500_sharpe = (sp500_annual_return - 0.04) / sp500_volatility if sp500_volatility > 0 else 0
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("S&P 500 Annual Return", f"{sp500_annual_return:.1%}")
                                with col2:
                                    st.metric("S&P 500 Volatility", f"{sp500_volatility:.1%}")
                                with col3:
                                    st.metric("S&P 500 Sharpe", f"{sp500_sharpe:.3f}")
                                with col4:
                                    st.metric("Portfolio Outperformance", f"{(perf_df['Annual_Return'].iloc[0] - sp500_annual_return):+.1%}")
                        
                        # Returns Distribution with VaR
                        st.subheader("📊 Portfolio Returns Distribution (Out-of-Sample)")
                        test_portfolio_returns = (oos_results['test_returns'] * best_weights).sum(axis=1)
                        
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
                        
                        # Monte Carlo Forecast
                        st.subheader("🎲 Monte Carlo Forecast (1 Year)")
                        mc_results = monte_carlo_forecast_streamlit(
                            returns, 
                            best_weights, 
                            n_simulations=mc_simulations,
                            n_days=252
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
                        
                        # Add VaR 95% line
                        var_95_value = np.percentile(mc_results['final_values'], 5)  # 5th percentile = 95% VaR
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
                                    x=list(range(252)),
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
                                x=list(range(252)),
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
                                x=list(range(252)),
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
                        
                        # Recommendations
                        st.subheader("💡 Investment Recommendations")
                        st.markdown(f"""
                        **Selected Strategy:** {best_method}
                        
                        **Key Insights:**
                        - Out-of-sample annual return: {perf_df['Annual_Return'].iloc[0]:.1%}
                        - Sharpe ratio: {perf_df['Sharpe_Ratio'].iloc[0]:.3f}
                        - Maximum drawdown: {perf_df['Max_Drawdown'].iloc[0]:.1%}
                        - Diversification (effective assets): {perf_df['Effective_N_Assets'].iloc[0]:.1f}
                        
                        **Risk Considerations:**
                        - Daily VaR (95%): {stress_results['var_95']:.2%}
                        - Probability of 1-year profit: {mc_results['prob_profit']:.1f}%
                        
                        **Next Steps:**
                        1. Consider rebalancing quarterly
                        2. Monitor drawdown limits
                        3. Review annually or when market conditions change significantly
                        """)
                
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Try with fewer tickers or different date range.")
