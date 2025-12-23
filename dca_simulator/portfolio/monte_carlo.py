"""
Monte Carlo simulation for portfolio forecasting.

Provides Monte Carlo simulation capabilities with Streamlit progress tracking.
Supports both Normal and t-Student distributions for more realistic modeling.
"""

import streamlit as st
import numpy as np
from scipy import stats


def monte_carlo_forecast_streamlit(returns, weights, n_simulations=5000, n_days=252, 
                                   initial_capital=100000, distribution='normal'):
    """
    Monte Carlo simulation for portfolio forecasting with Streamlit progress bar.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns for assets
    weights : pd.Series or array-like
        Portfolio weights
    n_simulations : int
        Number of Monte Carlo paths to simulate
    n_days : int
        Forecast horizon in days
    initial_capital : float
        Starting portfolio value
    distribution : str
        Distribution type: 'normal' or 't-student' (default: 'normal')
    
    Returns:
    --------
    dict with:
        - simulation_results: 2D array of all simulation paths
        - final_values: Array of final portfolio values
        - mean_final: Mean final value
        - median_final: Median final value
        - prob_profit: Probability of profit (%)
        - var_95: Value at Risk at 95% confidence
        - expected_shortfall: Expected shortfall (CVaR)
        - distribution_params: Dict with fitted distribution parameters
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    # Fit t-Student distribution if requested
    df_param = None
    loc_param = None
    scale_param = None
    
    if distribution == 't-student':
        df_param, loc_param, scale_param = stats.t.fit(portfolio_returns)
    
    simulation_results = np.zeros((n_simulations, n_days))
    final_values = []
    
    # Progress bar for Streamlit
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for sim in range(n_simulations):
            if sim % 500 == 0:
                progress = sim / n_simulations
                progress_bar.progress(progress)
                status_text.text(f"Running {distribution} simulation {sim}/{n_simulations}")
            
            # Generate returns based on selected distribution
            if distribution == 't-student':
                random_returns = stats.t.rvs(df_param, loc=loc_param, scale=scale_param, size=n_days)
            else:  # normal
                random_returns = np.random.normal(mean_return, std_return, n_days)
            
            capital_path = initial_capital * (1 + random_returns).cumprod()
            simulation_results[sim, :] = capital_path
            final_values.append(capital_path[-1])
    finally:
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
    
    final_values = np.array(final_values)
    
    # Calculate risk metrics
    var_95_percentile = np.percentile(final_values, 5)
    tail_losses = final_values[final_values <= var_95_percentile]
    expected_shortfall_value = initial_capital - tail_losses.mean() if len(tail_losses) > 0 else 0
    
    result = {
        'simulation_results': simulation_results,
        'final_values': final_values,
        'mean_final': final_values.mean(),
        'median_final': np.median(final_values),
        'prob_profit': (final_values > initial_capital).sum() / n_simulations * 100,
        'var_95': initial_capital - var_95_percentile,
        'expected_shortfall': expected_shortfall_value,
        'distribution_params': {}
    }
    
    if distribution == 't-student':
        result['distribution_params'] = {
            'df': df_param,
            'loc': loc_param,
            'scale': scale_param
        }
    else:
        result['distribution_params'] = {
            'mean': mean_return,
            'std': std_return
        }
    
    return result


def monte_carlo_forecast_no_ui(returns, weights, n_simulations=5000, n_days=252, 
                                initial_capital=100000, distribution='normal'):
    """
    Monte Carlo simulation without UI progress tracking (for non-Streamlit use).
    Vectorized implementation for faster execution.
    
    Parameters and Returns same as monte_carlo_forecast_streamlit.
    """
    portfolio_returns = (returns * weights).sum(axis=1)
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    # Fit t-Student distribution if requested
    df_param = None
    loc_param = None
    scale_param = None
    
    if distribution == 't-student':
        df_param, loc_param, scale_param = stats.t.fit(portfolio_returns)
    
    # Vectorized simulation (much faster)
    if distribution == 't-student':
        random_returns = stats.t.rvs(df_param, loc=loc_param, scale=scale_param, 
                                      size=(n_simulations, n_days))
    else:  # normal
        random_returns = np.random.normal(mean_return, std_return, (n_simulations, n_days))
    
    simulation_results = initial_capital * np.cumprod(1 + random_returns, axis=1)
    final_values = simulation_results[:, -1]
    
    # Calculate risk metrics
    var_95_percentile = np.percentile(final_values, 5)
    tail_losses = final_values[final_values <= var_95_percentile]
    expected_shortfall_value = initial_capital - tail_losses.mean() if len(tail_losses) > 0 else 0
    
    result = {
        'simulation_results': simulation_results,
        'final_values': final_values,
        'mean_final': final_values.mean(),
        'median_final': np.median(final_values),
        'prob_profit': (final_values > initial_capital).sum() / n_simulations * 100,
        'var_95': initial_capital - var_95_percentile,
        'expected_shortfall': expected_shortfall_value,
        'distribution_params': {}
    }
    
    if distribution == 't-student':
        result['distribution_params'] = {
            'df': df_param,
            'loc': loc_param,
            'scale': scale_param
        }
    else:
        result['distribution_params'] = {
            'mean': mean_return,
            'std': std_return
        }
    
    return result
