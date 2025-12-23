"""
Portfolio optimization and stress testing utilities.

Functions for comparing multiple portfolio construction methods
with out-of-sample testing and performance evaluation.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .hrp import HierarchicalRiskParity


def stress_test_analysis(returns, weights):
    """
    Perform basic stress-test analysis for portfolio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    weights : pd.Series or dict
        Portfolio weights
    
    Returns:
    --------
    dict : Risk metrics including VaR, CVaR, skewness, kurtosis
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
    kurtosis = portfolio_returns.kurtosis()  # Excess kurtosis (above 3)

    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }


def extract_optimal_weights_with_oos(returns, train_test_split=0.8, risk_free_rate=0.04):
    """
    Calculate optimal weights with true train/test split.
    Automatically selects best method based on OOS performance.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    train_test_split : float
        % of data for training (e.g., 0.8 = 80% train, 20% test)
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation
    
    Returns:
    --------
    dict with:
        - best_method: Name of best performing method
        - best_weights: Weights for best method
        - all_weights: Dict of all method weights
        - oos_performance: DataFrame of all method metrics
        - ranking: Composite score ranking
        - train_returns: Training set returns
        - test_returns: Test set returns
        - portfolio_series: Dict of cumulative return series
        - composite_scores: Composite scores for all methods
    """
    
    # Split data (handle Production Mode where train_test_split >= 0.99)
    is_production_mode = train_test_split >= 0.99
    
    if is_production_mode:
        # Production Mode: Use ALL data for training and evaluation
        train_returns = returns.copy()
        test_returns = returns.copy()  # Use all data for "evaluation" too
    else:
        # Backtesting Mode: True train/test split
        split_idx = int(len(returns) * train_test_split)
        train_returns = returns.iloc[:split_idx]
        test_returns = returns.iloc[split_idx:]
    
    # Calculate weights using different methods ON TRAINING DATA
    methods_weights = {}
    
    # 1. HRP
    try:
        hrp = HierarchicalRiskParity(train_returns)
        hrp_weights, _ = hrp.get_hrp_weights()
        methods_weights['HRP'] = hrp_weights
    except Exception as e:
        print(f"HRP failed: {e}")
    
    # 2. Equal Weight
    ew_weights = pd.Series(1/len(train_returns.columns), index=train_returns.columns)
    methods_weights['Equal Weight'] = ew_weights
    
    # 3. Inverse Volatility
    train_vols = train_returns.std()
    if train_vols.sum() > 0:
        inv_vol_weights = (1 / train_vols) / (1 / train_vols).sum()
        methods_weights['Inverse Volatility'] = inv_vol_weights
    
    # 4. Mean-Variance (Markowitz - Max Sharpe)
    try:
        mean_ret = train_returns.mean() * 252
        
        # Ledoit-Wolf shrinkage for more stable covariance matrix
        lw = LedoitWolf()
        cov_mat_lw = pd.DataFrame(
            lw.fit(train_returns).covariance_ * 252,
            index=train_returns.columns,
            columns=train_returns.columns
        )
        
        def neg_sharpe(weights, mean_returns, cov_matrix, rf):
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
            args=(mean_ret, cov_mat_lw, risk_free_rate),
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000}
        )
        if result_mv.success:
            mv_weights = pd.Series(result_mv.x, index=train_returns.columns)
            methods_weights['Mean-Variance (Markowitz)'] = mv_weights
    except Exception as e:
        print(f"Mean-Variance optimization failed: {e}")
    
    # 5. Minimum Variance
    try:
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
        if result_minvar.success:
            minvar_weights = pd.Series(result_minvar.x, index=train_returns.columns)
            methods_weights['Min Variance'] = minvar_weights
    except Exception as e:
        print(f"Min Variance optimization failed: {e}")
    
    # 6. Risk Parity
    try:
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
        if result_rp.success:
            rp_weights = pd.Series(result_rp.x / result_rp.x.sum(), index=train_returns.columns)
            methods_weights['Risk Parity'] = rp_weights
    except Exception as e:
        print(f"Risk Parity optimization failed: {e}")
    
    # OUT-OF-SAMPLE TESTING
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
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        max_dd = ((cumulative_returns - cumulative_returns.expanding().max()) / 
                  cumulative_returns.expanding().max()).min()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        downside_returns = test_portfolio_returns[test_portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0
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
    
    # SELECT BEST METHOD using multi-criteria scoring
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
        if col == 'Max_Drawdown':
            col_series = abs(col_series)
        
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
