"""
Hierarchical Risk Parity (HRP) implementation.

Exact implementation of the algorithm from Marcos López de Prado's paper:
"Building Diversified Portfolios that Outperform Out-of-Sample" (2016)

Algorithm steps:
1. Tree Clustering: Hierarchical clustering using correlation-based distance
2. Quasi-Diagonalization: Reorder covariance matrix by cluster hierarchy
3. Recursive Bisection: Top-down allocation using inverse-variance weighting

Reference: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678
"""

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class HierarchicalRiskParity:
    """
    Implementation of Hierarchical Risk Parity (HRP) algorithm.
    
    This follows the exact methodology from López de Prado (2016) with:
    - Proper correlation-based distance metric
    - Single-linkage hierarchical clustering
    - Recursive bisection with inverse-variance weighting
    """
    
    def __init__(self, returns):
        """
        Initialize HRP with returns data.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns for assets (columns = assets, rows = time periods)
        
        Raises:
        -------
        TypeError: If returns is not a DataFrame
        ValueError: If returns has less than 2 assets or invalid data
        """
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("Returns must be in pd.DataFrame format.")
        
        if returns.empty or returns.shape[1] < 2:
            raise ValueError("Returns must have at least 2 assets.")
        
        self.returns = returns
        self.cov_matrix = self.returns.cov()
        self.corr_matrix = self.returns.corr()
        self.tickers = self.returns.columns.tolist()
        
        # Validate covariance matrix (check for zero or negative variances)
        variances = np.diag(self.cov_matrix)
        if np.any(variances <= 0):
            raise ValueError("Invalid covariance matrix: zero or negative variances detected.")

    def get_linkage_matrix(self):
        """
        Step 1: Tree Clustering
        
        Calculate hierarchical clustering linkage matrix using correlation-based distance.
        
        Distance metric: d_ij = sqrt((1 - ρ_ij) / 2)
        where ρ_ij is the Pearson correlation between assets i and j.
        
        This distance metric:
        - Ranges from 0 (perfect correlation) to 1 (perfect negative correlation)
        - Satisfies metric properties (non-negative, symmetric, triangle inequality)
        - Is used in López de Prado's original implementation
        
        Returns:
        --------
        linkage_matrix : ndarray
            Hierarchical clustering linkage matrix using single-linkage method
        """
        # Correlation-based distance (López de Prado formula)
        dist = np.sqrt(0.5 * (1 - self.corr_matrix))
        
        # Clip to avoid numerical issues with floating point
        dist = np.clip(dist, 0, 1)
        
        # Convert to condensed distance matrix (required by scipy)
        dist_condensed = squareform(dist, checks=False)
        
        # Single linkage clustering (as specified in López de Prado)
        return linkage(dist_condensed, method='single')

    def get_quasi_diag_matrix(self, link):
        """
        Step 2: Quasi-Diagonalization
        
        Reorganize the covariance matrix to be quasi-diagonal by reordering assets
        according to the hierarchical clustering structure.
        
        This method traverses the dendrogram tree and produces a list of asset indices
        that, when used to reorder the covariance matrix, creates a block-diagonal structure
        where similar assets are grouped together.
        
        Parameters:
        -----------
        link : ndarray
            Linkage matrix from hierarchical clustering
        
        Returns:
        --------
        list : Ordered list of asset indices for quasi-diagonal covariance matrix
        
        Algorithm:
        ----------
        - Start from root node (contains all assets)
        - Recursively split into left and right children
        - Continue until reaching leaf nodes (individual assets)
        - Result is a permutation that groups similar assets together
        """
        link = link.astype(int)
        
        # Initialize with the two children of the root node
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]  # Total number of original items

        # Recursively expand cluster nodes into their children
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            
            # Find cluster nodes (indices >= num_items)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            
            # Replace cluster nodes with their left children
            sort_ix[i] = link[j, 0]
            
            # Insert right children
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def get_recursive_bisection_weights(self, sort_ix):
        """
        Step 3: Recursive Bisection
        
        Calculate portfolio weights using recursive bisection with inverse-variance weighting.
        
        This is the core HRP allocation algorithm that:
        1. Starts with equal weight (1.0) at the top level
        2. Recursively divides the portfolio into left and right halves
        3. At each split, allocates weight proportional to inverse-variance of each half
        4. Continues until individual asset weights are determined
        
        Parameters:
        -----------
        sort_ix : list
            Ordered list of asset indices from quasi-diagonalization
        
        Returns:
        --------
        pd.Series : Portfolio weights indexed by ticker symbols
        
        Algorithm (López de Prado 2016):
        --------------------------------
        For each cluster split:
        - Calculate left cluster inverse-variance weight: w_L = 1 / σ²_L
        - Calculate right cluster inverse-variance weight: w_R = 1 / σ²_R
        - Normalize: α_L = w_L / (w_L + w_R), α_R = w_R / (w_L + w_R)
        - Allocate: weight_L = α_L × parent_weight, weight_R = α_R × parent_weight
        - Recurse on left and right sub-clusters
        
        Where σ²_i is the portfolio variance of cluster i calculated as:
        σ²_i = w_i^T Σ_i w_i
        with w_i being the inverse-variance weights within cluster i
        """
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
                
                # Left cluster: Calculate inverse-variance portfolio variance
                cov_sub = self.cov_matrix.iloc[c_items0, c_items0]
                inv_diag = 1 / np.diag(cov_sub.values)
                w0 = inv_diag / inv_diag.sum()
                v0 = np.dot(w0, np.dot(cov_sub, w0))

                # Right cluster: Calculate inverse-variance portfolio variance
                cov_sub = self.cov_matrix.iloc[c_items1, c_items1]
                inv_diag = 1 / np.diag(cov_sub.values)
                w1 = inv_diag / inv_diag.sum()
                v1 = np.dot(w1, np.dot(cov_sub, w1))

                # Allocation factor: Lower variance gets higher weight
                # alpha = v1 / (v0 + v1) means left gets inverse proportion
                alpha = 1 - v0 / (v0 + v1) if v0 + v1 != 0 else 0.5
                
                weights[c_items0] *= alpha
                weights[c_items1] *= 1 - alpha
        
        return weights

    def get_hrp_weights(self):
        """
        Main function to calculate HRP weights.
        
        Executes the complete HRP algorithm:
        1. Build hierarchical clustering tree (tree clustering)
        2. Reorganize covariance matrix (quasi-diagonalization)
        3. Allocate weights recursively (recursive bisection)
        
        Returns:
        --------
        tuple : (hrp_weights, linkage_matrix)
            - hrp_weights: pd.Series of weights sorted by ticker name
            - linkage_matrix: scipy linkage matrix for dendrogram visualization
        
        Example:
        --------
        >>> hrp = HierarchicalRiskParity(returns_df)
        >>> weights, linkage = hrp.get_hrp_weights()
        >>> print(weights)
        AAPL    0.15
        MSFT    0.12
        ...
        """
        # Step 1: Tree Clustering
        link = self.get_linkage_matrix()
        
        # Step 2: Quasi-Diagonalization
        sort_ix = self.get_quasi_diag_matrix(link)
        
        # Step 3: Recursive Bisection
        sorted_tickers = [self.tickers[i] for i in sort_ix]
        hrp_weights = self.get_recursive_bisection_weights(sort_ix)
        hrp_weights.index = sorted_tickers
        
        # Ensure weights are sorted alphabetically like original data
        hrp_weights = hrp_weights.sort_index()
        
        return hrp_weights, link
