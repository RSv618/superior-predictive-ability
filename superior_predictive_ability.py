import numpy as np
import pandas as pd
from scipy.stats import norm

def superior_predictive_ability(log_returns_df: pd.DataFrame, risk_free_rate: float=0.0,
                                null_sharpe: float=0.0, alpha: float=0.05) -> list[str]:
    """
    Perform Hansen's Superior Predictive Ability (SPA) test on a set of strategies.

    The SPA test evaluates the predictive performance of multiple strategies by comparing their
    Sharpe ratios against a specified null Sharpe ratio, accounting for correlation between strategies.
    It adjusts for multiple comparisons and identifies strategies with statistically significant out-performance.
    This version doesn't use bootstrapping and uses a loglog correction.

    Args:
    log_returns_df (pd.DataFrame) : DataFrame of log returns, where each column represents a strategy and
    each row represents a time period (index should be timestamps or periods).

    risk_free_rate (float, optional) : The risk-free rate to compute excess returns.
    Default is 0, assuming returns are already excess returns.

    null_sharpe (float, optional) : The hypothesized Sharpe ratio under the null hypothesis.
    Default is 0, testing for performance above zero.

    alpha (float, optional) : The significance level for the test, default is 0.05.

    Returns:
    list[str] : A list of strategy names (column names from `log_returns_df`) that are statistically
    significant at the given alpha level. If no strategies meet the significance threshold, an empty list is returned.

    Notes:
    This function implements Hansen's SPA test by:
    1. Calculating excess Sharpe ratios.
    2. Transforming Sharpe ratios to account for correlation using
    Equation 15 from Pav, S. E. (2019) "Conditional inference on the asset with maximum Sharpe ratio".
    3. Applying a threshold to identify high-performing strategies.
    4. Adjusting for multiple comparisons and identifying statistically significant strategies.

    References:
    - Pav, S. E. (2019). "Conditional inference on the asset with maximum Sharpe ratio".
    - Hansen, P. R. (2005). "A Test for Superior Predictive Ability". Journal of Business & Economic Statistics.
    """
    n, n_strategies = log_returns_df.shape

    # Calculate excess returns
    excess_returns: pd.DataFrame = log_returns_df - risk_free_rate
    mean_returns: pd.Series = excess_returns.mean(axis=0)
    std_devs: pd.Series = excess_returns.std(axis=0)
    sharpe_ratios: pd.Series = mean_returns / std_devs
    strategies_list: list[str] = sharpe_ratios.sort_values(ascending=False).index.to_list()
    correlation_matrix: np.ndarray = np.corrcoef(log_returns_df[strategies_list].values, rowvar=False)

    # iterative removal (recursive hypothesis testing until all significant strategies are identified)
    statistically_significant: list[str] = []
    for i in range(n_strategies - 2):
        # define subset of the data
        subset_strategies_list: list[str] = strategies_list[i:]
        subset_n_strategies: int = len(subset_strategies_list)
        subset_sharpe_ratios: pd.Series = sharpe_ratios[subset_strategies_list]

        # Transform the array of sharpe_ratios to array xi
        zeta_bar: float = float(np.mean(subset_sharpe_ratios))
        zeta_ones_like: pd.Series = pd.Series(zeta_bar * np.ones_like(subset_sharpe_ratios),
                                              index=subset_strategies_list)
        rho: float = float(np.mean(correlation_matrix[i][i + 1:]))
        c: float = 1.0 / np.sqrt(1.0 + (subset_n_strategies - 1.0) * rho)
        term1: pd.Series = c * zeta_ones_like
        term2: pd.Series = (1.0 / np.sqrt(1.0 - rho)) * (subset_sharpe_ratios - zeta_ones_like)
        xi: pd.Series = term1 + term2

        # Rejection threshold with loglog correction
        initial_threshold: float = c * null_sharpe - np.sqrt(2 * np.log(np.log(n)) / n)
        strategies_passed_list: list = xi[xi>initial_threshold].index.to_list()
        k_tilde: int = len(strategies_passed_list)
        if k_tilde == 0:
            print(f'No statistically significant strategies found for the set {subset_strategies_list}.')
            return statistically_significant

        if max(xi) - c * null_sharpe >= norm.ppf(1.0 - alpha / k_tilde):
            statistically_significant.append(subset_strategies_list[0])
            print(f'{subset_strategies_list[0]} is statistically significant')
        else:
            if len(statistically_significant) == 0:
                print(f'No statistically significant strategies found.')
            return statistically_significant

if __name__=='__main__':
    # Suppose log_returns_df is your DataFrame of log returns with each column representing a strategy
    log_returns: pd.DataFrame = pd.read_csv(r'log_returns_matrix.csv', index_col='timestamp')
    statistically_significant_results: list[str] = superior_predictive_ability(log_returns, alpha=0.05)
