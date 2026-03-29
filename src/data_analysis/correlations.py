import pandas as pd
import numpy as np
from scipy.stats import kruskal, chi2_contingency




def correlation_quanti_def_KW(database: pd.DataFrame,
                              continuous_vars: list,
                              target: str) -> pd.DataFrame:
    """
    Compute Kruskal-Wallis test p-values between continuous variables
    and a categorical (binary or multi-class) target.

    Parameters
    ----------
    database : pd.DataFrame
        Input dataset
    continuous_vars : list
        List of continuous variable names
    target : str
        Target variable name (categorical)

    Returns
    -------
    pd.DataFrame
        Table with variables and corresponding p-values
    """

    results = []

    for var in continuous_vars:
        # Drop NA for current variable + target
        df = database[[var, target]].dropna()

        # Group values by target categories
        groups = [
            group[var].values
            for _, group in df.groupby(target)
        ]

        # Kruskal-Wallis requires at least 2 groups
        if len(groups) < 2:
            p_value = None
        else:
            try:
                stat, p_value = kruskal(*groups)
            except ValueError:
                # Handles edge cases (e.g., constant values)
                p_value = None

        results.append({
            "variable": var,
            "p_value": p_value,
            "stats_kw": stat if 'stat' in locals() else None
        })
       
    return pd.DataFrame(results).sort_values(by="p_value")


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(database: pd.DataFrame, var1: str, var2: str) -> float:
    """
    Compute Cramér's V between two categorical variables.

    Parameters
    ----------
    database : pd.DataFrame
        Input dataset
    var1 : str
        First categorical variable
    var2 : str
        Second categorical variable

    Returns
    -------
    float
        Cramér's V coefficient
    """

    # Drop missing values
    df = database[[var1, var2]].dropna()

    # Contingency table
    contingency_table = pd.crosstab(df[var1], df[var2])

    # Chi-square test
    chi2, _, _, _ = chi2_contingency(contingency_table)

    # Total sample size
    n = contingency_table.sum().sum()

    # Dimensions
    r, k = contingency_table.shape

    # Cramér's V
    v = np.sqrt((chi2 / n) / min(k - 1, r - 1))

    return v




def cramers_v_with_target(database: pd.DataFrame,
                          categorical_vars: list,
                          target: str) -> pd.DataFrame:
    """
    Compute Chi-square statistic and Cramér's V between multiple
    categorical variables and a target variable.

    Parameters
    ----------
    database : pd.DataFrame
        Input dataset
    categorical_vars : list
        List of categorical variables
    target : str
        Target variable (categorical)

    Returns
    -------
    pd.DataFrame
        Table with variable, chi2 and Cramér's V
    """

    results = []

    for var in categorical_vars:
        # Drop missing values
        df = database[[var, target]].dropna()

        # Contingency table
        contingency_table = pd.crosstab(df[var], df[target])

        # Skip if not enough data
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            results.append({
                "variable": var,
                "chi2": None,
                "cramers_v": None
            })
            continue

        try:
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.values.sum()
            r, k = contingency_table.shape

            v = np.sqrt((chi2 / n) / min(k - 1, r - 1))

        except Exception:
            chi2, v = None, None

        results.append({
            "variable": var,
            "chi2": chi2,
            "cramers_v": v
        })

    result_df = pd.DataFrame(results)

    # Option : tri par importance
    return result_df.sort_values(by="cramers_v", ascending=False)



import pandas as pd

def correlation_matrix_quanti(database: pd.DataFrame,
                              continuous_vars: list,
                              method: str = "spearman",
                              as_percent: bool = False) -> pd.DataFrame:
    """
    Compute correlation matrix for continuous variables.

    Parameters
    ----------
    database : pd.DataFrame
        Input dataset
    continuous_vars : list
        List of continuous variables
    method : str
        Correlation method ("pearson" or "spearman"), default = "spearman"
    as_percent : bool
        If True, return values in percentage

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """

    # Select relevant data and drop rows with NA
    df = database[continuous_vars].dropna()

    # Compute correlation matrix
    corr_matrix = df.corr(method=method)

    # Convert to percentage if required
    if as_percent:
        corr_matrix = corr_matrix * 100

    return corr_matrix



def cramers_v_matrix(database: pd.DataFrame,
                     categorical_vars: list,
                     corrected: bool = False,
                     as_percent: bool = False) -> pd.DataFrame:
    """
    Compute Cramér's V correlation matrix for categorical variables.

    Parameters
    ----------
    database : pd.DataFrame
        Input dataset
    categorical_vars : list
        List of categorical variables
    corrected : bool
        Apply bias correction (recommended)
    as_percent : bool
        Return values in percentage

    Returns
    -------
    pd.DataFrame
        Cramér's V matrix
    """

    def cramers_v(x, y):
        # Drop NA
        df = pd.DataFrame({"x": x, "y": y}).dropna()

        contingency_table = pd.crosstab(df["x"], df["y"])

        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            return np.nan

        chi2, _, _, _ = chi2_contingency(contingency_table)
        n = contingency_table.values.sum()
        r, k = contingency_table.shape

        phi2 = chi2 / n

        if corrected:
            # Bergsma correction
            phi2_corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
            r_corr = r - ((r-1)**2) / (n-1)
            k_corr = k - ((k-1)**2) / (n-1)
            denom = min(k_corr - 1, r_corr - 1)
        else:
            denom = min(k - 1, r - 1)

        if denom <= 0:
            return np.nan

        return np.sqrt(phi2_corr / denom) if corrected else np.sqrt(phi2 / denom)

    # Initialize matrix
    n = len(categorical_vars)
    matrix = pd.DataFrame(np.zeros((n, n)),
                          index=categorical_vars,
                          columns=categorical_vars)

    # Fill matrix
    for i, var1 in enumerate(categorical_vars):
        for j, var2 in enumerate(categorical_vars):
            if i <= j:
                value = cramers_v(database[var1], database[var2])
                matrix.loc[var1, var2] = value
                matrix.loc[var2, var1] = value

    # Convert to percentage
    if as_percent:
        matrix = matrix * 100

    return matrix