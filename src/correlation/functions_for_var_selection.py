from itertools import combinations
from functools import reduce
from pathlib import Path
import os
import pickle
import pandas as pd
from scipy.stats import kruskal


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def kruskal_pvalue(df: pd.DataFrame, var: str, target: str) -> float:
    """
    Compute the Kruskal-Wallis p-value between a continuous variable
    and a binary target. A low p-value indicates a strong association
    with the target (default).

    Args:
        df     : DataFrame containing both columns
        var    : Continuous variable name
        target : Binary target column name (e.g. 'def_year')

    Returns:
        p-value of the Kruskal-Wallis test
    """
    groups = [group[var].dropna().values for _, group in df.groupby(target)]
    _, pvalue = kruskal(*groups)
    return pvalue


def get_vars_to_drop(
    df: pd.DataFrame,
    variables: list[str],
    target: str,
    threshold: float,
) -> set[str]:
    """
    Identify variables to drop among highly correlated pairs.

    For each pair of variables whose Spearman correlation exceeds
    the threshold, drop the one that is least associated with the
    target — i.e. the one with the highest Kruskal-Wallis p-value.

    Args:
        df        : Training fold DataFrame
        variables : List of continuous candidate variables
        target    : Binary target column name
        threshold : Spearman correlation threshold (e.g. 0.5)

    Returns:
        Set of variable names to be dropped
    """
    # Compute Kruskal-Wallis p-values once per variable to avoid redundant calls
    pvalues = {v: kruskal_pvalue(df, v, target) for v in variables}

    # Absolute Spearman correlation matrix across all candidate variables
    corr_matrix = df[variables].corr(method="spearman").abs()

    # For each highly correlated pair, drop the variable least linked to default
    # i.e. the one with the highest p-value (least significant)
    return {
        max(v1, v2, key=lambda v: pvalues[v])
        for v1, v2 in combinations(variables, 2)
        if corr_matrix.loc[v1, v2] >= threshold
    }


# ─────────────────────────────────────────────
# FOLD CONSTRUCTION & PERSISTENCE
# ─────────────────────────────────────────────

def build_and_save_folds(
    df: pd.DataFrame,
    fold_col: str = "fold",
    save_dir: str = "folds/",
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build (train_i, test_i) pairs from a pre-computed fold column
    and persist each pair to disk as a pickle file.

    For fold i:
      - train_i = all rows where fold != i  (k-1 blocks)
      - test_i  = all rows where fold == i  (1 block)

    Args:
        df       : DataFrame containing the fold assignment column
        fold_col : Name of the column holding fold indices (default: 'fold')
        save_dir : Directory where fold files will be saved

    Returns:
        Dictionary mapping each fold index to its (train, test) tuple
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build all (train, test) pairs in a single comprehension
    folds = {
        fold_i: (
            df[df[fold_col] != fold_i].drop(columns=fold_col),  # train: all folds except i
            df[df[fold_col] == fold_i].drop(columns=fold_col),  # test:  fold i only
        )
        for fold_i in sorted(df[fold_col].unique())
    }

    # Persist each fold pair — pickle preserves dtypes and index integrity
    [
        pickle.dump(
            {"train": train, "test": test},
            open(f"{save_dir}fold_{fold_i}.pkl", "wb"),
        )
        for fold_i, (train, test) in folds.items()
    ]

    return folds


def load_folds(
    save_dir: str = "folds/",
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Reload persisted fold pairs from disk.

    Expects files named fold_0.pkl, fold_1.pkl, ... in save_dir,
    as produced by build_and_save_folds().

    Args:
        save_dir : Directory containing the fold pickle files

    Returns:
        Dictionary mapping each fold index to its (train, test) tuple
    """
    return {
        int(f.stem.split("_")[1]): (
            pickle.load(open(f, "rb"))["train"],
            pickle.load(open(f, "rb"))["test"],
        )
        for f in sorted(Path(save_dir).glob("fold_*.pkl"))
    }


# ─────────────────────────────────────────────
# VARIABLE FILTER
# ─────────────────────────────────────────────

def filter_correlated_variables_kfold(
    folds: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    variables: list[str],
    target: str,
    threshold: float = 0.5,
) -> list[str]:
    """
    Select variables whose pairwise Spearman correlation stays below
    the threshold across ALL folds.

    Robustness rule: a variable is dropped if it is flagged on AT LEAST
    ONE fold. Among a correlated pair, the variable with the weakest
    association to default (highest Kruskal-Wallis p-value) is removed.

    Args:
        folds     : Dictionary {fold_i: (train_i, test_i)} as returned
                    by build_and_save_folds() or load_folds()
        variables : List of continuous candidate variable names
        target    : Binary target column name (e.g. 'def_year')
        threshold : Spearman correlation threshold (default: 0.5)

    Returns:
        Filtered list of variable names
    """
    # Apply the correlation filter on the training set of each fold only —
    # the test set is never used during variable selection to avoid leakage
    vars_to_drop_per_fold = [
        get_vars_to_drop(train, variables, target, threshold)
        for train, _ in folds.values()
    ]

    # Union across folds: drop a variable if flagged on at least one fold
    vars_to_drop = reduce(lambda a, b: a | b, vars_to_drop_per_fold)

    # Preserve original variable ordering
    return [v for v in variables if v not in vars_to_drop]


def filter_uncorrelated_with_target(
    folds: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    variables: list[str],
    target: str,
    pvalue_threshold: float = 0.05,
) -> list[str]:
    """
    Apply Rule 1 of the filter method: drop any continuous variable that
    shows no significant association with the target on AT LEAST ONE fold.

    A variable is considered non-significant on a fold when its
    Kruskal-Wallis p-value against the binary target exceeds the threshold.

    Args:
        folds             : Dictionary {fold_i: (train_i, test_i)}
        variables         : List of continuous candidate variable names
        target            : Binary target column name (e.g. 'def_year')
        pvalue_threshold  : Significance threshold (default: 0.05)

    Returns:
        List of variables significantly associated with the target
        on ALL folds.
    """
    # Compute Kruskal-Wallis p-value for every (variable, fold) combination
    # Shape: dict {var: [pvalue_fold0, pvalue_fold1, ...]}
    pvalues_per_fold = {
        var: [
            kruskal_pvalue(train, var, target)
            for train, _ in folds.values()
        ]
        for var in variables
    }

    # Keep a variable only if it is significant (p < threshold) on ALL folds
    # i.e. drop it if p >= threshold on at least one fold
    return [
        var for var, pvalues in pvalues_per_fold.items()
        if all(p < pvalue_threshold for p in pvalues)
    ]


from scipy.stats import chi2_contingency
import numpy as np


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def cramers_v(df: pd.DataFrame, var: str, target: str) -> float:
    """
    Compute Cramér's V association measure between a categorical variable
    and a binary target.

    Cramér's V ranges from 0 (no association) to 1 (perfect association).
    It is derived from the Chi-squared statistic and corrected for table size.

    Args:
        df     : DataFrame containing both columns
        var    : Categorical variable name
        target : Binary target column name (e.g. 'def_year')

    Returns:
        Cramér's V coefficient (float between 0 and 1)
    """
    # Build contingency table between the categorical variable and the target
    contingency_table = pd.crosstab(df[var], df[target])

    # Chi-squared statistic and total number of observations
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()

    # Number of rows and columns in the contingency table
    r, k = contingency_table.shape

    # Cramér's V formula with bias correction
    phi2 = chi2 / n
    phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)

    return np.sqrt(phi2_corr / min(r_corr - 1, k_corr - 1))


# ─────────────────────────────────────────────
# RULE 2 — Categorical variables filter
# ─────────────────────────────────────────────

def filter_categorical_variables(
    folds: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    cat_variables: list[str],
    target: str,
    low_threshold: float = 0.10,
    high_threshold: float = 0.50,
) -> list[str]:
    """
    Apply Rule 2 of the filter method: drop any categorical variable whose
    association with the target is weak on AT LEAST ONE fold.

    Two thresholds structure the decision:
      - low_threshold  : below this, the association is considered negligible
                         → variable is dropped if V < low_threshold on any fold
      - high_threshold : above this, the association is considered strong
                         → used downstream for inter-variable redundancy (Rule 3)

    A variable is kept only if its Cramér's V with the target exceeds
    low_threshold on ALL folds.

    Args:
        folds          : Dictionary {fold_i: (train_i, test_i)}
        cat_variables  : List of categorical candidate variable names
        target         : Binary target column name (e.g. 'def_year')
        low_threshold  : Minimum Cramér's V to consider the variable relevant
                         (default: 0.10)
        high_threshold : Cramér's V above which association is deemed strong
                         (default: 0.50) — informational, logged only here

    Returns:
        List of categorical variables sufficiently associated with the target
        on ALL folds.
    """
    # Compute Cramér's V for every (variable, fold) combination
    # Shape: dict {var: [v_fold0, v_fold1, ...]}
    cramers_per_fold = {
        var: [
            cramers_v(train, var, target)
            for train, _ in folds.values()
        ]
        for var in cat_variables
    }

    # Log association strength for interpretability
    for var, v_values in cramers_per_fold.items():
        mean_v = np.mean(v_values)
        strength = (
            "strong"   if mean_v >= high_threshold else
            "moderate" if mean_v >= low_threshold  else
            "weak"
        )
        print(f"  {var:30s} | mean V = {mean_v:.3f} | {strength}")

    # Keep a variable only if V >= low_threshold on ALL folds
    # i.e. drop it if V < low_threshold on at least one fold
    return [
        var for var, v_values in cramers_per_fold.items()
        if all(v >= low_threshold for v in v_values)
    ]


def filter_correlated_categorical_variables(
    folds: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    cat_variables: list[str],
    target: str,
    high_threshold: float = 0.50,
) -> list[str]:
    """
    Apply Rule 4 of the filter method: drop one variable from each pair
    of categorical variables whose Cramér's V exceeds the threshold on
    AT LEAST ONE fold.

    Tiebreaker: among a highly associated pair, drop the variable with
    the weakest link to the target — i.e. the lowest mean Cramér's V
    with the target across all folds.

    Args:
        folds          : Dictionary {fold_i: (train_i, test_i)}
        cat_variables  : List of categorical variables retained after Rule 2
        target         : Binary target column name (e.g. 'def_year')
        high_threshold : Cramér's V threshold above which two variables
                         are considered redundant (default: 0.50)

    Returns:
        Filtered list of categorical variable names
    """
    # ── Step 1: mean Cramér's V with target across folds (tiebreaker) ──
    # Computed once per variable to avoid redundant calls
    mean_v_with_target = {
        var: np.mean([
            cramers_v(train, var, target)
            for train, _ in folds.values()
        ])
        for var in cat_variables
    }

    # ── Step 2: flag variables to drop from highly correlated pairs ──
    # For each pair, compute Cramér's V on every fold
    # Drop the variable least associated with target if V >= threshold
    # on at least one fold
    vars_to_drop_per_fold = [
        {
            # Drop the variable with the lowest mean Cramér's V with target
            min(v1, v2, key=lambda v: mean_v_with_target[v])
            for v1, v2 in combinations(cat_variables, 2)
            if cramers_v(train, v1, v2) >= high_threshold
        }
        for train, _ in folds.values()
    ]

    # ── Step 3: union across folds ──
    # A variable is dropped if flagged on at least one fold
    vars_to_drop = reduce(lambda a, b: a | b, vars_to_drop_per_fold)

    # ── Step 4: preserve original ordering ──
    return [v for v in cat_variables if v not in vars_to_drop]

