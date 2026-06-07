import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from typing import Callable


# Generic palette and markers. The lists cycle when there are more series.
_COLORS = ["#1a6faf", "#e07b39", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#17becf"]
_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]


def plot_default_by_bin(
    df: pd.DataFrame,
    default_var: str,
    continuous_var: str,
    year_var: str,
    bins: int = 3,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot default-rate trends by quantile bin for a continuous variable.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    default_var : str
        Binary default column.
    continuous_var : str
        Continuous variable split into quantile bins.
    year_var : str
        Time-period column.
    bins : int
        Number of quantile bins.
    title : str, optional
        Chart title. If None, a default title is generated.
    """
    df = df.copy()

    # Split the continuous variable into quantile bins.
    labels = list(range(1, bins + 1))
    df["_bin"] = pd.qcut(df[continuous_var], q=bins, labels=labels)

    # Aggregate default rates by period and bin.
    agg = (
        df.groupby([year_var, "_bin"], observed=True)[default_var]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={default_var: "dr"})
    )

    years = sorted(agg[year_var].unique())

    # Generate one plotting style per bin.
    styles = {
        b: dict(
            color=_COLORS[i % len(_COLORS)],
            lw=2,
            ls="-",
            marker=_MARKERS[i % len(_MARKERS)],
            label=f"Q{b}",
        )
        for i, b in enumerate(labels)
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for b in labels:
        data_b = agg[agg["_bin"] == b].set_index(year_var)["dr"]
        ax.plot(
            [y for y in years if y in data_b.index],
            [data_b[y] for y in years if y in data_b.index],
            **styles[b],
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel(year_var.capitalize(), fontsize=11)
    ax.set_ylabel("DR (%)", fontsize=11)
    ax.set_title(
        title or f"Default-rate trend by quantile\n({continuous_var})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(title="Percentile", framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xticks(years)
    plt.xticks(rotation=45 if len(years) > 8 else 0)

    fig.tight_layout()
    return fig

def plot_default_by_category(
    df: pd.DataFrame,
    default_var: str,
    qualitative_var: str,
    year_var: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot default-rate trends by category for a qualitative variable.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    default_var : str
        Binary default column.
    qualitative_var : str
        Categorical variable. Each category is plotted as a line.
    year_var : str
        Time-period column.
    title : str, optional
        Chart title. If None, a default title is generated.
    """
    df = df.copy()

    # Sort categories for stable plot ordering.
    modalities = sorted(df[qualitative_var].dropna().unique())

    # Aggregate default rates by period and category.
    agg = (
        df.groupby([year_var, qualitative_var], observed=True)[default_var]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={default_var: "dr"})
    )

    years = sorted(agg[year_var].unique())

    # Generate one plotting style per category.
    styles = {
        mod: dict(
            color=_COLORS[i % len(_COLORS)],
            lw=2,
            ls="-",
            marker=_MARKERS[i % len(_MARKERS)],
            label=str(mod),
        )
        for i, mod in enumerate(modalities)
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for mod in modalities:
        data_mod = agg[agg[qualitative_var] == mod].set_index(year_var)["dr"]
        ax.plot(
            [y for y in years if y in data_mod.index],
            [data_mod[y] for y in years if y in data_mod.index],
            **styles[mod],
        )

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel(year_var.capitalize(), fontsize=11)
    ax.set_ylabel("DR (%)", fontsize=11)
    ax.set_title(
        title or f"Default-rate trend by category\n({qualitative_var})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.legend(title=qualitative_var.capitalize(), framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_xticks(years)
    plt.xticks(rotation=45 if len(years) > 8 else 0)

    fig.tight_layout()
    return fig


def compute_psi_table(
    df: pd.DataFrame,
    continuous_vars: list[str],
    qualitative_vars: list[str],
    year_var: str,
    bins: int = 3,
) -> pd.DataFrame:
    """
    Produce a PSI table across rolling time windows.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    continuous_vars : list[str]
        Continuous variables discretized into quantile bins.
    qualitative_vars : list[str]
        Categorical variables, using their categories as natural bins.
    year_var : str
        Year or period column.
    bins : int
        Number of bins for continuous variables.
    """
    years = sorted(df[year_var].unique())
    windows = [(years[i], years[i + 1]) for i in range(len(years) - 1)]

    psi_fn = {
        **{v: lambda ref, curr, v=v: _psi_continuous(v, ref, curr, bins)
           for v in continuous_vars},
        **{v: lambda ref, curr, v=v: _psi_qualitative(v, ref, curr)
           for v in qualitative_vars},
    }

    return (
        pd.DataFrame(
            [
                {"variable": var} | {
                    f"{a}-{b}": psi_fn[var](
                        df[df[year_var] == a],
                        df[df[year_var] == b],
                    )
                    for a, b in windows
                }
                for var in continuous_vars + qualitative_vars
            ]
        )
        .set_index("variable")
        .pipe(lambda d: d.round(4))
        .pipe(_add_stability_flag)
    )


def _add_stability_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stability class based on the worst observed PSI value."""
    thresholds = [(0.10, "Stable"), (0.25, "Light drift"), (np.inf, "Strong drift")]
    classify = lambda v: next(label for bound, label in thresholds if v < bound)
    df["stability"] = df.max(axis=1).map(classify)
    return df


_PSI_THRESHOLDS = [(0.10, "Stable"), (0.25, "Light drift"), (np.inf, "Strong drift")]


def _classify_psi(value: float) -> str:
    return next(label for bound, label in _PSI_THRESHOLDS if value < bound)


def _psi_from_dist(ref: pd.Series, curr: pd.Series, var: str = "") -> float:
    """
    PSI = Σ (curr_i - ref_i) × ln(curr_i / ref_i)

    Raise explicit errors when:
      - ref_i == 0, because curr / ref is undefined
      - curr_i == 0, because log(0) is undefined
    """
    prefix = f"[PSI - {var}] " if var else "[PSI] "

    zero_ref = ref.index[ref == 0].tolist()
    zero_curr = curr.index[curr == 0].tolist()

    if zero_ref:
        raise ZeroDivisionError(
            f"{prefix}division by zero: zero proportion in REF "
            f"for the following bins: {zero_ref}.\n"
            "A bin is empty in the reference population. "
            "Reduce the number of bins or group rare categories."
        )
    if zero_curr:
        raise ValueError(
            f"{prefix}undefined log(0): zero proportion in CURR "
            f"for the following bins: {zero_curr}.\n"
            "A bin is empty in the current population. "
            "Reduce the number of bins or group rare categories."
        )

    return float(((curr - ref) * np.log(curr / ref)).sum())


def _dist_continuous(series: pd.Series, breakpoints: np.ndarray) -> pd.Series:
    """Return the actual distribution; zero bins trigger explicit downstream errors."""
    intervals = pd.IntervalIndex.from_breaks(breakpoints)
    return (
        pd.cut(series, bins=intervals, include_lowest=True)
        .value_counts(normalize=True)
        .reindex(intervals, fill_value=0)
    )


def _dist_qualitative(series: pd.Series, categories: np.ndarray) -> pd.Series:
    return (
        series.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )


def _psi_continuous(var: str, ref: pd.DataFrame, curr: pd.DataFrame,
                    bins: int) -> float:
    """Compute breakpoints on REF and apply them to CURR."""
    breakpoints = pd.qcut(ref[var], q=bins, retbins=True, duplicates="drop")[1]
    return _psi_from_dist(
        _dist_continuous(ref[var], breakpoints),
        _dist_continuous(curr[var], breakpoints),
        var=var,
    )


def _psi_qualitative(var: str, ref: pd.DataFrame, curr: pd.DataFrame) -> float:
    """Use the REF/CURR category union as natural bins."""
    categories = pd.concat([ref[var], curr[var]]).dropna().unique()
    return _psi_from_dist(
        _dist_qualitative(ref[var], categories),
        _dist_qualitative(curr[var], categories),
        var=var,
    )


def compute_psi_stability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame,
    continuous_vars: list[str],
    qualitative_vars: list[str],
    bins: int = 3,
) -> pd.DataFrame:
    """
    Compute PSI between Train, Test, and OOT for each variable.

    Output columns:
        PSI Train vs Test | PSI Train vs OOT | PSI Test vs OOT | Stability

    Parameters
    ----------
    train / test / oot : pd.DataFrame
        The three populations to compare.
    continuous_vars : list[str]
        Continuous variables discretized into bins using Train breakpoints.
    qualitative_vars : list[str]
        Categorical variables, using their categories as natural bins.
    bins : int
        Number of bins for continuous variables.

    Raises
    ------
    ZeroDivisionError
        Empty bin in the reference population.
    ValueError
        Empty bin in the current population.
    """
    scenarios: list[tuple[pd.DataFrame, pd.DataFrame, str]] = [
        (train, test, "PSI Train vs Test"),
        (train, oot, "PSI Train vs OOT"),
        (test, oot, "PSI Test vs OOT"),
    ]

    psi_fn: dict[str, Callable[[pd.DataFrame, pd.DataFrame], float]] = {
        **{v: lambda ref, curr, v=v: _psi_continuous(v, ref, curr, bins)
           for v in continuous_vars},
        **{v: lambda ref, curr, v=v: _psi_qualitative(v, ref, curr)
           for v in qualitative_vars},
    }

    return (
        pd.DataFrame(
            [
                {"Variable": var} | {
                    col: psi_fn[var](ref, curr)
                    for ref, curr, col in scenarios
                }
                for var in continuous_vars + qualitative_vars
            ]
        )
        .set_index("Variable")
        .pipe(lambda d: d.round(4))
        .pipe(_add_stability_column)
    )


def _add_stability_column(df: pd.DataFrame) -> pd.DataFrame:
    """Overall stability is based on the worst PSI scenario."""
    df["Stability"] = df.max(axis=1).map(_classify_psi)
    return df
