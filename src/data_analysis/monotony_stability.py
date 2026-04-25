import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


# Palette et marqueurs génériques — s'adapte à n'importe quel nombre de bins
_COLORS  = ["#1a6faf", "#e07b39", "#2ca02c", "#9467bd", "#d62728"]
_MARKERS = ["o", "s", "^", "D", "P"]


def plot_default_by_bin(
    df: pd.DataFrame,
    default_var: str,
    continuous_var: str,
    year_var: str,
    bins: int = 3,
    title: str | None = None,
) -> plt.Figure:
    """
    Représente l'évolution du taux de défaut par percentile
    d'une variable continue dans le temps.

    Parameters
    ----------
    df              : DataFrame source
    default_var     : colonne 0/1 indiquant le défaut
    continuous_var  : variable continue à découper en `bins` tranches égales
    year_var        : variable temporelle (année)
    bins            : nombre de tranches (3 → terciles, 4 → quartiles, …)
    title           : titre du graphique (auto-généré si None)
    """
    df = df.copy()

    # ── 1. Découpage en `bins` tranches égales ─────────────────────────────────
    labels = list(range(1, bins + 1))
    df["_bin"] = pd.qcut(df[continuous_var], q=bins, labels=labels)

    # ── 2. Taux de défaut agrégé par (année, tranche) ──────────────────────────
    agg = (
        df.groupby([year_var, "_bin"], observed=True)[default_var]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={default_var: "dr"})
    )

    years = sorted(agg[year_var].unique())

    # ── 3. Styles DRY : générés dynamiquement selon `bins` ────────────────────
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

    # ── 4. Tracé ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for b in labels:
        data_b = agg[agg["_bin"] == b].set_index(year_var)["dr"]
        ax.plot(
            [y for y in years if y in data_b.index],
            [data_b[y] for y in years if y in data_b.index],
            **styles[b],
        )

    # ── 5. Mise en forme ───────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel(year_var.capitalize(), fontsize=11)
    ax.set_ylabel("DR (%)", fontsize=11)
    ax.set_title(
        title or f"Évolution du taux de défaut par percentile\n({continuous_var})",
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




# Palette et marqueurs génériques — s'adapte à n'importe quel nombre de modalités
_COLORS  = ["#1a6faf", "#e07b39", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#17becf"]
_MARKERS = ["o", "s", "^", "D", "P", "X", "v"]


def plot_default_by_category(
    df: pd.DataFrame,
    default_var: str,
    qualitative_var: str,
    year_var: str,
    title: str | None = None,
) -> plt.Figure:
    """
    Représente l'évolution du taux de défaut par modalité
    d'une variable qualitative dans le temps.

    Parameters
    ----------
    df               : DataFrame source
    default_var      : colonne 0/1 indiquant le défaut
    qualitative_var  : variable catégorielle (modalités = courbes)
    year_var         : variable temporelle (année)
    title            : titre du graphique (auto-généré si None)
    """
    df = df.copy()

    # ── 1. Modalités triées (ordre naturel ou alphabétique) ────────────────────
    modalities = sorted(df[qualitative_var].dropna().unique())

    # ── 2. Taux de défaut agrégé par (année, modalité) ─────────────────────────
    agg = (
        df.groupby([year_var, qualitative_var], observed=True)[default_var]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={default_var: "dr"})
    )

    years = sorted(agg[year_var].unique())

    # ── 3. Styles DRY : générés dynamiquement selon les modalités ─────────────
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

    # ── 4. Tracé ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for mod in modalities:
        data_mod = agg[agg[qualitative_var] == mod].set_index(year_var)["dr"]
        ax.plot(
            [y for y in years if y in data_mod.index],
            [data_mod[y] for y in years if y in data_mod.index],
            **styles[mod],
        )

    # ── 5. Mise en forme ───────────────────────────────────────────────────────
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel(year_var.capitalize(), fontsize=11)
    ax.set_ylabel("DR (%)", fontsize=11)
    ax.set_title(
        title or f"Évolution du taux de défaut par modalité\n({qualitative_var})",
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



import pandas as pd
import numpy as np
from functools import reduce


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bin_distribution(series: pd.Series, bins: pd.IntervalIndex) -> pd.Series:
    """Retourne la distribution (proportions) d'une série sur des bins fixes."""
    return (
        pd.cut(series, bins=bins, include_lowest=True)
        .value_counts(normalize=True)
        .reindex(bins, fill_value=1e-4)   # évite log(0)
    )


def _psi_from_distributions(ref: pd.Series, curr: pd.Series) -> float:
    """Calcule le PSI entre deux distributions (séries de proportions)."""
    return float(((curr - ref) * np.log(curr / ref)).sum())


def _psi_continuous(series: pd.Series, ref_mask: pd.Series,
                    curr_mask: pd.Series, bins: int) -> float:
    """PSI pour une variable continue : discrétisation en `bins` terciles sur la ref."""
    breakpoints = pd.qcut(series[ref_mask], q=bins, retbins=True, duplicates="drop")[1]
    intervals   = pd.IntervalIndex.from_breaks(breakpoints)
    ref_dist    = _bin_distribution(series[ref_mask],  intervals)
    curr_dist   = _bin_distribution(series[curr_mask], intervals)
    return _psi_from_distributions(ref_dist, curr_dist)


def _psi_qualitative(series: pd.Series, ref_mask: pd.Series,
                     curr_mask: pd.Series) -> float:
    """PSI pour une variable qualitative : modalités comme bins naturels."""
    categories = series.dropna().unique()
    to_dist = lambda mask: (
        series[mask]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=1e-4)
    )
    return _psi_from_distributions(to_dist(ref_mask), to_dist(curr_mask))


# ── Fonction principale ────────────────────────────────────────────────────────

def compute_psi_table(
    df: pd.DataFrame,
    continuous_vars: list[str],
    qualitative_vars: list[str],
    year_var: str,
    bins: int = 3,
) -> pd.DataFrame:
    """
    Produit un tableau PSI (variables × fenêtres glissantes).

    Parameters
    ----------
    df               : DataFrame source
    continuous_vars  : variables continues (discrétisées en `bins` terciles)
    qualitative_vars : variables qualitatives (modalités comme bins)
    year_var         : colonne année
    bins             : nombre de tranches pour les variables continues
    """
    years   = sorted(df[year_var].unique())
    windows = [(years[i], years[i + 1]) for i in range(len(years) - 1)]
    col_names = [f"{a}-{b}" for a, b in windows]

    # ── Dispatch : associe chaque variable à sa fonction PSI ──────────────────

    psi_fn = {
        **{v: lambda ref, curr, v=v: _psi_continuous(v, ref, curr, bins)
        for v in continuous_vars},
        **{v: lambda ref, curr, v=v: _psi_qualitative(v, ref, curr)
        for v in qualitative_vars},
    }

    # Et l'appel :
    return (
        pd.DataFrame(
            [
                {"variable": var} | {
                    f"{a}-{b}": psi_fn[var](
                        df[df[year_var] == a],   # ← DataFrame filtré, pas un masque
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
    """Ajoute une colonne synthétique indiquant la stabilité maximale observée."""
    thresholds = [(0.10, "✅ Stable"), (0.25, "⚠️ Dérive légère"), (np.inf, "🔴 Dérive forte")]
    classify   = lambda v: next(label for bound, label in thresholds if v < bound)
    df["stabilité"] = df.max(axis=1).map(classify)
    return df


# ── Seuils PSI (source unique de vérité) ──────────────────────────────────────
_PSI_THRESHOLDS = [(0.10, "✅ Stable"), (0.25, "⚠️ Légère"), (np.inf, "🔴 Forte")]


# ── Helpers bas niveau ─────────────────────────────────────────────────────────

def _classify_psi(value: float) -> str:
    return next(label for bound, label in _PSI_THRESHOLDS if value < bound)


def _psi_from_dist(ref: pd.Series, curr: pd.Series, var: str = "") -> float:
    """
    PSI = Σ (curr_i - ref_i) × ln(curr_i / ref_i)

    Lève des erreurs explicites si :
      - ref_i  == 0  →  ZeroDivisionError  (curr / ref impossible)
      - curr_i == 0  →  ValueError         (log(0) indéfini)
    """
    prefix = f"[PSI – {var}] " if var else "[PSI] "

    zero_ref  = ref.index[ref  == 0].tolist()
    zero_curr = curr.index[curr == 0].tolist()

    if zero_ref:
        raise ZeroDivisionError(
            f"{prefix}Division par zéro : proportion nulle dans REF "
            f"pour les bins suivants → {zero_ref}.\n"
            "→ Un bin est vide dans la population de référence. "
            "Réduisez le nombre de bins ou regroupez les modalités rares."
        )
    if zero_curr:
        raise ValueError(
            f"{prefix}log(0) indéfini : proportion nulle dans CURR "
            f"pour les bins suivants → {zero_curr}.\n"
            "→ Un bin est vide dans la population courante. "
            "Réduisez le nombre de bins ou regroupez les modalités rares."
        )

    return float(((curr - ref) * np.log(curr / ref)).sum())


def _dist_continuous(series: pd.Series, breakpoints: np.ndarray) -> pd.Series:
    """Distribution réelle sans imputation — les zéros déclenchent l'erreur en aval."""
    intervals = pd.IntervalIndex.from_breaks(breakpoints)
    return (
        pd.cut(series, bins=intervals, include_lowest=True)
        .value_counts(normalize=True)
        .reindex(intervals, fill_value=0)   # 0 réel → erreur explicite dans _psi_from_dist
    )


def _dist_qualitative(series: pd.Series, categories: np.ndarray) -> pd.Series:
    return (
        series.value_counts(normalize=True)
        .reindex(categories, fill_value=0)  # idem
    )


# ── Calcul PSI par variable ────────────────────────────────────────────────────

def _psi_continuous(var: str, ref: pd.DataFrame, curr: pd.DataFrame,
                    bins: int) -> float:
    """Breakpoints calculés sur REF, appliqués à CURR."""
    breakpoints = pd.qcut(ref[var], q=bins, retbins=True, duplicates="drop")[1]
    return _psi_from_dist(
        _dist_continuous(ref[var],  breakpoints),
        _dist_continuous(curr[var], breakpoints),
        var=var,
    )


def _psi_qualitative(var: str, ref: pd.DataFrame, curr: pd.DataFrame) -> float:
    """Union des modalités REF ∪ CURR comme bins naturels."""
    categories = pd.concat([ref[var], curr[var]]).dropna().unique()
    return _psi_from_dist(
        _dist_qualitative(ref[var],  categories),
        _dist_qualitative(curr[var], categories),
        var=var,
    )


# ── Fonction principale ────────────────────────────────────────────────────────

def compute_psi_stability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oot: pd.DataFrame,
    continuous_vars: list[str],
    qualitative_vars: list[str],
    bins: int = 3,
) -> pd.DataFrame:
    """
    Calcule le PSI entre Train / Test / OOT pour chaque variable.

    Colonnes produites :
        PSI Train vs Test | PSI Train vs OOT | PSI Test vs OOT | Stabilité

    Parameters
    ----------
    train / test / oot   : DataFrames des trois populations
    continuous_vars      : discrétisées en `bins` tranches égales (breakpoints sur Train)
    qualitative_vars     : modalités utilisées comme bins naturels
    bins                 : nombre de tranches (défaut 3 → terciles)

    Raises
    ------
    ZeroDivisionError    : bin vide dans la population de référence (curr / ref)
    ValueError           : bin vide dans la population courante (log(0))
    """
    scenarios: list[tuple[pd.DataFrame, pd.DataFrame, str]] = [
        (train, test, "PSI Train vs Test"),
        (train, oot,  "PSI Train vs OOT"),
        (test,  oot,  "PSI Test vs OOT"),
    ]

    # Dispatch DRY : var → callable(ref, curr) → float
    psi_fn: dict[str, callable] = {
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
    """Stabilité globale = pire scénario parmi les 3 colonnes PSI."""
    df["Stabilité"] = df.max(axis=1).map(_classify_psi)
    return df



