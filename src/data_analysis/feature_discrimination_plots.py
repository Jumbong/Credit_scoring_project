import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.patches as mpatches

def plot_continuous_vs_categorical(
    df,
    continuous_var,
    categorical_var,
    category_labels=None,
    figsize=(7.24, 4.07),
    sample=None
):
    """
     Compare a continuous variable across categories
    using boxplot, KDE, and ECDF (2x2 layout).
    """

    sns.set_style("white")

    data = df[[continuous_var, categorical_var]].dropna().copy()

    # Optional sampling
    if sample:
        data = data.sample(sample, random_state=42)

    categories = sorted(data[categorical_var].unique())

    # Labels mapping (optional)
    if category_labels:
        labels = [category_labels.get(cat, str(cat)) for cat in categories]
    else:
        labels = [str(cat) for cat in categories]

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=100)

    # --- 1. Boxplot ---
    sns.boxplot(
        data=data,
        x=categorical_var,
        y=continuous_var,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Boxplot (median & spread)", loc="left")

    # --- 2. Boxplot comparaison médianes ---
    sns.boxplot(
        data=data,
        x=categorical_var,
        y=continuous_var,
        ax=axes[0, 1],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6
        }
    )

    axes[0, 1].set_title("Median comparison (Boxplot)", loc="left")
    medians = data.groupby(categorical_var)[continuous_var].median()

    for i, cat in enumerate(categories):
        axes[0, 1].text(
            i,
            medians[cat],
            f"{medians[cat]:.2f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    # --- 3. KDE only ---
    for cat, label in zip(categories, labels):
        subset = data[data[categorical_var] == cat][continuous_var]
        sns.kdeplot(
            subset,
            ax=axes[1, 0],
            label=label
        )
    axes[1, 0].set_title("Density comparison (KDE)", loc="left")
    axes[1, 0].legend()

    # --- 4. ECDF ---
    for cat, label in zip(categories, labels):
        subset = np.sort(data[data[categorical_var] == cat][continuous_var])
        y = np.arange(1, len(subset) + 1) / len(subset)
        axes[1, 1].plot(subset, y, label=label)
    axes[1, 1].set_title("Cumulative distribution (ECDF)", loc="left")
    axes[1, 1].legend()

    # Clean style (Storytelling with Data)
    for ax in axes.flat:
        sns.despine(ax=ax)
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.show()

def contingency_analysis(
    df,
    var1,
    var2,
    normalize=None,   # None, "index", "columns", "all"
    plot=True,
    figsize=(7.24, 4.07)
):
    """
    function to compute and visualize contingency table
    + Chi-square test + Cramér's V.
    """

    # --- Contingency table ---
    table = pd.crosstab(df[var1], df[var2])

    # --- Normalized version (optional) ---
    if normalize:
        table_norm = pd.crosstab(df[var1], df[var2], normalize=normalize, margins=True).round(3) * 100
    else:
        table_norm = None

    # --- Chi-square test ---
    chi2, p, dof, expected = chi2_contingency(table)

    # --- Cramér's V ---
    n = table.values.sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

    # --- Plot (heatmap) ---
    if plot:
        sns.set_style("white")
        plt.figure(figsize=figsize, dpi=100)

        data_to_plot = table_norm if table_norm is not None else table

        sns.heatmap(
            data_to_plot,
            annot=True,
            fmt=".2f" if normalize else "d",
            cbar=True
        )

        plt.title(f"{var1} vs {var2} (Contingency Table)", loc="left", weight="bold")
        plt.xlabel(var2)
        plt.ylabel(var1)

        sns.despine()
        plt.tight_layout()
        plt.show()

    # --- Output ---
    results = {
        "contingency_table": table,
        "normalized_table": table_norm,
        "chi2": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "cramers_v": cramers_v
    }

    return results

def plot_grouped_bar(df, cat_var, subcat_var,
                          normalize="index", title=""):
    ct = pd.crosstab(df[subcat_var], df[cat_var], normalize=normalize) * 100
    modalities = ct.index.tolist()
    categories = ct.columns.tolist()
    n_mod = len(modalities)
    n_cat = len(categories)
    x = np.arange(n_mod)
    width = 0.35

    colors = ['#0F6E56', '#993C1D']  # teal = non-défaut, coral = défaut

    fig, ax = plt.subplots(figsize=(7.24, 4.07), dpi=100)

    for i, (cat, color) in enumerate(zip(categories, colors)):
        offset = (i - n_cat / 2 + 0.5) * width
        ax.bar(x + offset, ct[cat], width=width, color=color, label=str(cat))

        # Annotations au-dessus de chaque barre
        for j, val in enumerate(ct[cat]):
            ax.text(x[j] + offset, val + 0.5, f"{val:.1f}%",
                    ha='center', va='bottom', fontsize=9, color='#444')

    # Style Cole
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=11)
    ax.set_ylabel("Taux (%)" if normalize else "Effectifs", fontsize=11, color='#555')
    ax.tick_params(left=False, colors='#555')


    handles = [mpatches.Patch(color=c, label=str(l))
               for c, l in zip(colors, categories)]
    ax.legend(handles=handles, title=cat_var, frameon=False,
              fontsize=10, loc='upper right')

    ax.set_title(title, fontsize=13, fontweight='normal', pad=14)
    plt.tight_layout()
    plt.savefig("default_by_ownership.png", dpi=150, bbox_inches='tight')
    plt.show()