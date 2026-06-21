import pickle
from pathlib import Path

import numpy as np
import pandas as pd


VARIABLE_LABELS = {
    "loan_int_rate_dis": "loan_int_rate",
    "loan_percent_income_dis": "loan_percent_income",
    "cb_person_default_on_file": "cb_person_default_on_file",
    "home_ownership_3": "home_ownership_3",
}


MODALITY_LABELS = {
    "loan_int_rate_dis": {
        "1": "(12.21, 21.825]",
        "2": "(9.91, 12.21]",
        "3": "(5.419, 9.91]",
    },
    "loan_percent_income_dis": {
        "1": "(0.2, 0.44]",
        "2": "(0.11, 0.2]",
        "3": "(-0.001, 0.11]",
    },
    "cb_person_default_on_file": {
        "N": "N",
        "Y": "Y",
    },
    "home_ownership_3": {
        "OWN": "OWN",
        "MORTGAGE": "MORTGAGE",
        "OTHER_RENT": "OTHER_RENT",
    },
}


DEFAULT_SCORE_CLASS_GROUPS = [
    {
        "score_class": 1,
        "risk_label": "Very high risk",
        "vingtiles": [1, 2, 3],
    },
    {
        "score_class": 2,
        "risk_label": "High risk",
        "vingtiles": [4, 5, 6],
    },
    {
        "score_class": 3,
        "risk_label": "Medium-high risk",
        "vingtiles": [7, 8],
    },
    {
        "score_class": 4,
        "risk_label": "Medium risk",
        "vingtiles": [9, 10, 11, 12],
    },
    {
        "score_class": 5,
        "risk_label": "Low risk",
        "vingtiles": [13, 14, 15, 16, 17],
    },
    {
        "score_class": 6,
        "risk_label": "Very low risk",
        "vingtiles": [18, 19, 20],
    },
]


def load_score_model_artifact(pickle_path):
    with Path(pickle_path).open("rb") as file:
        return pickle.load(file)


def _variable_label(variable):
    return VARIABLE_LABELS.get(variable, variable)


def _modality_label(variable, modality):
    return MODALITY_LABELS.get(variable, {}).get(str(modality), str(modality))


def _artifact_coefficients_by_variable(artifact):
    coefficients_by_variable = {}

    for variable, spec in artifact["variable_specs"].items():
        coefficients = {
            str(modality): float(coefficient)
            for modality, coefficient in spec["coefficients"].items()
        }
        reference = str(spec["reference"])
        coefficients[reference] = 0.0
        coefficients_by_variable[variable] = coefficients

    return coefficients_by_variable


def build_modality_score_table(artifact, score_scale=1000):
    """
    Compute score points by variable modality.

    For each variable j and modality i:
        SC(j,i) = score_scale * |c(j,i) - alpha_j| / sum_j alpha_j

    with alpha_j = max_i c(j,i). Since the selected model uses low-risk
    reference modalities, reference coefficients are 0 and non-reference
    coefficients are expected to be positive.
    """
    coefficients_by_variable = _artifact_coefficients_by_variable(artifact)
    alpha_by_variable = {
        variable: max(coefficients.values())
        for variable, coefficients in coefficients_by_variable.items()
    }
    denominator = sum(alpha_by_variable.values())

    if denominator <= 0:
        raise ValueError("La somme des coefficients maximum par variable est nulle.")

    rows = []

    for variable in artifact["variables"]:
        coefficients = coefficients_by_variable[variable]
        alpha = alpha_by_variable[variable]

        for modality, coefficient in coefficients.items():
            score_points = score_scale * abs(coefficient - alpha) / denominator

            rows.append({
                "variable": variable,
                "variable_label": _variable_label(variable),
                "modality": str(modality),
                "modality_label": _modality_label(variable, modality),
                "coefficient": coefficient,
                "alpha_variable": alpha,
                "score_points": score_points,
            })

    score_table = pd.DataFrame(rows)
    score_table = score_table.sort_values(
        ["variable", "coefficient"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return score_table


def score_dataset(df, modality_score_table, artifact, dataset_name=None):
    """Apply modality scores to a dataset and return individual scores."""
    scored_df = df.copy()
    missing_score_cols = []
    variable_score_cols = []

    for variable in artifact["variables"]:
        score_col = f"score_{_variable_label(variable)}"
        missing_col = f"missing_score_{_variable_label(variable)}"
        mapping = (
            modality_score_table
            .loc[modality_score_table["variable"].eq(variable)]
            .set_index("modality")["score_points"]
            .to_dict()
        )

        scored_df[score_col] = scored_df[variable].astype(str).map(mapping)
        scored_df[missing_col] = scored_df[score_col].isna()
        variable_score_cols.append(score_col)
        missing_score_cols.append(missing_col)

    scored_df["score_total"] = scored_df[variable_score_cols].sum(axis=1)
    scored_df.loc[scored_df[missing_score_cols].any(axis=1), "score_total"] = np.nan

    if dataset_name is not None:
        scored_df["dataset"] = dataset_name

    return scored_df


def score_datasets(datasets, modality_score_table, artifact):
    return {
        dataset_name: score_dataset(
            df=df,
            modality_score_table=modality_score_table,
            artifact=artifact,
            dataset_name=dataset_name,
        )
        for dataset_name, df in datasets.items()
    }


def compute_variable_contributions(train_df, modality_score_table, artifact):
    """
    Compute scale and score contributions by variable from train distribution.

    CTR_j = max_i SC(j,i) / 10
    q_j = weighted_sd(SC_j) / sum_k weighted_sd(SC_k)
    """
    rows = []

    for variable in artifact["variables"]:
        variable_scores = modality_score_table.loc[
            modality_score_table["variable"].eq(variable)
        ].copy()
        proportions = (
            train_df[variable]
            .astype(str)
            .value_counts(normalize=True, dropna=False)
            .rename("population_share")
            .reset_index()
            .rename(columns={variable: "modality", "index": "modality"})
        )
        proportions["modality"] = proportions["modality"].astype(str)
        merged = variable_scores.merge(
            proportions,
            on="modality",
            how="left",
        )
        merged["population_share"] = merged["population_share"].fillna(0.0)
        weighted_mean = (
            merged["population_share"] * merged["score_points"]
        ).sum()
        weighted_variance = (
            merged["population_share"]
            * (merged["score_points"] - weighted_mean) ** 2
        ).sum()
        weighted_sd = np.sqrt(weighted_variance)

        rows.append({
            "variable": variable,
            "variable_label": _variable_label(variable),
            "max_score_points": merged["score_points"].max(),
            "scale_contribution_pct": merged["score_points"].max() / 10,
            "weighted_mean_score": weighted_mean,
            "weighted_sd_score": weighted_sd,
        })

    contributions = pd.DataFrame(rows)
    denominator = contributions["weighted_sd_score"].sum()

    if denominator > 0:
        contributions["score_contribution_pct"] = (
            contributions["weighted_sd_score"] / denominator * 100
        )
    else:
        contributions["score_contribution_pct"] = np.nan

    return contributions


def build_score_outputs(artifact, train_df, test_df, oot_df, score_scale=1000):
    modality_score_table = build_modality_score_table(
        artifact=artifact,
        score_scale=score_scale,
    )
    scored_datasets = score_datasets(
        datasets={
            "Train": train_df,
            "Test": test_df,
            "OOT": oot_df,
        },
        modality_score_table=modality_score_table,
        artifact=artifact,
    )
    contributions = compute_variable_contributions(
        train_df=train_df,
        modality_score_table=modality_score_table,
        artifact=artifact,
    )

    return modality_score_table, scored_datasets, contributions


def build_conditional_density_table(
    scored_datasets,
    target="def",
    score_col="score_total",
    grid_size=300,
    score_min=0,
    score_max=1000,
):
    """Estimate conditional score densities by dataset and target value."""
    from scipy.stats import gaussian_kde

    grid = np.linspace(score_min, score_max, grid_size)
    density_rows = []

    for dataset_name, scored_df in scored_datasets.items():
        for target_value in sorted(scored_df[target].dropna().unique()):
            scores = (
                scored_df
                .loc[scored_df[target].eq(target_value), score_col]
                .dropna()
                .astype(float)
            )

            if len(scores) < 2 or scores.std() == 0:
                density = np.zeros_like(grid)
            else:
                kde = gaussian_kde(scores)
                density = kde(grid)

            for score_value, density_value in zip(grid, density):
                density_rows.append({
                    "dataset": dataset_name,
                    target: int(target_value),
                    "score": score_value,
                    "density": density_value,
                    "n": len(scores),
                })

    return pd.DataFrame(density_rows)


def build_score_vingtile_default_rate(
    scored_df,
    target="def",
    score_col="score_total",
    n_bins=20,
):
    """
    Compute default rate by score vingtile.

    Scores are ranked before binning so that repeated score values do not
    collapse quantile buckets. Vingtile 1 corresponds to the lowest scores
    and vingtile 20 to the highest scores.
    """
    required_cols = [target, score_col]
    missing_cols = [col for col in required_cols if col not in scored_df.columns]

    if missing_cols:
        raise ValueError(
            "Colonnes manquantes pour le calcul des vingtiles : "
            + ", ".join(missing_cols)
        )

    working_df = scored_df[required_cols].dropna().copy()

    if working_df.empty:
        raise ValueError("Aucune observation disponible pour calculer les vingtiles.")

    working_df = assign_score_vingtiles(
        df=working_df,
        score_col=score_col,
        n_bins=n_bins,
        vingtile_col="vingtile",
    )

    vingtile_table = (
        working_df
        .groupby("vingtile", observed=True)
        .agg(
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            n=(target, "size"),
            defaults=(target, "sum"),
            default_rate=(target, "mean"),
        )
        .reset_index()
    )

    vingtile_table["vingtile"] = vingtile_table["vingtile"].astype(int)

    return vingtile_table


def assign_score_vingtiles(
    df,
    score_col="score_total",
    n_bins=20,
    vingtile_col="score_vingtile",
):
    """
    Assign score vingtiles from low scores to high scores.

    Rank-based binning keeps 20 buckets even when several counterparties have
    the same discrete score.
    """
    assigned_df = df.copy()
    valid_mask = assigned_df[score_col].notna()
    score_rank = assigned_df.loc[valid_mask, score_col].rank(
        method="first",
        ascending=True,
    )
    assigned_df.loc[valid_mask, vingtile_col] = (
        pd.qcut(
            score_rank,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        + 1
    )
    assigned_df[vingtile_col] = assigned_df[vingtile_col].astype("Int64")

    return assigned_df


def build_score_class_table(
    base_final_with_score,
    score_class_groups=None,
    target="def",
    score_col="score_total",
    n_bins=20,
    min_population_share=0.01,
    min_default_count=500,
):
    """
    Build risk classes from score vingtiles and compute validation indicators.

    Classes are ordered from highest risk (class 1) to lowest risk.
    The relative gap is computed against the previous, riskier class:
        (DR_previous - DR_current) / DR_previous.
    """
    if score_class_groups is None:
        score_class_groups = DEFAULT_SCORE_CLASS_GROUPS

    scored_class_df = assign_score_vingtiles(
        df=base_final_with_score,
        score_col=score_col,
        n_bins=n_bins,
        vingtile_col="score_vingtile",
    )

    vingtile_to_class = {}
    class_metadata = {}

    for class_group in score_class_groups:
        score_class = class_group["score_class"]
        class_metadata[score_class] = {
            "risk_label": class_group["risk_label"],
            "vingtile_group": (
                f"{min(class_group['vingtiles'])}-"
                f"{max(class_group['vingtiles'])}"
            ),
        }

        for vingtile in class_group["vingtiles"]:
            vingtile_to_class[vingtile] = score_class

    scored_class_df["score_class"] = (
        scored_class_df["score_vingtile"]
        .map(vingtile_to_class)
        .astype("Int64")
    )
    scored_class_df["rating_expert"] = scored_class_df["score_class"]
    scored_class_df["risk_label"] = scored_class_df["score_class"].map(
        {
            score_class: metadata["risk_label"]
            for score_class, metadata in class_metadata.items()
        }
    )

    total_count = len(scored_class_df)

    class_table = (
        scored_class_df
        .dropna(subset=["score_class"])
        .groupby("score_class", observed=True)
        .agg(
            min_score=(score_col, "min"),
            max_score=(score_col, "max"),
            n=(target, "size"),
            defaults=(target, "sum"),
            default_rate=(target, "mean"),
            min_vingtile=("score_vingtile", "min"),
            max_vingtile=("score_vingtile", "max"),
        )
        .reset_index()
    )

    class_table["score_class"] = class_table["score_class"].astype(int)
    vingtile_rates = (
        scored_class_df
        .dropna(subset=["score_class", "score_vingtile"])
        .groupby(["score_class", "score_vingtile"], observed=True)
        .agg(vingtile_default_rate=(target, "mean"))
        .reset_index()
        .groupby("score_class", observed=True)
        .agg(
            min_vingtile_default_rate=("vingtile_default_rate", "min"),
            max_vingtile_default_rate=("vingtile_default_rate", "max"),
        )
        .reset_index()
    )
    vingtile_rates["score_class"] = vingtile_rates["score_class"].astype(int)
    vingtile_rates["range_vingtile_default_rate"] = (
        vingtile_rates["max_vingtile_default_rate"]
        - vingtile_rates["min_vingtile_default_rate"]
    )
    class_table = class_table.merge(
        vingtile_rates,
        on="score_class",
        how="left",
    )
    class_table["risk_label"] = class_table["score_class"].map(
        {
            score_class: metadata["risk_label"]
            for score_class, metadata in class_metadata.items()
        }
    )
    class_table["vingtile_group"] = class_table["score_class"].map(
        {
            score_class: metadata["vingtile_group"]
            for score_class, metadata in class_metadata.items()
        }
    )
    class_table["population_share"] = class_table["n"] / total_count
    class_table["score_interval"] = class_table.apply(
        lambda row: f"[{row['min_score']:.2f} ; {row['max_score']:.2f}]",
        axis=1,
    )
    class_table["relative_gap_vs_previous"] = (
        class_table["default_rate"].shift(1) - class_table["default_rate"]
    ) / class_table["default_rate"].shift(1)
    class_table["relative_gap_ok"] = (
        class_table["relative_gap_vs_previous"].isna()
        | class_table["relative_gap_vs_previous"].ge(0.30)
    )
    class_table["population_min_ok"] = (
        class_table["population_share"].ge(min_population_share)
    )
    class_table["default_count_min_ok"] = (
        class_table["defaults"].ge(min_default_count)
    )
    class_table["min_size_or_defaults_ok"] = (
        class_table["population_min_ok"] | class_table["default_count_min_ok"]
    )

    class_table = class_table[
        [
            "score_class",
            "risk_label",
            "vingtile_group",
            "score_interval",
            "min_score",
            "max_score",
            "n",
            "defaults",
            "population_share",
            "default_rate",
            "min_vingtile_default_rate",
            "max_vingtile_default_rate",
            "range_vingtile_default_rate",
            "relative_gap_vs_previous",
            "relative_gap_ok",
            "population_min_ok",
            "default_count_min_ok",
            "min_size_or_defaults_ok",
        ]
    ]

    scored_class_df["score_interval"] = scored_class_df["score_class"].map(
        class_table.set_index("score_class")["score_interval"].to_dict()
    )

    return class_table, scored_class_df


def build_score_class_stability_tables(
    base_final_with_score_class,
    target="def",
    period_col="year",
    class_col="score_class",
):
    """
    Compute risk and volume stability by period and score class.

    Risk stability is measured through the default rate over time.
    Volume stability is measured through each class population share over time.
    """
    required_cols = [target, period_col, class_col]
    missing_cols = [
        col
        for col in required_cols
        if col not in base_final_with_score_class.columns
    ]

    if missing_cols:
        raise ValueError(
            "Colonnes manquantes pour la stabilité des classes : "
            + ", ".join(missing_cols)
        )

    working_df = base_final_with_score_class.dropna(
        subset=[target, period_col, class_col]
    ).copy()
    period_totals = (
        working_df
        .groupby(period_col, observed=True)
        .size()
        .rename("period_total")
        .reset_index()
    )

    optional_cols = [
        col
        for col in ["risk_label", "score_interval", "vingtile_group"]
        if col in working_df.columns
    ]
    grouping_cols = [period_col, class_col] + optional_cols

    stability_by_period = (
        working_df
        .groupby(grouping_cols, observed=True)
        .agg(
            n=(target, "size"),
            defaults=(target, "sum"),
            default_rate=(target, "mean"),
        )
        .reset_index()
        .merge(period_totals, on=period_col, how="left")
    )
    stability_by_period["volume_share"] = (
        stability_by_period["n"] / stability_by_period["period_total"]
    )

    class_metadata = (
        working_df[[class_col] + optional_cols]
        .drop_duplicates(subset=[class_col])
    )

    stability_summary = (
        stability_by_period
        .groupby(class_col, observed=True)
        .agg(
            n_periods=(period_col, "nunique"),
            min_n=("n", "min"),
            max_n=("n", "max"),
            total_n=("n", "sum"),
            min_defaults=("defaults", "min"),
            total_defaults=("defaults", "sum"),
            mean_default_rate=("default_rate", "mean"),
            min_default_rate=("default_rate", "min"),
            max_default_rate=("default_rate", "max"),
            std_default_rate=("default_rate", "std"),
            mean_volume_share=("volume_share", "mean"),
            min_volume_share=("volume_share", "min"),
            max_volume_share=("volume_share", "max"),
            std_volume_share=("volume_share", "std"),
        )
        .reset_index()
    )

    stability_summary["cv_default_rate"] = (
        stability_summary["std_default_rate"]
        / stability_summary["mean_default_rate"]
    )
    stability_summary["cv_volume_share"] = (
        stability_summary["std_volume_share"]
        / stability_summary["mean_volume_share"]
    )
    stability_summary = class_metadata.merge(
        stability_summary,
        on=class_col,
        how="right",
    )

    risk_pivot = stability_by_period.pivot(
        index=period_col,
        columns=class_col,
        values="default_rate",
    ).reset_index()
    volume_pivot = stability_by_period.pivot(
        index=period_col,
        columns=class_col,
        values="volume_share",
    ).reset_index()

    return stability_by_period, stability_summary, risk_pivot, volume_pivot
