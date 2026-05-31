from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor


def auc_score(y_true, y_score):
    """Compute AUC without sklearn, using average ranks for ties."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    ranks = y_score.rank(method="average")
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return np.nan

    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def gini_score(y_true, y_score):
    auc = auc_score(y_true, y_score)

    if pd.isna(auc):
        return np.nan

    return 2 * auc - 1


def create_stratified_folds_without_sklearn(
    train_df,
    n_splits=4,
    target="def",
    year_col="year",
    fold_col="fold",
    random_state=42,
):
    """Create folds stratified by target and year without sklearn."""
    df = train_df.copy()
    rng = np.random.default_rng(random_state)

    df[fold_col] = -1
    df["_strat_col"] = df[target].astype(str) + "_" + df[year_col].astype(str)

    for _, idx in df.groupby("_strat_col").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)

        assigned_folds = np.resize(np.arange(1, n_splits + 1), len(idx))
        df.loc[idx, fold_col] = assigned_folds

    return df.drop(columns="_strat_col")


def get_reference_modalities(train_df, variables, target="def"):
    """Return the lowest-risk modality for each variable."""
    references = {}

    for var in variables:
        rates = train_df.groupby(var, dropna=False)[target].mean().sort_values()
        references[var] = rates.index[0]

    return references


def build_logit_formula(vars_combo, references, target="def"):
    terms = []

    for var in vars_combo:
        ref = references[var]

        if hasattr(ref, "item"):
            ref = ref.item()

        terms.append(f'C(Q("{var}"), Treatment(reference={repr(ref)}))')

    return f'Q("{target}") ~ ' + " + ".join(terms)


def generate_logit_formulas(train_df, variables, k, target="def"):
    """Yield formula dictionaries for all k-variable combinations."""
    for vars_combo in combinations(variables, k):
        references = get_reference_modalities(
            train_df=train_df,
            variables=list(vars_combo),
            target=target,
        )

        formula = build_logit_formula(
            vars_combo=vars_combo,
            references=references,
            target=target,
        )

        yield {
            "variables": vars_combo,
            "references": references,
            "formula": formula,
        }


def get_coef_names_for_variable(model, variable):
    return [
        name
        for name in model.params.index
        if name != "Intercept"
        and (
            f'Q("{variable}")' in name
            or f"C({variable}" in name
            or name.startswith(f"{variable}[")
            or name == variable
        )
    ]


def get_variable_joint_pvalues(model, model_variables):
    """Compute one joint Wald p-value per model variable."""
    params = model.params
    cov = model.cov_params()
    pvalues = {}

    for var in model_variables:
        coef_names = get_coef_names_for_variable(model, var)

        if len(coef_names) == 0:
            pvalues[var] = np.nan
            continue

        beta = params.loc[coef_names].values
        cov_beta = cov.loc[coef_names, coef_names].values

        stat = float(beta.T @ np.linalg.pinv(cov_beta) @ beta)
        pvalues[var] = chi2.sf(stat, df=len(coef_names))

    return pvalues


def get_modality_pvalues(model, model_variables):
    """Return p-values for every non-reference modality coefficient."""
    pvalues = {}

    for var in model_variables:
        coef_names = get_coef_names_for_variable(model, var)

        for coef_name in coef_names:
            pvalues[coef_name] = model.pvalues.get(coef_name, np.nan)

    return pvalues


def flag_pvalue_lt_alpha(pvalue, alpha=0.05):
    if pd.isna(pvalue):
        return "KO"

    return "OK" if pvalue < alpha else "KO"


def check_global_significance(model, alpha=0.05):
    pvalue = model.llr_pvalue
    flag = flag_pvalue_lt_alpha(pvalue, alpha)

    if flag == "OK":
        label = f"OK - modele globalement significatif, p-value={pvalue:.4g}"
    else:
        label = f"KO - modele non significatif globalement, p-value={pvalue:.4g}"

    return flag, label, pvalue


def check_dict_pvalues(pvalues, alpha=0.05, object_name="elements"):
    ko_items = [
        name
        for name, pvalue in pvalues.items()
        if pd.isna(pvalue) or pvalue >= alpha
    ]

    if len(ko_items) == 0:
        return "OK", f"OK - tous les {object_name} sont significatifs"

    return "KO", f"KO - {object_name} non significatifs : " + " | ".join(ko_items)


def calculate_vif(model):
    """Compute VIF for every explanatory coefficient except the intercept."""
    exog = pd.DataFrame(
        model.model.exog,
        columns=model.model.exog_names,
    )

    vif_values = {}

    for column in exog.columns:
        if column == "Intercept":
            continue

        column_index = exog.columns.get_loc(column)
        vif_values[column] = variance_inflation_factor(exog.values, column_index)

    return vif_values


def check_vif(vif_values, threshold=10):
    ko_items = [
        name
        for name, value in vif_values.items()
        if pd.isna(value) or np.isinf(value) or value >= threshold
    ]

    if len(ko_items) == 0:
        return "OK", f"OK - tous les VIF sont < {threshold}"

    return "KO", f"KO - VIF >= {threshold} : " + " | ".join(ko_items)


def evaluate_auc_gini(model, df, target):
    y_score = model.predict(df)

    auc = auc_score(df[target], y_score)
    gini = gini_score(df[target], y_score)

    return auc, gini, y_score


def _prepare_fold_datasets(train_df_with_folds, fold_col):
    fold_datasets = {}

    for fold_id in sorted(train_df_with_folds[fold_col].unique()):
        train_fold = (
            train_df_with_folds
            .loc[train_df_with_folds[fold_col] != fold_id]
            .drop(columns=[fold_col])
        )

        valid_fold = (
            train_df_with_folds
            .loc[train_df_with_folds[fold_col] == fold_id]
            .drop(columns=[fold_col])
        )

        fold_datasets[fold_id] = {
            "train": train_fold,
            "valid": valid_fold,
        }

    return fold_datasets


def _add_train_error_metrics(row, error_message):
    row.update({
        "global_signif_flag": "KO",
        "global_signif_label": f"KO - erreur entrainement train : {error_message}",
        "global_pvalue": np.nan,
        "variables_signif_flag": "KO",
        "variables_signif_label": f"KO - erreur entrainement train : {error_message}",
        "variables_pvalues": {},
        "modalities_signif_flag": "KO",
        "modalities_signif_label": f"KO - erreur entrainement train : {error_message}",
        "modalities_pvalues": {},
        "vif_flag": "KO",
        "vif_label": f"KO - erreur entrainement train : {error_message}",
        "vif_values": {},
        "AUC": np.nan,
        "Gini": np.nan,
        "AUC_test": np.nan,
        "Gini_test": np.nan,
        "AUC_OOT": np.nan,
        "Gini_OOT": np.nan,
        "error": error_message,
    })


def train_models_with_all_checks(
    formula_combinations,
    train_df_with_folds,
    test_df,
    oot_df,
    variables,
    target="def",
    fold_col="fold",
    alpha=0.05,
    vif_threshold=10,
    maxiter=100,
):
    """
    Train candidate logit models and return one wide result row per formula.

    The full-train model carries significance checks and train/test/OOT
    AUC-Gini. Fold models only return AUC and Gini on their validation fold.
    """
    rows = []

    train_full = train_df_with_folds.drop(columns=[fold_col])
    fold_datasets = _prepare_fold_datasets(train_df_with_folds, fold_col)

    for item in formula_combinations:
        formula = item["formula"]
        vars_combo = tuple(item["variables"])
        references = item.get("references", {})

        unknown_vars = set(vars_combo) - set(variables)

        if len(unknown_vars) > 0:
            raise ValueError(f"Variables inconnues dans la formule : {unknown_vars}")

        row = {
            "formula": formula,
            "variables": vars_combo,
            "references": references,
        }

        try:
            model_train = smf.logit(
                formula=formula,
                data=train_full,
            ).fit(disp=False, maxiter=maxiter)

            global_flag, global_label, global_pvalue = check_global_significance(
                model_train,
                alpha=alpha,
            )

            variable_pvalues_train = get_variable_joint_pvalues(
                model_train,
                vars_combo,
            )

            variables_flag, variables_label = check_dict_pvalues(
                variable_pvalues_train,
                alpha=alpha,
                object_name="variables",
            )

            modality_pvalues_train = get_modality_pvalues(
                model_train,
                vars_combo,
            )

            modalities_flag, modalities_label = check_dict_pvalues(
                modality_pvalues_train,
                alpha=alpha,
                object_name="modalites",
            )

            vif_values = calculate_vif(model_train)
            vif_flag, vif_label = check_vif(
                vif_values,
                threshold=vif_threshold,
            )

            auc_train, gini_train, _ = evaluate_auc_gini(
                model_train,
                train_full,
                target,
            )

            auc_test, gini_test, _ = evaluate_auc_gini(
                model_train,
                test_df,
                target,
            )

            auc_oot, gini_oot, _ = evaluate_auc_gini(
                model_train,
                oot_df,
                target,
            )

            row.update({
                "global_signif_flag": global_flag,
                "global_signif_label": global_label,
                "global_pvalue": global_pvalue,
                "variables_signif_flag": variables_flag,
                "variables_signif_label": variables_label,
                "variables_pvalues": variable_pvalues_train,
                "modalities_signif_flag": modalities_flag,
                "modalities_signif_label": modalities_label,
                "modalities_pvalues": modality_pvalues_train,
                "vif_flag": vif_flag,
                "vif_label": vif_label,
                "vif_values": vif_values,
                "AUC": auc_train,
                "Gini": gini_train,
                "AUC_test": auc_test,
                "Gini_test": gini_test,
                "AUC_OOT": auc_oot,
                "Gini_OOT": gini_oot,
                "error": None,
            })

        except Exception as exc:
            _add_train_error_metrics(row, str(exc))

        for fold_id, fold_data in fold_datasets.items():
            try:
                model_fold = smf.logit(
                    formula=formula,
                    data=fold_data["train"],
                ).fit(disp=False, maxiter=maxiter)

                auc_fold, gini_fold, _ = evaluate_auc_gini(
                    model_fold,
                    fold_data["valid"],
                    target,
                )

                row.update({
                    f"AUC_fold{fold_id}": auc_fold,
                    f"Gini_fold{fold_id}": gini_fold,
                    f"error_fold{fold_id}": None,
                })

            except Exception as exc:
                row.update({
                    f"AUC_fold{fold_id}": np.nan,
                    f"Gini_fold{fold_id}": np.nan,
                    f"error_fold{fold_id}": str(exc),
                })

        rows.append(row)

    results = pd.DataFrame(rows)

    gini_fold_cols = [
        col for col in results.columns
        if col.startswith("Gini_fold")
    ]

    auc_fold_cols = [
        col for col in results.columns
        if col.startswith("AUC_fold")
    ]

    results["AUC_folds_mean"] = results[auc_fold_cols].mean(axis=1)
    results["Gini_folds_mean"] = results[gini_fold_cols].mean(axis=1)
    results["Gini_folds_min"] = results[gini_fold_cols].min(axis=1)
    results["Gini_folds_std"] = results[gini_fold_cols].std(axis=1)

    # Stability heuristic: reward validation Gini and penalize train drift.
    results["Gini_penalized"] = (
        results["Gini_folds_mean"]
        - (results["Gini"] - results["Gini_test"]).abs()
        - (results["Gini"] - results["Gini_OOT"]).abs()
    )

    flag_cols = [
        col
        for col in results.columns
        if col.endswith("_flag") or "_flag_fold" in col
    ]

    results["all_checks_OK"] = results[flag_cols].eq("OK").all(axis=1)

    return results
