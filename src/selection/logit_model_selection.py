from itertools import combinations
import ast
import pickle
import re

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


def roc_curve_without_sklearn(y_true, y_score):
    """Return ROC curve points sorted by decreasing score."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return pd.DataFrame({
            "threshold": [np.nan],
            "fpr": [np.nan],
            "tpr": [np.nan],
        })

    curve_df = (
        pd.DataFrame({"y_true": y_true, "y_score": y_score})
        .sort_values("y_score", ascending=False)
        .groupby("y_score", sort=False)["y_true"]
        .agg(["sum", "count"])
    )

    tps = curve_df["sum"].cumsum()
    fps = (curve_df["count"] - curve_df["sum"]).cumsum()

    roc_df = pd.DataFrame({
        "threshold": curve_df.index.to_numpy(),
        "fpr": (fps / n_neg).to_numpy(),
        "tpr": (tps / n_pos).to_numpy(),
    })

    start = pd.DataFrame({"threshold": [np.inf], "fpr": [0.0], "tpr": [0.0]})
    end = pd.DataFrame({"threshold": [-np.inf], "fpr": [1.0], "tpr": [1.0]})

    return pd.concat([start, roc_df, end], ignore_index=True)


def pr_auc_score(y_true, y_score):
    """Compute non-interpolated Precision-Recall AUC, also called AP."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    n_pos = (y_true == 1).sum()

    if n_pos == 0:
        return np.nan

    curve_df = (
        pd.DataFrame({"y_true": y_true, "y_score": y_score})
        .sort_values("y_score", ascending=False)
        .groupby("y_score", sort=False)["y_true"]
        .agg(["sum", "count"])
    )

    tps = curve_df["sum"].cumsum()
    fps = (curve_df["count"] - curve_df["sum"]).cumsum()
    precision = tps / (tps + fps)
    recall = tps / n_pos
    recall_step = recall.diff().fillna(recall)

    return float((precision * recall_step).sum())


def recall_score(y_true, y_score, threshold=0.5):
    """Compute recall/sensitivity at the selected score threshold."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    n_pos = (y_true == 1).sum()

    if n_pos == 0:
        return np.nan

    y_pred = y_score >= threshold
    true_positives = ((y_true == 1) & y_pred).sum()

    return true_positives / n_pos


def f1_score(y_true, y_score, threshold=0.5):
    """Compute F1-score at the selected score threshold."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)

    y_pred = y_score >= threshold
    true_positives = ((y_true == 1) & y_pred).sum()
    false_positives = ((y_true == 0) & y_pred).sum()
    false_negatives = ((y_true == 1) & ~y_pred).sum()
    denominator = 2 * true_positives + false_positives + false_negatives

    if denominator == 0:
        return np.nan

    return 2 * true_positives / denominator


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


def evaluate_model_metrics(model, df, target, recall_threshold=0.5):
    y_score = model.predict(df)

    metrics = {
        "AUC": auc_score(df[target], y_score),
        "Gini": gini_score(df[target], y_score),
        "PR_AUC": pr_auc_score(df[target], y_score),
        "Recall": recall_score(df[target], y_score, threshold=recall_threshold),
        "F1": f1_score(df[target], y_score, threshold=recall_threshold),
    }

    return metrics, y_score


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
        "global_signif_label": f"KO - train fitting error: {error_message}",
        "global_pvalue": np.nan,
        "variables_signif_flag": "KO",
        "variables_signif_label": f"KO - train fitting error: {error_message}",
        "variables_pvalues": {},
        "modalities_signif_flag": "KO",
        "modalities_signif_label": f"KO - train fitting error: {error_message}",
        "modalities_pvalues": {},
        "vif_flag": "KO",
        "vif_label": f"KO - train fitting error: {error_message}",
        "vif_values": {},
        "AUC": np.nan,
        "Gini": np.nan,
        "PR_AUC": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "AUC_test": np.nan,
        "Gini_test": np.nan,
        "PR_AUC_test": np.nan,
        "Recall_test": np.nan,
        "F1_test": np.nan,
        "AUC_OOT": np.nan,
        "Gini_OOT": np.nan,
        "PR_AUC_OOT": np.nan,
        "Recall_OOT": np.nan,
        "F1_OOT": np.nan,
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
    recall_threshold=0.5,
    maxiter=100,
):
    """
    Train candidate logit models and return one wide result row per formula.

    The full-train model carries significance checks and train/test/OOT
    AUC-Gini, Precision-Recall AUC, Recall/Sensitivity and F1-score. Fold
    models only return performance metrics on their validation fold.
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

            train_metrics, _ = evaluate_model_metrics(
                model_train,
                train_full,
                target,
                recall_threshold=recall_threshold,
            )

            test_metrics, _ = evaluate_model_metrics(
                model_train,
                test_df,
                target,
                recall_threshold=recall_threshold,
            )

            oot_metrics, _ = evaluate_model_metrics(
                model_train,
                oot_df,
                target,
                recall_threshold=recall_threshold,
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
                "AUC": train_metrics["AUC"],
                "Gini": train_metrics["Gini"],
                "PR_AUC": train_metrics["PR_AUC"],
                "Recall": train_metrics["Recall"],
                "F1": train_metrics["F1"],
                "AUC_test": test_metrics["AUC"],
                "Gini_test": test_metrics["Gini"],
                "PR_AUC_test": test_metrics["PR_AUC"],
                "Recall_test": test_metrics["Recall"],
                "F1_test": test_metrics["F1"],
                "AUC_OOT": oot_metrics["AUC"],
                "Gini_OOT": oot_metrics["Gini"],
                "PR_AUC_OOT": oot_metrics["PR_AUC"],
                "Recall_OOT": oot_metrics["Recall"],
                "F1_OOT": oot_metrics["F1"],
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

                fold_metrics, _ = evaluate_model_metrics(
                    model_fold,
                    fold_data["valid"],
                    target,
                    recall_threshold=recall_threshold,
                )

                row.update({
                    f"AUC_fold{fold_id}": fold_metrics["AUC"],
                    f"Gini_fold{fold_id}": fold_metrics["Gini"],
                    f"PR_AUC_fold{fold_id}": fold_metrics["PR_AUC"],
                    f"Recall_fold{fold_id}": fold_metrics["Recall"],
                    f"F1_fold{fold_id}": fold_metrics["F1"],
                    f"error_fold{fold_id}": None,
                })

            except Exception as exc:
                row.update({
                    f"AUC_fold{fold_id}": np.nan,
                    f"Gini_fold{fold_id}": np.nan,
                    f"PR_AUC_fold{fold_id}": np.nan,
                    f"Recall_fold{fold_id}": np.nan,
                    f"F1_fold{fold_id}": np.nan,
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

    pr_auc_fold_cols = [
        col for col in results.columns
        if col.startswith("PR_AUC_fold")
    ]

    recall_fold_cols = [
        col for col in results.columns
        if col.startswith("Recall_fold")
    ]

    f1_fold_cols = [
        col for col in results.columns
        if col.startswith("F1_fold")
    ]

    results["AUC_folds_mean"] = results[auc_fold_cols].mean(axis=1)
    results["Gini_folds_mean"] = results[gini_fold_cols].mean(axis=1)
    results["Gini_folds_min"] = results[gini_fold_cols].min(axis=1)
    results["Gini_folds_std"] = results[gini_fold_cols].std(axis=1)
    results["PR_AUC_folds_mean"] = results[pr_auc_fold_cols].mean(axis=1)
    results["PR_AUC_folds_min"] = results[pr_auc_fold_cols].min(axis=1)
    results["PR_AUC_folds_std"] = results[pr_auc_fold_cols].std(axis=1)
    results["Recall_folds_mean"] = results[recall_fold_cols].mean(axis=1)
    results["Recall_folds_min"] = results[recall_fold_cols].min(axis=1)
    results["Recall_folds_std"] = results[recall_fold_cols].std(axis=1)
    results["F1_folds_mean"] = results[f1_fold_cols].mean(axis=1)
    results["F1_folds_min"] = results[f1_fold_cols].min(axis=1)
    results["F1_folds_std"] = results[f1_fold_cols].std(axis=1)

    # Stability heuristic: reward validation Gini and penalize train drift.
    results["Gini_penalized"] = (
        results["Gini_folds_mean"]
        - (results["Gini"] - results["Gini_test"]).abs()
        - (results["Gini"] - results["Gini_OOT"]).abs()
    )
    results["PR_AUC_penalized"] = (
        results["PR_AUC_folds_mean"]
        - (results["PR_AUC"] - results["PR_AUC_test"]).abs()
        - (results["PR_AUC"] - results["PR_AUC_OOT"]).abs()
    )
    results["Recall_penalized"] = (
        results["Recall_folds_mean"]
        - (results["Recall"] - results["Recall_test"]).abs()
        - (results["Recall"] - results["Recall_OOT"]).abs()
    )
    results["F1_penalized"] = (
        results["F1_folds_mean"]
        - (results["F1"] - results["F1_test"]).abs()
        - (results["F1"] - results["F1_OOT"]).abs()
    )

    flag_cols = [
        col
        for col in results.columns
        if col.endswith("_flag") or "_flag_fold" in col
    ]

    results["all_checks_OK"] = results[flag_cols].eq("OK").all(axis=1)

    return results


def _as_python_scalar(value):
    if hasattr(value, "item"):
        return value.item()

    return value


def _reference_to_label(value):
    value = _as_python_scalar(value)

    if pd.isna(value):
        return "nan"

    return str(value)


def _parse_reference(raw_reference):
    raw_reference = str(raw_reference).strip()

    if (
        (raw_reference.startswith("'") and raw_reference.endswith("'"))
        or (raw_reference.startswith('"') and raw_reference.endswith('"'))
    ):
        return raw_reference[1:-1]

    try:
        return ast.literal_eval(raw_reference)
    except (ValueError, SyntaxError):
        return raw_reference


def extract_formula_references(formula):
    """Extract variable references from a statsmodels categorical formula."""
    references = {}
    pattern = re.compile(
        r'C\(Q\("([^"]+)"\),\s*Treatment\(reference=([^)]+)\)\)'
    )

    for variable, raw_reference in pattern.findall(formula):
        references[variable] = _parse_reference(raw_reference)

    return references


def extract_formula_variables(formula):
    """Extract categorical variable order from a statsmodels formula."""
    return list(extract_formula_references(formula).keys())


def select_best_checked_model(summary_df, n_variables=None):
    """Select the best checked model by maximum penalized Gini."""
    candidates = summary_df.copy()

    if n_variables is not None and "variables" in candidates.columns:
        candidates = candidates.loc[
            candidates["variables"].map(lambda value: _count_variables(value) == n_variables)
        ]

    checked_mask = (
        candidates["global_signif_flag"].eq("OK")
        & candidates["variables_signif_flag"].eq("OK")
        & candidates["modalities_signif_flag"].eq("OK")
        & candidates["vif_flag"].eq("OK")
    )

    if "all_checks_OK" in candidates.columns:
        checked_mask = checked_mask & candidates["all_checks_OK"].astype(bool)

    candidates = candidates.loc[checked_mask].copy()

    if candidates.empty:
        raise ValueError("Aucun modele ne respecte tous les criteres de selection.")

    return candidates.sort_values("Gini_penalized", ascending=False).iloc[0]


def _count_variables(value):
    if isinstance(value, (tuple, list)):
        return len(value)

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (tuple, list)):
            return len(parsed)
    except (ValueError, SyntaxError):
        pass

    return len(extract_formula_variables(str(value)))


def build_scorecard_table(model, formula):
    """Build a variable/modality/coefficient table for a categorical logit."""
    references = extract_formula_references(formula)
    params = model.params.to_dict()
    rows = []
    coef_pattern = re.compile(
        r'C\(Q\("([^"]+)"\),\s*Treatment\(reference=[^)]+\)\)\[T\.(.+)\]'
    )
    coefficients_by_variable = {variable: {} for variable in references}

    for coef_name, coefficient in params.items():
        match = coef_pattern.match(coef_name)

        if match is None:
            continue

        variable = match.group(1)
        modality = match.group(2)
        coefficients_by_variable.setdefault(variable, {})[str(modality)] = {
            "coef_name": coef_name,
            "coefficient": float(coefficient),
        }

    for variable_index, (variable, reference) in enumerate(references.items(), start=1):
        reference_label = _reference_to_label(reference)
        rows.append({
            "variable_number": variable_index,
            "variable": variable,
            "modality": reference_label,
            "coefficient": 0.0,
            "coef_name": "REFERENCE",
            "is_reference": True,
        })

        for modality, coef_info in coefficients_by_variable.get(variable, {}).items():
            rows.append({
                "variable_number": variable_index,
                "variable": variable,
                "modality": modality,
                "coefficient": coef_info["coefficient"],
                "coef_name": coef_info["coef_name"],
                "is_reference": False,
            })

    rows.append({
        "variable_number": "",
        "variable": "Intercept",
        "modality": "Intercept",
        "coefficient": float(params["Intercept"]),
        "coef_name": "Intercept",
        "is_reference": False,
    })

    return pd.DataFrame(rows)


def build_score_model_artifact(model, selected_row, target="def"):
    """Build a pickle-safe score model artifact for future scoring."""
    formula = selected_row["formula"]
    scorecard_table = build_scorecard_table(model, formula)
    variable_specs = {}

    for variable, group in scorecard_table.groupby("variable", sort=False):
        if variable == "Intercept":
            continue

        reference_rows = group.loc[group["is_reference"]]
        reference = reference_rows["modality"].iloc[0] if not reference_rows.empty else None
        coefficients = (
            group
            .loc[~group["is_reference"], ["modality", "coefficient"]]
            .set_index("modality")["coefficient"]
            .to_dict()
        )
        variable_specs[variable] = {
            "reference": reference,
            "coefficients": {str(key): float(value) for key, value in coefficients.items()},
        }

    return {
        "artifact_type": "categorical_logit_scorecard",
        "formula": formula,
        "target": target,
        "variables": extract_formula_variables(formula),
        "selection_rule": "checks OK puis Gini_penalized maximal",
        "intercept": float(model.params["Intercept"]),
        "params": {key: float(value) for key, value in model.params.to_dict().items()},
        "scorecard_table": scorecard_table.to_dict(orient="records"),
        "variable_specs": variable_specs,
        "metrics": {
            "Gini_train": selected_row.get("Gini"),
            "Gini_test": selected_row.get("Gini_test"),
            "Gini_OOT": selected_row.get("Gini_OOT"),
            "Gini_folds_mean": selected_row.get("Gini_folds_mean"),
            "Gini_penalized": selected_row.get("Gini_penalized"),
            "Recall_penalized": selected_row.get("Recall_penalized"),
            "F1_penalized": selected_row.get("F1_penalized"),
            "PR_AUC_penalized": selected_row.get("PR_AUC_penalized"),
        },
    }


def save_best_model_pickle(
    summary_df,
    train_df,
    output_path,
    n_variables=4,
    target="def",
    maxiter=100,
):
    """Fit, save and return the best checked categorical logit score artifact."""
    selected_row = select_best_checked_model(summary_df, n_variables=n_variables)
    model = smf.logit(
        formula=selected_row["formula"],
        data=train_df,
    ).fit(disp=False, maxiter=maxiter)
    artifact = build_score_model_artifact(
        model=model,
        selected_row=selected_row,
        target=target,
    )

    with open(output_path, "wb") as file:
        pickle.dump(artifact, file)

    return artifact


def load_score_model_pickle(input_path):
    with open(input_path, "rb") as file:
        return pickle.load(file)


def predict_score_model_proba(artifact, df):
    """Score observations from a saved categorical logit score artifact."""
    linear_score = np.full(len(df), artifact["intercept"], dtype=float)

    for variable, spec in artifact["variable_specs"].items():
        values = df[variable].astype(str)

        for modality, coefficient in spec["coefficients"].items():
            linear_score += (values == str(modality)).to_numpy() * coefficient

    return pd.Series(
        1 / (1 + np.exp(-linear_score)),
        index=df.index,
        name="pd_score",
    )


def evaluate_saved_score_model(artifact, datasets, target="def"):
    """Compute AUC, Gini and ROC curves for named datasets."""
    metric_rows = []
    roc_curves = {}

    for dataset_name, df in datasets.items():
        y_score = predict_score_model_proba(artifact, df)
        auc = auc_score(df[target], y_score)
        gini = gini_score(df[target], y_score)

        metric_rows.append({
            "dataset": dataset_name,
            "AUC": auc,
            "Gini": gini,
        })
        roc_curves[dataset_name] = roc_curve_without_sklearn(df[target], y_score)

    return pd.DataFrame(metric_rows), roc_curves
