from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(
    "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring"
)
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "summary_models.xlsx"
TRAIN_PATH = PROJECT_ROOT / "data" / "train_discretized.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_selection"
OUTPUT_PATH = OUTPUT_DIR / "best_4_variables_logit_model.pkl"


def is_ok_model(row):
    return (
        row["global_signif_flag"] == "OK"
        and row["variables_signif_flag"] == "OK"
        and row["modalities_signif_flag"] == "OK"
        and row["vif_flag"] == "OK"
    )


summary_4 = pd.read_excel(SUMMARY_PATH, sheet_name="summary_4")
eligible = summary_4.loc[summary_4.apply(is_ok_model, axis=1)].copy()

if eligible.empty:
    raise ValueError("Aucun modele a 4 variables ne respecte tous les criteres.")

eligible = eligible.sort_values("Gini_penalized", ascending=False)
best_row = eligible.iloc[0]
formula = best_row["formula"]

train_df = pd.read_csv(TRAIN_PATH)
model_result = smf.logit(formula=formula, data=train_df).fit(
    disp=False,
    maxiter=100,
)


def parse_reference(value):
    value = str(value).strip()

    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        return value


def build_variable_specs(formula, params):
    references = {
        match.group(1): parse_reference(match.group(2))
        for match in re.finditer(
            r'C\(Q\("([^"]+)"\),\s*Treatment\(reference=([^)]+)\)\)',
            formula,
        )
    }

    specs = {
        variable: {
            "reference": reference,
            "coefficients": {},
        }
        for variable, reference in references.items()
    }

    coef_pattern = re.compile(
        r'C\(Q\("([^"]+)"\),\s*Treatment\(reference=[^)]+\)\)\[T\.(.+)\]'
    )

    for name, value in params.items():
        match = coef_pattern.match(name)

        if match is None:
            continue

        variable = match.group(1)
        modality = match.group(2)
        specs[variable]["coefficients"][str(modality)] = float(value)

    return specs


def predict_proba_from_artifact(artifact, df):
    linear_score = np.full(len(df), artifact["intercept"], dtype=float)

    for variable, spec in artifact["variable_specs"].items():
        values = df[variable].astype(str)

        for modality, coefficient in spec["coefficients"].items():
            linear_score += (values == str(modality)).to_numpy() * coefficient

    return 1 / (1 + np.exp(-linear_score))

artifact = {
    "artifact_type": "logit_score_model",
    "formula": formula,
    "variables": best_row["variables"],
    "references": best_row["references"],
    "intercept": float(model_result.params["Intercept"]),
    "params": model_result.params.to_dict(),
    "variable_specs": build_variable_specs(formula, model_result.params.to_dict()),
    "selection_rule": "checks OK puis Gini_penalized maximal parmi les modeles a 4 variables",
    "metrics": {
        "Gini_train": best_row["Gini"],
        "Gini_test": best_row["Gini_test"],
        "Gini_OOT": best_row["Gini_OOT"],
        "Gini_folds_mean": best_row["Gini_folds_mean"],
        "Gini_penalized": best_row["Gini_penalized"],
        "Recall_penalized": best_row.get("Recall_penalized"),
        "F1_penalized": best_row.get("F1_penalized"),
        "PR_AUC_penalized": best_row.get("PR_AUC_penalized"),
    },
    "target": "def",
    "train_source": str(TRAIN_PATH),
    "score_usage": (
        "linear_score = intercept + somme des coefficients des modalites "
        "observees; probability = 1 / (1 + exp(-linear_score))."
    ),
}

artifact["sample_train_probabilities"] = [
    float(value)
    for value in predict_proba_from_artifact(artifact, train_df.head(5))
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("wb") as file:
    pickle.dump(artifact, file)

print(f"Saved {OUTPUT_PATH}")
print(f"Formula: {formula}")
print(f"Gini penalized: {best_row['Gini_penalized']:.6f}")
print(
    "Sample probabilities:",
    [round(value, 6) for value in artifact["sample_train_probabilities"]],
)
