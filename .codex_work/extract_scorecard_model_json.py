from pathlib import Path
import json
import pickle


PROJECT_ROOT = Path(
    "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring"
)
PICKLE_PATH = PROJECT_ROOT / "outputs" / "model_selection" / "best_4_variables_logit_scorecard.pkl"
JSON_PATH = PROJECT_ROOT / ".codex_work" / "best_4_variables_scorecard.json"


with PICKLE_PATH.open("rb") as file:
    artifact = pickle.load(file)

scorecard_rows = artifact["scorecard_table"]

payload = {
    "artifact_type": artifact["artifact_type"],
    "formula": artifact["formula"],
    "target": artifact["target"],
    "variables": artifact["variables"],
    "metrics": artifact["metrics"],
    "scorecard_table": scorecard_rows,
}

with JSON_PATH.open("w", encoding="utf-8") as file:
    json.dump(payload, file, ensure_ascii=True, indent=2)

print(JSON_PATH)
