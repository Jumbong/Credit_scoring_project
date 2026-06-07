# PD Credit Scoring Project

This project builds a probability of default model with a staged workflow:

1. `01_exploratory_data_analysis.qmd`
2. `02_data_cleaning_imputation.qmd`
3. `03_correlation_analysis.qmd`
4. `04_variable_selection.qmd`
5. `05_monotonicity_stability.qmd`
6. `06_variable_discretization.qmd`
7. `07_feature_discrimination_visualization.qmd`
8. `08_logistic_model_selection.qmd`

Main folders:

- `data/`: input and intermediate datasets.
- `outputs/`: reports, model summaries, charts, stability tables, and analysis outputs.
- `cv_folds_pickle/`: persisted cross-validation folds in pickle format.
- `cv_folds_csv/`: persisted cross-validation folds in CSV format.
- `src/`: reusable Python functions.
