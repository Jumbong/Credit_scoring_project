
import pandas as pd



def build_distribution_summary(datasets: dict, target="def"):

    table = pd.DataFrame({
        name: {
            "Nb obs": len(df),
            "Nb def": df[target].sum(),
            "Default rate": df[target].mean()
        }
        for name, df in datasets.items()
    }).T.reset_index().rename(columns={"index": "Database"})

    total = pd.DataFrame({
        "Database": ["Total"],
        "Nb obs": [table["Nb obs"].sum()],
        "Nb def": [table["Nb def"].sum()],
        "Default rate": [table["Nb def"].sum()/table["Nb obs"].sum()]
    })

    return pd.concat([table, total], ignore_index=True)


def build_bounds_table(df, lower=0.01, upper=0.99):

    bounds = (
        df.quantile([lower, upper])
        .T
        .reset_index()
        .rename(columns={
            "index": "Variable",
            lower: "Lower Bound",
            upper: "Upper Bound"
        })
    )

    return bounds


def apply_iqr_bounds(train, test, oot, variables):

    train = train.copy()
    test = test.copy()
    oot = oot.copy()

    bounds = []

    for var in variables:

        Q1 = train[var].quantile(0.25)
        Q3 = train[var].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        bounds.append({
            "Variable": var,
            "Lower Bound": lower,
            "Upper Bound": upper
        })

        for df in [train, test, oot]:
            df[var] = df[var].clip(lower, upper)

    bounds_table = pd.DataFrame(bounds)

    return bounds_table, train, test, oot

def impute_missing_values(train, test, oot,
                          emp_var="person_emp_length",
                          rate_var="loan_int_rate",
                          emp_value=0):
    """
    Impute missing values using statistics computed on the training dataset.

    Parameters
    ----------
    train, test, oot : pandas.DataFrame
        Datasets to process.
        
    emp_var : str
        Variable representing employment length.
        
    rate_var : str
        Variable representing interest rate.
        
    emp_value : int or float
        Value used to impute employment length (conservative strategy).

    Returns
    -------
    train_imp, test_imp, oot_imp : pandas.DataFrame
        Imputed datasets.
    """

    # Copy datasets to avoid modifying originals
    train_imp = train.copy()
    test_imp = test.copy()
    oot_imp = oot.copy()

    # ----------------------------
    # Compute statistics on TRAIN
    # ----------------------------

    rate_median = train_imp[rate_var].median()

    # ----------------------------
    # Create missing indicators
    # ----------------------------

    for df in [train_imp, test_imp, oot_imp]:

        df[f"{emp_var}_missing"] = df[emp_var].isnull().astype(int)
        df[f"{rate_var}_missing"] = df[rate_var].isnull().astype(int)

    # ----------------------------
    # Apply imputations
    # ----------------------------

    for df in [train_imp, test_imp, oot_imp]:

        df[emp_var] = df[emp_var].fillna(emp_value)
        df[rate_var] = df[rate_var].fillna(rate_median)

    return train_imp, test_imp, oot_imp

