
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def plot_default_rate_ax(data, variable, target, bins=10, ax=None):
    """
    Plot default rate by binned numerical variable on a given matplotlib axis.
    """

    df = data[[variable, target]].copy()

    # Create bins
    df[f"{variable}_bin"] = pd.qcut(
        df[variable],
        q=bins,
        duplicates="drop"
    )

    # Compute default rate by bin
    summary = (
        df.groupby(f"{variable}_bin", observed=True)[target]
        .mean()
        .reset_index()
    )

    # Convert intervals to strings for plotting
    summary[f"{variable}_bin"] = summary[f"{variable}_bin"].astype(str)

    # Plot
    ax.plot(
        summary[f"{variable}_bin"],
        summary[target],
        marker="o"
    )

    ax.set_title(f"Default rate by {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", rotation=45)

    return ax


def tx_rsq_par_var(df, categ_vars, date, target, cols=2, sharey=False):
    """
    Generate a grid of line charts showing the average event rate by category over time
    for a list of categorical variables.

    Parameters
    ----------
    df : pandas DataFrame
        Input dataset.

    categ_vars : list of str
        List of categorical variables to analyze.

    date : str
        Name of the date or time-period column.

    target : str
        Name of the binary target variable.
        The target should be coded as 1 for event/default and 0 otherwise.

    cols : int, default=2
        Number of columns in the subplot grid.

    sharey : bool, default=False
        Whether all subplots should share the same y-axis scale.

    Returns
    -------
    None
        The function displays the plots directly.
    """

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Check whether all required columns are present in the DataFrame
    missing_cols = [col for col in [date] + categ_vars if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following columns are missing from the DataFrame: {missing_cols}"
        )

    # Remove rows with missing values in the date column or categorical variables
    df = df.dropna(subset=[date] + categ_vars)

    # Determine the number of variables and the required number of subplot rows
    num_vars = len(categ_vars)
    rows = math.ceil(num_vars / cols)

    # Create the subplot grid
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 6, rows * 4),
        sharex=False,
        sharey=sharey
    )

    # Flatten the axes array to make iteration easier
    axes = axes.flatten()

    # Loop over each categorical variable and create one plot per variable
    for i, categ_var in enumerate(categ_vars):

        # Compute the average target value by date and category
        df_times_series = (
            df.groupby([date, categ_var])[target]
            .mean()
            .reset_index()
        )

        # Reshape the data so that each category becomes one line in the plot
        df_pivot = df_times_series.pivot(
            index=date,
            columns=categ_var,
            values=target
        )

        # Select the axis corresponding to the current variable
        ax = axes[i]

        # Plot one line per category
        for category in df_pivot.columns:
            ax.plot(
                df_pivot.index,
                df_pivot[category],
                label=str(category).strip()
            )

        # Set chart title and axis labels
        ax.set_title(f"{categ_var.strip()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Default rate (%)")

        # Adjust the legend depending on the number of categories
        if len(df_pivot.columns) > 10:
            ax.legend(
                title="Categories",
                fontsize="x-small",
                loc="upper left",
                ncol=2
            )
        else:
            ax.legend(
                title="Categories",
                fontsize="small",
                loc="upper left"
            )

    # Remove unused subplot axes when the grid is larger than the number of variables
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a global title to the figure
    fig.suptitle(
        "Default Rate by Categorical Variable",
        fontsize=10,
        x=0.5,
        y=1.02,
        ha="center"
    )

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()

    # Display the final figure
    plt.show()

def combined_barplot_lineplot(df, cat_vars, cible, cols=2):
    """
    Generate a grid of combined bar plots and line plots for a list of categorical variables.

    For each categorical variable:
    - the bar plot shows the relative frequency of each category;
    - the line plot shows the average target rate for each category.

    Parameters
    ----------
    df : pandas DataFrame
        Input dataset.

    cat_vars : list of str
        List of categorical variables to analyze.

    cible : str
        Name of the binary target variable.
        The target should be coded as 1 for event/default and 0 otherwise.

    cols : int, default=2
        Number of columns in the subplot grid.

    Returns
    -------
    None
        The function displays the plots directly.
    """

    # Count the number of categorical variables to plot
    num_vars = len(cat_vars)

    # Compute the number of rows needed for the subplot grid
    rows = math.ceil(num_vars / cols)

    # Create the subplot grid
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 6, rows * 4)
    )

    # Flatten the axes array to make iteration easier
    axes = axes.flatten()

    # Loop over each categorical variable
    for i, cat_col in enumerate(cat_vars):

        # Select the current subplot axis for the bar plot
        ax1 = axes[i]

        # Convert categorical dtype variables to string if needed
        # This avoids plotting issues with categorical intervals or ordered categories
        if pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype(str)

        # Compute the average target rate by category
        tx_rsq = (
            df.groupby([cat_col])[cible]
            .mean()
            .reset_index()
        )

        # Compute the relative frequency of each category
        effectifs = (
            df[cat_col]
            .value_counts(normalize=True)
            .reset_index()
        )

        # Rename columns for clarity
        effectifs.columns = [cat_col, "count"]

        # Merge category frequencies with target rates
        merged_data = (
            effectifs
            .merge(tx_rsq, on=cat_col)
            .sort_values(by=cible, ascending=True)
        )

        # Create a secondary y-axis for the line plot
        ax2 = ax1.twinx()

        # Plot category frequencies as bars
        sns.barplot(
            data=merged_data,
            x=cat_col,
            y="count",
            color="grey",
            ax=ax1
        )

        # Plot the average target rate as a line
        sns.lineplot(
            data=merged_data,
            x=cat_col,
            y=cible,
            color="red",
            marker="o",
            ax=ax2
        )

        # Set the subplot title and axis labels
        ax1.set_title(f"{cat_col}")
        ax1.set_xlabel("")
        ax1.set_ylabel("Category frequency")
        ax2.set_ylabel("Risk rate (%)")

        # Rotate x-axis labels for better readability
        ax1.tick_params(axis="x", rotation=45)

    # Remove unused subplot axes if the grid is larger than the number of variables
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a global title for the full figure
    fig.suptitle(
        "Combined Bar Plots and Line Plots for Categorical Variables",
        fontsize=10,
        x=0.0,
        y=1.02,
        ha="left"
    )

    # Adjust layout to reduce overlapping elements
    plt.tight_layout()

    # Display the final figure
    plt.show()

def test_freq_by_group(data, qualitative_vars, threshold=0.05):
    """
    Identifie les variables qualitatives qui ont au moins une modalité avec une fréquence relative
    inférieure ou égale au seuil spécifié.
    """
    # Liste pour stocker les variables correspondant au critère
    unique_mod_result = []

    for var in qualitative_vars:
        # Vérifie si la variable existe dans le DataFrame
        if var not in data.columns:
            print(f"Attention : la variable '{var}' n'existe pas dans le DataFrame.")
            continue
        
        # Calcul des fréquences relatives des modalités
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences
        print("\nFréquences relatives des modalités pour la variable :", var,"\n")
        print(value_counts)

        # Vérifie si au moins une modalité a une fréquence <= threshold
        if (value_counts <= threshold).any():
            unique_mod_result.append(var)

    # Message si aucune variable ne satisfait le critère
    if len(unique_mod_result) == 0:
        print("Aucune variable n'a de modalités avec moins de 5% d'effectifs.")
    else :
        print(f"Les variables suivantes ont au moins une modalité avec une fréquence <= {threshold * 100}% :")
        print(unique_mod_result)

    return unique_mod_result

def calculate_relative_difference(df, cat_var, cible):
    """
    Calcule l'écart relatif entre la modalité actuelle et celle précédente
    en fonction du taux de la variable cible.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        cat_var (str): Nom de la variable catégorielle.
        cible (str): Nom de la variable cible binaire (0 ou 1).
        
    Returns:
        pd.DataFrame: DataFrame avec les modalités, taux de la cible,
                      et écarts relatifs entre les modalités.
    """
    # Calculer le taux de la variable cible par modalité
    taux_cible = (
        df.groupby(cat_var)[cible]
        .mean()
        .reset_index()
        .rename(columns={cible: "taux_cible"})
    )
    
    
    # Trier les modalités par taux cible croissant
    taux_cible = taux_cible.sort_values("taux_cible").reset_index(drop=True) 
    
    # Calculer l'écart relatif entre la modalité actuelle et la précédente
    taux_cible["ecart_relatif"] = taux_cible["taux_cible"].pct_change().fillna(0) *100
    
    return taux_cible

def iv_woe(data, target, bins=5, show_woe=False, epsilon=1e-16):
    """
    Compute the Information Value (IV) and Weight of Evidence (WoE)
    for all explanatory variables in a dataset.

    Numerical variables with more than 10 unique values are first discretized
    into quantile-based bins. Categorical variables and numerical variables
    with few unique values are used as they are.

    Parameters
    ----------
    data : pandas DataFrame
        Input dataset containing the explanatory variables and the target.

    target : str
        Name of the binary target variable.
        The target should be coded as 1 for event/default and 0 for non-event/non-default.

    bins : int, default=5
        Number of quantile bins used to discretize continuous variables.

    show_woe : bool, default=False
        If True, display the detailed WoE table for each variable.

    epsilon : float, default=1e-16
        Small value used to avoid division by zero and log(0).

    Returns
    -------
    newDF : pandas DataFrame
        Summary table containing the Information Value of each variable.

    woeDF : pandas DataFrame
        Detailed WoE table for all variables and all groups.
    """

    # Initialize output DataFrames
    newDF = pd.DataFrame()
    woeDF = pd.DataFrame()

    # Get all column names
    cols = data.columns

    # Run WoE and IV calculation on all explanatory variables
    for ivars in cols[~cols.isin([target])]:

        # If the variable is numerical and has many unique values,
        # discretize it into quantile-based bins
        if (data[ivars].dtype.kind in "bifc") and (len(np.unique(data[ivars].dropna())) > 10):
            binned_x = pd.qcut(
                data[ivars],
                bins,
                duplicates="drop"
            )

            d0 = pd.DataFrame({
                "x": binned_x,
                "y": data[target]
            })

        # Otherwise, use the variable as it is
        else:
            d0 = pd.DataFrame({
                "x": data[ivars],
                "y": data[target]
            })

        # Compute the number of observations and events in each group
        d = (
            d0.groupby("x", as_index=False, observed=True)
            .agg({"y": ["count", "sum"]})
        )

        # Rename columns
        d.columns = ["Cutoff", "N", "Events"]

        # Compute the percentage of events in each group
        d["% of Events"] = (
            np.maximum(d["Events"], epsilon)
            / (d["Events"].sum() + epsilon)
        )

        # Compute the number of non-events in each group
        d["Non-Events"] = d["N"] - d["Events"]

        # Compute the percentage of non-events in each group
        d["% of Non-Events"] = (
            np.maximum(d["Non-Events"], epsilon)
            / (d["Non-Events"].sum() + epsilon)
        )

        # Compute Weight of Evidence
        # Here, WoE is defined as log(%Events / %Non-Events)
        # With this convention, positive WoE indicates higher default/event risk
        d["WoE"] = np.log(
            d["% of Events"] / d["% of Non-Events"]
        )

        # Compute the IV contribution of each group
        d["IV"] = d["WoE"] * (
            d["% of Events"] - d["% of Non-Events"]
        )

        # Add the variable name to the detailed table
        d.insert(
            loc=0,
            column="Variable",
            value=ivars
        )

        # Print the global Information Value of the variable
        print("=" * 30 + "\n")
        print(
            "Information Value of variable "
            + ivars
            + " is "
            + str(round(d["IV"].sum(), 6))
        )

        # Store the global IV of the variable
        temp = pd.DataFrame(
            {
                "Variable": [ivars],
                "IV": [d["IV"].sum()]
            },
            columns=["Variable", "IV"]
        )

        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Display the detailed WoE table if requested
        if show_woe:
            print(d)

    return newDF, woeDF


import numpy as np
import pandas as pd


def apply_cutoffs_from_woe_table(df, variable, woe_table, new_col=None, labels=None):
    """
    Apply cutoffs from a WoE table to discretize a continuous variable.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset containing the variable to discretize.

    variable : str
        Name of the continuous variable to discretize.

    woe_table : pandas DataFrame
        WoE table containing at least the columns 'Variable' and 'Cutoff'.

    new_col : str, optional
        Name of the new discretized variable.
        If None, the new column will be named '<variable>_binned'.

    labels : list, optional
        Labels to assign to the bins.
        If None, interval labels are kept.

    Returns
    -------
    pandas DataFrame
        DataFrame with the new discretized variable.
    """

    df = df.copy()

    if new_col is None:
        new_col = f"{variable}_binned"

    # Filter the WoE table for the selected variable
    cutoffs = (
        woe_table
        .loc[woe_table["Variable"] == variable, "Cutoff"]
        .dropna()
        .tolist()
    )

    if len(cutoffs) == 0:
        raise ValueError(f"No cutoffs found for variable '{variable}'.")

    # Case 1: Cutoff values are pandas Interval objects
    if isinstance(cutoffs[0], pd.Interval):
        edges = [cutoffs[0].left]

        for interval in cutoffs:
            edges.append(interval.right)

    # Case 2: Cutoff values are stored as strings
    else:
        intervals = pd.IntervalIndex.from_tuples([
            tuple(
                float(x.strip())
                for x in str(cutoff)
                .replace("(", "")
                .replace("]", "")
                .split(",")
            )
            for cutoff in cutoffs
        ])

        edges = [intervals[0].left]

        for interval in intervals:
            edges.append(interval.right)

    # Use -inf and +inf for safer application on new datasets
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Apply the binning
    df[new_col] = pd.cut(
        df[variable],
        bins=edges,
        labels=labels,
        include_lowest=True
    )

    return df


