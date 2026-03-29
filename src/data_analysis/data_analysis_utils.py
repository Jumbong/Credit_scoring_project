import pandas as pd
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def build_default_summary(
    df: pd.DataFrame,
    category_col: str,
    default_col: str,
    category_label: Optional[str] = None,
    include_na: bool = False,
    sort_by: str = "count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Construit un tableau de synthèse pour une variable catégorielle.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame source.
    category_col : str
        Nom de la variable catégorielle.
    default_col : str
        Colonne binaire indiquant le défaut (0/1 ou booléen).
    category_label : str, optionnel
        Libellé à afficher pour la première colonne.
        Par défaut : category_col.
    include_na : bool, default=False
        Si True, conserve les valeurs manquantes comme catégorie.
    sort_by : str, default="count"
        Colonne de tri logique parmi {"count", "defaults", "prop", "default_rate", "category"}.
    ascending : bool, default=False
        Sens du tri.

    Retour
    ------
    pd.DataFrame
        Tableau prêt à exporter.
    """

    if category_col not in df.columns:
        raise KeyError(f"La colonne catégorielle '{category_col}' est introuvable.")
    if default_col not in df.columns:
        raise KeyError(f"La colonne défaut '{default_col}' est introuvable.")

    data = df[[category_col, default_col]].copy()

    # Validation minimale sur la cible
    # On convertit bool -> int ; sinon on suppose 0/1 documenté
    if pd.api.types.is_bool_dtype(data[default_col]):
        data[default_col] = data[default_col].astype(int)

    # Gestion des NA de la variable catégorielle
    if include_na:
        data[category_col] = data[category_col].astype("object").fillna("Missing")
    else:
        data = data[data[category_col].notna()].copy()

    grouped = (
        data.groupby(category_col, dropna=False,observed=False)[default_col]
        .agg(count="size", defaults="sum")
        .reset_index()
    )

    total_obs = grouped["count"].sum()
    total_def = grouped["defaults"].sum()

    grouped["prop"] = grouped["count"] / total_obs if total_obs > 0 else 0.0
    grouped["default_rate"] = grouped["defaults"] / grouped["count"]

    sort_mapping = {
        "count": "count",
        "defaults": "defaults",
        "prop": "prop",
        "default_rate": "default_rate",
        "category": category_col,
    }
    if sort_by not in sort_mapping:
        raise ValueError(
            "sort_by doit être parmi {'count', 'defaults', 'prop', 'default_rate', 'category'}."
        )

    grouped = grouped.sort_values(sort_mapping[sort_by], ascending=ascending).reset_index(drop=True)

    total_row = pd.DataFrame(
        {
            category_col: ["Total"],
            "count": [total_obs],
            "defaults": [total_def],
            "prop": [1.0 if total_obs > 0 else 0.0],
            "default_rate": [total_def / total_obs if total_obs > 0 else 0.0],
        }
    )

    summary = pd.concat([grouped, total_row], ignore_index=True)

    

    summary = summary.rename(
        columns={
            category_col: category_label or category_col,
            "count": "Nb of obs",
            "defaults": "Nb def",
            "prop": "Prop",
            "default_rate": "Default rate",
        }
    )
    summary = summary[[category_label or category_col, "Nb of obs", "Prop", "Nb def", "Default rate"]]
    return summary


def export_summary_to_excel(
    summary: pd.DataFrame,
    output_path: str,
    sheet_name: str = "Summary",
    title: str = "All perimeters",
) -> None:
    """
    Exporte le tableau de synthèse dans un fichier Excel avec mise en forme.
    Nécessite le moteur xlsxwriter.
    """

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        #

        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)

        nrows, ncols = summary.shape
        total_excel_row = 2 + nrows  # +1 implicite Excel car index 0-based côté xlsxwriter pour set_row
        # Détail :
        # ligne 0 : titre fusionné
        # ligne 2 : header
        # données commencent ligne 3 (Excel visuel), mais xlsxwriter manipule en base 0

        # -------- Formats --------
        border_color = "#4F4F4F"
        header_bg = "#D9EAF7"
        title_bg = "#CFE2F3"
        total_bg = "#D9D9D9"
        white_bg = "#FFFFFF"

        title_fmt = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "font_size": 14,
            "border": 1,
            "bg_color": title_bg,
        })

        header_fmt = workbook.add_format({
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "border": 1,
            "bg_color": header_bg,
        })

        text_fmt = workbook.add_format({
            "border": 1,
            "align": "left",
            "valign": "vcenter",
            "bg_color": white_bg,
        })

        int_fmt = workbook.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "num_format": "# ##0",
            "bg_color": white_bg,
        })

        pct_fmt = workbook.add_format({
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "num_format": "0.00%",
            "bg_color": white_bg,
        })

        total_text_fmt = workbook.add_format({
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "bg_color": total_bg,
        })

        total_int_fmt = workbook.add_format({
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "num_format": "# ##0",
            "bg_color": total_bg,
        })

        total_pct_fmt = workbook.add_format({
            "bold": True,
            "border": 1,
            "align": "center",
            "valign": "vcenter",
            "num_format": "0.00%",
            "bg_color": total_bg,
        })

        # -------- Titre fusionné --------
        worksheet.merge_range(0, 0, 0, ncols - 1, title, title_fmt)

        # -------- Header --------
        worksheet.set_row(2, 28)
        for col_idx, col_name in enumerate(summary.columns):
            worksheet.write(1, col_idx, col_name, header_fmt)

        # -------- Largeurs de colonnes --------
        column_widths = {
            0: 24,  # catégorie
            1: 14,  # Nb of obs
            2: 12,  # Nb def
            3: 10,  # Prop
            4: 14,  # Default rate
        }
        for col_idx in range(ncols):
            worksheet.set_column(col_idx, col_idx, column_widths.get(col_idx, 15))

        # -------- Mise en forme cellule par cellule --------
        last_row_idx = nrows - 1

        for row_idx in range(nrows):
            excel_row = 2 + row_idx  # données à partir de la ligne 3 (0-based xlsxwriter)

            is_total = row_idx == last_row_idx

            for col_idx, col_name in enumerate(summary.columns):
                value = summary.iloc[row_idx, col_idx]

                if col_idx == 0:
                    fmt = total_text_fmt if is_total else text_fmt
                elif col_name in ["Nb of obs", "Nb def"]:
                    fmt = total_int_fmt if is_total else int_fmt
                elif col_name in ["Prop", "Default rate"]:
                    fmt = total_pct_fmt if is_total else pct_fmt
                else:
                    fmt = total_text_fmt if is_total else text_fmt

                worksheet.write(excel_row, col_idx, value, fmt)

        # Optionnel : figer le header
        #worksheet.freeze_panes(3, 1)

        # Optionnel : encadrement global déjà géré par cellule
        # mais on peut aussi améliorer la lisibilité
        worksheet.set_default_row(24)


def generate_categorical_report_excel(
    df: pd.DataFrame,
    category_col: str,
    default_col: str,
    output_path: str,
    sheet_name: str = "Summary",
    title: str = "All perimeters",
    category_label: Optional[str] = None,
    include_na: bool = False,
    sort_by: str = "count",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Fonction principale DRY :
    1. calcule le tableau
    2. l'exporte vers Excel
    3. renvoie aussi le DataFrame récapitulatif
    """
    summary = build_default_summary(
        df=df,
        category_col=category_col,
        default_col=default_col,
        category_label=category_label,
        include_na=include_na,
        sort_by=sort_by,
        ascending=ascending,
    )

    export_summary_to_excel(
        summary=summary,
        output_path=output_path,
        sheet_name=sheet_name,
        title=title,
    )

    return summary

def discretize_variable_by_quartiles(
    df: pd.DataFrame,
    variable: str,
    new_var: str | None = None
) -> pd.DataFrame:
    """
    Discretize a continuous variable into four intervals based on its quartiles.

    The function computes Q1, Q2 (median), and Q3 of the selected variable and
    creates four bins corresponding to the following intervals:

        ]min ; Q1], ]Q1 ; Q2], ]Q2 ; Q3], ]Q3 ; max]

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the variable to discretize.

    variable : str
        Name of the continuous variable to be discretized.

    new_var : str, optional
        Name of the new categorical variable created. If None,
        the name "<variable>_quartile" is used.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with the new quartile-based categorical variable.
    """

    # Create a copy of the dataframe to avoid modifying the original dataset
    data = df.copy()

    # If no name is provided for the new variable, create one automatically
    if new_var is None:
        new_var = f"{variable}_quartile"

    # Compute the quartiles of the variable
    q1, q2, q3 = data[variable].quantile([0.25, 0.50, 0.75])

    # Retrieve the minimum and maximum values of the variable
    vmin = data[variable].min()
    vmax = data[variable].max()

    # Define the bin edges
    # A small epsilon is subtracted from the minimum value to ensure it is included
    bins = [vmin - 1e-9, q1, q2, q3, vmax]

    # Define human-readable labels for each interval
    labels = [
        f"]{vmin:.2f} ; {q1:.2f}]",
        f"]{q1:.2f} ; {q2:.2f}]",
        f"]{q2:.2f} ; {q3:.2f}]",
        f"]{q3:.2f} ; {vmax:.2f}]",
    ]

    # Use pandas.cut to assign each observation to a quartile-based interval
    data[new_var] = pd.cut(
        data[variable],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Return the dataframe with the new discretized variable
    return data



def plot_series(years, data):
    
    fig, ax = plt.subplots()

    for label, values in data.items():
        ax.plot(years, values, marker="o", label=label)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.show()

def create_quartile_bins(
    df: pd.DataFrame,
    variable: str,
    new_var: str | None = None
) -> pd.DataFrame:

    data = df.copy()

    if new_var is None:
        new_var = f"{variable}_quartile"

    q1, q2, q3 = data[variable].quantile([0.25, 0.50, 0.75])
    vmin = data[variable].min()
    vmax = data[variable].max()

    bins = [vmin - 1e-9, q1, q2, q3, vmax]  # petit epsilon pour inclure le min

    labels = [
        f"]{vmin:.2f} ; {q1:.2f}]",
        f"]{q1:.2f} ; {q2:.2f}]",
        f"]{q2:.2f} ; {q3:.2f}]",
        f"]{q3:.2f} ; {vmax:.2f}]",
    ]

    data[new_var] = pd.cut(
        data[variable],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return data