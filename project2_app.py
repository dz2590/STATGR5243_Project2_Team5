from shiny import App, reactive, render, ui
import pandas as pd
import pyreadr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_diabetes,
    load_breast_cancer,
)

# File Loading

def load_uploaded_file(file_path: str) -> pd.DataFrame:
    ext = Path(file_path).suffix.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    elif ext == ".parquet":
        return pd.read_parquet(file_path)

    elif ext == ".json":
        return pd.read_json(file_path)

    elif ext == ".rds":
        result = pyreadr.read_r(file_path)

        if not result:
            raise ValueError("RDS file is empty or unreadable.")

        first_key = next(iter(result.keys()))
        obj = result[first_key]

        if isinstance(obj, pd.DataFrame):
            return obj

        return pd.DataFrame(obj)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

# Built-in datasets

def load_builtin_dataset(name: str) -> pd.DataFrame:
    if name == "diabetes":
        return load_diabetes(as_frame=True).frame.copy()

    elif name == "breast_cancer":
        return load_breast_cancer(as_frame=True).frame.copy()

    else:
        raise ValueError("Unknown dataset")

# Summary helpers

def summarize_dataframe(df: pd.DataFrame) -> str:
    lines = [
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Missing values (total): {int(df.isna().sum().sum())}",
        "",
        "Column types:",
    ]

    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {dtype}")

    return "\n".join(lines)

def standard_scale_series(s: pd.Series) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std == 0:
        return s
    return (s - s.mean()) / std


def minmax_scale_series(s: pd.Series) -> pd.Series:
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return s
    return (s - s_min) / (s_max - s_min)


def preprocess_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool,
    missing_strategy: str,
    scale_method: str,
    encode_categorical: bool,
) -> pd.DataFrame:
    df_clean = df.copy()

    # 1) Remove duplicates
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()

    # 2) Missing value handling
    if missing_strategy == "drop_rows":
        df_clean = df_clean.dropna()

    elif missing_strategy == "mean":
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

        non_numeric_cols = df_clean.select_dtypes(exclude=["number"]).columns
        for col in non_numeric_cols:
            if df_clean[col].isna().any():
                mode = df_clean[col].mode(dropna=True)
                if not mode.empty:
                    df_clean[col] = df_clean[col].fillna(mode.iloc[0])

    elif missing_strategy == "median":
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        non_numeric_cols = df_clean.select_dtypes(exclude=["number"]).columns
        for col in non_numeric_cols:
            if df_clean[col].isna().any():
                mode = df_clean[col].mode(dropna=True)
                if not mode.empty:
                    df_clean[col] = df_clean[col].fillna(mode.iloc[0])

    elif missing_strategy == "mode":
        for col in df_clean.columns:
            if df_clean[col].isna().any():
                mode = df_clean[col].mode(dropna=True)
                if not mode.empty:
                    df_clean[col] = df_clean[col].fillna(mode.iloc[0])

    # 3) Scaling numeric features
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns

    if scale_method == "standard":
        for col in numeric_cols:
            df_clean[col] = standard_scale_series(df_clean[col])

    elif scale_method == "minmax":
        for col in numeric_cols:
            df_clean[col] = minmax_scale_series(df_clean[col])

    # 4) Encode categorical variables
    if encode_categorical:
        cat_cols = df_clean.select_dtypes(exclude=["number"]).columns.tolist()
        if len(cat_cols) > 0:
            df_clean = pd.get_dummies(df_clean, columns=cat_cols, drop_first=False)

    return df_clean


def compare_dataframes(before_df: pd.DataFrame, after_df: pd.DataFrame) -> str:
    lines = [
        "Before vs After",
        f"Rows: {before_df.shape[0]} -> {after_df.shape[0]}",
        f"Columns: {before_df.shape[1]} -> {after_df.shape[1]}",
        f"Missing values: {int(before_df.isna().sum().sum())} -> {int(after_df.isna().sum().sum())}",
        f"Duplicate rows: {int(before_df.duplicated().sum())} -> {int(after_df.duplicated().sum())}",
    ]
    return "\n".join(lines)

# UI

app_ui = ui.page_fluid(
    ui.busy_indicators.use(spinners=True, pulse=True, fade=True),

    ui.div(
        ui.h2(
            "End-to-end Data Analysis Tool",
            style="margin-bottom: 6px;"
        ),

        ui.tags.small(
            "A workflow tool for loading, cleaning, transforming, and analyzing datasets",
            style="color: #6c757d;"
        ),

        style="""
            padding-top: 16px;
            padding-bottom: 20px;
            margin-bottom: 12px;
            border-bottom: 1px solid #dee2e6;
        """
    ),
    ui.div(
        ui.output_ui("dataset_status_bar"),
        style="margin-bottom: 18px;",
    ),

    ui.navset_tab(

        # TAB 1 — LOAD DATA

        ui.nav_panel(
            "Load Data",

            ui.layout_sidebar(

                ui.sidebar(
                    ui.h4("Data Input"),
                    ui.p(
                        "Upload a dataset or choose a built-in dataset to begin.",
                        style="color: #6c757d; margin-bottom: 16px;",
                    ),

                    ui.tooltip(
                        ui.input_select(
                            "data_source",
                            "Load Dataset",
                            {
                                "upload": "Upload file",
                                "builtin": "Built-in dataset",
                            },
                            selected="upload",
                        ),
                        "Choose whether to upload your own file or use a built-in dataset.",
                    ),

                    ui.panel_conditional(
                        "input.data_source === 'upload'",

                        ui.div(
                            ui.tooltip(
                                ui.input_file(
                                    "file",
                                    "Upload dataset",
                                    accept=[
                                        ".csv",
                                        ".xlsx",
                                        ".xls",
                                        ".parquet",
                                        ".json",
                                        ".rds",
                                    ],
                                    multiple=False,
                                ),
                                "Accepted formats: CSV, Excel, Parquet, JSON, and RDS.",
                            ),
                            style="margin-bottom: 18px;",
                        ),

                        ui.hr(),

                        ui.h5("Supported File Formats"),
                        ui.tags.ul(
                            ui.tags.li("CSV (.csv)"),
                            ui.tags.li("Excel (.xlsx, .xls)"),
                            ui.tags.li("Parquet (.parquet)"),
                            ui.tags.li("JSON (.json)"),
                            ui.tags.li("RDS (.rds)"),
                        ),
                    ),

                    ui.panel_conditional(
                        "input.data_source === 'builtin'",

                        ui.div(
                            ui.tooltip(
                                ui.input_select(
                                    "builtin_name",
                                    "Select built-in dataset",
                                    {
                                        "diabetes": "Diabetes",
                                        "breast_cancer": "Breast Cancer",
                                    },
                                    selected="diabetes",
                                ),
                                "Use a sample dataset to test the workflow without uploading a file.",
                            ),
                            style="margin-bottom: 18px;",
                        ),
                    ),
                ),

                ui.card(
                    ui.card_header("Dataset Summary"),
                    ui.output_ui("dataset_info_ui"),
                    style="margin-bottom: 20px;",
                ),

                ui.card(
                    ui.card_header("Preview"),
                    ui.output_ui("preview_notice"),
                    ui.output_ui("preview_ui"),
                    style="margin-bottom: 20px;",
                ),
            ),
        ),

        # TAB 2 — CLEAN & PREPROCESS

        ui.nav_panel(
            "Clean & Preprocess",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Data Cleaning and Preprocessing"),
                    ui.p(
                "Choose preprocessing steps and view the cleaned dataset in real time.",
                style="color: #6c757d; margin-bottom: 16px;",
                ),

                ui.input_checkbox(
                    "remove_duplicates",
                    "Remove duplicate rows",
                    value=False,
                ),

                ui.input_select(
                    "missing_strategy",
                    "Missing value handling",
                    {
                        "none": "Do nothing",
                        "drop_rows": "Drop rows with missing values",
                        "mean": "Fill numeric with mean",
                        "median": "Fill numeric with median",
                        "mode": "Fill all columns with mode",
                    },
                    selected="none",
                ),
            ui.input_select(
                "scale_method",
                "Scale numeric features",
                {
                    "none": "No scaling",
                    "standard": "Standard scaling (z-score)",
                    "minmax": "Min-max scaling (0 to 1)",
                },
                selected="none",
            ),

            ui.input_checkbox(
                "encode_categorical",
                "One-hot encode categorical columns",
                value=False,
            ),

            ui.hr(),

            ui.tags.small(
                "Tip: preprocessing is applied to a copy of the loaded dataset.",
                style="color: #6c757d;",
            ),
        ),

        ui.div(
            ui.card(
                ui.card_header("Preprocessing Summary"),
                ui.output_ui("preprocess_summary_ui"),
                style="margin-bottom: 20px;",
            ),

            ui.card(
                ui.card_header("Changes Before vs After"),
                ui.output_ui("preprocess_changes_ui"),
                style="margin-bottom: 20px;",
            ),

            ui.card(
                ui.card_header("Processed Data Preview"),
                ui.output_ui("processed_preview_notice"),
                ui.output_ui("processed_preview_ui"),
                style="margin-bottom: 20px;",
            ),
        ),
    ),
),
        # TAB 3 — FEATURE ENGINEERING

        ui.nav_panel(
            "Feature Engineering",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Feature Engineering"),
                   
                    ui.hr(),
                ),

                ui.card(
                    ui.card_header("Output"),
                  
                    style="margin-bottom: 20px;",
                ),
            ),
        ),

        # TAB 4 — EDA

                # TAB 4 — EDA

        ui.nav_panel(
            "Exploratory Data Analysis",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("EDA"),
                    ui.p(
                        "Interactively explore the processed dataset with plots, filters, and correlation analysis.",
                        style="color: #6c757d; margin-bottom: 16px;",
                    ),

                    ui.input_select(
                        "eda_plot_type",
                        "Plot type",
                        {
                            "hist": "Histogram",
                            "box": "Boxplot",
                            "scatter": "Scatterplot",
                            "bar": "Bar chart",
                            "corr": "Correlation heatmap",
                        },
                        selected="hist",
                    ),

                    ui.output_ui("eda_x_var_ui"),
                    ui.output_ui("eda_y_var_ui"),

                    ui.hr(),

                    ui.h5("Optional Filter"),
                    ui.output_ui("eda_filter_var_ui"),
                    ui.output_ui("eda_filter_range_ui"),

                    ui.hr(),

                    ui.tags.small(
                        "EDA is performed on the processed dataset, so preprocessing changes will be reflected here.",
                        style="color: #6c757d;",
                    ),
                ),

                ui.div(
                    ui.card(
                        ui.card_header("Visualization"),
                        ui.output_plot("eda_plot"),
                        style="margin-bottom: 20px;",
                    ),

                    ui.card(
                        ui.card_header("Summary Statistics"),
                        ui.output_ui("eda_summary_notice"),
                        ui.output_table("eda_summary_table"),
                        style="margin-bottom: 20px;",
                    ),

                    ui.card(
                        ui.card_header("Correlation Matrix"),
                        ui.output_ui("eda_corr_notice"),
                        ui.output_table("eda_corr_table"),
                        style="margin-bottom: 20px;",
                    ),

                    ui.card(
                        ui.card_header("Filtered Data Preview"),
                        ui.output_ui("eda_preview_notice"),
                        ui.output_table("eda_preview_table"),
                        style="margin-bottom: 20px;",
                    ),
                ),
            ),
        ),
    ),
)

# SERVER

def server(input, output, session):

    @reactive.calc
    def dataset():
        source = input.data_source()

        try:
            if source == "upload":
                file_info = input.file()

                if not file_info:
                    return None

                file_path = file_info[0]["datapath"]
                return load_uploaded_file(file_path)

            elif source == "builtin":
                return load_builtin_dataset(input.builtin_name())

            return None

        except Exception as e:
            return f"ERROR: {str(e)}"

    @reactive.calc
    def dataset_name():
        source = input.data_source()

        if source == "upload":
            file_info = input.file()
            if not file_info:
                return "No dataset loaded"
            return file_info[0]["name"]

        elif source == "builtin":
            name_map = {
                "diabetes": "Built-in: Diabetes",
                "breast_cancer": "Built-in: Breast Cancer",
            }
            return name_map.get(input.builtin_name(), "Built-in dataset")

        return "No dataset loaded"

    @render.ui
    def dataset_status_bar():
        df = dataset()

        if df is None:
            return ui.div(
                ui.strong("Dataset:"), " No dataset loaded",
                " | ",
                ui.strong("Rows:"), " —",
                " | ",
                ui.strong("Columns:"), " —",
                " | ",
                ui.strong("Missing:"), " —",
                style="""
                    background-color: #f8f9fa;
                    padding: 10px 14px;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                """,
            )

        if isinstance(df, str):
            return ui.div(
                ui.strong("Dataset:"), f" {dataset_name()}",
                " | ",
                ui.strong("Status:"), f" {df}",
                style="""
                    background-color: #fff3cd;
                    padding: 10px 14px;
                    border: 1px solid #ffe69c;
                    border-radius: 8px;
                """,
            )

        return ui.div(
            ui.strong("Dataset:"), f" {dataset_name()}",
            " | ",
            ui.strong("Rows:"), f" {df.shape[0]}",
            " | ",
            ui.strong("Columns:"), f" {df.shape[1]}",
            " | ",
            ui.strong("Missing:"), f" {int(df.isna().sum().sum())}",
            style="""
                background-color: #f8f9fa;
                padding: 10px 14px;
                border: 1px solid #dee2e6;
                border-radius: 8px;
            """,
        )

    @render.ui
    def dataset_info_ui():
        df = dataset()

        if df is None:
            return ui.p(
                "No dataset loaded yet. Upload a file or choose a built-in dataset to get started.",
                style="color: #6c757d; margin: 0;",
            )

        if isinstance(df, str):
            return ui.p(
                df,
                style="color: #842029; margin: 0;",
            )

        return ui.tags.pre(
            summarize_dataframe(df),
            style="margin: 0; white-space: pre-wrap;",
        )

    @render.ui
    def preview_notice():
        df = dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        return ui.tags.small(
            "Showing first 10 rows",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.ui
    def preview_ui():
        df = dataset()

        if df is None:
            return ui.p(
                "Preview will appear here after a dataset is loaded.",
                style="color: #6c757d; margin: 0;",
            )

        if isinstance(df, str):
            return ui.p(
                "Unable to preview the dataset because an error occurred.",
                style="color: #842029; margin: 0;",
            )

        return ui.output_table("preview")

    @render.table
    def preview():
        df = dataset()

        if df is None:
            return pd.DataFrame()

        if isinstance(df, str):
            return pd.DataFrame()

        return df.head(10)
    @reactive.calc
    def processed_dataset():
        df = dataset()

        if df is None:
            return None

        if isinstance(df, str):
            return df

        try:
            return preprocess_dataframe(
                df=df,
                remove_duplicates=input.remove_duplicates(),
                missing_strategy=input.missing_strategy(),
                scale_method=input.scale_method(),
                encode_categorical=input.encode_categorical(),
            )
        except Exception as e:
            return f"ERROR: {str(e)}"

    @render.ui
    def preprocess_summary_ui():
        df = processed_dataset()

        if df is None:
            return ui.p(
                "Load a dataset first to use preprocessing tools.",
                style="color: #6c757d; margin: 0;",
            )

        if isinstance(df, str):
            return ui.p(
                df,
                style="color: #842029; margin: 0;",
            )

        return ui.tags.pre(
            summarize_dataframe(df),
            style="margin: 0; white-space: pre-wrap;",
        )

    @render.ui
    def preprocess_changes_ui():
        before_df = dataset()
        after_df = processed_dataset()

        if before_df is None:
            return ui.p(
                "No dataset loaded yet.",
                style="color: #6c757d; margin: 0;",
            )

        if isinstance(before_df, str):
            return ui.p(
                before_df,
                style="color: #842029; margin: 0;",
            )

        if isinstance(after_df, str):
            return ui.p(
                after_df,
                style="color: #842029; margin: 0;",
            )

        return ui.tags.pre(
            compare_dataframes(before_df, after_df),
            style="margin: 0; white-space: pre-wrap;",
        )

    @render.ui
    def processed_preview_notice():
        df = processed_dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        return ui.tags.small(
            "Showing first 10 rows after preprocessing",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.ui
    def processed_preview_ui():
        df = processed_dataset()

        if df is None:
            return ui.p(
                "Processed data preview will appear here.",
                style="color: #6c757d; margin: 0;",
            )

        if isinstance(df, str):
            return ui.p(
                "Unable to preview processed data because an error occurred.",
                style="color: #842029; margin: 0;",
            )

        return ui.output_table("processed_preview")

    @render.table
    def processed_preview():
        df = processed_dataset()

        if df is None or isinstance(df, str):
            return pd.DataFrame()

        return df.head(10)
    
    @reactive.calc
    def eda_dataset():
        df = processed_dataset()

        if df is None:
            return None

        if isinstance(df, str):
            return df

        return df.copy()

    @reactive.calc
    def eda_numeric_cols():
        df = eda_dataset()

        if df is None or isinstance(df, str):
            return []

        return df.select_dtypes(include=["number"]).columns.tolist()

    @reactive.calc
    def eda_categorical_cols():
        df = eda_dataset()

        if df is None or isinstance(df, str):
            return []

        return df.select_dtypes(exclude=["number"]).columns.tolist()

    @render.ui
    def eda_x_var_ui():
        plot_type = input.eda_plot_type()
        num_cols = eda_numeric_cols()
        cat_cols = eda_categorical_cols()

        if plot_type in ["hist", "box"]:
            if not num_cols:
                return ui.p("No numeric columns available.", style="color: #6c757d;")
            return ui.input_select(
                "eda_x_var",
                "Numeric variable",
                {col: col for col in num_cols},
                selected=num_cols[0],
            )

        if plot_type == "scatter":
            if len(num_cols) < 2:
                return ui.p("At least two numeric columns are required for a scatterplot.", style="color: #6c757d;")
            return ui.input_select(
                "eda_x_var",
                "X variable",
                {col: col for col in num_cols},
                selected=num_cols[0],
            )

        if plot_type == "bar":
            if not cat_cols:
                return ui.p("No categorical columns available for a bar chart.", style="color: #6c757d;")
            return ui.input_select(
                "eda_x_var",
                "Categorical variable",
                {col: col for col in cat_cols},
                selected=cat_cols[0],
            )

        return ui.div()

    @render.ui
    def eda_y_var_ui():
        plot_type = input.eda_plot_type()
        num_cols = eda_numeric_cols()

        if plot_type == "scatter":
            if len(num_cols) < 2:
                return ui.div()

            default_y = num_cols[1] if len(num_cols) > 1 else num_cols[0]

            return ui.input_select(
                "eda_y_var",
                "Y variable",
                {col: col for col in num_cols},
                selected=default_y,
            )

        return ui.div()

    @render.ui
    def eda_filter_var_ui():
        num_cols = eda_numeric_cols()

        choices = {"none": "No filter"}
        for col in num_cols:
            choices[col] = col

        return ui.input_select(
            "eda_filter_var",
            "Filter numeric column",
            choices,
            selected="none",
        )

    @render.ui
    def eda_filter_range_ui():
        df = eda_dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        filter_var = input.eda_filter_var()

        if filter_var is None or filter_var == "none":
            return ui.div()

        if filter_var not in df.columns:
            return ui.div()

        s = pd.to_numeric(df[filter_var], errors="coerce").dropna()

        if s.empty:
            return ui.p("Selected filter column has no usable numeric values.", style="color: #6c757d;")

        min_val = float(s.min())
        max_val = float(s.max())

        if min_val == max_val:
            return ui.p("Selected filter column has only one unique value.", style="color: #6c757d;")

        return ui.input_slider(
            "eda_filter_range",
            f"Range for {filter_var}",
            min=min_val,
            max=max_val,
            value=(min_val, max_val),
        )

    @reactive.calc
    def eda_filtered_dataset():
        df = eda_dataset()

        if df is None or isinstance(df, str):
            return df

        filter_var = input.eda_filter_var()

        if filter_var is None or filter_var == "none":
            return df

        if filter_var not in df.columns:
            return df

        rng = input.eda_filter_range()

        if rng is None:
            return df

        low, high = rng

        s = pd.to_numeric(df[filter_var], errors="coerce")
        mask = s.between(low, high, inclusive="both")

        return df.loc[mask].copy()

    @render.plot
    def eda_plot():
        df = eda_filtered_dataset()
        plot_type = input.eda_plot_type()

        fig, ax = plt.subplots(figsize=(8, 5))

        if df is None:
            ax.text(0.5, 0.5, "Load a dataset first.", ha="center", va="center")
            ax.axis("off")
            return fig

        if isinstance(df, str):
            ax.text(0.5, 0.5, df, ha="center", va="center")
            ax.axis("off")
            return fig

        if df.empty:
            ax.text(0.5, 0.5, "No rows available after filtering.", ha="center", va="center")
            ax.axis("off")
            return fig

        try:
            if plot_type == "hist":
                x = input.eda_x_var()
                if x not in df.columns:
                    ax.text(0.5, 0.5, "Please select a valid numeric variable.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                s = pd.to_numeric(df[x], errors="coerce").dropna()
                if s.empty:
                    ax.text(0.5, 0.5, "No valid numeric data for histogram.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                ax.hist(s, bins=30, edgecolor="black")
                ax.set_title(f"Histogram of {x}")
                ax.set_xlabel(x)
                ax.set_ylabel("Frequency")

            elif plot_type == "box":
                x = input.eda_x_var()
                if x not in df.columns:
                    ax.text(0.5, 0.5, "Please select a valid numeric variable.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                s = pd.to_numeric(df[x], errors="coerce").dropna()
                if s.empty:
                    ax.text(0.5, 0.5, "No valid numeric data for boxplot.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                ax.boxplot(s)
                ax.set_title(f"Boxplot of {x}")
                ax.set_ylabel(x)

            elif plot_type == "scatter":
                x = input.eda_x_var()
                y = input.eda_y_var()

                if x not in df.columns or y not in df.columns:
                    ax.text(0.5, 0.5, "Please select valid numeric variables.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                x_data = pd.to_numeric(df[x], errors="coerce")
                y_data = pd.to_numeric(df[y], errors="coerce")
                plot_df = pd.DataFrame({"x": x_data, "y": y_data}).dropna()

                if plot_df.empty:
                    ax.text(0.5, 0.5, "No valid numeric pairs for scatterplot.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                ax.scatter(plot_df["x"], plot_df["y"], alpha=0.7)
                ax.set_title(f"{y} vs {x}")
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            elif plot_type == "bar":
                x = input.eda_x_var()
                if x not in df.columns:
                    ax.text(0.5, 0.5, "Please select a valid categorical variable.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                counts = df[x].astype(str).value_counts(dropna=False).head(20)
                if counts.empty:
                    ax.text(0.5, 0.5, "No valid data for bar chart.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                ax.bar(counts.index.astype(str), counts.values)
                ax.set_title(f"Bar Chart of {x}")
                ax.set_xlabel(x)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

            elif plot_type == "corr":
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()

                if len(num_cols) < 2:
                    ax.text(0.5, 0.5, "At least two numeric columns are required for a correlation heatmap.", ha="center", va="center")
                    ax.axis("off")
                    return fig

                corr = df[num_cols].corr()

                im = ax.imshow(corr.values, aspect="auto")
                ax.set_xticks(range(len(num_cols)))
                ax.set_xticklabels(num_cols, rotation=90)
                ax.set_yticks(range(len(num_cols)))
                ax.set_yticklabels(num_cols)
                ax.set_title("Correlation Heatmap")
                fig.colorbar(im, ax=ax)

            plt.tight_layout()
            return fig

        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f"Plot error: {str(e)}", ha="center", va="center")
            ax.axis("off")
            return fig

    @render.ui
    def eda_summary_notice():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        return ui.tags.small(
            "Summary statistics are based on the currently filtered processed dataset.",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.table
    def eda_summary_table():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str) or df.empty:
            return pd.DataFrame()

        summary = df.describe(include="all").transpose().reset_index()
        summary = summary.rename(columns={"index": "column"})
        return summary.fillna("")

    @render.ui
    def eda_corr_notice():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        return ui.tags.small(
            "Correlation is computed using numeric columns only.",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.table
    def eda_corr_table():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str) or df.empty:
            return pd.DataFrame()

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if len(num_cols) < 2:
            return pd.DataFrame({"message": ["At least two numeric columns are required."]})

        return df[num_cols].corr().round(3)

    @render.ui
    def eda_preview_notice():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str):
            return ui.div()

        return ui.tags.small(
            "Showing first 10 rows of the filtered processed dataset.",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.table
    def eda_preview_table():
        df = eda_filtered_dataset()

        if df is None or isinstance(df, str):
            return pd.DataFrame()

        return df.head(10)

app = App(app_ui, server)