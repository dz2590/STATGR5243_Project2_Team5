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
    columns_to_drop: list,
) -> pd.DataFrame:
    df_clean = df.copy()

    # 0) Drop selected columns
    if columns_to_drop:
        existing = [c for c in columns_to_drop if c in df_clean.columns]
        df_clean = df_clean.drop(columns=existing)

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

# Feature Engineering helpers

def apply_unary_transform(series: pd.Series, method: str) -> pd.Series:
    if method == "log":
        return np.log(series.replace(0, np.nan))
    elif method == "log1p":
        return np.log1p(series)
    elif method == "sqrt":
        return np.sqrt(series.clip(lower=0))
    elif method == "square":
        return series ** 2
    elif method == "reciprocal":
        return 1 / series.replace(0, np.nan)
    elif method == "abs":
        return series.abs()
    elif method == "zscore":
        return (series - series.mean()) / series.std()
    elif method == "minmax":
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn) if mx != mn else series * 0
    else:
        raise ValueError(f"Unknown transform: {method}")


def apply_binary_transform(s1: pd.Series, s2: pd.Series, op: str) -> pd.Series:
    if op == "add":
        return s1 + s2
    elif op == "subtract":
        return s1 - s2
    elif op == "multiply":
        return s1 * s2
    elif op == "divide":
        return s1 / s2.replace(0, np.nan)
    elif op == "ratio_pct":
        total = s1 + s2
        return (s1 / total.replace(0, np.nan)) * 100
    else:
        raise ValueError(f"Unknown operation: {op}")


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

                ui.tooltip(
                    ui.input_selectize(
                        "drop_columns",
                        "Select columns to remove",
                        choices=[],
                        multiple=True,
                    ),
                    "Selected columns will be removed from the dataset before other preprocessing steps.",
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
                    ui.p(
                        "Create new features from existing columns. "
                        "Apply transformations or combine two columns.",
                        style="color: #6c757d; margin-bottom: 16px;",
                    ),

                    ui.hr(),
                    ui.h5("① Single-Column Transform"),
                    ui.p("Apply a mathematical transformation to one numeric column.",
                        style="color: #6c757d; font-size: 13px; margin-bottom: 10px;"),
                    ui.tooltip(
                        ui.input_select("fe_col", "Select column", choices=[], selected=None),
                        "Only numeric columns are shown.",
                    ),
                    ui.tooltip(
                        ui.input_select(
                            "fe_method", "Transformation",
                            {
                                "log": "Log (ln x)", "log1p": "Log1p (ln(x+1))",
                                "sqrt": "Square root (√x)", "square": "Square (x²)",
                                "reciprocal": "Reciprocal (1/x)", "abs": "Absolute value (|x|)",
                                "zscore": "Z-score standardization", "minmax": "Min-Max normalization",
                            },
                            selected="log1p",
                        ),
                        "Choose how to transform the selected column.",
                    ),
                    ui.tooltip(
                        ui.input_text("fe_new_col_name", "New column name", placeholder="e.g. log_age"),
                        "Leave blank to auto-generate a name.",
                    ),
                    ui.input_action_button("fe_apply", "Apply Transform",
                        class_="btn btn-primary btn-sm", style="width:100%;margin-top:6px;"),

                    ui.hr(),
                    ui.h5("② Two-Column Interaction"),
                    ui.p("Combine two numeric columns to create an interaction feature.",
                        style="color: #6c757d; font-size: 13px; margin-bottom: 10px;"),
                    ui.tooltip(
                        ui.input_select("fe_col_a", "Column A", choices=[], selected=None),
                        "First column for the interaction.",
                    ),
                    ui.tooltip(
                        ui.input_select(
                            "fe_op", "Operation",
                            {
                                "add": "A + B", "subtract": "A − B", "multiply": "A × B",
                                "divide": "A ÷ B", "ratio_pct": "A / (A+B) × 100%",
                            },
                            selected="multiply",
                        ),
                        "How to combine the two columns.",
                    ),
                    ui.tooltip(
                        ui.input_select("fe_col_b", "Column B", choices=[], selected=None),
                        "Second column for the interaction.",
                    ),
                    ui.tooltip(
                        ui.input_text("fe_inter_name", "New column name", placeholder="e.g. age_x_bmi"),
                        "Leave blank to auto-generate a name.",
                    ),
                    ui.input_action_button("fe_apply_inter", "Apply Interaction",
                        class_="btn btn-primary btn-sm", style="width:100%;margin-top:6px;"),

                    ui.hr(),
                    ui.h5("③ Binning (Discretization)"),
                    ui.p("Convert a continuous column into discrete bins.",
                        style="color: #6c757d; font-size: 13px; margin-bottom: 10px;"),
                    ui.tooltip(
                        ui.input_select("fe_bin_col", "Select column", choices=[], selected=None),
                        "Numeric column to discretize.",
                    ),
                    ui.tooltip(
                        ui.input_slider("fe_bin_n", "Number of bins", min=2, max=20, value=5, step=1),
                        "How many equal-width bins to create.",
                    ),
                    ui.tooltip(
                        ui.input_text("fe_bin_name", "New column name", placeholder="e.g. age_bin"),
                        "Leave blank to auto-generate a name.",
                    ),
                    ui.input_action_button("fe_apply_bin", "Apply Binning",
                        class_="btn btn-primary btn-sm", style="width:100%;margin-top:6px;"),

                    ui.hr(),
                    ui.input_action_button("fe_reset", "Reset All Features",
                        class_="btn btn-outline-danger btn-sm", style="width:100%;"),
                ),

                ui.div(
                    ui.output_ui("fe_status"),
                    ui.card(
                        ui.card_header("Before / After Distribution"),
                        ui.output_plot("fe_plot"),
                        style="margin-bottom: 20px;",
                    ),
                    ui.card(
                        ui.card_header("Added Features"),
                        ui.output_ui("fe_added_list"),
                        style="margin-bottom: 20px;",
                    ),
                    ui.card(
                        ui.card_header("Dataset Preview (with new features)"),
                        ui.output_ui("fe_preview_notice"),
                        ui.output_ui("fe_preview_table"),
                        style="margin-bottom: 20px;",
                    ),
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
                        ui.output_ui("eda_summary_table"),
                        style="margin-bottom: 20px;",
                    ),

                    ui.card(
                        ui.card_header("Correlation Matrix"),
                        ui.output_ui("eda_corr_notice"),
                        ui.output_ui("eda_corr_table"),
                        style="margin-bottom: 20px;",
                    ),

                    ui.card(
                        ui.card_header("Filtered Data Preview"),
                        ui.output_ui("eda_preview_notice"),
                        ui.output_ui("eda_preview_table"),
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
                columns_to_drop=list(input.drop_columns() or []),
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

        preview = df.head(10)
        th = "".join(
            f"<th style='padding:6px 12px;text-align:left;border-bottom:2px solid #dee2e6;white-space:nowrap'>{c}</th>"
            for c in preview.columns
        )
        rows = "".join(
            "<tr>" + "".join(
                f"<td style='padding:5px 12px;text-align:left;border-bottom:1px solid #f0f0f0;white-space:nowrap'>{v}</td>"
                for v in row
            ) + "</tr>"
            for _, row in preview.iterrows()
        )
        return ui.HTML(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    
    @reactive.effect
    def _update_drop_columns_choices():
        df = dataset()
        if not isinstance(df, pd.DataFrame):
            return
        choices = {c: c for c in df.columns.tolist()}
        ui.update_selectize("drop_columns", choices=choices, session=session)

    # ── FEATURE ENGINEERING ────────────────────────────────────

    fe_df    = reactive.value(None)
    fe_added = reactive.value([])
    fe_last  = reactive.value(None)

    @reactive.effect
    def _fe_sync_base():
        df = processed_dataset()
        if isinstance(df, pd.DataFrame):
            fe_df.set(df.copy())
            fe_added.set([])
            fe_last.set(None)

    @reactive.effect
    def _fe_update_col_choices():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return
        num_cols = df.select_dtypes(include="number").columns.tolist()
        choices = {c: c for c in num_cols}
        ui.update_select("fe_col",     choices=choices, session=session)
        ui.update_select("fe_col_a",   choices=choices, session=session)
        ui.update_select("fe_col_b",   choices=choices, session=session)
        ui.update_select("fe_bin_col", choices=choices, session=session)

    @reactive.effect
    @reactive.event(input.fe_apply)
    def _fe_apply_transform():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return
        col    = input.fe_col()
        method = input.fe_method()
        name   = input.fe_new_col_name().strip() or f"{method}_{col}"
        if not col:
            return
        try:
            df = df.copy()
            df[name] = apply_unary_transform(df[col], method)
            fe_df.set(df)
            log = fe_added.get().copy()
            log.append({"type": "transform", "desc": f"{name}  ←  {method}({col})", "before": col, "after": name})
            fe_added.set(log)
            fe_last.set({"before": col, "after": name, "error": None})
        except Exception as e:
            fe_last.set({"error": str(e)})

    @reactive.effect
    @reactive.event(input.fe_apply_inter)
    def _fe_apply_interaction():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return
        col_a = input.fe_col_a()
        col_b = input.fe_col_b()
        op    = input.fe_op()
        op_sym = {"add": "+", "subtract": "-", "multiply": "x", "divide": "div", "ratio_pct": "pct"}
        name  = input.fe_inter_name().strip() or f"{col_a}_{op_sym.get(op,'op')}_{col_b}"
        if not col_a or not col_b:
            return
        if col_a == col_b:
            fe_last.set({"error": "Column A and Column B must be different."})
            return
        try:
            df = df.copy()
            df[name] = apply_binary_transform(df[col_a], df[col_b], op)
            fe_df.set(df)
            op_label = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷", "ratio_pct": "/(A+B)×100%"}
            log = fe_added.get().copy()
            log.append({"type": "interaction", "desc": f"{name}  ←  {col_a} {op_label.get(op,op)} {col_b}", "before": col_a, "after": name})
            fe_added.set(log)
            fe_last.set({"before": col_a, "after": name, "error": None})
        except Exception as e:
            fe_last.set({"error": str(e)})

    @reactive.effect
    @reactive.event(input.fe_apply_bin)
    def _fe_apply_binning():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return
        col  = input.fe_bin_col()
        n    = input.fe_bin_n()
        name = input.fe_bin_name().strip() or f"{col}_bin{n}"
        if not col:
            return
        try:
            df = df.copy()
            df[name] = pd.cut(df[col], bins=n, labels=False)
            fe_df.set(df)
            log = fe_added.get().copy()
            log.append({"type": "binning", "desc": f"{name}  ←  cut({col}, bins={n})", "before": col, "after": name})
            fe_added.set(log)
            fe_last.set({"before": col, "after": name, "error": None})
        except Exception as e:
            fe_last.set({"error": str(e)})

    @reactive.effect
    @reactive.event(input.fe_reset)
    def _fe_reset():
        df = processed_dataset()
        if isinstance(df, pd.DataFrame):
            fe_df.set(df.copy())
        fe_added.set([])
        fe_last.set(None)

    @render.ui
    def fe_status():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return ui.p("No dataset loaded. Go to the Load Data tab first.", style="color: #6c757d;")
        last = fe_last.get()
        if last is None:
            return ui.div(
                ui.strong("Ready. "), "Use the sidebar controls to add new features.",
                style="background-color:#f8f9fa;padding:10px 14px;border:1px solid #dee2e6;border-radius:8px;margin-bottom:16px;",
            )
        if last.get("error"):
            return ui.div(
                ui.strong("Error: "), last["error"],
                style="background-color:#f8d7da;padding:10px 14px;border:1px solid #f5c2c7;border-radius:8px;color:#842029;margin-bottom:16px;",
            )
        log = fe_added.get()
        last_desc = log[-1]["desc"] if log else ""
        return ui.div(
            ui.strong("✓ Feature added: "), last_desc,
            " | ", ui.strong("Total new features: "), str(len(log)),
            style="background-color:#d1e7dd;padding:10px 14px;border:1px solid #badbcc;border-radius:8px;color:#0f5132;margin-bottom:16px;",
        )

    @render.plot
    def fe_plot():
        last = fe_last.get()
        df   = fe_df.get()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if last is None or last.get("error") or not isinstance(df, pd.DataFrame):
            for ax in axes:
                ax.text(0.5, 0.5, "Apply a transformation to see the before/after chart.",
                    ha="center", va="center", fontsize=10)
                ax.axis("off")
            return fig
        col_before = last.get("before")
        col_after  = last.get("after")
        if col_before not in df.columns or col_after not in df.columns:
            for ax in axes:
                ax.axis("off")
            return fig
        before_vals = pd.to_numeric(df[col_before], errors="coerce").dropna()
        after_vals  = pd.to_numeric(df[col_after],  errors="coerce").dropna()
        axes[0].hist(before_vals, bins=30, color="#6c8ebf", edgecolor="white")
        axes[0].set_title(f"Before: {col_before}", fontsize=11)
        axes[0].set_ylabel("Frequency")
        axes[1].hist(after_vals, bins=30, color="#82b366", edgecolor="white")
        axes[1].set_title(f"After: {col_after}", fontsize=11)
        plt.tight_layout()
        return fig

    @render.ui
    def fe_added_list():
        log = fe_added.get()
        if not log:
            return ui.p("No features added yet.", style="color: #6c757d; margin: 0;")
        type_colors = {
            "transform":   ("#cfe2ff", "#084298"),
            "interaction": ("#d1e7dd", "#0f5132"),
            "binning":     ("#fff3cd", "#664d03"),
        }
        items = []
        for i, feat in enumerate(log, 1):
            bg, fg = type_colors.get(feat["type"], ("#f8f9fa", "#212529"))
            items.append(ui.div(
                ui.tags.span(feat["type"].upper(),
                    style=f"background-color:{bg};color:{fg};font-size:11px;padding:1px 7px;border-radius:10px;font-weight:600;margin-right:8px;"),
                f"{i}.  {feat['desc']}",
                style="padding:6px 0;font-size:14px;border-bottom:1px solid #f0f0f0;",
            ))
        return ui.div(*items)

    @render.ui
    def fe_preview_notice():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return ui.div()
        n = len(fe_added.get())
        msg = f"Showing first 10 rows · {n} new feature(s) added" if n else "Showing first 10 rows"
        return ui.tags.small(msg, style="color: #6c757d; display: block; margin-bottom: 10px;")

    @render.ui
    def fe_preview_table():
        df = fe_df.get()
        if not isinstance(df, pd.DataFrame):
            return ui.p("Preview will appear here after a dataset is loaded.", style="color: #6c757d;")
        log = fe_added.get()
        new_cols = [f["after"] for f in log if f.get("after") in df.columns]
        preview = df.head(10)
        th = "".join(
            f"<th style='padding:6px 12px;text-align:left;border-bottom:2px solid #dee2e6;white-space:nowrap;background:{'#d1e7dd' if c in new_cols else ''};color:{'#0f5132' if c in new_cols else ''}'>{c}{' ✦' if c in new_cols else ''}</th>"
            for c in preview.columns
        )
        rows = ""
        for _, row in preview.iterrows():
            tds = "".join(
                f"<td style='padding:5px 12px;text-align:left;border-bottom:1px solid #f0f0f0;white-space:nowrap;background:{'#f0faf4' if c in new_cols else ''}'>{v}</td>"
                for c, v in row.items()
            )
            rows += f"<tr>{tds}</tr>"
        return ui.HTML(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    @reactive.calc
    def eda_dataset():
        # Tab3 → Tab4 联动：优先使用 feature engineering 后的数据
        fe = fe_df.get()
        if isinstance(fe, pd.DataFrame):
            return fe.copy()
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

    @render.ui
    def eda_summary_table():
        df = eda_filtered_dataset()
        if df is None or isinstance(df, str) or df.empty:
            return ui.p("No data available.", style="color: #6c757d;")
        summary = df.describe(include="all").transpose().reset_index()
        summary = summary.rename(columns={"index": "column"}).fillna("")
        th = "".join(
            f"<th style='padding:6px 12px;text-align:left;border-bottom:2px solid #dee2e6;white-space:nowrap'>{c}</th>"
            for c in summary.columns
        )
        rows = "".join(
            "<tr>" + "".join(
                f"<td style='padding:5px 12px;text-align:left;border-bottom:1px solid #f0f0f0;white-space:nowrap'>{v}</td>"
                for v in row
            ) + "</tr>"
            for _, row in summary.iterrows()
        )
        return ui.HTML(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    @render.ui
    def eda_corr_notice():
        df = eda_filtered_dataset()
        if df is None or isinstance(df, str):
            return ui.div()
        return ui.tags.small(
            "Correlation is computed using numeric columns only.",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.ui
    def eda_corr_table():
        df = eda_filtered_dataset()
        if df is None or isinstance(df, str) or df.empty:
            return ui.p("No data available.", style="color: #6c757d;")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(num_cols) < 2:
            return ui.p("At least two numeric columns are required.", style="color: #6c757d;")
        corr = df[num_cols].corr().round(3).reset_index().rename(columns={"index": ""})
        th = "".join(
            f"<th style='padding:6px 12px;text-align:left;border-bottom:2px solid #dee2e6;white-space:nowrap'>{c}</th>"
            for c in corr.columns
        )
        rows = "".join(
            "<tr>" + "".join(
                f"<td style='padding:5px 12px;text-align:left;border-bottom:1px solid #f0f0f0;white-space:nowrap'>{v}</td>"
                for v in row
            ) + "</tr>"
            for _, row in corr.iterrows()
        )
        return ui.HTML(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    @render.ui
    def eda_preview_notice():
        df = eda_filtered_dataset()
        if df is None or isinstance(df, str):
            return ui.div()
        return ui.tags.small(
            "Showing first 10 rows of the filtered processed dataset.",
            style="color: #6c757d; display: block; margin-bottom: 10px;",
        )

    @render.ui
    def eda_preview_table():
        df = eda_filtered_dataset()
        if df is None or isinstance(df, str):
            return ui.p("No data available.", style="color: #6c757d;")
        preview = df.head(10)
        th = "".join(
            f"<th style='padding:6px 12px;text-align:left;border-bottom:2px solid #dee2e6;white-space:nowrap'>{c}</th>"
            for c in preview.columns
        )
        rows = "".join(
            "<tr>" + "".join(
                f"<td style='padding:5px 12px;text-align:left;border-bottom:1px solid #f0f0f0;white-space:nowrap'>{v}</td>"
                for v in row
            ) + "</tr>"
            for _, row in preview.iterrows()
        )
        return ui.HTML(
            f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:13px;width:100%'>"
            f"<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>"
        )

app = App(app_ui, server)