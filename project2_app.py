from shiny import App, reactive, render, ui
import pandas as pd
import pyreadr
from pathlib import Path

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
                
                    ui.hr(),
                ),

                ui.card(
                    ui.card_header("Output"),
    
                    style="margin-bottom: 20px;",
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

        ui.nav_panel(
            "Exploratory Data Analysis",

            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("EDA"),
                  
                    ui.hr(),
                ),

                ui.card(
                    ui.card_header("Visualization"),
                   
                    style="margin-bottom: 20px;",
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


app = App(app_ui, server)