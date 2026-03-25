from shiny import App, reactive, render, ui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.datasets import load_diabetes, load_breast_cancer
from pathlib import Path
import pyreadr


# 1. HELPERS


def load_uploaded_file(file_path: str, original_name: str) -> pd.DataFrame:
    ext = Path(original_name).suffix.lower()
    if ext == ".csv": return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]: return pd.read_excel(file_path)
    elif ext == ".parquet": return pd.read_parquet(file_path)
    elif ext == ".rds":
        result = pyreadr.read_r(file_path)
        return pd.DataFrame(result[next(iter(result.keys()))])
    return pd.read_csv(file_path)


# 2. UI


app_ui = ui.page_fluid(
    ui.tags.style("""
        .stats-container { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-card { background: #ffffff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 8px 16px; min-width: 130px; border-top: 3px solid #0d6efd; }
        .stat-card.missing { border-top-color: #dc3545; }
        .dictionary-pre { background: #f8f9fa; padding: 15px; border: 1px solid #ddd; font-family: monospace; font-size: 0.85rem; overflow-x: auto; white-space: pre; }
        .log-box { font-size: 0.85rem; padding: 12px; border-radius: 6px; margin-top: 10px; border-left: 5px solid; background: #e8f5e9; border-color: #4caf50; color: #1b5e20; }
        .scaling-note { background: #fff3cd; color: #856404; padding: 12px; border-radius: 6px; border: 1px solid #ffeeba; margin-bottom: 10px; font-size: 0.9rem; }
        .target-lock-box { background: #f1f8ff; border: 1px solid #0366d6; padding: 10px; border-radius: 6px; margin-bottom: 15px; font-size: 0.9rem; }
        .compact-table table { font-size: 0.82rem !important; width: 100% !important; margin-bottom: 0 !important; }
        .smart-logic-banner { background: #e7f3ff; padding: 10px; border-radius: 8px; border: 1px solid #b6d4fe; margin-bottom: 15px; font-size: 0.85rem; }
        .fe-section { border: 1px solid #ddd; padding: 15px; border-radius: 8px; margin-bottom: 20px; background: #f9f9f9; }
        .fe-section-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; border-bottom: 2px solid #0d6efd; padding-bottom: 8px; }
        .feature-controls { display: flex; flex-direction: column; gap: 10px; }
        .slider-with-label { display: flex; align-items: center; gap: 10px; }
        .slider-with-label .shiny-input-slider { flex: 1; }
        .slider-value { background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-weight: 600; min-width: 40px; text-align: center; }
        .fe-undo-reset { display: flex; gap: 5px; margin-top: 10px; }
        .fe-undo-reset .btn { flex: 1; }
    """),

    ui.h2("STATGR5243 Data Analysis Tool", style="margin-top: 20px; font-weight: 700;"),
    ui.output_ui("summary_stats_banner"),

    ui.navset_tab(
        ui.nav_panel("0. User Guide",
            ui.card(

                ui.div(
                    ui.h3("Team 5"),
                    ui.p("Fangyi Lin (fl2748), Zhe Lin (zl3613), Zhanhang Shi (zs2741), Daisy Zhou (dz2590)"),
                    ui.h4("Overview"),
                    ui.p("This tool provides a workflow for data analysis from loading raw data to performing feature engineering. Please read the user guide to understand how to use the tool before proceeding to the next tab."),
                    
                    ui.h5("1. Load Data"),
                    ui.p("Start here by uploading your own dataset or using one of the built-in sample datasets."),
                    ui.tags.ul(
                        ui.tags.li("Under Data Source, choose 'Upload File' to load either a CSV, Excel, Parquet, or RDS file"),
                        ui.tags.li("Or, select 'Use Built-in Data' to test out the Diabetes or Breast Cancer dataset"),
                        ui.tags.li("You can see a preview of the dataset on the right to confirm it has loaded correctly")
                    ),
                    
                    ui.h5("2. EDA (Exploratory Data Analysis)"),
                    ui.p("The goal of EDA is to understand your dataset before cleaning and preprocessing."),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Data Dictionary"), " - See column names, data types, missing value and unique counts"),
                        ui.tags.li(ui.strong("Dataset Explorer"), " - Browse and filter your data with row limits and column filters"),
                        ui.tags.li(ui.strong("Summary Statistics"), " - View min, max, mean, std, and percentiles for selected variables")
                    ),
                    
                    ui.h5("3. Visualizations"),
                    ui.p("Create visualizations to understand relationships and distributions of the variables in your dataset."),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Interactive Plot"), " - Select a variable to view as a histogram, boxplot, scatterplot, or bar chart. An error warning will pop up if the variable type is not compatible with the plot type."),
                        ui.tags.li(ui.strong("Correlation Matrix"), " - View correlations between numeric variables to understand strong vs weak relationships.")
                    ),
                    
                    ui.h5("4. Clean & Preprocess"),
                    ui.p("Prepare your data for analysis and modeling."),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Data Cleaning"), " - Remove duplicates, handle outliers, and drop columns where necessary"),
                        ui.tags.li(ui.strong("Imputation"), " - Handle missing values by either dropping the rows altogether or imputing via mean, median, or mode."),
                        ui.tags.li(ui.strong("Transformation"), " - Label encode the categorical variables. One hot is not an option here to prevent the creation of an extremely large number of columns that would make the dataset hard to view in the tool."),
                        ui.tags.li(ui.strong("Scaling"), " - Standardize (Z-Score) or normalize (Min-Max) numeric features. Note that only one method should be applied consistently to all of the numeric features."),
                        ui.tags.li(ui.strong("Undo & Reset"), " - Revert the last change made or start over."),
                        ui.tags.li(ui.strong("Export"), " - Export the cleaned dataset to a csv file if needed after all methods have been applied.")
                    ),
                    
                    ui.h5("5. Feature Engineering"),
                    ui.p("Create new features to improve model performance."),
                    ui.tags.ul(
                        ui.tags.li(ui.strong("Single Column Transform"), " - Transform one variable at a time using the following transformations: log, sqrt, square, reciprocal, power, etc."),
                        ui.tags.li(ui.strong("Interaction Terms"), " - Multiply, add, divide, or create ratios from two columns of your choice"),
                        ui.tags.li(ui.strong("Binning"), " - Discretize continuous variables into bins"),
                        ui.tags.li(ui.strong("Before/After Plots"), " - See how your transformations change the distributions with a before and after"),
                        ui.tags.li(ui.strong("Undo & Reset"), " - Undo the last transformation or reset altogether."),
                        ui.tags.li(ui.strong("Export"), " - Export the final dataset to a csv file if needed after all feature engineering transformations have been performed.")
                    ),
                    
                    ui.h4("Tips"),
                    ui.tags.ul(
                        ui.tags.li("Always start with EDA to understand your data"),
                        ui.tags.li("Remove duplicates and outliers before imputation"),
                        ui.tags.li("Choose consistent scaling method across all features"),
                        ui.tags.li("Test transformations and observe the changes in the before/after plots"),
                        ui.tags.li("Use the Undo button if you want to try a different approach"),
                        ui.tags.li("Download your final dataset for use in other tools"),
                        ui.tags.li("As you perform data cleaning and preprocessing, you can track the number of missing values remaining in the top grey bar")
                    ),
                    
                    ui.hr(),
                    ui.p(ui.em("To get started, go to '1. Load Data' and upload your dataset!"), style="text-align: center; color: #666; font-size: 1.1rem; font-weight: 500;"),
                    
                    class_="mb-3"
                )
            )
        ),

        ui.nav_panel("1. Load Data",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("data_source", "Data Source", {
                        "upload": "Upload File",
                        "builtin": "Use Built-in Data"
                    }),
                    ui.panel_conditional("input.data_source === 'upload'", ui.input_file("file", "Choose File")),
                    ui.panel_conditional("input.data_source === 'builtin'", 
                        ui.input_select("builtin_data", "Select Built-in Data", {
                            "diabetes": "Diabetes (Regression)",
                            "breast_cancer": "Breast Cancer (Classification)"
                        })
                    ),
                ),
                ui.card(ui.card_header("Raw Data Preview"), ui.output_table("preview_table")),
            )
        ),
        
        ui.nav_panel("2. EDA",
            ui.card(ui.card_header("Data Dictionary"), ui.output_ui("data_dict_ui")),
            ui.card(
                ui.card_header("Dataset Explorer - Filter & Explore"),
                ui.div(
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.h6("Display Options"),
                            ui.input_select("row_limit", "Rows to Display", {
                                "50": "50 rows",
                                "100": "100 rows",
                                "150": "150 rows",
                                "200": "200 rows",
                                "300": "300 rows",
                                "500": "500 rows"
                            }, selected="100"),
                            ui.hr(),
                            ui.h6("Filter Options"),
                            ui.input_selectize("filter_col", "Filter by Column", choices=[], multiple=False),
                            ui.input_selectize("filter_values", "Select Values", choices=[], multiple=True),
                            ui.input_action_button("clear_filter", "Clear Filters", class_="btn-outline-secondary w-100"),
                            ui.div(
                                ui.tags.small("Showing: ", ui.tags.b(ui.output_ui("row_count_display"))),
                                style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px;"
                            ),
                            width="250px"
                        ),
                        ui.div(ui.output_data_frame("dataset_table"), style="overflow-x: auto;")
                    ),
                    style="padding: 0;"
                )
            ),
            ui.card(
                ui.card_header("Summary Statistics"),
                ui.div(
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.h6("Select Variables"),
                            ui.input_selectize("stats_vars", "Variables to Display", choices=[], multiple=True),
                            ui.input_action_button("stats_all", "Select All", class_="btn-outline-secondary w-100"),
                            ui.input_action_button("stats_clear", "Clear All", class_="btn-outline-secondary w-100"),
                            width="220px"
                        ),
                        ui.div(ui.output_table("math_summary_table"), style="overflow-x: auto;")
                    ),
                    style="padding: 0;"
                )
            ),
        ),

        ui.nav_panel("3. Visualizations",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("plot_type", "Plot Type", {"hist": "Histogram", "box": "Boxplot", "scatter": "Scatterplot", "bar": "Bar Chart"}),
                    ui.input_slider("bar_limit", "Max Categories (Bar)", 5, 30, 10),
                    ui.hr(),
                    ui.input_selectize("eda_x", "Select X Variable", choices=[]),
                    ui.panel_conditional("input.plot_type === 'scatter'", ui.input_selectize("eda_y", "Select Y Variable", choices=[])),
                ),
                ui.card(ui.card_header("Interactive Plot"), ui.output_plot("main_plot")),
                ui.card(ui.card_header("Correlation Matrix"), ui.div(ui.output_table("correlation_heatmap"), style="overflow-x: auto;")),
            )
        ),

        ui.nav_panel("4. Clean & Preprocess",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h6("Data Cleaning"),
                    ui.input_checkbox("remove_dupes", "Remove Duplicate Rows", False),
                    ui.input_checkbox("remove_outliers", "Remove Outliers (IQR)", False),
                    ui.input_selectize("drop_cols", "Drop Columns", choices=[], multiple=True),
                    ui.input_select("target_col", "Target Variable", choices=[]),
                    ui.hr(),
                    ui.h6("Imputation"),
                    ui.input_select("impute_method", "Method", {
                        "none": "Keep Missing", "mean": "Mean", "median": "Median", "mode": "Mode", "drop": "Drop Rows"
                    }),
                    ui.input_selectize("impute_cols", "Apply To:", choices=[], multiple=True),
                    ui.hr(),
                    ui.h6("Transformation"),
                    ui.input_checkbox("label_encode", "Label Encode Categorical", False),
                    ui.input_select("scaling_method", "Scaling Method", {"none": "None", "zscore": "Z-Score", "minmax": "Min-Max"}),
                    ui.input_selectize("scaling_cols", "Apply Scaling to:", choices=[], multiple=True),
                    ui.hr(),
                    ui.input_action_button("apply_prep", "Apply All Changes", class_="btn-primary w-100"),
                    ui.div(
                        ui.input_action_button("undo_prep", "Undo Last", class_="btn-warning flex-grow-1"),
                        ui.input_action_button("reset_prep", "Reset All", class_="btn-outline-secondary flex-grow-1"),
                        style="display: flex; gap: 5px; margin-top: 10px;"
                    ),
                    ui.download_button("download_csv", "Download CSV", class_="btn-success w-100 mt-2"),
                ),
                ui.card(ui.card_header("Preprocessing Changes"), ui.output_ui("transformation_summary_ui")),
                ui.card(
                    ui.card_header("Dataset Summary and Preprocessing Recommendations"),
                    ui.div(
                        ui.tags.b("Note: Scaling Consistency"),
                        ui.p("Apply ONE scaling method (Z-Score or Min-Max) across all features to keep model weights balanced.", class_="mb-0"),
                        class_="scaling-note"
                    ),
                    ui.div(ui.output_table("prep_diagnostics"), class_="compact-table")
                ),
                ui.card(ui.card_header("Cleaned Dataset Preview"), ui.div(ui.output_table("prep_table"), class_="compact-table")),
            )
        ),

        ui.nav_panel("5. Feature Engineering",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.output_ui("fe_target_display"),
                    
                    # SECTION 1: Single Column Transform
                    ui.div(
                        ui.div("1) Single Column Transform", class_="fe-section-title"),
                        ui.input_text("fe_new_name", "New Column Name", placeholder="e.g., log_price"),
                        ui.input_select("fe_col", "Source Column", choices=[]),
                        ui.input_select("fe_math", "Transformation", {
                            "log": "log(x)",
                            "log1p": "log1p: ln(x+1)",
                            "sqrt": "sqrt(x)",
                            "square": "square: x²",
                            "abs": "abs: |x|",
                            "reciprocal": "reciprocal: 1/x",
                            "zscore": "z-score standardization",
                            "minmax": "min-max normalization"
                        }),
                        ui.input_action_button("apply_fe", "Create Feature", class_="btn-success w-100"),
                        class_="fe-section"
                    ),
                    
                    # SECTION 2: Two Column Interaction
                    ui.div(
                        ui.div("2) Two Column Interaction", class_="fe-section-title"),
                        ui.input_select("int_col_a", "Column A", choices=[]),
                        ui.input_select("int_op", "Operation", {
                            "multiply": "A × B",
                            "add": "A + B",
                            "subtract": "A - B",
                            "divide": "A / B",
                            "ratio": "A / (A+B) × 100%"
                        }),
                        ui.input_select("int_col_b", "Column B", choices=[]),
                        ui.input_text("int_new_name", "New Column Name", placeholder="e.g., feature_interaction"),
                        ui.input_action_button("apply_interaction", "Apply Interaction", class_="btn-info w-100"),
                        class_="fe-section"
                    ),
                    
                    # SECTION 3: Binning (Discretization)
                    ui.div(
                        ui.div("3) Binning (Discretization)", class_="fe-section-title"),
                        ui.input_select("bin_col", "Select Column", choices=[]),
                        ui.div(
                            ui.input_slider("bin_count", "Number of Bins", 2, 20, 5),
                            class_="slider-with-label"
                        ),
                        ui.input_text("bin_new_name", "New Column Name", placeholder="e.g., price_bins"),
                        ui.input_action_button("apply_binning", "Apply Binning", class_="btn-warning w-100"),
                        class_="fe-section"
                    ),
                    
                    # Undo/Reset buttons for Feature Engineering
                    ui.div(
                        ui.input_action_button("undo_fe", "↶ Undo Last Feature", class_="btn-warning flex-grow-1"),
                        ui.input_action_button("reset_fe", "↺ Reset All Features", class_="btn-outline-secondary flex-grow-1"),
                        class_="fe-undo-reset"
                    ),
                    ui.hr(),
                    ui.download_button("download_fe_csv", "Download Engineered CSV", class_="btn-success w-100"),
                ),
                ui.card(ui.card_header("Engineered Dataset Preview"), ui.div(ui.output_table("fe_preview_table"), class_="compact-table")),
                ui.card(ui.card_header("Before & After Distribution"), ui.output_plot("fe_distribution_plot")),
                ui.card(ui.card_header("Feature Engineering Log"), ui.output_ui("fe_log_ui")),
            )
        )
    )
)

# 3. SERVER


def server(input, output, session):
    raw_df = reactive.value(None)
    proc_df = reactive.value(None)
    prev_df = reactive.value(None)
    execution_log = reactive.value([])
    
    # Feature Engineering specific tracking
    fe_df = reactive.value(None)
    fe_prev_df = reactive.value(None)
    fe_execution_log = reactive.value([])
    last_fe_column = reactive.value(None)  # Track the last engineered column name
    last_fe_source_column = reactive.value(None)  # Track the source column
    
    # Target variable locking
    locked_target = reactive.value(None)
    
    # Data Explorer filtering
    explorer_filter_col = reactive.value(None)
    explorer_filter_values = reactive.value([])

    @reactive.calc
    def active_data():
        return proc_df.get() if proc_df.get() is not None else raw_df.get()
    
    @reactive.calc
    def fe_active_data():
        """Data after feature engineering"""
        return fe_df.get() if fe_df.get() is not None else active_data()

    @reactive.effect
    def _data_load():
        if input.data_source() == "upload" and input.file() is not None:
            f = input.file()[0]; raw_df.set(load_uploaded_file(f["datapath"], f["name"]))
        elif input.data_source() == "builtin":
            builtin_choice = input.builtin_data()
            if builtin_choice == "diabetes":
                raw_df.set(pd.DataFrame(load_diabetes(as_frame=True).data))
            elif builtin_choice == "breast_cancer":
                raw_df.set(pd.DataFrame(load_breast_cancer(as_frame=True).data))
        proc_df.set(None); execution_log.set([])
        fe_df.set(None); fe_execution_log.set([])

    @reactive.effect
    def _update_choices():
        df = active_data()
        if df is None: return
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        current_target = locked_target.get()
        
        ui.update_selectize("eda_x", choices=all_cols)
        ui.update_selectize("eda_y", choices=all_cols)
        ui.update_select("fe_col", choices=num_cols)
        ui.update_select("int_col_a", choices=num_cols)
        ui.update_select("int_col_b", choices=num_cols)
        ui.update_select("bin_col", choices=num_cols)
        ui.update_selectize("drop_cols", choices=all_cols, selected=[])
        ui.update_selectize("impute_cols", choices=all_cols, selected=[])
        ui.update_selectize("scaling_cols", choices=num_cols, selected=[])
        
        # Maintain target selection if it still exists o.w. update choices only
        if current_target and current_target in all_cols:
            ui.update_select("target_col", choices=all_cols, selected=current_target)
        else:
            ui.update_select("target_col", choices=all_cols)

    @render.ui
    def data_dict_ui():
        df = active_data()
        if df is None: return ui.p("No data loaded.")
        lines = []
        for c in df.columns:
            dtype = df[c].dtype
            missing = df[c].isna().sum()
            unique = df[c].nunique()
            lines.append(f"{c:<30} | Type: {str(dtype):<12} | Missing: {missing:<6} | Unique: {unique}")
        header = f"{'Column':<30} | Type: {'dtype':<12} | Missing: {'count':<6} | Unique: {'#'}"
        separator = "-" * 85
        full_text = header + "\n" + separator + "\n" + "\n".join(lines)
        return ui.pre(full_text, class_="dictionary-pre")

    @render.ui
    def summary_stats_banner():
        df = active_data()
        if df is None: return None
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isna().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        return ui.div(
            ui.div(
                ui.tags.span("Rows: ", style="font-weight: 600;"),
                f"{len(df):,} | ",
                ui.tags.span("Columns: ", style="font-weight: 600;"),
                f"{len(df.columns)} | ",
                ui.tags.span("Missing Cells: ", style="font-weight: 600;"),
                f"{missing_cells:,} ({missing_pct:.1f}%)",
                style="padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #0d6efd; font-size: 0.95rem; margin-bottom: 15px;"
            )
        )

    @render.table
    def prep_diagnostics():
        df = active_data()
        if df is None or len(df) == 0: return None
        diag = []
        for c in df.columns:
            m = df[c].isna().sum(); mp = m / len(df) if len(df) > 0 else 0
            col_l = c.lower()
            unique_ratio = df[c].nunique() / len(df) if len(df) > 0 else 0
            is_cat = (not pd.api.types.is_numeric_dtype(df[c]) or (unique_ratio < 0.10) or any(k in col_l for k in ["id", "type"])) and not any(k in col_l for k in ["tax", "fee", "amount", "fare", "total", "count"])
            is_dt = pd.api.types.is_datetime64_any_dtype(df[c])
            is_n = pd.api.types.is_numeric_dtype(df[c])
            i_rec = "-"
            if mp > 0.70:
                i_rec = "Remove Variable"
            elif m == 0 or is_dt: i_rec = "-"
            elif is_cat: i_rec = "Mode"
            else: i_rec = "Median"
            s_rec = "-"
            if is_n and not is_cat and not is_dt:
                sk = df[c].skew() if df[c].notna().any() else 0
                s_rec = "Z-Score" if abs(sk) < 1 else "Min-Max"
            diag.append({"Column": c, "Missing Count": f"{m:,}", "Missing %": f"{mp*100:.1f}%", "Logic": "Categorical" if is_cat else ("Datetime" if is_dt else "Numeric"), "Impute Rec": i_rec, "Scale Rec": s_rec})
        return pd.DataFrame(diag).style.map(lambda v: 'background-color: #f8d7da; color: #842029; font-weight: bold;' if v == 'Remove Variable' else '', subset=['Impute Rec'])

    @reactive.effect
    @reactive.event(input.apply_prep)
    def _prep():
        df_start = active_data()
        if df_start is None: return
        prev_df.set(df_start.copy())
        df = df_start.copy()
        log = list(execution_log.get())
        current_target = input.target_col()

        if input.remove_dupes():
            initial = len(df); df.drop_duplicates(inplace=True)
            log.append(f"Removed {initial - len(df)} duplicate rows")
        if input.remove_outliers():
            initial_rows = len(df)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for c in num_cols:
                # Calculate quartiles only on non-missing values
                Q1 = df[c].quantile(0.25)
                Q3 = df[c].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only apply if IQR is valid
                    # Remove outliers but keep rows with missing values
                    lower_bound = Q1 - 1.5*IQR
                    upper_bound = Q3 + 1.5*IQR
                    df = df[(df[c].isna()) | ((df[c] >= lower_bound) & (df[c] <= upper_bound))]
            outlier_removed = initial_rows - len(df)
            log.append(f"Outliers removed (IQR 1.5x): {outlier_removed} rows")

        dropped = list(input.drop_cols())
        if dropped:
            df.drop(columns=dropped, inplace=True)
            log.append(f"Dropped columns: {dropped}")
        
        method, imp_cols = input.impute_method(), list(input.impute_cols())
        if method != "none" and imp_cols:
            for c in imp_cols:
                col_l = c.lower()
                is_cat = (not pd.api.types.is_numeric_dtype(df[c]) or (df[c].nunique()/len(df) < 0.10) or any(k in col_l for k in ["id", "type"])) and not any(k in col_l for k in ["tax", "fee", "amount", "fare", "total", "count"])
                if method == "mode" or is_cat: v = df[c].mode()[0] if not df[c].mode().empty else np.nan
                elif method == "mean": v = df[c].mean()
                else: v = df[c].median()
                df[c] = df[c].fillna(v)
            log.append(f"Imputed {imp_cols} via {method}")
        
        if input.label_encode():
            le = LabelEncoder(); cat_cols = df.select_dtypes(exclude=[np.number]).columns
            for c in cat_cols: df[c] = le.fit_transform(df[c].astype(str))
            log.append("Categorical columns Label Encoded")

        sm, sc = input.scaling_method(), list(input.scaling_cols())
        if sm != "none" and sc:
            scaler = StandardScaler() if sm == "zscore" else MinMaxScaler()
            df[sc] = scaler.fit_transform(df[sc])
            log.append(f"Scaled {sc} using {sm}")
            
        proc_df.set(df)
        execution_log.set(log)
        fe_df.set(None)  # Reset FE when prep changes
        fe_execution_log.set([])

        new_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ui.update_selectize("drop_cols", choices=new_cols, selected=[])
        ui.update_selectize("impute_cols", choices=new_cols, selected=[])
        ui.update_selectize("scaling_cols", choices=num_cols, selected=[])
        ui.update_select("fe_col", choices=num_cols)
        ui.update_select("int_col_a", choices=num_cols)
        ui.update_select("int_col_b", choices=num_cols)
        ui.update_select("bin_col", choices=num_cols)
        # Retain target_col if it still exists, otherwise update choices
        if current_target in new_cols:
            ui.update_select("target_col", choices=new_cols, selected=current_target)
        else:
            ui.update_select("target_col", choices=new_cols)

    @reactive.effect
    @reactive.event(input.undo_prep)
    def _undo():
        if prev_df.get() is not None:
            proc_df.set(prev_df.get())
            log = list(execution_log.get())
            log.append("Last action undone.")
            execution_log.set(log)

    @reactive.effect
    @reactive.event(input.reset_prep)
    def _reset():
        proc_df.set(None); execution_log.set(["Reset to raw data."])
        fe_df.set(None); fe_execution_log.set([])

    @render.ui
    def transformation_summary_ui():
        log = execution_log.get()
        if not log: return ui.p("No changes applied yet.")
        return ui.div(ui.tags.ul([ui.tags.li(x) for x in log]), class_="log-box")

    @render.plot
    def main_plot():
        df = active_data(); x = input.eda_x(); pt = input.plot_type()
        if df is None or not x: return None
        fig, ax = plt.subplots(); is_num = pd.api.types.is_numeric_dtype(df[x])
        if pt in ["hist", "box", "scatter"] and not is_num:
            ax.text(0.5, 0.5, "Categorical Data: Use Bar Chart", ha='center', va='center', color="red"); ax.set_axis_off(); return fig
        if pt == "bar": df[x].value_counts().head(input.bar_limit()).plot(kind='bar', ax=ax, color="#198754")
        elif pt == "hist": ax.hist(df[x].dropna(), bins=30, color="#0d6efd")
        elif pt == "box": sns.boxplot(y=df[x], ax=ax, color="#0dcaf0")
        elif pt == "scatter" and input.eda_y(): sns.scatterplot(x=df[x], y=df[input.eda_y()], ax=ax)
        return fig

    @render.table
    def correlation_heatmap():
        """Display correlation matrix of numeric columns"""
        df = active_data()
        if df is None: return None
        
        # Select only numeric columns
        num_cols = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        
        if num_cols.empty or len(num_cols.columns) < 2:
            return None
        
        # Calculate and return correlation matrix with styling
        corr = num_cols.corr()
        return corr.style.background_gradient(cmap='coolwarm', axis=None).format(precision=2)

    @render.table
    def preview_table(): return raw_df.get().head(10) if raw_df.get() is not None else None
    @render.table
    def prep_table(): return active_data().head(10) if active_data() is not None else None
    @render.table
    def fe_preview_table(): return fe_active_data().head(10) if fe_active_data() is not None else None
    
    @reactive.effect
    def _update_stats_vars():
        """Update stats variable choices"""
        df = active_data()
        if df is None: return
        all_cols = df.columns.tolist()
        ui.update_selectize("stats_vars", choices=all_cols, selected=[])
    
    @reactive.effect
    @reactive.event(input.stats_all)
    def _stats_select_all():
        """Select all variables for stats"""
        df = active_data()
        if df is None: return
        all_cols = df.columns.tolist()
        ui.update_selectize("stats_vars", selected=all_cols)
    
    @reactive.effect
    @reactive.event(input.stats_clear)
    def _stats_clear_all():
        """Clear all variables for stats"""
        ui.update_selectize("stats_vars", selected=[])
    
    @render.table
    def math_summary_table():
        df = active_data()
        if df is None: return None
        
        selected_vars = input.stats_vars()
        
        # If no variables selected, show all
        if not selected_vars or len(selected_vars) == 0:
            return df.describe(include='all').transpose().reset_index()
        
        # Show only selected variables
        df_selected = df[list(selected_vars)]
        return df_selected.describe(include='all').transpose().reset_index()
    
    @render.table
    def corr_matrix_table():
        df = active_data()
        if df is None: return None
        num = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        return num.corr().style.background_gradient(cmap='coolwarm', axis=None).format(precision=2) if not num.empty else None

    # DATA EXPLORER SECTION 
    
    @reactive.effect
    def _update_filter_columns():
        """Update filter column choices"""
        df = active_data()
        if df is None: return
        all_cols = df.columns.tolist()
        ui.update_selectize("filter_col", choices=all_cols, selected=[])
    
    @reactive.effect
    def _update_filter_values():
        """Update filter values based on selected column"""
        df = active_data()
        filter_col = input.filter_col()
        
        print(f"DEBUG _update_filter_values: filter_col='{filter_col}'")
        
        if df is None or not filter_col or filter_col not in df.columns:
            ui.update_selectize("filter_values", choices=[], selected=[])
            return
        
        # Get unique values for the selected column
        unique_vals = df[filter_col].dropna().unique()
        # Convert to strings for display
        unique_vals_str = sorted([str(v) for v in unique_vals])
        print(f"DEBUG: Found {len(unique_vals_str)} unique values for {filter_col}")
        ui.update_selectize("filter_values", choices=unique_vals_str, selected=[])
    
    @reactive.effect
    @reactive.event(input.clear_filter)
    def _clear_filters():
        """Clear all filters"""
        ui.update_selectize("filter_col", selected=[])
        ui.update_selectize("filter_values", selected=[])
    
    @reactive.calc
    def filtered_data():
        """Get filtered dataset with row limit"""
        df = active_data()
        if df is None: return None
        
        filter_col = input.filter_col()
        filter_vals = input.filter_values()
        row_limit = int(input.row_limit()) if input.row_limit() else 100
        
        print(f"DEBUG filtered_data: filter_col='{filter_col}', filter_vals={filter_vals}, len={len(filter_vals) if filter_vals else 0}")
        
        # Start with all data
        result_df = df.copy()
        
        # Apply row limit first
        result_df = result_df.head(row_limit)
        
        # Then apply filter if both column and values are selected
        if filter_col and filter_vals and len(filter_vals) > 0:
            print(f"DEBUG: Applying filter - col='{filter_col}', values={list(filter_vals)}")
            if filter_col in result_df.columns:
                try:
                    # Convert filter_vals to list 
                    filter_vals_list = list(filter_vals)
                    
                    print(f"DEBUG: Filtering {filter_col} with values {filter_vals_list}")
                    
                    # Apply filter
                    result_df = result_df[result_df[filter_col].astype(str).isin(filter_vals_list)]
                    
                    print(f"DEBUG: Result rows after filter: {len(result_df)}")
                except Exception as e:
                    print(f"Filter error: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"DEBUG: No filter applied (col={bool(filter_col)}, vals={bool(filter_vals and len(filter_vals) > 0)})")
        
        return result_df
    
    @render.data_frame
    def dataset_table():
        """Display filtered dataset as interactive table"""
        df = filtered_data()
        if df is None or len(df) == 0:
            return None
        return render.DataGrid(df, filters=False)
    
    @render.ui
    def row_count_display():
        """Display count of filtered rows"""
        df_filtered = filtered_data()
        df_total = active_data()
        
        if df_total is None:
            return "No data"
        
        filtered_count = len(df_filtered) if df_filtered is not None and len(df_filtered) > 0 else 0
        total_count = len(df_total)
        
        row_limit = int(input.row_limit()) if input.row_limit() else 100
        filter_col = input.filter_col()
        filter_vals = input.filter_values()
        
        # Show different messages based on what's being displayed
        if filter_col and len(filter_vals) > 0:
            return ui.tags.span(f"{filtered_count:,} filtered (limit: {row_limit})")
        else:
            return ui.tags.span(f"{filtered_count:,} of {total_count:,} (limit: {row_limit})")

    @render.ui
    def fe_target_display():
        t = locked_target.get()
        if t:
            return ui.div(
                ui.tags.b("Target:"), 
                ui.div(f"Target: {t}", style="margin-top: 8px; padding: 8px; background: #e8f5e9; border-radius: 4px; font-weight: 600;"),
                class_="target-lock-box"
            )
        else:
            return ui.div(
                ui.tags.b("No Target Selected"),
                ui.div("Select a target variable in the Clean & Preprocess tab", style="margin-top: 8px; color: #666; font-size: 0.9rem;"),
                class_="target-lock-box"
            )
    
    @reactive.effect
    def _lock_target():
        """Lock the target variable when selected"""
        t = input.target_col()
        if t:
            locked_target.set(t)

    @render.download(filename="processed_data.csv")
    async def download_csv():
        df = active_data()
        if df is not None: yield df.to_csv(index=False)

    @render.download(filename="engineered_data.csv")
    async def download_fe_csv():
        df = fe_active_data()
        if df is not None: yield df.to_csv(index=False)

    # FEATURE ENGINEERING SECTION 
    
    @reactive.effect
    @reactive.event(input.apply_fe)
    def _apply_fe():
        """Apply single column transformation"""
        df = fe_active_data().copy()
        col = input.fe_col()
        math = input.fe_math()
        user_name = input.fe_new_name()
        
        if col and math:
            new_name = user_name if user_name else f"{col}_{math}"
            
            try:
                if math == "log": df[new_name] = np.log(df[col])
                elif math == "log1p": df[new_name] = np.log1p(df[col])
                elif math == "sqrt": df[new_name] = np.sqrt(df[col])
                elif math == "square": df[new_name] = df[col]**2
                elif math == "abs": df[new_name] = np.abs(df[col])
                elif math == "reciprocal": df[new_name] = 1 / df[col].replace(0, np.nan)
                elif math == "zscore":
                    mean = df[col].mean()
                    std = df[col].std()
                    df[new_name] = (df[col] - mean) / std if std > 0 else df[col]
                elif math == "minmax":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[new_name] = (df[col] - min_val) / (max_val - min_val) if max_val > min_val else df[col]
                
                fe_df.set(df)
                last_fe_column.set(new_name)  # Track the new column
                last_fe_source_column.set(col)  # Track source for comparison
                log = list(fe_execution_log.get())
                log.append(f"✓ Feature '{new_name}' created from {col} ({math})")
                fe_execution_log.set(log)
                
                new_cols = df.columns.tolist()
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                ui.update_selectize("eda_x", choices=new_cols)
                ui.update_selectize("drop_cols", choices=new_cols)
                ui.update_select("fe_col", choices=num_cols)
                ui.update_select("int_col_a", choices=num_cols)
                ui.update_select("int_col_b", choices=num_cols)
                ui.update_select("bin_col", choices=num_cols)
                ui.update_text("fe_new_name", value="")
            except Exception as e:
                log = list(fe_execution_log.get())
                log.append(f"✗ Error creating '{new_name}': {str(e)}")
                fe_execution_log.set(log)
    
    @reactive.effect
    @reactive.event(input.apply_interaction)
    def _apply_interaction():
        """Apply two-column interaction"""
        df = fe_active_data().copy()
        col_a = input.int_col_a()
        col_b = input.int_col_b()
        op = input.int_op()
        user_name = input.int_new_name()
        
        if col_a and col_b and op:
            new_name = user_name if user_name else f"{col_a}_{op}_{col_b}"
            
            try:
                if op == "multiply":
                    df[new_name] = df[col_a] * df[col_b]
                    op_display = "×"
                elif op == "add":
                    df[new_name] = df[col_a] + df[col_b]
                    op_display = "+"
                elif op == "subtract":
                    df[new_name] = df[col_a] - df[col_b]
                    op_display = "-"
                elif op == "divide":
                    df[new_name] = df[col_a] / df[col_b].replace(0, np.nan)
                    op_display = "/"
                elif op == "ratio":
                    df[new_name] = (df[col_a] / (df[col_a] + df[col_b]) * 100).replace([np.inf, -np.inf], np.nan)
                    op_display = "% ratio"
                
                fe_df.set(df)
                last_fe_column.set(new_name)  # Track the new column
                last_fe_source_column.set(f"{col_a} & {col_b}")  # Track source columns
                log = list(fe_execution_log.get())
                log.append(f"✓ Interaction '{new_name}' created: {col_a} {op_display} {col_b}")
                fe_execution_log.set(log)
                
                new_cols = df.columns.tolist()
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                ui.update_selectize("eda_x", choices=new_cols)
                ui.update_select("fe_col", choices=num_cols)
                ui.update_select("int_col_a", choices=num_cols)
                ui.update_select("int_col_b", choices=num_cols)
                ui.update_select("bin_col", choices=num_cols)
                ui.update_text("int_new_name", value="")
            except Exception as e:
                log = list(fe_execution_log.get())
                log.append(f"✗ Error creating interaction: {str(e)}")
                fe_execution_log.set(log)
    
    @reactive.effect
    @reactive.event(input.apply_binning)
    def _apply_binning():
        """Apply binning/discretization"""
        df = fe_active_data().copy()
        col = input.bin_col()
        n_bins = input.bin_count()
        user_name = input.bin_new_name()
        
        if col and n_bins >= 2:
            new_name = user_name if user_name else f"{col}_binned"
            
            try:
                df[new_name] = pd.cut(df[col], bins=n_bins, labels=False, duplicates='drop')
                
                fe_df.set(df)
                last_fe_column.set(new_name)  # Track the new column
                last_fe_source_column.set(col)  # Track source column
                log = list(fe_execution_log.get())
                log.append(f"✓ Binned '{col}' into {n_bins} bins → '{new_name}'")
                fe_execution_log.set(log)
                
                new_cols = df.columns.tolist()
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                ui.update_selectize("eda_x", choices=new_cols)
                ui.update_select("fe_col", choices=num_cols)
                ui.update_select("int_col_a", choices=num_cols)
                ui.update_select("int_col_b", choices=num_cols)
                ui.update_select("bin_col", choices=num_cols)
                ui.update_text("bin_new_name", value="")
            except Exception as e:
                log = list(fe_execution_log.get())
                log.append(f"✗ Error binning '{col}': {str(e)}")
                fe_execution_log.set(log)
    
    @reactive.effect
    @reactive.event(input.undo_fe)
    def _undo_fe():
        """Undo last feature engineering step - removes only the last added feature"""
        log = list(fe_execution_log.get())
        last_col = last_fe_column.get()
        
        if not log or not last_col:
            return
        
        # Get current engineered dataframe
        df = fe_active_data()
        if df is None:
            return
        
        # Remove only the last engineered column if it exists
        if last_col in df.columns:
            df = df.drop(columns=[last_col])
            fe_df.set(df)
        
        # Remove last log entry and update tracking
        last_fe_column.set(None)
        last_fe_source_column.set(None)
        log.pop()
        
        # Add undo message to log
        if log:
            log.append(f"↶ Removed '{last_col}'")
        else:
            log.append("↶ All features removed")
        
        fe_execution_log.set(log)
    
    @reactive.effect
    @reactive.event(input.reset_fe)
    def _reset_fe():
        """Reset all feature engineering"""
        fe_df.set(None)
        last_fe_column.set(None)
        last_fe_source_column.set(None)
        fe_execution_log.set(["↺ Feature engineering reset"])

    @render.ui
    def fe_log_ui():
        """Display feature engineering log"""
        log = fe_execution_log.get()
        if not log: return ui.p("No features engineered yet.")
        return ui.div(ui.tags.ul([ui.tags.li(x) for x in log]), class_="log-box")

    @render.plot
    def fe_distribution_plot():
        """Plot before and after distribution of last engineered feature"""
        new_col = last_fe_column.get()
        source_col = last_fe_source_column.get()
        
        if new_col is None or source_col is None:
            return None
        
        df_fe = fe_active_data()
        df_active = active_data()
        
        if df_fe is None or df_active is None:
            return None
        
        # For interaction features, don't show before/after
        if " & " in str(source_col):
            fig, ax = plt.subplots(figsize=(10, 4))
            if new_col in df_fe.columns:
                ax.hist(df_fe[new_col].dropna(), bins=30, color="#0d6efd", alpha=0.7, edgecolor='black')
                ax.set_title(f"Distribution of '{new_col}' (Interaction Feature)", fontsize=12, fontweight='bold')
                ax.set_xlabel(new_col)
                ax.set_ylabel("Frequency")
            return fig
        
        # For single column transforms and binning, show before and after
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before (source column)
        if source_col in df_active.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df_active[source_col])
            if is_numeric:
                ax1.hist(df_active[source_col].dropna(), bins=30, color="#198754", alpha=0.7, edgecolor='black')
            else:
                df_active[source_col].value_counts().head(10).plot(kind='bar', ax=ax1, color="#198754")
                ax1.tick_params(axis='x', rotation=45)
            ax1.set_title(f"Before: '{source_col}'", fontsize=12, fontweight='bold')
            ax1.set_xlabel(source_col)
            ax1.set_ylabel("Frequency")
        
        # After (new engineered column)
        if new_col in df_fe.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df_fe[new_col])
            if is_numeric:
                ax2.hist(df_fe[new_col].dropna(), bins=30, color="#0d6efd", alpha=0.7, edgecolor='black')
            else:
                df_fe[new_col].value_counts().head(10).plot(kind='bar', ax=ax2, color="#0d6efd")
                ax2.tick_params(axis='x', rotation=45)
            ax2.set_title(f"After: '{new_col}'", fontsize=12, fontweight='bold')
            ax2.set_xlabel(new_col)
            ax2.set_ylabel("Frequency")
        
        fig.tight_layout()
        return fig

app = App(app_ui, server)