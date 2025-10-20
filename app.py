import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Intelligent Data Assistant", layout="wide")


# -----------------------------
# Session State Initialization
# -----------------------------
def initialize_session_state() -> None:
    if "raw_df" not in st.session_state:
        st.session_state.raw_df: Optional[pd.DataFrame] = None
    if "processed_df" not in st.session_state:
        st.session_state.processed_df: Optional[pd.DataFrame] = None
    if "impute_strategies" not in st.session_state:
        st.session_state.impute_strategies: Dict[str, str] = {}
    if "encoding_strategies" not in st.session_state:
        st.session_state.encoding_strategies: Dict[str, str] = {}
    if "scaler_choice" not in st.session_state:
        st.session_state.scaler_choice: Optional[str] = None
    if "target_column" not in st.session_state:
        st.session_state.target_column: Optional[str] = None
    if "output_choice" not in st.session_state:
        st.session_state.output_choice: str = "Download Cleaned CSV"
    if "explain_log" not in st.session_state:
        st.session_state.explain_log: List[str] = []


initialize_session_state()


# -----------------------------
# Helper Utilities
# -----------------------------
def compute_data_overview(dataframe: pd.DataFrame) -> Tuple[int, int, int]:
    num_rows: int = dataframe.shape[0]
    num_columns: int = dataframe.shape[1]
    total_missing: int = int(dataframe.isna().sum().sum())
    return num_rows, num_columns, total_missing


def build_info_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    column_names: List[str] = list(dataframe.columns)
    dtypes: List[str] = [str(dtype) for dtype in dataframe.dtypes]
    missing_percentages: List[float] = (
        dataframe.isna().mean().fillna(0.0).values * 100.0
    )
    info_df: pd.DataFrame = pd.DataFrame(
        {
            "column": column_names,
            "dtype": dtypes,
            "missing_%": np.round(missing_percentages, 2),
        }
    )
    return info_df


def get_numerical_columns(dataframe: pd.DataFrame) -> List[str]:
    return list(dataframe.select_dtypes(include=[np.number]).columns)


def get_categorical_columns(dataframe: pd.DataFrame) -> List[str]:
    return list(dataframe.select_dtypes(include=["object", "category"]).columns)


def add_log(message: str) -> None:
    st.session_state.explain_log.append(message)


def generate_processing_code(
    impute_strategies: Dict[str, str],
    encoding_strategies: Dict[str, str],
    scaler_choice: Optional[str],
) -> str:
    """
    Create a Python function as a string that reproduces the user's selected
    preprocessing steps. The function will be named `apply_cleaning` and take a
    pandas DataFrame as input, returning a cleaned DataFrame.
    """

    lines: List[str] = []
    lines.append("import pandas as pd")
    lines.append("import numpy as np")
    # Only import scalers if needed
    if scaler_choice in {"StandardScaler", "MinMaxScaler"}:
        lines.append("from sklearn.preprocessing import StandardScaler, MinMaxScaler")
    lines.append("")
    lines.append("def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:")
    lines.append("    \"\"\"Apply the selected preprocessing steps to a copy of the input DataFrame.\"\"\"")
    lines.append("    cleaned = df.copy()")
    lines.append("")

    # Imputation
    if len(impute_strategies) > 0:
        lines.append("    # 1) Missing Value Imputation")
        for column_name, strategy in impute_strategies.items():
            if strategy == "Mean":
                lines.append(
                    f"    if cleaned['{column_name}'].dtype.kind in 'biufc':\n"
                    f"        cleaned['{column_name}'] = cleaned['{column_name}'].fillna(cleaned['{column_name}'].mean())\n"
                    f"    else:\n"
                    f"        # Non-numeric column selected for 'Mean'; fallback to mode\n"
                    f"        mode_val = cleaned['{column_name}'].mode(dropna=True).iloc[0] if not cleaned['{column_name}'].mode(dropna=True).empty else None\n"
                    f"        cleaned['{column_name}'] = cleaned['{column_name}'].fillna(mode_val)"
                )
            elif strategy == "Median":
                lines.append(
                    f"    if cleaned['{column_name}'].dtype.kind in 'biufc':\n"
                    f"        cleaned['{column_name}'] = cleaned['{column_name}'].fillna(cleaned['{column_name}'].median())\n"
                    f"    else:\n"
                    f"        # Non-numeric column selected for 'Median'; fallback to mode\n"
                    f"        mode_val = cleaned['{column_name}'].mode(dropna=True).iloc[0] if not cleaned['{column_name}'].mode(dropna=True).empty else None\n"
                    f"        cleaned['{column_name}'] = cleaned['{column_name}'].fillna(mode_val)"
                )
            else:  # Mode
                lines.append(
                    f"    mode_val = cleaned['{column_name}'].mode(dropna=True).iloc[0] if not cleaned['{column_name}'].mode(dropna=True).empty else None\n"
                    f"    cleaned['{column_name}'] = cleaned['{column_name}'].fillna(mode_val)"
                )
        lines.append("")

    # Encoding
    one_hot_columns: List[str] = [
        col for col, strategy in encoding_strategies.items() if strategy == "One-Hot Encode"
    ]
    label_encode_columns: List[str] = [
        col for col, strategy in encoding_strategies.items() if strategy == "Label Encode"
    ]

    if len(one_hot_columns) > 0:
        cols_literal = ", ".join([f"'{c}'" for c in one_hot_columns])
        lines.append("    # 2) Categorical Encoding - One-Hot")
        lines.append(
            f"    cleaned = pd.get_dummies(cleaned, columns=[{cols_literal}], drop_first=False)"
        )
        lines.append("")

    if len(label_encode_columns) > 0:
        lines.append("    # 3) Categorical Encoding - Label Encode via category codes")
        for col in label_encode_columns:
            lines.append(
                f"    if cleaned['{col}'].dtype.name in ['object', 'category']:\n"
                f"        cleaned['{col}'] = cleaned['{col}'].astype('category').cat.codes\n"
                f"    else:\n"
                f"        # If not categorical, keep as is (no-op)\n"
                f"        cleaned['{col}'] = cleaned['{col}']"
            )
        lines.append("")

    # Scaling
    if scaler_choice in {"StandardScaler", "MinMaxScaler"}:
        lines.append("    # 4) Numerical Feature Scaling")
        lines.append("    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()")
        if scaler_choice == "StandardScaler":
            lines.append("    scaler = StandardScaler()")
        else:
            lines.append("    scaler = MinMaxScaler()")
        lines.append("    if len(numeric_cols) > 0:")
        lines.append("        cleaned[numeric_cols] = scaler.fit_transform(cleaned[numeric_cols])")
        lines.append("")

    lines.append("    return cleaned")
    return "\n".join(lines)


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.title("Intelligent Data Assistant")
st.sidebar.markdown("### 1) Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        st.session_state.raw_df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        st.session_state.raw_df = pd.read_csv(uploaded_file, encoding_errors="ignore")


raw_df: Optional[pd.DataFrame] = st.session_state.raw_df

if raw_df is not None:
    # Dynamic options based on data
    numerical_columns: List[str] = get_numerical_columns(raw_df)
    categorical_columns: List[str] = get_categorical_columns(raw_df)
    columns_with_missing: List[str] = [
        col for col in raw_df.columns if raw_df[col].isna().any()
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 2) Preprocessing Options")

    # Missing Value Imputation
    st.sidebar.markdown("**2.1 Missing Value Imputation**")
    for col in columns_with_missing:
        if col in numerical_columns:
            choices = ["Mean", "Median", "Mode"]
            default_index = 0
        else:
            choices = ["Mode"]
            default_index = 0
        selected = st.sidebar.selectbox(
            f"Impute '{col}'",
            choices,
            index=default_index,
            key=f"impute_{col}",
        )
        st.session_state.impute_strategies[col] = selected

    # Categorical Variable Encoding
    st.sidebar.markdown("**2.2 Categorical Variable Encoding**")
    for col in categorical_columns:
        enc_choice = st.sidebar.selectbox(
            f"Encode '{col}'",
            ["One-Hot Encode", "Label Encode"],
            index=0,
            key=f"encode_{col}",
        )
        st.session_state.encoding_strategies[col] = enc_choice

    # Numerical Feature Scaling
    st.sidebar.markdown("**2.3 Numerical Feature Scaling**")
    st.sidebar.radio(
        "Select scaler for all numerical features",
        ["StandardScaler", "MinMaxScaler"],
        index=0,
        key="scaler_choice",
    )

    # Target Variable Selection
    st.sidebar.markdown("### 3) Target Variable Selection")
    target_options: List[str] = ["None (Unsupervised Learning)"] + list(raw_df.columns)
    target_selected = st.sidebar.selectbox(
        "Select target column (optional)", target_options, index=0
    )
    st.session_state.target_column = (
        None if target_selected == "None (Unsupervised Learning)" else target_selected
    )

    # Final Output Choice
    st.sidebar.markdown("### 4) Final Output Choice")
    st.sidebar.radio(
        "Choose your output",
        ["Download Cleaned CSV", "Generate Python Code"],
        index=0,
        key="output_choice",
    )

    # Process Button
    st.sidebar.markdown("---")
    process_clicked: bool = st.sidebar.button("Process Data & Get Insights", type="primary")
else:
    process_clicked = False


# -----------------------------
# Main Page - Dashboard & Results
# -----------------------------
st.title("Intelligent Data Assistant")

if raw_df is None:
    st.info("Upload a CSV file from the sidebar to get started.")
else:
    # Visual Data Summary
    st.markdown("### Data Overview")
    rows, cols, missing = compute_data_overview(raw_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Number of Rows", f"{rows:,}")
    c2.metric("Number of Columns", f"{cols:,}")
    c3.metric("Total Missing Values", f"{missing:,}")

    st.markdown("#### Column Summary (like df.info())")
    st.dataframe(build_info_table(raw_df), use_container_width=True)

    # Interactive Charting
    st.markdown("### Interactive Charts")
    chart_tabs = st.tabs(["Histogram", "Scatter", "Box Plot", "Correlation Heatmap"])

    # Histogram
    with chart_tabs[0]:
        if len(numerical_columns) > 0:
            hist_col = st.selectbox("Select numerical column", numerical_columns, key="hist_col")
            fig = px.histogram(raw_df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available for histogram.")

    # Scatter
    with chart_tabs[1]:
        if len(numerical_columns) >= 2:
            scatter_x = st.selectbox("X-axis", numerical_columns, key="scatter_x")
            scatter_y = st.selectbox("Y-axis", numerical_columns, key="scatter_y")
            color_col = st.selectbox(
                "Color (optional)", [None] + categorical_columns, index=0, key="scatter_color"
            )
            fig = px.scatter(
                raw_df,
                x=scatter_x,
                y=scatter_y,
                color=color_col if color_col else None,
                title=f"Scatter: {scatter_x} vs {scatter_y}",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least two numerical columns for scatter plot.")

    # Box Plot
    with chart_tabs[2]:
        if len(numerical_columns) > 0:
            box_num = st.selectbox("Numerical column", numerical_columns, key="box_num")
            box_cat = st.selectbox(
                "Category (optional)", [None] + categorical_columns, index=0, key="box_cat"
            )
            fig = px.box(
                raw_df,
                x=box_cat if box_cat else None,
                y=box_num,
                points="outliers",
                title=f"Box Plot of {box_num}" + (f" by {box_cat}" if box_cat else ""),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns available for box plot.")

    # Correlation Heatmap
    with chart_tabs[3]:
        corr_df = raw_df.select_dtypes(include=[np.number])
        if corr_df.shape[1] > 0:
            corr = corr_df.corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                origin="lower",
                title="Correlation Heatmap (Numerical Features)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical features available for correlation heatmap.")

    # Processing & Results
    if process_clicked:
        st.session_state.explain_log = []

        working_df = raw_df.copy()

        # 1) Imputation
        for col, strategy in st.session_state.impute_strategies.items():
            missing_before = int(working_df[col].isna().sum())
            if missing_before == 0:
                continue
            if strategy == "Mean" and working_df[col].dtype.kind in "biufc":
                fill_value = working_df[col].mean()
                working_df[col] = working_df[col].fillna(fill_value)
                add_log(f"Filled {missing_before} missing in '{col}' using Mean ({fill_value:.4f}).")
            elif strategy == "Median" and working_df[col].dtype.kind in "biufc":
                fill_value = working_df[col].median()
                working_df[col] = working_df[col].fillna(fill_value)
                add_log(f"Filled {missing_before} missing in '{col}' using Median ({fill_value:.4f}).")
            else:
                mode_series = working_df[col].mode(dropna=True)
                fill_value = mode_series.iloc[0] if not mode_series.empty else None
                working_df[col] = working_df[col].fillna(fill_value)
                add_log(
                    f"Filled {missing_before} missing in '{col}' using Mode ({fill_value})."
                )

        # 2) Categorical Encoding
        # One-Hot Encoding first (creates new columns and removes originals)
        one_hot_cols = [
            col
            for col, strategy in st.session_state.encoding_strategies.items()
            if strategy == "One-Hot Encode" and col in working_df.columns
        ]
        if len(one_hot_cols) > 0:
            working_df = pd.get_dummies(working_df, columns=one_hot_cols, drop_first=False)
            add_log(
                "Applied One-Hot Encoding to: " + ", ".join([f"'{c}'" for c in one_hot_cols])
            )

        # Label Encoding (category codes)
        label_cols = [
            col
            for col, strategy in st.session_state.encoding_strategies.items()
            if strategy == "Label Encode"
        ]
        for col in label_cols:
            if col in working_df.columns and working_df[col].dtype.name in ["object", "category"]:
                working_df[col] = working_df[col].astype("category").cat.codes
                add_log(f"Label-encoded column '{col}' using category codes.")

        # 3) Scaling
        numeric_cols_after_encoding = list(working_df.select_dtypes(include=[np.number]).columns)
        if st.session_state.scaler_choice == "StandardScaler":
            scaler = StandardScaler()
            if len(numeric_cols_after_encoding) > 0:
                working_df[numeric_cols_after_encoding] = scaler.fit_transform(
                    working_df[numeric_cols_after_encoding]
                )
                add_log(
                    f"Scaled {len(numeric_cols_after_encoding)} numerical features using StandardScaler."
                )
        elif st.session_state.scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
            if len(numeric_cols_after_encoding) > 0:
                working_df[numeric_cols_after_encoding] = scaler.fit_transform(
                    working_df[numeric_cols_after_encoding]
                )
                add_log(
                    f"Scaled {len(numeric_cols_after_encoding)} numerical features using MinMaxScaler."
                )

        # Save results
        st.session_state.processed_df = working_df

        # Model Suggestions
        suggestions: List[str] = []
        if st.session_state.target_column is None:
            suggestions.append("Clustering: KMeans, DBSCAN, GaussianMixture")
        else:
            target_series = raw_df[st.session_state.target_column]
            if pd.api.types.is_numeric_dtype(target_series):
                suggestions.append("Regression: LinearRegression, RandomForestRegressor, XGBoostRegressor")
            else:
                suggestions.append("Classification: LogisticRegression, RandomForestClassifier, XGBoostClassifier")

        # Display Results
        st.markdown("### Processed DataFrame")
        st.dataframe(st.session_state.processed_df.head(100), use_container_width=True)

        st.markdown("### Explainability Log")
        if len(st.session_state.explain_log) == 0:
            st.info("No cleaning actions were needed.")
        else:
            for item in st.session_state.explain_log:
                st.write(f"- {item}")

        st.markdown("### ML Model Suggestions")
        for s in suggestions:
            st.write(f"- {s}")

        st.markdown("### Final Output")
        if st.session_state.output_choice == "Download Cleaned CSV":
            csv_buffer = io.StringIO()
            st.session_state.processed_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
            )
        else:
            code_str = generate_processing_code(
                st.session_state.impute_strategies,
                st.session_state.encoding_strategies,
                st.session_state.scaler_choice,
            )
            st.code(code_str, language="python")


