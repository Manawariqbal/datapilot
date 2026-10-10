
import hashlib

import pandas as pd
import streamlit as st

from datapilot.profiling import profile_dataframe
from datapilot.experiments import run_experiments, save_run


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="DataPilot | AI Data Scientist",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            120deg,
            rgba(91, 112, 240, 0.13),
            rgba(32, 180, 160, 0.08)
        );
    }

    .hero h1 {
        margin: 0;
        padding: 0;
        font-size: 2.2rem;
        font-weight: 750;
    }

    .hero p {
        margin-top: 0.6rem;
        margin-bottom: 0;
        opacity: 0.8;
        font-size: 1rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.22);
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }

    .section-note {
        opacity: 0.72;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def dataframe_signature(file_bytes):
    """Identify the uploaded dataset so stale results aren't displayed."""
    return hashlib.sha256(file_bytes).hexdigest()


def show_empty_state(title, description):
    st.markdown(f"### {title}")
    st.info(description)


def format_number(value):
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, float)):
        return f"{value:,.4f}" if abs(value) < 1 else f"{value:,.2f}"

    return str(value)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🧭 DataPilot")
    st.caption("Autonomous data science workspace")

    st.divider()

    st.markdown("**Workflow**")
    st.markdown("1. Upload a dataset")
    st.markdown("2. Select the prediction target")
    st.markdown("3. Run experiments")
    st.markdown("4. Review evidence and results")

    st.divider()

    st.markdown("**Current capabilities**")
    st.markdown("- Dataset profiling")
    st.markdown("- Exploratory data analysis")
    st.markdown("- Baseline model")
    st.markdown("- TabPFN challenger")
    st.markdown("- Model comparison")

    st.divider()
    st.caption("Built with Python, Streamlit and ML tools.")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🧭 DataPilot</h1>
        <p>
            Your autonomous data scientist for tabular data.
            Upload a dataset, run experiments, and explore
            evidence-backed findings in one workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# DATASET UPLOAD
# --------------------------------------------------

upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Upload your dataset",
        type=["csv"],
        help="Upload a CSV file containing features and a target column.",
    )

with info_col:
    st.markdown("#### Supported input")
    st.markdown("CSV files containing numerical or categorical columns.")
    st.caption("Choose a target column after uploading your data.")


if not uploaded:
    st.markdown("")
    st.markdown("### Get started")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 01 · Inspect")
        st.write("Preview your data and identify missing values and duplicates.")

    with c2:
        st.markdown("#### 02 · Experiment")
        st.write("Run EDA, a baseline model, and a TabPFN challenger.")

    with c3:
        st.markdown("#### 03 · Understand")
        st.write("Compare model metrics and inspect data signals.")

    st.stop()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:
    file_bytes = uploaded.getvalue()
    signature = dataframe_signature(file_bytes)

    df = pd.read_csv(uploaded)

    if df.empty:
        st.error("This CSV contains no rows. Please upload a non-empty dataset.")
        st.stop()

    if len(df.columns) == 0:
        st.error("No columns were found in this CSV.")
        st.stop()

except Exception as exc:
    st.error(f"Unable to read the CSV: {exc}")
    st.stop()


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

profile = profile_dataframe(df)

missing_values = profile.get("missing_values", {})
missing_cells = int(sum(missing_values.values()))
duplicate_rows = int(profile.get("duplicate_rows", 0))

st.markdown("## Dataset overview")
st.caption(
    f"File: {uploaded.name} · "
    f"{len(df):,} rows · {len(df.columns):,} columns"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Total rows", f"{len(df):,}")
m2.metric("Features / columns", f"{len(df.columns):,}")
m3.metric("Duplicate rows", f"{duplicate_rows:,}")
m4.metric("Missing cells", f"{missing_cells:,}")

with st.expander("Preview dataset", expanded=True):
    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=350,
    )

    st.caption(
        f"Showing the first {min(20, len(df)):,} of {len(df):,} rows."
    )


# --------------------------------------------------
# DATA QUALITY
# --------------------------------------------------

with st.expander("Data quality summary"):
    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.markdown("**Missing values by column**")

        missing_df = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Missing values": int(count),
                    "Missing %": round(count / len(df) * 100, 2),
                }
                for column, count in missing_values.items()
            ]
        )

        if not missing_df.empty:
            missing_df = missing_df.sort_values(
                "Missing values", ascending=False
            )
            st.dataframe(
                missing_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No missing-value details available.")

    with quality_col2:
        st.markdown("**Column types**")

        dtype_df = pd.DataFrame(
            {
                "Column": df.columns.astype(str),
                "Data type": df.dtypes.astype(str).values,
                "Unique values": [
                    df[column].nunique(dropna=True)
                    for column in df.columns
                ],
            }
        )

        st.dataframe(
            dtype_df,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# EXPERIMENT CONFIGURATION
# --------------------------------------------------

st.divider()
st.markdown("## Configure experiment")

target = st.selectbox(
    "Choose the target column",
    options=list(df.columns),
    help="The column DataPilot will investigate as the prediction target.",
)

run_tabpfn_model = st.checkbox(
    "Run TabPFN challenger",
    value=True,
    help="Disable this to run only the standard experiment workflow.",
)

experiment_key = f"{signature}:{target}:{run_tabpfn_model}"

# Clear old results when the dataset or experiment configuration changes.
if st.session_state.get("experiment_key") != experiment_key:
    st.session_state.pop("results", None)
    st.session_state.pop("run_dir", None)
    st.session_state["experiment_key"] = experiment_key

run_clicked = st.button(
    "🚀 Run DataPilot experiment",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# RUN EXPERIMENT
# --------------------------------------------------

if run_clicked:
    try:
        with st.spinner(
            "DataPilot is profiling the data and running experiments..."
        ):
            results = run_experiments(
                df,
                target,
                run_tabpfn_model=run_tabpfn_model,
            )

        st.session_state["results"] = results
        st.session_state["run_dir"] = None

        try:
            run_dir = save_run(results)
            st.session_state["run_dir"] = str(run_dir)
            st.success(f"Experiment completed. Artifacts saved to `{run_dir}`")

        except Exception as exc:
            st.warning(
                "Experiments completed, but saving artifacts failed: "
                f"{exc}"
            )

    except Exception as exc:
        st.error(f"The experiment failed: {exc}")


# --------------------------------------------------
# RESULTS DASHBOARD
# --------------------------------------------------

results = st.session_state.get("results")

if results and st.session_state.get("experiment_key") == experiment_key:
    st.divider()

    st.markdown("## Investigation results")
    st.caption(
        "Review the experiment evidence below. "
        "Model metrics are calculated by the experiment engine."
    )

    # ---------------- OVERVIEW ----------------

    overview_tab, quality_tab, eda_tab, models_tab, details_tab = st.tabs(
        [
            "📊 Overview",
            "🧹 Data quality",
            "🔎 EDA insights",
            "🤖 Model comparison",
            "🧪 Experiment details",
        ]
    )

    with overview_tab:
        problem_type = results.get("problem_type", "Not available")
        result_target = results.get("target", target)

        st.markdown("### Experiment summary")

        a1, a2, a3 = st.columns(3)

        a1.metric("Problem type", str(problem_type).replace("_", " ").title())
        a2.metric("Target column", str(result_target))
        a3.metric("Models evaluated", len(results.get("models", [])))

        models = results.get("models", [])
        model_df = pd.DataFrame(models)

        if not model_df.empty:
            st.markdown("### Performance at a glance")
            st.dataframe(
                model_df,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Key findings")

        eda = results.get("eda", {})
        quality_flags = eda.get("quality_flags", [])

        if quality_flags:
            for flag in quality_flags[:5]:
                st.warning(str(flag))
        else:
            st.success("No major deterministic data-quality flags detected.")

        comparison = results.get("model_comparison")

        if comparison:
            st.markdown("### Model comparison conclusion")
            st.write(
                comparison.get(
                    "conclusion",
                    "Comparison results are available in the model tab.",
                )
            )

        if st.session_state.get("run_dir"):
            st.caption(
                f"Saved artifacts: {st.session_state['run_dir']}"
            )

    # ---------------- DATA QUALITY ----------------

    with quality_tab:
        st.markdown("### Data quality findings")

        flags = results.get("eda", {}).get("quality_flags", [])

        if flags:
            for flag in flags:
                st.warning(str(flag))
        else:
            st.success("No major deterministic quality flags detected.")

        st.markdown("### Missing values")

        if missing_cells:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Column": column,
                            "Missing values": int(count),
                            "Missing %": round(count / len(df) * 100, 2),
                        }
                        for column, count in missing_values.items()
                        if count > 0
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("The uploaded dataset has no missing cells.")

        st.markdown("### Duplicate records")
        st.metric("Duplicate rows", f"{duplicate_rows:,}")

    # ---------------- EDA ----------------

    with eda_tab:
        st.markdown("### Target analysis")

        target_summary = results.get("eda", {}).get("target_summary", {})

        if target_summary:
            st.json(target_summary)
        else:
            show_empty_state(
                "Target summary unavailable",
                "The experiment engine did not return target summary data.",
            )

        st.markdown("### Strongest signals")

        signals = results.get("eda", {}).get(
            "correlations_with_target", []
        )

        if signals:
            signals_df = pd.DataFrame(signals)
            st.dataframe(
                signals_df.head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            show_empty_state(
                "No numeric target signals",
                "No numeric correlations with the target were returned.",
            )

        st.caption(
            "Correlations indicate associations, not causation. "
            "A low correlation does not necessarily mean a feature is useless."
        )

    # ---------------- MODEL COMPARISON ----------------

    with models_tab:
        st.markdown("### Model performance")

        models = results.get("models", [])
        model_df = pd.DataFrame(models)

        if model_df.empty:
            show_empty_state(
                "No model results",
                "No model metrics were returned by this experiment.",
            )
        else:
            st.dataframe(
                model_df,
                use_container_width=True,
                hide_index=True,
            )

            # Plot only common predictive performance metrics.
            metric_candidates = [
                "accuracy",
                "f1_weighted",
                "precision_weighted",
                "recall_weighted",
                "roc_auc",
            ]

            available_metrics = [
                metric
                for metric in metric_candidates
                if metric in model_df.columns
            ]

            if available_metrics:
                model_name_col = next(
                    (
                        column
                        for column in ["model", "model_name", "name"]
                        if column in model_df.columns
                    ),
                    None,
                )

                if model_name_col:
                    chart_df = model_df[
                        [model_name_col] + available_metrics
                    ].copy()

                    chart_df = chart_df.set_index(model_name_col)

                    for column in available_metrics:
                        chart_df[column] = pd.to_numeric(
                            chart_df[column], errors="coerce"
                        )

                    chart_df = chart_df.dropna(how="all", subset=available_metrics)

                    if not chart_df.empty:
                        st.markdown("### Compare metrics")
                        selected_metric = st.selectbox(
                            "Select metric",
                            available_metrics,
                            format_func=lambda value: value.replace(
                                "_", " "
                            ).title(),
                        )

                        st.bar_chart(chart_df[[selected_metric]])

            comparison = results.get("model_comparison")

            if comparison:
                st.markdown("### Comparison evidence")
                st.json(comparison)

    # ---------------- EXPERIMENT DETAILS ----------------

    with details_tab:
        st.markdown("### Errors and warnings")

        errors = results.get("errors", [])

        if errors:
            st.json(errors)
        else:
            st.success("No experiment errors were reported.")

        st.markdown("### Raw experiment output")

        with st.expander("View complete results object"):
            st.json(results)

        run_dir = st.session_state.get("run_dir")

        if run_dir:
            st.markdown("### Artifacts")
            st.code(run_dir)

    st.divider()

    if st.button("Clear current results"):
        st.session_state.pop("results", None)
        st.session_state.pop("run_dir", None)
        st.rerun()
