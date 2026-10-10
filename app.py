import hashlib
import json
import numbers
from datetime import datetime
from pathlib import Path

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
# REPORT BUILDER
# --------------------------------------------------
# Builds the full experiment report as Markdown. It is shown inside the
# app (📄 Full report tab), offered as a download, and saved next to the
# run artifacts. Every lookup uses .get() so a missing key in the results
# never breaks the report; that section simply says "not available".

# Preferred metric used to name the "best" model (first one present wins).
PRIMARY_METRICS = ["roc_auc", "f1_weighted", "accuracy", "r2", "rmse", "mae"]
# For these metrics a smaller value is better.
LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "log_loss"}


def format_value(value):
    """Human-readable text for any scalar that may appear in the results."""
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, numbers.Integral):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real):
        return f"{float(value):,.4f}"
    return str(value)


def df_to_markdown(frame):
    """Render a DataFrame as a Markdown table (no extra dependencies)."""
    if frame is None or frame.empty:
        return "_No data available._"

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False):
        cells = [
            format_value(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dict_to_markdown(data):
    """Render a dict as bullets; nested values become JSON code blocks."""
    if not data:
        return "_No data available._"

    lines = []
    for key, value in data.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, (dict, list, tuple)):
            lines.append(f"- **{label}**:")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, default=str))
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"- **{label}**: {format_value(value)}")
    return "\n".join(lines)


def pick_best_model(model_df):
    """Return the best model by the first available primary metric."""
    if model_df is None or model_df.empty:
        return None

    name_col = next(
        (c for c in ["model", "model_name", "name"] if c in model_df.columns),
        None,
    )
    if name_col is None:
        return None

    for metric in PRIMARY_METRICS:
        if metric not in model_df.columns:
            continue
        scores = pd.to_numeric(model_df[metric], errors="coerce")
        if not scores.notna().any():
            continue
        lower_is_better = metric in LOWER_IS_BETTER
        best_idx = scores.idxmin() if lower_is_better else scores.idxmax()
        return {
            "model": str(model_df.loc[best_idx, name_col]),
            "metric": metric,
            "score": float(scores.loc[best_idx]),
            "lower_is_better": lower_is_better,
        }
    return None


def build_report_markdown(results, df, profile, dataset_name, run_dir=None):
    """Assemble the complete experiment report as a Markdown string."""
    eda = results.get("eda", {}) or {}
    models = results.get("models", []) or []
    model_df = pd.DataFrame(models)
    comparison = results.get("model_comparison")
    errors = results.get("errors", []) or []
    flags = eda.get("quality_flags", []) or []

    missing_values = profile.get("missing_values", {}) or {}
    missing_cells = int(sum(missing_values.values()))
    duplicate_rows = int(profile.get("duplicate_rows", 0))

    problem_type = (
        str(results.get("problem_type", "Not available"))
        .replace("_", " ")
        .title()
    )
    target = results.get("target", "Not available")
    best = pick_best_model(model_df)

    conclusion = None
    comparison_details = None
    if isinstance(comparison, dict):
        conclusion = comparison.get("conclusion")
        comparison_details = {
            key: value for key, value in comparison.items() if key != "conclusion"
        }
    elif comparison:
        conclusion = str(comparison)

    out = []

    # ---- Title ----
    out.append("# DataPilot Experiment Report")
    out.append("")
    out.append(
        f"**Dataset:** `{dataset_name}` · "
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    out.append("")

    # ---- 1. Executive summary ----
    out.append("## 1. Executive summary")
    out.append("")
    out.append(f"- **Dataset size:** {len(df):,} rows × {len(df.columns):,} columns")
    out.append(f"- **Target column:** `{target}`")
    out.append(f"- **Problem type:** {problem_type}")
    out.append(f"- **Models evaluated:** {len(models)}")
    if best:
        direction = "lower is better" if best["lower_is_better"] else "higher is better"
        out.append(
            f"- **Best model:** {best['model']} "
            f"({best['metric']} = {best['score']:.4f}, {direction})"
        )
    out.append(f"- **Data-quality flags:** {len(flags)}")
    out.append(f"- **Experiment errors:** {len(errors)}")
    out.append("")
    if conclusion:
        out.append("**Model comparison conclusion**")
        out.append("")
        out.append(str(conclusion))
        out.append("")

    # ---- 2. Dataset overview ----
    out.append("## 2. Dataset overview")
    out.append("")
    out.append(f"- **Rows:** {len(df):,}")
    out.append(f"- **Columns:** {len(df.columns):,}")
    out.append(f"- **Duplicate rows:** {duplicate_rows:,}")
    out.append(f"- **Missing cells:** {missing_cells:,}")
    out.append("")
    out.append("### Columns")
    out.append("")
    column_df = pd.DataFrame(
        {
            "Column": df.columns.astype(str),
            "Data type": df.dtypes.astype(str).values,
            "Unique values": [df[c].nunique(dropna=True) for c in df.columns],
            "Missing values": [int(df[c].isna().sum()) for c in df.columns],
            "Missing %": [
                f"{df[c].isna().sum() / len(df) * 100:.2f}%" for c in df.columns
            ],
        }
    )
    out.append(df_to_markdown(column_df))
    out.append("")

    # ---- 3. Data quality ----
    out.append("## 3. Data quality")
    out.append("")
    out.append("### Quality flags")
    out.append("")
    if flags:
        for flag in flags:
            out.append(f"- ⚠️ {flag}")
    else:
        out.append("No major deterministic data-quality flags detected.")
    out.append("")
    out.append("### Missing values")
    out.append("")
    if missing_cells:
        missing_df = pd.DataFrame(
            [
                {
                    "Column": column,
                    "Missing values": int(count),
                    "Missing %": f"{count / len(df) * 100:.2f}%",
                }
                for column, count in missing_values.items()
                if count > 0
            ]
        ).sort_values("Missing values", ascending=False)
        out.append(df_to_markdown(missing_df))
    else:
        out.append("The dataset has no missing cells.")
    out.append("")
    out.append("### Duplicate records")
    out.append("")
    out.append(f"{duplicate_rows:,} duplicate row(s) found.")
    out.append("")

    # ---- 4. EDA ----
    out.append("## 4. Exploratory data analysis")
    out.append("")
    out.append("### Target summary")
    out.append("")
    out.append(dict_to_markdown(eda.get("target_summary", {})))
    out.append("")
    out.append("### Strongest signals (correlation with target)")
    out.append("")
    signals = eda.get("correlations_with_target", [])
    if signals:
        out.append(df_to_markdown(pd.DataFrame(signals).head(15)))
    else:
        out.append("No numeric correlations with the target were returned.")
    out.append("")
    out.append(
        "_Correlations indicate associations, not causation. "
        "A low correlation does not necessarily mean a feature is useless._"
    )
    out.append("")

    # ---- 5. Models ----
    out.append("## 5. Model results")
    out.append("")
    if model_df.empty:
        out.append("No model metrics were returned by this experiment.")
    else:
        out.append(df_to_markdown(model_df))
        if best:
            out.append("")
            out.append(
                f"**Best model by `{best['metric']}`:** {best['model']} "
                f"({best['score']:.4f})"
            )
    out.append("")
    if comparison_details:
        out.append("### Comparison evidence")
        out.append("")
        out.append(dict_to_markdown(comparison_details))
        out.append("")

    # ---- 6. Errors ----
    out.append("## 6. Errors and warnings")
    out.append("")
    if errors:
        for error in errors:
            if isinstance(error, (dict, list)):
                out.append("```json")
                out.append(json.dumps(error, indent=2, default=str))
                out.append("```")
            else:
                out.append(f"- {error}")
    else:
        out.append("No experiment errors were reported.")
    out.append("")

    # ---- 7. Artifacts ----
    out.append("## 7. Saved artifacts")
    out.append("")
    if run_dir:
        out.append(f"Run artifacts are saved in `{run_dir}`.")
    else:
        out.append("Run artifacts were not saved for this experiment.")
    out.append("")

    return "\n".join(out)


def save_report_file(report_md, run_dir):
    """Write the report next to the run artifacts as report.md."""
    path = Path(run_dir) / "report.md"
    path.write_text(report_md, encoding="utf-8")
    return path


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
    st.markdown("5. Read or download the full report")

    st.divider()

    st.markdown("**Current capabilities**")
    st.markdown("- Dataset profiling")
    st.markdown("- Exploratory data analysis")
    st.markdown("- Baseline model")
    st.markdown("- TabPFN challenger")
    st.markdown("- Model comparison")
    st.markdown("- Full downloadable report")

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
        st.write("Compare model metrics, inspect data signals, and get a full report.")

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
        except Exception as exc:
            st.warning(
                "Experiments completed, but saving artifacts failed: "
                f"{exc}"
            )
        else:
            # Save the full report alongside the run artifacts.
            try:
                report_md = build_report_markdown(
                    results, df, profile, uploaded.name, str(run_dir)
                )
                save_report_file(report_md, run_dir)
                st.success(
                    f"Experiment completed. Artifacts and report saved to `{run_dir}`"
                )
            except Exception as exc:
                st.success(f"Experiment completed. Artifacts saved to `{run_dir}`")
                st.warning(f"The report file could not be saved: {exc}")

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

    # ---------------- TABS ----------------

    (
        overview_tab,
        quality_tab,
        eda_tab,
        models_tab,
        details_tab,
        report_tab,
    ) = st.tabs(
        [
            "📊 Overview",
            "🧹 Data quality",
            "🔎 EDA insights",
            "🤖 Model comparison",
            "🧪 Experiment details",
            "📄 Full report",
        ]
    )

    # ---------------- OVERVIEW ----------------

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

        st.info(
            "Want every detail in one place? Open the **📄 Full report** tab "
            "to read the complete report or download it."
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

    # ---------------- FULL REPORT ----------------

    with report_tab:
        report_md = build_report_markdown(
            results,
            df,
            profile,
            uploaded.name,
            st.session_state.get("run_dir"),
        )

        report_stem = Path(uploaded.name).stem

        st.markdown("### Full experiment report")
        st.caption(
            "Every detail from this run in one place: dataset overview, "
            "data quality, EDA, model results, comparison evidence and errors."
        )

        d1, d2 = st.columns(2)

        d1.download_button(
            "⬇️ Download report (.md)",
            data=report_md,
            file_name=f"datapilot_report_{report_stem}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        d2.download_button(
            "⬇️ Download raw results (.json)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"datapilot_results_{report_stem}.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()

        st.markdown(report_md)

        with st.expander("View report as Markdown source"):
            st.code(report_md, language="markdown")
