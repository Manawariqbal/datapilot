import pandas as pd
import streamlit as st

from datapilot.profiling import profile_dataframe
from datapilot.experiments import run_experiments, save_run

st.set_page_config(page_title="DataPilot", layout="wide")

st.title("DataPilot")
st.caption("An open-source autonomous data scientist for tabular data")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.subheader("Dataset")
    st.dataframe(df.head(20), width="stretch")

    profile = profile_dataframe(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", profile["rows"])
    c2.metric("Columns", profile["columns"])
    c3.metric("Duplicate rows", profile["duplicate_rows"])
    c4.metric(
        "Missing cells",
        sum(profile["missing_values"].values()),
    )

    with st.expander("Data quality details"):
        st.json(profile["missing_values"])

    target = st.selectbox("Choose target column", df.columns)

    if st.button("Run DataPilot experiment", type="primary"):
        with st.spinner("Running profiling, EDA, baseline, and TabPFN..."):
            results = run_experiments(df, target, run_tabpfn_model=True)

        st.session_state["results"] = results

        try:
            run_dir = save_run(results)
            st.success(f"Experiment completed. Artifacts saved to `{run_dir}`")
        except Exception as exc:
            st.warning(f"Results computed, but artifact saving failed: {exc}")

    results = st.session_state.get("results")

    if results:
        st.subheader("Investigation")

        c1, c2 = st.columns(2)
        c1.metric("Problem type", results["problem_type"])
        c2.metric("Target", results["target"])

        st.markdown("### Model comparison")
        model_df = pd.DataFrame(results["models"])
        st.dataframe(model_df, use_container_width=True)

        st.markdown("### Target analysis")
        st.json(results["eda"]["target_summary"])

        st.markdown("### Strongest signals")
        signals = results["eda"]["correlations_with_target"][:10]
        if signals:
            st.dataframe(pd.DataFrame(signals), use_container_width=True)
        else:
            st.info("No numeric target signals were available.")

        st.markdown("### Data quality flags")
        flags = results["eda"]["quality_flags"]
        if flags:
            for flag in flags:
                st.warning(flag)
        else:
            st.success("No major deterministic quality flags detected.")

        if results["errors"]:
            st.markdown("### Experiment errors")
            st.json(results["errors"])

        st.info(
            "Next milestone: connect this deterministic experiment engine to "
            "LangGraph + local Gemma so the agent can choose and explain the next experiment."
        )
