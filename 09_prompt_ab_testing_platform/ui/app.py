"""Streamlit dashboard for the Prompt A/B Testing Platform."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import random
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.hashing import assign_variant, compute_traffic_distribution, fixed_hash
from src.models import Experiment, ExperimentVariant, PromptVersion, VariantAllocation, ExperimentOutcome
from src.statistics import compute_statistics, evaluate_kill_switch

st.set_page_config(page_title="Prompt A/B Testing", page_icon=":ab:", layout="wide")

st.title("Prompt Versioning and A/B Testing Platform")
st.markdown("Fixed hash traffic splitting with scipy.stats statistical analysis.")

tab1, tab2, tab3 = st.tabs(["Traffic Splitting", "Statistical Analysis", "Kill Switch"])

with tab1:
    st.subheader("Deterministic Traffic Splitting")
    st.markdown("See how users get assigned to variants using fixed hashing.")

    col1, col2 = st.columns(2)
    traffic_split = col1.slider("Treatment Traffic %", 0, 100, 50)
    sample_size = col2.slider("Sample Size", 100, 10000, 1000, step=100)

    if "experiment" not in st.session_state:
        control_pv = PromptVersion(prompt_id="test", version="1.0.0", system_prompt="v1", user_template="t1", model="gpt-4o")
        treatment_pv = PromptVersion(prompt_id="test", version="2.0.0", system_prompt="v2", user_template="t2", model="gpt-4o")
        st.session_state["experiment"] = Experiment(
            experiment_id="exp-001",
            name="Test Experiment",
            variants=[
                ExperimentVariant(variant_id="control", allocation=VariantAllocation.CONTROL, prompt_version=control_pv, traffic_percentage=1 - traffic_split / 100),
                ExperimentVariant(variant_id="treatment", allocation=VariantAllocation.TREATMENT, prompt_version=treatment_pv, traffic_percentage=traffic_split / 100),
            ],
        )

    exp = st.session_state["experiment"]
    exp.variants[0].traffic_percentage = 1 - traffic_split / 100
    exp.variants[1].traffic_percentage = traffic_split / 100

    user_id = st.text_input("User ID", value="user_12345")
    if st.button("Assign Variant", type="primary"):
        assignment = assign_variant(user_id, exp)
        st.write(f"**User:** {assignment.user_id}")
        st.write(f"**Hash Value:** {assignment.hash_value}")
        st.write(f"**Variant:** {assignment.variant_id}")
        st.write(f"**Allocation:** {assignment.allocation.value}")
        st.write(f"**Reason:** {assignment.evaluation_reason}")

    if st.button("Simulate Distribution"):
        dist = compute_traffic_distribution(exp, sample_size)
        st.write("**Actual Distribution:**")
        for vid, count in dist.items():
            pct = count / sample_size * 100
            st.write(f"  {vid}: {count} ({pct:.1f}%)")

        fig = go.Figure(data=[go.Bar(
            x=list(dist.keys()),
            y=list(dist.values()),
            marker_color=["#4a9eff", "#2ecc71"],
        )])
        fig.update_layout(title=f"Traffic Distribution (n={sample_size})", xaxis_title="Variant", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Statistical Analysis")
    st.markdown("Simulate experiment outcomes and run t-test analysis.")

    col1, col2 = st.columns(2)
    control_mean = col1.slider("Control Mean Score", 0.0, 1.0, 0.75, 0.05)
    treatment_mean = col2.slider("Treatment Mean Score", 0.0, 1.0, 0.82, 0.05)

    col1, col2 = st.columns(2)
    control_n = col1.number_input("Control Sample Size", 50, 5000, 500, step=50)
    treatment_n = col2.number_input("Treatment Sample Size", 50, 5000, 500, step=50)

    if st.button("Run Analysis", type="primary"):
        random.seed(42)
        np.random.seed(42)
        control_outcomes = [
            ExperimentOutcome(experiment_id="exp", variant_id="control", user_id=f"u{i}",
                              success=random.random() < control_mean, score=np.random.normal(control_mean, 0.15))
            for i in range(control_n)
        ]
        treatment_outcomes = [
            ExperimentOutcome(experiment_id="exp", variant_id="treatment", user_id=f"u{i}",
                              success=random.random() < treatment_mean, score=np.random.normal(treatment_mean, 0.15))
            for i in range(treatment_n)
        ]

        result = compute_statistics("exp", control_outcomes, treatment_outcomes)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Control Mean", f"{result.control_mean:.4f}")
        col2.metric("Treatment Mean", f"{result.treatment_mean:.4f}")
        col3.metric("P-value", f"{result.p_value:.6f}")
        col4.metric("Winner", result.winner.upper())

        if result.is_significant:
            st.success(f"Statistically significant! Winner: {result.winner}")
        else:
            st.info("Not statistically significant. Need more data.")

        st.write(f"**Effect Size (Cohen's d):** {result.effect_size:.4f}")
        st.write(f"**Confidence Interval:** [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
        st.write(f"**T-statistic:** {result.t_statistic:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=[o.score for o in control_outcomes], name="Control", opacity=0.7, marker_color="#4a9eff"))
        fig.add_trace(go.Histogram(x=[o.score for o in treatment_outcomes], name="Treatment", opacity=0.7, marker_color="#2ecc71"))
        fig.update_layout(title="Score Distributions", barmode="overlay", xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Kill Switch Evaluator")
    st.markdown("Check if performance drops should trigger automatic kill switches.")

    col1, col2 = st.columns(2)
    threshold = col1.slider("Kill Switch Threshold", 0.01, 0.20, 0.05, 0.01)
    error_count = col2.slider("Error Count in Window", 0, 50, 3)

    if st.button("Evaluate Kill Switch", type="primary"):
        exp.kill_switch_threshold = threshold
        exp.error_count_window = 100

        control_outcomes = [
            ExperimentOutcome(experiment_id="exp", variant_id="control", user_id=f"u{i}",
                              success=True, score=0.85)
            for i in range(100)
        ]
        treatment_outcomes = [
            ExperimentOutcome(experiment_id="exp", variant_id="treatment", user_id=f"u{i}",
                              success=i >= error_count, score=0.85 if i >= error_count else 0.5)
            for i in range(100)
        ]

        result = evaluate_kill_switch(exp, control_outcomes, treatment_outcomes)

        if result.triggered:
            st.error(f"KILL SWITCH TRIGGERED! Variant killed: {result.variant_killed}")
            st.write(f"**Reason:** {result.reason}")
        else:
            st.success("Kill switch not triggered. Performance is acceptable.")

        st.write(f"**Performance Delta:** {result.performance_delta:+.4f}")
        st.write(f"**Threshold:** {result.threshold}")
