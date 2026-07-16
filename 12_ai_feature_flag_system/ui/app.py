"""Streamlit dashboard for the AI Feature Flag System."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import random
import streamlit as st
import plotly.graph_objects as go

from src.models import FeatureFlag, FlagStatus, QualityMetric, RolloutStrategy
from src.flag_sdk import FeatureFlagSDK, consistent_hash, assign_variant
from src.rollout import evaluate_rollout, apply_rollout_decision, check_error_spike

st.set_page_config(page_title="AI Feature Flags", page_icon=":checkered_flag:", layout="wide")

st.title("AI Feature Flag System")
st.markdown("Canary rollout with LLM-as-judge quality monitoring and auto-rollback.")

if "sdk" not in st.session_state:
    st.session_state["sdk"] = FeatureFlagSDK()
    st.session_state["flag"] = FeatureFlag(
        flag_id="new_llm_routing",
        name="New LLM Routing Algorithm",
        description="A/B test the new cost-optimized routing logic",
        rollout_strategy=RolloutStrategy.CANARY,
        rollout_percentage=0.05,
        target_percentage=1.0,
        incremental_step=0.05,
        canary_metrics=[
            QualityMetric(name="response_quality", baseline_score=0.85, canary_score=0.82, threshold=0.05, is_passing=True),
            QualityMetric(name="latency", baseline_score=0.90, canary_score=0.88, threshold=0.05, is_passing=True),
        ],
    )
    st.session_state["sdk"].register_flag(st.session_state["flag"])
    st.session_state["rollout_history"] = []

tab1, tab2, tab3 = st.tabs(["Flag Evaluator", "Rollout Control", "Quality Monitor"])

with tab1:
    st.subheader("Flag Evaluator")
    st.markdown("Check if a specific user gets the flag enabled.")

    flag = st.session_state["flag"]
    user_id = st.text_input("User ID", value="user_12345")

    if st.button("Evaluate Flag", type="primary"):
        result = st.session_state["sdk"].evaluate(flag.flag_id, user_id)
        col1, col2 = st.columns(2)
        col1.metric("Enabled", "YES" if result.enabled else "NO")
        col2.metric("Rollout %", f"{result.rollout_percentage:.0%}")
        st.write(f"**Hash Value:** {result.hash_value}")
        st.write(f"**Reason:** {result.evaluation_reason}")

    st.write("---")
    st.write("**Batch Evaluation (100 users):**")
    if st.button("Evaluate 100 Users"):
        enabled_count = 0
        for i in range(100):
            result = st.session_state["sdk"].evaluate(flag.flag_id, f"user_{i}")
            if result.enabled:
                enabled_count += 1
        st.metric("Users with Flag Enabled", f"{enabled_count}/100")
        st.write(f"Expected: ~{flag.rollout_percentage * 100:.0f}%")

with tab2:
    st.subheader("Rollout Control")
    flag = st.session_state["flag"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Rollout", f"{flag.rollout_percentage:.0%}")
    col2.metric("Target", f"{flag.target_percentage:.0%}")
    col3.metric("Status", flag.status.value.upper())

    st.write(f"**Strategy:** {flag.rollout_strategy.value}")
    st.write(f"**Increment Step:** {flag.incremental_step:.0%}")
    st.write(f"**Error Threshold:** {flag.error_threshold:.0%}")
    st.write(f"**Auto Rollback:** {'Enabled' if flag.auto_rollback else 'Disabled'}")

    col1, col2 = st.columns(2)
    error_count = col1.slider("Simulated Error Count", 0, 50, 0)

    if col2.button("Evaluate Rollout Decision", type="primary"):
        decision = evaluate_rollout(flag, flag.canary_metrics, error_count)
        st.write(f"**Action:** {decision.action.upper()}")
        st.write(f"**New Percentage:** {decision.new_percentage:.0%}")
        st.write(f"**Reason:** {decision.reason}")

        if decision.action == "advance":
            st.success(f"Advancing from {decision.current_percentage:.0%} to {decision.new_percentage:.0%}")
        elif decision.action == "rollback":
            st.error("ROLLBACK triggered!")
        elif decision.action == "hold":
            st.warning("Holding at current percentage.")
        else:
            st.info("Rollout complete!")

        st.session_state["rollout_history"].append({
            "Action": decision.action,
            "From": f"{decision.current_percentage:.0%}",
            "To": f"{decision.new_percentage:.0%}",
            "Reason": decision.reason[:80],
        })

        flag = apply_rollout_decision(flag, decision)

    if st.button("Advance Manually"):
        new_pct = min(flag.target_percentage, flag.rollout_percentage + flag.incremental_step)
        flag.rollout_percentage = new_pct
        st.success(f"Advanced to {new_pct:.0%}")
        st.session_state["rollout_history"].append({"Action": "manual_advance", "From": f"{flag.rollout_percentage - flag.incremental_step:.0%}", "To": f"{new_pct:.0%}", "Reason": "Manual override"})

    if st.button("Rollback Now"):
        flag.rollout_percentage = 0.0
        flag.status = FlagStatus.ROLLED_BACK
        st.error("Rolled back to 0%!")
        st.session_state["rollout_history"].append({"Action": "manual_rollback", "From": f"{flag.rollout_percentage:.0%}", "To": "0%", "Reason": "Manual rollback"})

    if st.session_state["rollout_history"]:
        st.write("**Rollout History:**")
        st.dataframe(st.session_state["rollout_history"], use_container_width=True)

with tab3:
    st.subheader("Canary Quality Metrics")
    flag = st.session_state["flag"]

    for metric in flag.canary_metrics:
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{metric.name}**")
        col2.write("PASSING" if metric.is_passing else "FAILING")

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline", f"{metric.baseline_score:.2f}")
        col2.metric("Canary", f"{metric.canary_score:.2f}")
        col3.metric("Delta", f"{metric.canary_score - metric.baseline_score:+.2f}")

        if metric.is_passing:
            st.success(f"{metric.name} is passing the threshold.")
        else:
            st.error(f"{metric.name} is below threshold!")

    fig = go.Figure()
    for metric in flag.canary_metrics:
        fig.add_trace(go.Bar(name=f"{metric.name} (base)", x=[metric.name], y=[metric.baseline_score], marker_color="#4a9eff"))
        fig.add_trace(go.Bar(name=f"{metric.name} (canary)", x=[metric.name], y=[metric.canary_score], marker_color="#2ecc71"))
    fig.update_layout(title="Baseline vs Canary Quality", barmode="group", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
