"""Streamlit dashboard for the LLM Output Arbitration System."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import streamlit as st
import plotly.graph_objects as go

from src.models import CriticOutput, CriticType, AdjudicationInput
from src.adjudicator import adjudicate, compute_weighted_score, compute_consensus_level, determine_verdict

st.set_page_config(page_title="Output Arbitration", page_icon=":scales:", layout="wide")

st.title("LLM Output Arbitration System")
st.markdown("Parallel multi-critic judge with central adjudicator scoring 1 to 10.")

tab1, tab2 = st.tabs(["Arbitration Simulator", "Critic Analysis"])

with tab1:
    st.subheader("Simulate Arbitration")
    st.markdown("Set critic scores and see the adjudicator's final verdict.")

    col1, col2, col3 = st.columns(3)

    fa_score = col1.slider("Factual Accuracy (GPT-4o)", 0.0, 10.0, 8.5, 0.1)
    fa_conf = col1.slider("FA Confidence", 0.0, 1.0, 0.9, 0.05)

    lc_score = col2.slider("Logical Consistency (Claude)", 0.0, 10.0, 7.5, 0.1)
    lc_conf = col2.slider("LC Confidence", 0.0, 1.0, 0.85, 0.05)

    cp_score = col3.slider("Completeness (Llama)", 0.0, 10.0, 6.0, 0.1)
    cp_conf = col3.slider("CP Confidence", 0.0, 1.0, 0.75, 0.05)

    if st.button("Run Adjudicator", type="primary"):
        outputs = [
            CriticOutput(critic_type=CriticType.FACTUAL_ACCURACY, score=fa_score, confidence=fa_conf, evidence=["fact1"], critique="Mostly accurate", model_used="gpt-4o"),
            CriticOutput(critic_type=CriticType.LOGICAL_CONSISTENCY, score=lc_score, confidence=lc_conf, evidence=["logic1"], critique="Consistent reasoning", model_used="claude-3-5-sonnet"),
            CriticOutput(critic_type=CriticType.COMPLETENESS, score=cp_score, confidence=cp_conf, evidence=["complete1"], critique="Missing some details", model_used="llama3.1-8b"),
        ]

        result = adjudicate(AdjudicationInput(query="test", response="test", critic_outputs=outputs))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Score", f"{result.final_score}/10")
        col2.metric("Verdict", result.verdict.upper())
        col3.metric("Consensus", f"{result.consensus_level:.0%}")
        col4.metric("Critics", len(result.critic_scores))

        if result.verdict == "accept":
            st.success(f"Verdict: ACCEPT (score {result.final_score}/10)")
        elif result.verdict == "revise":
            st.warning(f"Verdict: REVISE (score {result.final_score}/10)")
        else:
            st.error(f"Verdict: REJECT (score {result.final_score}/10)")

        st.write("**Reasoning:**")
        st.write(result.reasoning)

        st.write("**Critic Scores:**")
        scores_data = []
        for critic, score in result.critic_scores.items():
            deviation = result.score_deviations.get(critic, 0)
            scores_data.append({"Critic": critic, "Score": score, "Deviation": f"{deviation:+.1f}"})
        st.dataframe(scores_data, use_container_width=True)

        fig = go.Figure(data=[go.Bar(
            x=list(result.critic_scores.keys()),
            y=list(result.critic_scores.values()),
            marker_color=["#4a9eff", "#2ecc71", "#f5a623"],
        )])
        fig.add_hline(y=result.final_score, line_dash="dash", line_color="red", annotation_text=f"Final: {result.final_score}")
        fig.update_layout(title="Critic Scores vs Final Adjudication", yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Critic Weight Analysis")
    st.markdown("Understand how critic weights affect the final score.")

    weights = {
        CriticType.FACTUAL_ACCURACY: 0.40,
        CriticType.LOGICAL_CONSISTENCY: 0.35,
        CriticType.COMPLETENESS: 0.25,
    }

    st.write("**Default Weights:**")
    weight_data = [{"Critic": ct.value, "Weight": f"{w:.0%}"} for ct, w in weights.items()]
    st.dataframe(weight_data, use_container_width=True)

    fig = go.Figure(data=[go.Pie(
        labels=[ct.value for ct in weights],
        values=[w for w in weights.values()],
        hole=0.4,
    )])
    fig.update_layout(title="Critic Weight Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Consensus Level Simulator:**")
    scores = []
    for i in range(3):
        scores.append(st.slider(f"Critic {i+1} Score", 0.0, 10.0, 7.0 + i, 0.1, key=f"consensus_{i}"))

    test_outputs = [
        CriticOutput(critic_type=list(CriticType)[i], score=scores[i], confidence=0.9, evidence=[], critique="", model_used="test")
        for i in range(3)
    ]
    consensus = compute_consensus_level(test_outputs)
    weighted = compute_weighted_score(test_outputs)
    verdict = determine_verdict(round(weighted))

    col1, col2, col3 = st.columns(3)
    col1.metric("Weighted Score", f"{weighted:.2f}")
    col2.metric("Consensus", f"{consensus:.0%}")
    col3.metric("Verdict", verdict.upper())
