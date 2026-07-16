"""Streamlit dashboard for the Model Regression Detection System."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from src.models import EmailCategory, ExpectedDifficulty, PromptConfig
from src.regressor import classify_severity, compute_accuracy, compute_per_difficulty, load_ground_truth

st.set_page_config(page_title="Model Regression System", page_icon=":bar_chart:", layout="wide")

st.title("Model Regression Detection System")
st.markdown("CI/CD-style prompt regression testing pipeline for LLM classification tasks.")

tab1, tab2, tab3 = st.tabs(["Prompt Config", "Ground Truth Data", "Regression Simulator"])

with tab1:
    st.subheader("Prompt Configuration")
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "classifier_v1.yaml"
    if prompt_path.exists():
        import yaml
        raw = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        st.code(raw.get("system_prompt", ""), language="text")
        st.write(f"**Model:** {raw.get('model', '')}")
        st.write(f"**Version:** {raw.get('version', '')}")
        st.write(f"**Temperature:** {raw.get('temperature', 0)}")
        st.write(f"**Categories:** {', '.join(raw.get('categories', []))}")
    else:
        st.warning("Prompt file not found. Showing default config.")
        st.json({"prompt_id": "classifier_v1", "version": "1.0.0", "model": "gpt-4o-mini"})

with tab2:
    st.subheader("Ground Truth Test Items")
    gt_path = Path(__file__).resolve().parent.parent / "tests" / "ground_truth.json"
    if gt_path.exists():
        items = load_ground_truth(gt_path)
        data = []
        for item in items:
            data.append({
                "ID": item.id,
                "Subject": item.subject[:60],
                "Expected": item.expected_category.value,
                "Difficulty": item.expected_difficulty.value,
                "Tags": ", ".join(item.tags[:3]),
            })
        st.dataframe(data, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        cats = [i.expected_category.value for i in items]
        diffs = [i.expected_difficulty.value for i in items]
        col1.metric("Total Items", len(items))
        col2.metric("Categories", len(set(cats)))
        col3.metric("Difficulties", len(set(diffs)))

        fig = go.Figure(data=[go.Bar(
            x=list(set(cats)),
            y=[cats.count(c) for c in set(cats)],
            marker_color=["#4a9eff", "#2ecc71", "#f5a623", "#e74c3c"],
        )])
        fig.update_layout(title="Category Distribution", xaxis_title="Category", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(data=[go.Bar(
            x=list(set(diffs)),
            y=[diffs.count(d) for d in set(diffs)],
            marker_color=["#2ecc71", "#f5a623", "#e74c3c"],
        )])
        fig2.update_layout(title="Difficulty Distribution", xaxis_title="Difficulty", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Ground truth file not found.")

with tab3:
    st.subheader("Regression Simulator")
    st.markdown("Simulate model predictions and see regression detection in action.")

    gt_path = Path(__file__).resolve().parent.parent / "tests" / "ground_truth.json"
    if gt_path.exists():
        items = load_ground_truth(gt_path)

        col1, col2 = st.columns(2)
        accuracy_target = col1.slider("Simulated Accuracy (%)", 0, 100, 85)
        baseline_acc = col2.slider("Baseline Accuracy (%)", 0, 100, 90)

        if st.button("Run Simulation", type="primary"):
            import random
            random.seed(42)
            cats = [EmailCategory.BILLING, EmailCategory.TECHNICAL, EmailCategory.ACCOUNT, EmailCategory.GENERAL]
            predictions = []
            for item in items:
                if random.randint(1, 100) <= accuracy_target:
                    pred_cat = item.expected_category
                else:
                    pred_cat = random.choice([c for c in cats if c != item.expected_category])
                from src.models import ModelPrediction
                predictions.append(ModelPrediction(item_id=item.id, predicted_category=pred_cat))

            correct, total = compute_accuracy(items, predictions)
            actual_acc = correct / total
            delta = actual_acc - (baseline_acc / 100)
            severity = classify_severity(delta)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{actual_acc:.1%}")
            col2.metric("Baseline", f"{baseline_acc / 100:.1%}")
            col3.metric("Delta", f"{delta:+.1%}")
            sev_color = {"none": "off", "warning": "on", "critical": "on"}
            col4.metric("Severity", severity.value.upper())

            if severity.value == "critical":
                st.error("CRITICAL: Merge would be blocked. Regression exceeds 8% threshold.")
            elif severity.value == "warning":
                st.warning("WARNING: Regression exceeds 3% threshold. Review before merge.")
            else:
                st.success("No regression detected. Safe to merge.")

            per_diff = compute_per_difficulty(items, predictions)
            diff_data = []
            for diff, stats in per_diff.items():
                diff_data.append({"Difficulty": diff, "Accuracy": f"{stats['accuracy']:.1%}", "Correct": f"{int(stats['correct'])}/{int(stats['total'])}"})
            st.dataframe(diff_data, use_container_width=True)

            results = []
            for item, pred in zip(items, predictions, strict=True):
                is_correct = pred.predicted_category == item.expected_category
                results.append({
                    "ID": item.id,
                    "Expected": item.expected_category.value,
                    "Predicted": pred.predicted_category.value if pred.predicted_category else "NONE",
                    "Correct": "Yes" if is_correct else "No",
                })
            st.dataframe(results, use_container_width=True)
