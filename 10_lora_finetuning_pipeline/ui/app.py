"""Streamlit dashboard for the LoRA Fine-tuning Pipeline."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from src.models import TrainingConfig
from src.data_preprocessor import split_dataset, DatasetItem, check_leakage

st.set_page_config(page_title="LoRA Fine-tuning", page_icon=":brain:", layout="wide")

st.title("LoRA Fine-tuning Pipeline")
st.markdown("PEFT/QLoRA training harness with W&B logging and catastrophic forgetting checks.")

tab1, tab2, tab3 = st.tabs(["Training Config", "Dataset Splitter", "Forgetting Check"])

with tab1:
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)
    base_model = col1.text_input("Base Model", value="meta-llama/Llama-3.1-8B")
    lora_r = col2.number_input("LoRA Rank (r)", 1, 256, 16)
    lora_alpha = col1.number_input("LoRA Alpha", 1, 512, 32)
    lora_dropout = col2.slider("LoRA Dropout", 0.0, 0.5, 0.05, 0.01)
    learning_rate = col1.text_input("Learning Rate", value="0.0002")
    num_epochs = col2.number_input("Epochs", 1, 100, 3)
    use_qlora = col1.checkbox("Use QLoRA (4-bit quantization)", value=True)
    batch_size = col2.number_input("Batch Size", 1, 32, 4)

    st.write("**Target Modules (LoRA applied to):**")
    target_modules = st.multiselect(
        "Linear Attention Modules",
        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        default=["q_proj", "v_proj"],
    )

    if st.button("Build Config", type="primary"):
        config = TrainingConfig(
            base_model=base_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            learning_rate=float(learning_rate),
            num_train_epochs=num_epochs,
            use_qlora=use_qlora,
            per_device_train_batch_size=batch_size,
        )
        st.success("Configuration built successfully!")
        st.json(config.model_dump())

        st.write("**PEFT Config:**")
        st.json(config.to_peft_config())

with tab2:
    st.subheader("Dataset Splitter")
    st.markdown("Split data into 80/10/10 folds with leakage prevention.")

    num_items = st.slider("Number of Dataset Items", 10, 500, 100, step=10)

    if st.button("Split Dataset", type="primary"):
        items = []
        for i in range(num_items):
            items.append(DatasetItem(
                id=f"item_{i}",
                text=f"This is training example number {i} with unique content {i * 7919}",
                label="positive" if i % 2 == 0 else "negative",
            ))

        result = split_dataset(items)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train", len(result.train))
        col2.metric("Validation", len(result.validation))
        col3.metric("Test", len(result.test))
        col4.metric("Leakage Check", "PASSED" if result.leakage_check_passed else "FAILED")

        if result.leakage_check_passed:
            st.success("No data leakage detected across splits.")
        else:
            st.error("Data leakage detected! Review dataset for duplicates.")

        fig = go.Figure(data=[go.Pie(
            labels=["Train", "Validation", "Test"],
            values=[len(result.train), len(result.validation), len(result.test)],
            hole=0.4,
            marker_colors=["#4a9eff", "#2ecc71", "#f5a623"],
        )])
        fig.update_layout(title="Dataset Split (80/10/10)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Catastrophic Forgetting Check")
    st.markdown("Compare base model vs fine-tuned model on general benchmarks.")

    benchmarks = ["MMLU", "HellaSwag", "ARC Challenge", "TruthfulQA", "Winogrande"]

    col1, col2 = st.columns(2)
    base_scores = {}
    ft_scores = {}
    for bench in benchmarks:
        base_scores[bench] = col1.slider(f"{bench} - Base", 0.0, 1.0, 0.78 + np.random.random() * 0.05, 0.01, key=f"base_{bench}")
        ft_scores[bench] = col2.slider(f"{bench} - Fine-tuned", 0.0, 1.0, 0.75 + np.random.random() * 0.05, 0.01, key=f"ft_{bench}")

    threshold = st.slider("Forgetting Threshold", 0.01, 0.20, 0.05, 0.01)

    if st.button("Check Forgetting", type="primary"):
        from src.evaluation import check_catastrophic_forgetting

        results = []
        for bench in benchmarks:
            base_list = [base_scores[bench] + np.random.normal(0, 0.02) for _ in range(50)]
            ft_list = [ft_scores[bench] + np.random.normal(0, 0.02) for _ in range(50)]
            r = check_catastrophic_forgetting(bench, base_list, ft_list, threshold)
            results.append({
                "Benchmark": bench,
                "Base Score": f"{r.base_model_score:.4f}",
                "FT Score": f"{r.finetuned_model_score:.4f}",
                "Delta": f"{r.performance_delta:+.4f}",
                "Forgetting": "YES" if r.forgetting_detected else "NO",
            })

        st.dataframe(results, use_container_width=True)

        forgetting_count = sum(1 for r in results if r["Forgetting"] == "YES")
        if forgetting_count > 0:
            st.warning(f"Forgetting detected on {forgetting_count}/{len(results)} benchmarks.")
        else:
            st.success("No catastrophic forgetting detected on any benchmark.")

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Base Model", x=benchmarks, y=[base_scores[b] for b in benchmarks], marker_color="#4a9eff"))
        fig.add_trace(go.Bar(name="Fine-tuned", x=benchmarks, y=[ft_scores[b] for b in benchmarks], marker_color="#2ecc71"))
        fig.update_layout(title="Benchmark Comparison", barmode="group", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
