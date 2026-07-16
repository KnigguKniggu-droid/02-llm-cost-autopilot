"""Streamlit dashboard for the LLM Cost Autopilot."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import ComplexityTier, ProxyRequest
from src.proxy_router import classify_complexity, load_model_registry, load_routing_rules, estimate_cost, select_model

st.set_page_config(page_title="LLM Cost Autopilot", page_icon=":moneybag:", layout="wide")

st.title("LLM Cost Autopilot")
st.markdown("Intelligent multi-provider routing proxy with cost optimization.")

try:
    registry = load_model_registry()
    rules = load_routing_rules()
except Exception:
    registry = {}
    rules = {}

tab1, tab2, tab3 = st.tabs(["Model Registry", "Complexity Classifier", "Cost Estimator"])

with tab1:
    st.subheader("Registered Models")
    if registry:
        data = []
        for model_id, cfg in registry.items():
            data.append({
                "Model": cfg.display_name,
                "Provider": cfg.provider.value,
                "Input $/1K": cfg.input_cost_per_1k,
                "Output $/1K": cfg.output_cost_per_1k,
                "Quality": f"{cfg.quality_score:.2f}",
                "Tiers": ", ".join(str(t.value) for t in cfg.supported_tiers),
                "Efficiency": f"{cfg.cost_efficiency:.2f}",
            })
        st.dataframe(data, use_container_width=True)

        models = list(registry.keys())
        input_costs = [registry[m].input_cost_per_1k for m in models]
        output_costs = [registry[m].output_cost_per_1k for m in models]
        quality = [registry[m].quality_score for m in models]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cost per 1K Tokens", "Quality Score"))
        fig.add_trace(go.Bar(name="Input", x=models, y=input_costs, marker_color="#4a9eff"), row=1, col=1)
        fig.add_trace(go.Bar(name="Output", x=models, y=output_costs, marker_color="#2ecc71"), row=1, col=1)
        fig.add_trace(go.Bar(name="Quality", x=models, y=quality, marker_color="#f5a623"), row=1, col=2)
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model registry not loaded. Check config/models.yaml exists.")

with tab2:
    st.subheader("Complexity Classifier")
    st.markdown("Paste a request to see which complexity tier it gets routed to.")

    user_msg = st.text_area("User Message", value="Explain how neural networks work step by step with code examples.", height=100)
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    if st.button("Classify", type="primary"):
        request = ProxyRequest(
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        result = classify_complexity(request)

        tier_names = {
            ComplexityTier.TIER_1_EXTRACTION: "Tier 1: Extraction",
            ComplexityTier.TIER_2_CLASSIFICATION: "Tier 2: Classification",
            ComplexityTier.TIER_3_MULTI_STEP_LOGIC: "Tier 3: Multi-Step Logic",
        }

        col1, col2 = st.columns(2)
        col1.metric("Tier", tier_names.get(result.tier, "Unknown"))
        col2.metric("Confidence", f"{result.confidence:.0%}")

        st.write("**Detected Signals:**")
        signal_data = []
        for signal, value in result.signals.items():
            signal_data.append({"Signal": signal, "Value": str(value)})
        st.dataframe(signal_data, use_container_width=True)

        if registry and rules:
            from src.models import ComplexityClassification
            model, reason = select_model(result, registry, rules)
            st.success(f"**Routed to:** {model.display_name} ({model.model_id})")
            st.write(f"**Reason:** {reason}")
            cost = estimate_cost(model, result)
            st.info(f"**Estimated Cost:** ${cost:.6f}")

with tab3:
    st.subheader("Cost Estimator")
    st.markdown("Compare estimated costs across all models for a given request.")

    est_tokens = st.number_input("Estimated Input Tokens", 10, 100000, 500)
    est_output = st.number_input("Estimated Output Tokens", 10, 100000, 1000)

    if registry and st.button("Compare Costs"):
        from src.models import ComplexityClassification
        costs = []
        for model_id, cfg in registry.items():
            cls = ComplexityClassification(
                tier=ComplexityTier.TIER_3_MULTI_STEP_LOGIC,
                confidence=0.9,
                signals={},
                estimated_input_tokens=est_tokens,
                estimated_output_tokens=est_output,
            )
            cost = estimate_cost(cfg, cls)
            costs.append({"Model": cfg.display_name, "Provider": cfg.provider.value, "Cost (USD)": round(cost, 6)})

        costs.sort(key=lambda x: x["Cost (USD)"])
        st.dataframe(costs, use_container_width=True)

        fig = go.Figure(data=[go.Bar(
            x=[c["Model"] for c in costs],
            y=[c["Cost (USD)"] for c in costs],
            marker_color=["#2ecc71" if i == 0 else "#4a9eff" for i in range(len(costs))],
        )])
        fig.update_layout(title="Cost Comparison", xaxis_title="Model", yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)
