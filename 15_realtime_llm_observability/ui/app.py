"""Streamlit dashboard for the Realtime LLM Observability system."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import asyncio
import time
import random

import streamlit as st
import plotly.graph_objects as go

from src.metrics import export_metrics, record_ttft, record_inter_token_latency, set_token_velocity, record_drift_anomaly, set_active_streams
from src.server import ObservabilityServer, LiveMetricEvent

st.set_page_config(page_title="Realtime LLM Observability", page_icon=":satellite_antenna:", layout="wide")

st.title("Realtime LLM Observability Dashboard")
st.markdown("Streaming P95 latency, token drift, and live metric tracking.")

if "server" not in st.session_state:
    st.session_state["server"] = ObservabilityServer()
    st.session_state["sim_data"] = []

server = st.session_state["server"]

tab1, tab2, tab3 = st.tabs(["Live Metrics", "Stream Simulator", "Prometheus Export"])

with tab1:
    st.subheader("Live Metric Snapshot")

    snapshot = server.get_snapshot()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Streams", snapshot["active_streams"])
    col2.metric("TTFT P50", f"{snapshot['ttft']['p50_ms']:.1f}ms")
    col3.metric("TTFT P95", f"{snapshot['ttft']['p95_ms']:.1f}ms")
    col4.metric("TTFT P99", f"{snapshot['ttft']['p99_ms']:.1f}ms")

    col1, col2, col3 = st.columns(3)
    col1.metric("ITL P50", f"{snapshot['itl']['p50_ms']:.1f}ms")
    col2.metric("ITL P95", f"{snapshot['itl']['p95_ms']:.1f}ms")
    col3.metric("ITL Samples", snapshot["itl"]["samples"])

    if snapshot["ttft"]["samples"] > 0:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="P50", x=["TTFT", "ITL"], y=[snapshot["ttft"]["p50_ms"], snapshot["itl"]["p50_ms"]], marker_color="#4a9eff"))
        fig.add_trace(go.Bar(name="P95", x=["TTFT", "ITL"], y=[snapshot["ttft"]["p95_ms"], snapshot["itl"]["p95_ms"]], marker_color="#f5a623"))
        fig.add_trace(go.Bar(name="P99", x=["TTFT", "ITL"], y=[snapshot["ttft"]["p99_ms"], snapshot["itl"]["p99_ms"]], marker_color="#e74c3c"))
        fig.update_layout(title="Latency Percentiles (ms)", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics yet. Run the Stream Simulator to generate data.")

with tab2:
    st.subheader("Streaming Request Simulator")
    st.markdown("Simulate LLM streaming requests and watch metrics update in real time.")

    col1, col2 = st.columns(2)
    model_name = col1.text_input("Model", value="gpt-4o")
    num_chunks = col2.slider("Number of Chunks", 5, 50, 15)
    base_ttft = st.slider("Base TTFT (ms)", 10, 2000, 150)
    chunk_delay = st.slider("Inter-chunk delay (ms)", 1, 100, 15)
    expected_tokens = st.slider("Expected Tokens", 10, 500, 100)

    if st.button("Simulate Stream", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        request_id = f"sim_{int(time.time())}"
        asyncio.run(server.start_stream(request_id, model_name))

        total_tokens = 0
        for i in range(num_chunks):
            chunk_tokens = random.randint(3, 10)
            total_tokens += chunk_tokens

            if i == 0:
                ttft = base_ttft + random.uniform(-20, 20)
                asyncio.run(server.record_first_chunk(request_id, model_name, "openai"))
                record_ttft(model=model_name, vendor="openai", ttft_seconds=ttft / 1000)
                status.write(f"First chunk: TTFT={ttft:.1f}ms")
            else:
                itl = chunk_delay + random.uniform(-3, 3)
                asyncio.run(server.record_chunk(request_id, model_name, i, chunk_tokens))
                record_inter_token_latency(model=model_name, itl_seconds=itl / 1000)

            time.sleep(0.02)
            progress.progress((i + 1) / num_chunks)

        asyncio.run(server.end_stream(request_id, model_name, total_tokens, expected_tokens))
        status.success(f"Stream complete: {total_tokens} tokens in {num_chunks} chunks")

        if expected_tokens > 0 and abs(total_tokens / expected_tokens - 1.0) > 0.2:
            direction = "over" if total_tokens > expected_tokens else "under"
            record_drift_anomaly(model=model_name, drift_direction=direction)
            st.warning(f"Drift anomaly: {total_tokens} actual vs {expected_tokens} expected ({direction})")

        st.rerun()

    if st.button("Simulate 5 Rapid Streams"):
        for s in range(5):
            request_id = f"rapid_{s}_{int(time.time())}"
            asyncio.run(server.start_stream(request_id, f"model_{s}"))
            asyncio.run(server.record_first_chunk(request_id, f"model_{s}", "openai"))
            for c in range(random.randint(5, 15)):
                asyncio.run(server.record_chunk(request_id, f"model_{s}", c, random.randint(2, 8)))
                time.sleep(0.01)
            asyncio.run(server.end_stream(request_id, f"model_{s}", random.randint(20, 80)))
        st.success("5 streams simulated!")
        st.rerun()

    if st.button("Clear All Metrics"):
        server = ObservabilityServer()
        st.session_state["server"] = server
        st.success("Metrics cleared!")
        st.rerun()

with tab3:
    st.subheader("Prometheus Metrics Export")
    st.markdown("Raw Prometheus text exposition format (served at /metrics in production).")

    metrics_text = export_metrics()
    st.code(metrics_text, language="text")

    st.write("**Metric Definitions:**")
    st.write("- `llm_time_to_first_token_seconds`: Histogram with ms-level buckets")
    st.write("- `llm_token_generation_velocity_per_second`: Gauge per model")
    st.write("- `llm_semantic_drift_anomalies_total`: Counter with drift_direction label")
    st.write("- `llm_inter_token_latency_seconds`: Histogram with ms-level buckets")
    st.write("- `llm_requests_total`: Counter per model and status")
    st.write("- `llm_errors_total`: Counter per model and error_type")
    st.write("- `llm_active_streams`: Gauge for current stream count")

    st.write("---")
    st.write("**Docker Deployment:**")
    st.code("""docker build -t realtime-observability .
docker run -p 8000:8000 -p 9090:9090 realtime-observability

# Grafana scrapes http://localhost:8000/metrics
# Dashboard SSE at http://localhost:8000/v1/observability/dashboard/events""", language="bash")
