"""Streamlit dashboard for the LLM API Gateway."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import time
import random
import streamlit as st
import plotly.graph_objects as go

from src.circuit_breaker import CircuitBreaker
from src.models import CircuitState, VendorConfig, GatewayConfig
from src.token_bucket import TokenBucketRateLimiter

st.set_page_config(page_title="LLM API Gateway", page_icon=":rocket:", layout="wide")

st.title("LLM API Gateway")
st.markdown("Rate limiting, fallback routing, and circuit breaker observability.")

tab1, tab2, tab3 = st.tabs(["Circuit Breakers", "Rate Limiter", "Vendor Routing"])

with tab1:
    st.subheader("Circuit Breaker Simulator")
    st.markdown("Simulate circuit breaker state transitions under failure conditions.")

    if "cb" not in st.session_state:
        st.session_state["cb"] = CircuitBreaker("openai-primary", failure_threshold=5, recovery_timeout_s=10)
        st.session_state["cb_events"] = []

    cb = st.session_state["cb"]

    col1, col2, col3 = st.columns(3)

    if col1.button("Record Success", type="primary"):
        cb.record_success("openai-primary")
        state = cb.get_state("openai-primary")
        st.session_state["cb_events"].append({"Event": "Success", "State": state.state.value, "Failures": state.failure_count, "Time": time.strftime("%H:%M:%S")})

    if col2.button("Record Failure"):
        cb.record_failure("openai-primary")
        state = cb.get_state("openai-primary")
        st.session_state["cb_events"].append({"Event": "Failure", "State": state.state.value, "Failures": state.failure_count, "Time": time.strftime("%H:%M:%S")})

    if col3.button("Can Request?"):
        can = cb.can_request("openai-primary")
        state = cb.get_state("openai-primary")
        st.session_state["cb_events"].append({"Event": "Check", "Can Request": str(can), "State": state.state.value, "Time": time.strftime("%H:%M:%S")})

    state = cb.get_state("openai-primary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("State", state.state.value.upper())
    col2.metric("Failures", state.failure_count)
    col3.metric("Threshold", state.failure_threshold)
    col4.metric("Successes", state.consecutive_successes)

    if state.state == CircuitState.CLOSED:
        st.success("Circuit is CLOSED: requests flow normally.")
    elif state.state == CircuitState.HALF_OPEN:
        st.warning("Circuit is HALF-OPEN: testing if vendor recovered.")
    else:
        st.error("Circuit is OPEN: all requests to this vendor are blocked.")

    if st.session_state["cb_events"]:
        st.write("**Event Log:**")
        st.dataframe(st.session_state["cb_events"][-20:], use_container_width=True)

with tab2:
    st.subheader("Token Bucket Rate Limiter")
    st.markdown("Simulate Redis token bucket rate limiting.")

    if "limiter" not in st.session_state:
        st.session_state["limiter"] = TokenBucketRateLimiter()
        st.session_state["rate_log"] = []

    capacity = st.slider("Bucket Capacity", 1, 100, 10)
    refill_rate = st.slider("Refill Rate (tokens/sec)", 1, 50, 5)

    if st.button("Send Request", type="primary"):
        result = st.session_state["limiter"].check_rate_limit(
            "openai-primary", capacity=capacity, refill_rate=refill_rate, requested=1
        )
        st.session_state["rate_log"].append({
            "Allowed": str(result.allowed),
            "Remaining": f"{result.remaining_tokens:.2f}",
            "Retry After (ms)": f"{result.retry_after_ms:.1f}",
            "Time": time.strftime("%H:%M:%S"),
        })

        if result.allowed:
            st.success("Request allowed!")
        else:
            st.error(f"Rate limited! Retry after {result.retry_after_ms:.0f}ms")

    if st.button("Send 10 Rapid Requests"):
        for _ in range(10):
            result = st.session_state["limiter"].check_rate_limit(
                "openai-primary", capacity=capacity, refill_rate=refill_rate, requested=1
            )
            st.session_state["rate_log"].append({
                "Allowed": str(result.allowed),
                "Remaining": f"{result.remaining_tokens:.2f}",
                "Retry After (ms)": f"{result.retry_after_ms:.1f}",
                "Time": time.strftime("%H:%M:%S"),
            })

    if st.session_state["rate_log"]:
        st.write("**Rate Limit Log:**")
        st.dataframe(st.session_state["rate_log"][-20:], use_container_width=True)

        allowed = [1 if r["Allowed"] == "True" else 0 for r in st.session_state["rate_log"][-50:]]
        fig = go.Figure(data=[go.Bar(
            x=list(range(len(allowed))),
            y=allowed,
            marker_color=["#2ecc71" if a else "#e74c3c" for a in allowed],
        )])
        fig.update_layout(title="Request Results (green=allowed, red=rate limited)", yaxis=dict(tickvals=[0, 1], ticktext=["Blocked", "Allowed"]))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Vendor Routing Configuration")

    vendors = [
        {"id": "openai-primary", "name": "OpenAI Primary", "priority": 0, "rps": 50, "burst": 100},
        {"id": "anthropic-fallback", "name": "Anthropic Fallback", "priority": 1, "rps": 30, "burst": 60},
        {"id": "ollama-local", "name": "Ollama Local", "priority": 2, "rps": 100, "burst": 200},
    ]

    data = []
    for v in vendors:
        data.append({
            "Vendor": v["name"],
            "ID": v["id"],
            "Priority": v["priority"],
            "Rate (RPS)": v["rps"],
            "Burst": v["burst"],
        })
    st.dataframe(data, use_container_width=True)

    st.write("**Fallback Chain:**")
    chain = [v["name"] for v in sorted(vendors, key=lambda x: x["priority"])]
    for i, name in enumerate(chain):
        prefix = "  -> " if i > 0 else ""
        st.write(f"  {i+1}. {prefix}{name}")

    st.write("**Simulated Request Routing:**")
    if st.button("Simulate 20 Requests"):
        routes = []
        for i in range(20):
            vendor_idx = i % 3 if random.random() > 0.1 else 1
            routes.append({"Request": i + 1, "Vendor": vendors[vendor_idx]["name"], "Status": random.choice(["200 OK", "200 OK", "200 OK", "429 Rate Limited"])})
        st.dataframe(routes, use_container_width=True)
