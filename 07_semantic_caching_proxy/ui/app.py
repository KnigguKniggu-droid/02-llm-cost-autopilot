"""Streamlit dashboard for the Semantic Caching Proxy."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import streamlit as st
import plotly.graph_objects as go

from src.cache import SemanticCache, compute_cache_key, cosine_similarity
from src.models import CacheEntry, CacheStatus

st.set_page_config(page_title="Semantic Caching", page_icon=":zap:", layout="wide")

st.title("Semantic Caching Layer for LLM APIs")
st.markdown("Redis VL-backed semantic cache with 0.95 similarity boundary.")

tab1, tab2, tab3 = st.tabs(["Cache Key Generator", "Similarity Playground", "Cache Simulator"])

with tab1:
    st.subheader("Cache Key Generator")
    st.markdown("See how unique cache keys are computed from request metadata.")

    col1, col2 = st.columns(2)
    model = col1.text_input("Model", value="gpt-4o")
    temperature = col2.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    system_prompt = st.text_area("System Prompt", value="You are a helpful assistant.", height=80)

    if st.button("Generate Cache Key", type="primary"):
        key = compute_cache_key(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": "test"}],
        )

        st.code(f"System Hash:    {key.system_hash}", language="text")
        st.code(f"Temp Hash:      {key.temperature_hash}", language="text")
        st.code(f"Model Hash:     {key.model_hash}", language="text")
        st.code(f"Combined Hash:  {key.combined_hash}", language="text")
        st.code(f"Redis Key:      {key.to_redis_key()}", language="text")

        st.write("**Verify determinism:**")
        key2 = compute_cache_key(model=model, temperature=temperature, system_prompt=system_prompt, messages=[{"role": "user", "content": "different"}])
        if key.combined_hash == key2.combined_hash:
            st.success("Keys are deterministic: same metadata produces same key regardless of user message.")
        else:
            st.error("Keys differ: something is wrong.")

with tab2:
    st.subheader("Cosine Similarity Playground")
    st.markdown("Test similarity scores to understand the 0.95 cache hit boundary.")

    col1, col2 = st.columns(2)
    vec_a = col1.text_input("Query Embedding", value="1.0, 0.8, 0.5, 0.3, 0.1")
    vec_b = col2.text_input("Cached Embedding", value="0.95, 0.75, 0.48, 0.28, 0.09")

    threshold = st.slider("Similarity Boundary", 0.5, 1.0, 0.95, 0.01)

    if st.button("Compare"):
        try:
            a = [float(x.strip()) for x in vec_a.split(",")]
            b = [float(x.strip()) for x in vec_b.split(",")]
            sim = cosine_similarity(a, b)

            st.metric("Cosine Similarity", f"{sim:.6f}")
            st.write(f"**Threshold:** {threshold}")

            if sim >= threshold:
                st.success(f"CACHE HIT: similarity {sim:.4f} >= threshold {threshold}")
            elif sim >= threshold * 0.9:
                st.warning(f"PARTIAL HIT: similarity {sim:.4f} is close to threshold")
            else:
                st.info(f"CACHE MISS: similarity {sim:.4f} < threshold {threshold}")

            fig = go.Figure(data=[go.Bar(
                x=["Similarity", "Threshold"],
                y=[sim, threshold],
                marker_color=["#2ecc71" if sim >= threshold else "#e74c3c", "#4a9eff"],
            )])
            fig.update_layout(title="Similarity vs Threshold", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Parse error: {exc}")

with tab3:
    st.subheader("Cache Simulator")
    st.markdown("Simulate cache lookups and see hit/miss behavior.")

    if "cache" not in st.session_state:
        st.session_state["cache"] = SemanticCache(similarity_boundary=0.95)
        st.session_state["cache_entries"] = []

    query = st.text_input("Query Text", value="What is machine learning?")
    query_emb = st.text_input("Query Embedding (comma-separated)", value="1.0, 0.5, 0.3")

    col1, col2 = st.columns(2)

    if col1.button("Store in Cache", type="primary"):
        try:
            emb = [float(x.strip()) for x in query_emb.split(",")]
            key = compute_cache_key("gpt-4o", 0.7, "system", [{"role": "user", "content": query}])
            entry = CacheEntry(
                cache_key=key.to_redis_key(),
                query=query,
                query_embedding=emb,
                response=f"Response to: {query}",
                model="gpt-4o",
                temperature=0.7,
            )
            st.session_state["cache"].store(entry)
            st.session_state["cache_entries"].append({"query": query, "response": entry.response, "key": key.combined_hash[:16]})
            st.success("Stored in cache!")
        except Exception as exc:
            st.error(f"Error: {exc}")

    if col2.button("Lookup in Cache"):
        try:
            emb = [float(x.strip()) for x in query_emb.split(",")]
            key = compute_cache_key("gpt-4o", 0.7, "system", [{"role": "user", "content": query}])
            result = st.session_state["cache"].lookup(emb, key)

            if result.status == CacheStatus.HIT:
                st.success(f"CACHE HIT! Similarity: {result.similarity_score:.4f}")
                st.write(f"**Cached Response:** {result.entry.response}")
            elif result.status == CacheStatus.PARTIAL_HIT:
                st.warning(f"PARTIAL HIT. Similarity: {result.similarity_score:.4f}")
            else:
                st.info(f"CACHE MISS. Best similarity: {result.similarity_score:.4f}")

            st.write(f"Lookup latency: {result.lookup_latency_ms:.2f}ms")
        except Exception as exc:
            st.error(f"Error: {exc}")

    stats = st.session_state["cache"].stats()
    st.write(f"**Cache Stats:** {stats}")

    if st.session_state["cache_entries"]:
        st.write("**Stored Entries:**")
        st.dataframe(st.session_state["cache_entries"], use_container_width=True)

    if st.button("Clear Cache"):
        st.session_state["cache"].clear()
        st.session_state["cache_entries"] = []
        st.success("Cache cleared!")
