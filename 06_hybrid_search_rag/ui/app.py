"""Streamlit dashboard for the Hybrid Search RAG Pipeline."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from src.models import Document, DocumentSource, FusedResult, RetrievalResult
from src.hybrid_retriever import reciprocal_rank_fusion, BM25Retriever
from src.reranker import CrossEncoderReranker
from src.citation_verifier import extract_citations, verify_citations
from src.ingestion import chunk_text, hash_content

st.set_page_config(page_title="Hybrid Search RAG", page_icon=":books:", layout="wide")

st.title("Hybrid Search RAG Pipeline")
st.markdown("BM25 + Vector Fusion RAG with RRF, cross-encoder reranking, and citation verification.")

tab1, tab2, tab3 = st.tabs(["Document Ingestion", "RRF Fusion", "Citation Verifier"])

with tab1:
    st.subheader("Document Chunking")
    st.markdown("Paste text to see how it gets chunked for RAG indexing.")

    text = st.text_area("Document Text", value="This is a sample document about machine learning. " * 10, height=150)
    chunk_size = st.slider("Chunk Size", 100, 1000, 512)
    overlap = st.slider("Chunk Overlap", 0, 200, 64)

    if st.button("Chunk Document", type="primary"):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        st.write(f"**Produced {len(chunks)} chunks:**")
        for i, chunk in enumerate(chunks[:10]):
            with st.expander(f"Chunk {i} ({len(chunk)} chars)"):
                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        if len(chunks) > 10:
            st.info(f"...and {len(chunks) - 10} more chunks")

        hashes = [hash_content(c) for c in chunks]
        st.write(f"**Content Hashes:** {len(set(hashes))} unique out of {len(hashes)} total")

with tab2:
    st.subheader("Reciprocal Rank Fusion")
    st.markdown("See how BM25 and vector search results get fused via RRF.")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**BM25 Results**")
        bm25_docs = st.text_area("BM25 doc IDs (one per line)", value="doc_a\ndoc_b\ndoc_c\ndoc_d", height=100, key="bm25")

    with col2:
        st.write("**Vector Results**")
        vec_docs = st.text_area("Vector doc IDs (one per line)", value="doc_b\ndoc_e\ndoc_a\ndoc_f", height=100, key="vec")

    rrf_k = st.slider("RRF k parameter", 1, 100, 60)

    if st.button("Fuse Results", type="primary"):
        bm25_ids = [d.strip() for d in bm25_docs.split("\n") if d.strip()]
        vec_ids = [d.strip() for d in vec_docs.split("\n") if d.strip()]

        bm25_results = [
            RetrievalResult(doc_id=did, content=f"content of {did}", source_file=f"{did}.txt",
                            score=10 - i, rank=i + 1, retrieval_method="bm25")
            for i, did in enumerate(bm25_ids)
        ]
        vec_results = [
            RetrievalResult(doc_id=did, content=f"content of {did}", source_file=f"{did}.txt",
                            score=0.9 - i * 0.1, rank=i + 1, retrieval_method="vector")
            for i, did in enumerate(vec_ids)
        ]

        fused = reciprocal_rank_fusion(bm25_results, vec_results, top_k=20, k=rrf_k)

        st.write("**Fused Results:**")
        data = []
        for f in fused:
            data.append({
                "Doc ID": f.doc_id,
                "RRF Score": f"{f.rrf_score:.6f}",
                "BM25 Rank": f.bm25_rank or "-",
                "Vector Rank": f.vector_rank or "-",
                "Final Rank": f.final_rank,
            })
        st.dataframe(data, use_container_width=True)

        fig = go.Figure(data=[go.Bar(
            x=[f.doc_id for f in fused],
            y=[f.rrf_score for f in fused],
            marker_color=["#2ecc71" if f.bm25_rank and f.vector_rank else "#4a9eff" for f in fused],
        )])
        fig.update_layout(title="RRF Scores (green = found by both methods)", xaxis_title="Doc ID", yaxis_title="RRF Score")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Citation Verification")
    st.markdown("Check if bracketed citations in generated text reference real documents.")

    gen_text = st.text_area("Generated Text with Citations", value="According to [doc_a], the API is stable. See [doc_b] for details. This claim comes from [fake_doc].", height=100)

    doc_ids = st.text_input("Available Document IDs (comma-separated)", value="doc_a,doc_b,doc_c")

    if st.button("Verify Citations", type="primary"):
        from src.models import RerankedResult
        available_ids = [d.strip() for d in doc_ids.split(",")]
        docs = [
            RerankedResult(doc_id=did, content=f"content of {did}", source_file=f"{did}.txt",
                           cross_encoder_score=0.9, rrf_rank=1, final_rank=1)
            for did in available_ids
        ]

        result = verify_citations(gen_text, docs)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Citations", result.total_citations)
        col2.metric("Verified", result.verified_citations)
        col3.metric("Unverified", result.unverified_citations)
        col4.metric("Hallucinated", result.hallucinated_citations)

        if result.verification_score == 1.0:
            st.success("All citations verified!")
        elif result.verification_score > 0.5:
            st.warning("Some citations could not be verified.")
        else:
            st.error("Multiple hallucinated citations detected!")

        st.write("**Details:**")
        for detail in result.details:
            if "Verified" in detail:
                st.success(detail)
            elif "Hallucinated" in detail:
                st.error(detail)
            else:
                st.warning(detail)
