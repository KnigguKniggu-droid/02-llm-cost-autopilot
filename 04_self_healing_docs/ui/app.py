"""Streamlit dashboard for the Self-Healing Documentation system."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from src.ast_parser import parse_file
from src.doc_linker import parse_markdown_file, cosine_similarity
from src.diff_checker import parse_git_diff

st.set_page_config(page_title="Self-Healing Docs", page_icon=":wrench:", layout="wide")

st.title("Self-Healing Technical Documentation")
st.markdown("AST code-to-documentation linker with git diff staleness detection.")

tab1, tab2, tab3 = st.tabs(["AST Parser", "Markdown Linker", "Git Diff Checker"])

with tab1:
    st.subheader("Python AST Parser")
    st.markdown("Paste Python code to see extracted semantic tokens.")

    code = st.text_area("Python Code", value='''def hello(name: str) -> str:
    """Greet a person."""
    return f"Hello, {name}"

class Calculator:
    """A simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
''', height=200)

    if st.button("Parse AST", type="primary"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            f.flush()
            tokens = parse_file(Path(f.name))

        st.write(f"**Extracted {len(tokens)} semantic tokens:**")
        data = []
        for t in tokens:
            data.append({
                "Type": t.element_type.value,
                "Name": t.name,
                "Lines": f"{t.line_start}-{t.line_end}",
                "Docstring": t.docstring[:60] + "..." if len(t.docstring) > 60 else t.docstring,
                "Hash": t.source_hash[:16],
            })
        st.dataframe(data, use_container_width=True)

        type_counts = {}
        for t in tokens:
            type_counts[t.element_type.value] = type_counts.get(t.element_type.value, 0) + 1
        fig = go.Figure(data=[go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.4,
        )])
        fig.update_layout(title="Token Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Cosine Similarity Linker")
    st.markdown("Test cosine similarity between code and documentation embeddings.")

    col1, col2 = st.columns(2)
    vec_a = col1.text_input("Vector A (comma-separated)", value="1.0, 0.5, 0.3, 0.8")
    vec_b = col2.text_input("Vector B (comma-separated)", value="0.9, 0.4, 0.2, 0.7")

    if st.button("Compute Similarity"):
        try:
            a = [float(x.strip()) for x in vec_a.split(",")]
            b = [float(x.strip()) for x in vec_b.split(",")]
            sim = cosine_similarity(a, b)
            st.metric("Cosine Similarity", f"{sim:.4f}")
            if sim > 0.95:
                st.warning("Similarity > 0.95: potential duplicate detected!")
            elif sim > 0.5:
                st.success("Moderate to high similarity: likely a documentation link.")
            else:
                st.info("Low similarity: no strong link.")
        except Exception as exc:
            st.error(f"Parse error: {exc}")

with tab3:
    st.subheader("Git Diff Parser")
    st.markdown("Paste a git diff to see parsed file changes.")

    diff_text = st.text_area("Git Diff", value='''diff --git a/src/foo.py b/src/foo.py
index 1234567..abcdefg 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,3 +10,4 @@
 def foo():
-    return 1
+    return 2
+    # new line
''', height=150)

    if st.button("Parse Diff", type="primary"):
        entries = parse_git_diff(diff_text)
        st.write(f"**Found {len(entries)} changed file(s):**")
        for entry in entries:
            with st.expander(f"{entry.file_path} ({entry.change_type})"):
                st.write(f"Removed lines: {entry.removed_lines}")
                st.write(f"Added lines: {entry.added_lines}")
                st.code(entry.diff_content, language="diff")
