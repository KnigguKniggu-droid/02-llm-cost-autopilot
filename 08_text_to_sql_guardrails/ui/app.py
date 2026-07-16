"""Streamlit dashboard for the Text-to-SQL Guardrails system."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import streamlit as st

from src.guardrails import validate_sql, classify_statement, sanitize_sql
from src.models import SQLStatementType, GuardrailViolation

st.set_page_config(page_title="Text-to-SQL Guardrails", page_icon=":shield:", layout="wide")

st.title("Text-to-SQL Interface with Guardrails")
st.markdown("SQLAlchemy introspection + sqlparse security middleware + hallucination detection.")

tab1, tab2 = st.tabs(["SQL Validator", "Guardrail Tester"])

with tab1:
    st.subheader("SQL Query Validator")
    st.markdown("Test any SQL query against the security guardrails.")

    sql = st.text_area("SQL Query", value="SELECT * FROM users WHERE age > 18;", height=100)

    if st.button("Validate", type="primary"):
        result = validate_sql(sql)

        col1, col2, col3 = st.columns(3)
        col1.metric("Statement Type", result.statement_type.value.upper())
        col2.metric("Is Safe", "YES" if result.is_safe else "NO")
        col3.metric("Violations", len(result.violations))

        if result.is_safe:
            st.success("Query passed all guardrails.")
            st.code(result.sanitized_sql, language="sql")
        else:
            st.error(f"Query REJECTED: {result.rejected_reason}")
            st.write("**Violations:**")
            for v in result.violations:
                st.write(f"- {v.value}: {v.name}")

with tab2:
    st.subheader("Guardrail Test Suite")
    st.markdown("Test various SQL patterns to see which ones pass or fail.")

    test_cases = [
        ("SELECT * FROM orders", "Valid SELECT", True),
        ("INSERT INTO users VALUES (1, 'admin')", "DML INSERT", False),
        ("UPDATE users SET role='admin'", "DML UPDATE", False),
        ("DELETE FROM logs", "DML DELETE", False),
        ("DROP TABLE users", "DDL DROP", False),
        ("CREATE TABLE hack (id INT)", "DDL CREATE", False),
        ("ALTER TABLE users ADD COLUMN pw TEXT", "DDL ALTER", False),
        ("SELECT 1; DROP TABLE users", "Multi-statement injection", False),
        ("SELECT pg_sleep(100)", "Dangerous function", False),
        ("SELECT * FROM products WHERE price < 50", "Valid SELECT with WHERE", True),
    ]

    results = []
    for sql, description, expected_safe in test_cases:
        result = validate_sql(sql)
        status = "PASS" if result.is_safe == expected_safe else "FAIL"
        results.append({
            "Test": description,
            "SQL": sql[:50],
            "Expected": "Safe" if expected_safe else "Blocked",
            "Actual": "Safe" if result.is_safe else "Blocked",
            "Status": status,
        })

    st.dataframe(results, use_container_width=True)

    passed = sum(1 for r in results if r["Status"] == "PASS")
    st.metric("Tests Passed", f"{passed}/{len(results)}")

    st.write("**Statement Type Classification:**")
    type_data = []
    for sql, desc, _ in test_cases:
        stmt_type = classify_statement(sql)
        type_data.append({"Description": desc, "SQL": sql[:40], "Type": stmt_type.value})
    st.dataframe(type_data, use_container_width=True)
