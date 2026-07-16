"""Streamlit dashboard for the Agentic Workflow Orchestrator."""

from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "..")

import asyncio
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from src.state import WorkflowState, ExecutionNodeName, NodeStatus, Milestone
from src.orchestrator import WorkflowManager
from src.sample_loop import task_planner_node, code_executor_node

st.set_page_config(page_title="Agentic Workflow Orchestrator", page_icon=":robot:", layout="wide")

st.title("Agentic Workflow Orchestrator")
st.markdown("Multi-agent task dispatcher with state checkpoints and fault-tolerant recovery.")

tab1, tab2, tab3 = st.tabs(["Run Workflow", "State Inspector", "Checkpoint Browser"])

with tab1:
    st.subheader("Run a Workflow")
    st.markdown("Enter an objective and watch the orchestrator plan and execute it.")

    objective = st.text_area(
        "Task Objective",
        value="Write a Python function that validates email addresses",
        height=80,
    )
    max_retries = st.slider("Max Retries", 1, 10, 3)

    if st.button("Run Workflow", type="primary"):
        with st.spinner("Running workflow..."):
            manager = WorkflowManager(
                task_objective=objective,
                max_retries=max_retries,
            )
            manager.register_node(ExecutionNodeName.PLANNER, task_planner_node)
            manager.register_node(ExecutionNodeName.CODE_EXECUTOR, code_executor_node)

            result = asyncio.run(manager.run())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", result["status"].upper())
        col2.metric("Milestones", f"{result['milestones_completed']}/{result['milestones_total']}")
        col3.metric("Checkpoints", result["checkpoints_saved"])
        col4.metric("Retries", result["retries_used"])

        if result["status"] == "succeeded":
            st.success(result["message"])
        else:
            st.error(result["message"])

        st.write(f"**Duration:** {result['duration_seconds']:.3f}s")
        st.write(f"**Final Temperature:** {result['final_temperature']}")
        st.write(f"**Shared Memory Keys:** {', '.join(result['shared_memory_keys'])}")

        st.write("**State Summary:**")
        st.json(result["summary"])

with tab2:
    st.subheader("State Inspector")
    st.markdown("Inspect the WorkflowState object fields.")

    state = WorkflowState(
        session_id="demo-session-001",
        task_objective="Demo objective for inspection",
    )

    st.write("**Core Fields:**")
    st.json({
        "session_id": state.session_id,
        "task_objective": state.task_objective,
        "execution_node": state.execution_node.value,
        "checkpoint_version": state.checkpoint_version,
        "retry_index": state.retry_index,
        "status": state.status.value,
        "max_retries": state.max_retries,
        "hyperparameters": state.hyperparameters,
    })

    st.write("**Milestone Simulator:**")
    num_milestones = st.slider("Number of Milestones", 1, 10, 4)
    completed = st.slider("Completed Milestones", 0, num_milestones, 2)

    state.milestones = []
    for i in range(num_milestones):
        status = NodeStatus.SUCCEEDED if i < completed else NodeStatus.IDLE
        state.milestones.append(Milestone(
            milestone_id=f"m_{i}",
            description=f"Milestone {i+1}",
            status=status,
        ))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", len(state.milestones))
    col2.metric("Completed", len(state.completed_milestones))
    col3.metric("Pending", len(state.pending_milestones))

    st.write(f"**Is Complete:** {state.is_complete}")
    st.write(f"**Has Exhausted Retries:** {state.has_exhausted_retries}")

    fig = go.Figure(data=[go.Pie(
        labels=["Completed", "Pending"],
        values=[len(state.completed_milestones), len(state.pending_milestones)],
        hole=0.4,
        marker_colors=["#2ecc71", "#f5a623"],
    )])
    fig.update_layout(title="Milestone Progress")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Checkpoint Browser")
    st.markdown("View saved checkpoint files from the last workflow run.")

    checkpoint_dir = Path(__file__).resolve().parent.parent / "data" / "checkpoints"
    if checkpoint_dir.exists():
        files = list(checkpoint_dir.glob("checkpoint_*.json"))
        if files:
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            selected = st.selectbox(
                "Select Checkpoint",
                options=[f.name for f in files[:20]],
            )
            if selected:
                data = json.loads((checkpoint_dir / selected).read_text(encoding="utf-8"))
                st.json({
                    "session_id": data.get("session_id"),
                    "checkpoint_version": data.get("checkpoint_version"),
                    "retry_index": data.get("retry_index"),
                    "execution_node": data.get("execution_node"),
                    "status": data.get("status"),
                    "milestones_count": len(data.get("milestones", [])),
                    "history_count": len(data.get("execution_history", [])),
                    "shared_memory_keys": list(data.get("shared_memory_variables", {}).keys()),
                })

                if data.get("milestones"):
                    st.write("**Milestones:**")
                    ms_data = []
                    for m in data["milestones"]:
                        ms_data.append({"ID": m["milestone_id"], "Description": m["description"][:60], "Status": m["status"]})
                    st.dataframe(ms_data, use_container_width=True)

                if data.get("execution_history"):
                    st.write("**Execution History:**")
                    hist_data = []
                    for h in data["execution_history"][-10:]:
                        hist_data.append({
                            "Node": h["node"],
                            "Status": h["status"],
                            "Duration (ms)": f"{h['duration_ms']:.1f}",
                            "Temperature": h["temperature_used"],
                        })
                    st.dataframe(hist_data, use_container_width=True)
        else:
            st.info("No checkpoint files found. Run a workflow first.")
    else:
        st.info("Checkpoint directory does not exist yet. Run a workflow from the 'Run Workflow' tab.")
