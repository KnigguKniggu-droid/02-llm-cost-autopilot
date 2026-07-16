"""Master dashboard launcher for the AI Engineering Workspace.

Run this script to open a menu of all 15 project dashboards.
Each dashboard opens in a new browser tab via Streamlit.

Usage:
    python launch_dashboards.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = Path(__file__).resolve().parent
PYTHON = sys.executable

PROJECTS = {
    "01": ("Model Regression Detection System", "01_model_regression_system"),
    "02": ("LLM Cost Autopilot", "02_llm_cost_autopilot"),
    "03": ("Failure Forensics Tool", "03_failure_forensics_tool"),
    "04": ("Self-Healing Documentation", "04_self_healing_docs"),
    "05": ("LLM Output Arbitration System", "05_output_arbitration_system"),
    "06": ("Hybrid Search RAG Pipeline", "06_hybrid_search_rag"),
    "07": ("Semantic Caching Proxy", "07_semantic_caching_proxy"),
    "08": ("Text-to-SQL Guardrails", "08_text_to_sql_guardrails"),
    "09": ("Prompt A/B Testing Platform", "09_prompt_ab_testing_platform"),
    "10": ("LoRA Fine-tuning Pipeline", "10_lora_finetuning_pipeline"),
    "11": ("LLM API Gateway", "11_llm_api_gateway"),
    "12": ("AI Feature Flag System", "12_ai_feature_flag_system"),
    "13": ("Prod Log Dataset Generator", "13_prod_log_dataset_generator"),
    "14": ("Agentic Workflow Orchestrator", "14_agentic_workflow_orchestrator"),
    "15": ("Realtime LLM Observability", "15_realtime_llm_observability"),
}


def main() -> None:
    print()
    print("=" * 60)
    print("  AI Engineering Workspace - Dashboard Launcher")
    print("=" * 60)
    print()
    print("  Select a project dashboard to launch:")
    print()

    for num, (name, folder) in PROJECTS.items():
        ui_path = WORKSPACE / folder / "ui" / "app.py"
        status = "READY" if ui_path.exists() else "NO UI"
        print(f"  {num}. {name:<45s} [{status}]")

    print()
    print("  q. Quit")
    print()

    choice = input("  Enter number: ").strip().lower()

    if choice == "q" or choice == "":
        print("  Goodbye.")
        return

    if choice not in PROJECTS:
        print(f"  Invalid choice: {choice}")
        return

    name, folder = PROJECTS[choice]
    ui_path = WORKSPACE / folder / "ui" / "app.py"

    if not ui_path.exists():
        print(f"  Dashboard file not found: {ui_path}")
        return

    project_dir = WORKSPACE / folder
    port = 8501 + int(choice)

    print(f"  Launching: {name}")
    print(f"  Directory: {project_dir}")
    print(f"  URL: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")
    print()

    os.chdir(str(project_dir))
    subprocess.run([
        PYTHON, "-m", "streamlit", "run", "ui/app.py",
        "--server.port", str(port),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()
