---
title: AI Workflow Discovery Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

*(Scroll down for the French version / Voir plus bas pour la version française)*

<div align="center"> <img src="data/MermaidChart.png" width="900" alt="Diagram"> </div>

# AI Workflow Discovery Agent (English)

This repository contains a hybrid business process analysis system. The architecture couples a deterministic clustering engine (NLP) with a multi-agent LLM inference pipeline to diagnose operational inefficiencies, design optimized automation architectures, and generate execution code.

## 🏗 System Architecture

The system is built on the principle of Separation of Concerns, dividing heavy processing, logical orchestration, and the presentation layer.

```text
├── agents/
│   ├── agent_analyst.py      # ROI calculation and diagnosis
│   ├── agent_mapper.py       # Process architecture generation
│   └── agent_advisor.py      # Tech stack selection
├── data/                     # JSON/CSV demonstration files & Tool Catalog
├── assets/
│   └── style.css             # Presentation layer isolation
├── templates/
│   └── airflow_dag.jinja2    # Apache Airflow DAG template
├── tests/
│   ├── pytest.ini            # Pytest configuration
│   └── test_suite.py         # Pytest coverage (AST validation, mocking, etc.)
├── app.py                    # Internal API Gateway and Streamlit UI
├── data_engine.py            # HDBSCAN clustering engine
├── data_refiner.py           # Data Sentinel (Input Normalization)
├── schemas.py                # Pydantic data contracts
├── workflow_viz.py           # Graphviz rendering engine (SVG)
├── code_generator.py         # Jinja2 to Python Airflow compiler
├── collect_telemetry.py      # Inference observability engine
├── Dockerfile                # Environment configuration (Python 3.11-slim, UID 1000)
├── .dockerignore             # Build filters
└── requirements.txt          # Python dependencies

**Deterministic Engine (DataEngine):**

   -Ingests raw exports (CSV/JSON).
   -Vectorizes text via sentence-transformers (all-MiniLM-L6-v2) on CPU.
   -Applies the HDBSCAN clustering algorithm with dynamic density adjustment to isolate noise and semantically group process recurrences.
   -Extracts mathematical centroids to reduce the payload sent to the LLM.

**Multi-Agent Pipeline (instructor + Gemini 3.1 Flash-Lite):**

   -Analyst Agent: Evaluates frictions, the ratio of manual steps, and integrates the deterministic ROI of automation.
   -Mapper Agent: Receives the diagnosis and generates a directed acyclic graph (DAG) representing the optimized target workflow.
   -Advisor Agent: Receives the workflow and selects the implementation technology stack enforcing a "Single Source of Truth" architecture.

**Strict Validation (Pydantic) & Telemetry:**

   -Zero manual parsing of probabilistic JSON. Data transfers between agents and the interface are guaranteed by strict Pydantic  contracts (schemas.py).
   -Inference latency and token consumption are tracked per agent via collect_telemetry.py.

4. Presentation & Code Generation

**Presentation & Rendering Layer (Streamlit + Graphviz):**

   -User interface decoupled from business logic via Streamlit.
   -Dynamic translation of Pydantic objects into interactive SVG diagrams via graphviz.
   -Code Generation: Compiles the optimized workflow into a valid Apache Airflow DAG (.py) using Jinja2 templates and Abstract Syntax Tree (AST) validation.

# 🛠️ How it works

1. Data Ingestion & Sentinel Validation: Choose a pre-loaded dataset or upload your own (CSV/JSON). The "Sentinel" (DataRefiner) analyzes the entire source to ensure atomicity, automatically converting unstructured logs (like Slack or Notion) into clean, processable events before analysis.

2. ROI Configuration: Use the Average Hourly Rate ($/h) slider in the sidebar to define estimated employer costs. This parameter is critical: it allows the deterministic engine to calculate precise savings and annual projections mathematically, eliminating the risk of numerical hallucinations.

3. Semantic Clustering (DataEngine): The system vectorizes text via all-MiniLM-L6-v2 and applies the HDBSCAN algorithm with dynamic density adjustment. This isolates noise and semantically groups process recurrences to extract mathematical centroids, reducing the payload sent to the LLM.

4. Multi-Agent Pipeline & Code Gen: A 3-agent pipeline (Analyst, Mapper, Advisor) powered by Gemini 3.1 Flash-Lite diagnoses frictions, generates an optimized SVG workflow diagram, and compiles a deployment-ready Apache Airflow DAG (.py) using Jinja2 templates.

⚙️ Deployment (Docker)
The application is containerized to guarantee environment idempotency, especially for system dependencies like Graphviz. The container runs securely under a non-root user (UID 1000) and pre-downloads ML models during the build phase to eliminate cold starts.

### Prerequisites

-Docker installed.
-A Google AI Studio API key (GOOGLE_AI_STUDIO_KEY).

### Local Execution
1.Clone the repository:

Bash
git clone [https://github.com/Maxime-Gagne/AI_Workflow_Discovery_Agent.git](https://github.com/Maxime-Gagne/AI_Workflow_Discovery_Agent.git)
cd AI_Workflow_Discovery_Agent

2.Configure environment variables:

**Create a .env file at the root:**
GOOGLE_AI_STUDIO_KEY=your_key_here

**Build and run the container:**

docker rm -f wf-agent
docker build -t ai_workflow_agent .
docker run -d -p 7860:7860 --env-file .env --name wf-agent ai_workflow_agent

Access the interface: http://localhost:7860

🔒 Security and Constraints
The LLM (Gemini 3.1 Flash-Lite) is queried in GEMINI_JSON mode via instructor, forcing structured output without formatting hallucinations.

HDBSCAN processing is executed locally (or on the container's host instance); only aggregated metrics and centroids are sent to the LLM API, limiting token consumption and securing raw data.


----------------------------------------------------------------------------------------------------


