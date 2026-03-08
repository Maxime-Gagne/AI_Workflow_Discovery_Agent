import streamlit as st
import json
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
import instructor
from code_generator import generate_airflow_dag
from collect_telemetry import MetricsTracker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_analyst import analyze, analyze_sop
from agents.agent_mapper import map_workflow
from agents.agent_advisor import advise
from data_engine import DataEngine
from data_refiner import refine_source_data
from workflow_viz import render_workflow


st.set_page_config(
    page_title="AI Workflow Discovery Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chargement propre du CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. The interface will be displayed without the custom style..")

load_css("assets/style.css")
load_dotenv()
api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_json(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)

SOURCES = [
    {"key": "tickets_support", "icon": "🎫", "title": "Customer Support", "subtitle": "Export Zendesk — 20 tickets", "file": "tickets_support.json"},
    {"key": "recrutement_pipeline", "icon": "👥", "title": "RH Recrutement", "subtitle": "HubSpot Pipeline — 47 applications", "file": "pipeline_recrutement.json"},
    {"key": "marketing_pipeline", "icon": "📝", "title": "Marketing Content", "subtitle": "Notion + Slack — 12 contents", "file": "pipeline_marketing.json"},
    {"key": "sop_text", "icon": "📄", "title": "Procedure (SOP)", "subtitle": "raw text (Analyse LLM directe)", "file": None},
    {"key": "custom_upload", "icon": "📂", "title": "Personalized Data", "subtitle": "Upload CSV/JSON (DataEngine)", "file": None}
]

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    if api_key:
        st.success("✓ Gemini AI Studio key configured")
    else:
        st.error("CRITICAL: Clé GOOGLE_AI_STUDIO_KEY missing.")
        st.stop()
    st.markdown("---")
    taux_horaire = st.slider(
        "💰 Average hourly rate ($/h)",
        min_value=15,
        max_value=150,
        value=25,
        step=5,
        help="Estimated employer cost. Used for deterministic ROI calculation."
    )
    st.session_state["taux_horaire"] = taux_horaire

    st.markdown("---")
    st.markdown("### 🧮 HDBSCAN Configs")

    epsilon = st.slider(
        "Distance epsilon (Cluster Selection)",
        min_value=0.1, max_value=1.5, value=0.5, step=0.1,
        help="Euclidean distance threshold for cluster merging."
    )

    auto_cluster_size = st.checkbox("Dynamic density calculation (Heuristic)", value=True, help="Adjust min_cluster_size according to max(2, 1.5 * ln(n))")

    if auto_cluster_size:
        min_cluster_size = None
    else:
        min_cluster_size = st.slider(
            "Minimal Cluster Size (min_cluster_size)",
            min_value=2, max_value=20, value=3, step=1
        )

    st.markdown("---")
    st.markdown("""
### 🛠️ How it works?
1. Select your data source
Choose one of the pre-loaded business datasets or import your own file (CSV, JSON) via the Custom Data option. The system also accepts raw Standard Operating Procedures (SOP) for direct inference analysis.

2. Configure ROI parameters
In the sidebar, use the 💰 Average Hourly Rate ($/h) slider to define the estimated employer cost. This parameter is fundamental: it allows the deterministic engine to mathematically calculate potential savings and annual projections, eliminating the risk of numerical hallucinations often found in standard LLM outputs.

3. Validation & Restructuring (The Sentinel)
Once the source is loaded, a qualification step analyzes the entire dataset. If the source is unstructured (chat logs, fragmented notes), the "Sentinel" (DataRefiner) normalizes the content into Atomic Actions before transmitting them to the clustering engine. This ensures the mathematical analysis is based on clean, discrete events.

4. Multi-Agent Pipeline Execution
Click 🚀 Launch analysis to trigger the asynchronous inference chain
""")

st.markdown("""
<div style='padding: 32px 0 16px 0;'>
    <h1 style='font-size: 2rem; margin-bottom: 4px;'>🔍 AI Workflow Discovery Agent</h1>
    <p style='color: #64748B; font-size: 1.05rem;'>Analyse hybride (Moteur Déterministe + Inférence LLM)</p>
</div>
<hr class='section-divider'>
""", unsafe_allow_html=True)

cols = st.columns(len(SOURCES))
selected_source = st.session_state.get("selected_source", None)

for i, source in enumerate(SOURCES):
    with cols[i]:
        is_selected = selected_source == source["key"]
        card_class = "source-card selected" if is_selected else "source-card"
        st.markdown(f"""
        <div class='{card_class}'>
            <div class='source-icon'>{source['icon']}</div>
            <div class='source-title'>{source['title']}</div>
            <div class='source-subtitle'>{source['subtitle']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✓ Selected" if is_selected else "Analyse →", key=f"btn_{source['key']}", use_container_width=True, type="primary" if is_selected else "secondary"):
            st.session_state["selected_source"] = source["key"]
            st.session_state["results"] = None
            st.rerun()

if selected_source:
    source_info = next(s for s in SOURCES if s["key"] == selected_source)
    raw_data = None

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Aperçu des données de la source sélectionnée
    if source_info.get("file"):
        preview_data = load_json(source_info["file"])
        with st.expander(f"👁 Source datas preview — {source_info['title']}", expanded=True):
            st.json(preview_data, expanded=False)

    col_info, col_launch = st.columns([3, 1])

    with col_info:
        st.markdown(f"### {source_info['icon']} Source : **{source_info['title']}**")

        # Routage de l'ingestion de données
        if source_info["key"] == "custom_upload":
            st.info("Import your own raw dataset.")
            uploaded_file = st.file_uploader("Data File", type=["json", "csv"])
            if uploaded_file is not None:
                # Lecture intégrale pour l'analyse globale de la sentinelle
                raw_content = uploaded_file.read().decode("utf-8")

                with st.status("Source validation and restructuration...", expanded=True) as status:
                    # Appel au DataRefiner pour garantir l'atomicité
                    report = refine_source_data(raw_content, api_key)

                    if not report.is_processable:
                        st.error(f"Source rejected : {report.explanation}")
                        st.stop()

                    st.write(f"✅ Detected Format : {report.detected_format}")
                    st.info(report.explanation)
                    status.update(label="Data successfully normalized", state="complete")

                # Conversion des objets Pydantic en dictionnaires pour le DataEngine
                data_for_engine = [action.model_dump() for action in report.structured_data]

                # Initialisation du moteur avec la donnée "propre"
                engine = DataEngine() # Initialise le modèle
                raw_data = data_for_engine # Prépare la donnée pour process_pipeline

        elif source_info["key"] == "sop_text":
            st.info("Paste the raw text of your standard operating procedure (SOP) below.")
            raw_data = st.text_area("Standard Operating Procedure raw text (SOP)", height=250)

        else:
            raw_data = load_json(source_info["file"])
            st.text("Ingestion of the hierarchical JSON payload...")
            payload = raw_data

    with col_launch:
        st.markdown("<br><br>", unsafe_allow_html=True)
        launch_btn = st.button("🚀 Launch Analysis", use_container_width=True, type="primary", disabled=(raw_data is None))

    if launch_btn or st.session_state.get("results"):
        if launch_btn:
            st.session_state["results"] = None

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Pipeline d'agents")
        col_a1, col_a2, col_a3 = st.columns(3)

        def agent_card(col, icon, name, desc, status):
            with col:
                if status == "done": st.success(f"{icon} **{name}** ✓\n\n{desc}")
                elif status == "active": st.info(f"{icon} **{name}** ⏳\n\n{desc}")
                else: st.markdown(f"<div class='agent-step'><strong>{icon} {name}</strong><br><small style='color:#94A3B8'>{desc}</small></div>", unsafe_allow_html=True)

        if launch_btn and not st.session_state.get("results"):
            progress = st.progress(0, text="Initialisation...")
            tracker = MetricsTracker()

            try:
                agent_card(col_a1, "🔬", "Analyst", "Clustering & Diagnostic…", "active")
                agent_card(col_a2, "🗺", "Mapper", "En attente…", "waiting")
                agent_card(col_a3, "💡", "Advisor", "En attente…", "waiting")

                taux_horaire = st.session_state.get("taux_horaire", 25.0)
                if source_info["key"] == "sop_text":
                    agent_card(col_a1, "🔬", "Analyste", "SOP extraction in progress…", "active")
                    diagnostic, lat_analyst, meta_analyst = analyze_sop(raw_data, source_info["title"], taux_horaire)
                else:
                    # On utilise l'instance existante d'engine si elle a été créée (custom_upload)
                    # Sinon on en crée une nouvelle pour les fichiers de démo
                    if 'engine' not in locals():
                        engine = DataEngine()

                    payload, roi_metrics = engine.process_pipeline(
                        raw_data=raw_data,
                        source_name=source_info["title"],
                        taux_horaire=taux_horaire,
                        min_cluster_size=min_cluster_size,
                        epsilon=epsilon
                    )
                    diagnostic, lat_analyst, meta_analyst = analyze(payload, source_info["title"], roi_metrics)

                tracker.record("Analyst", lat_analyst, meta_analyst)
                progress.progress(33, text="Agent Mapper in progress…")

                agent_card(col_a1, "🔬", "Analyst", "Diagnosis done", "done")
                agent_card(col_a2, "🗺", "Mapper", "Construction of the workflow…", "active")

                workflow, resume, lat_mapper, meta_mapper = map_workflow(diagnostic)
                tracker.record("Mapper", lat_mapper, meta_mapper)
                progress.progress(66, text="Agent Advisor en cours…")

                agent_card(col_a2, "🗺", "Mapper", "Buil workflow", "done")
                agent_card(col_a3, "💡", "Advisor", "Tools selection…", "active")

                recommendations, lat_advisor, meta_advisor = advise(workflow)
                tracker.record("Advisor", lat_advisor, meta_advisor)
                progress.progress(100, text="Analyse complète ✓")

                st.session_state["results"] = {
                    "diagnosis": diagnostic,
                    "workflow": workflow,
                    "transformation_resume": resume,
                    "recommendations": recommendations,
                    "telemetry": tracker.get_summary() # Injection des données de télémétrie
                }
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Engineering Error : {str(e)}")

        if st.session_state.get("results"):
            results = st.session_state["results"]
            diag = results["diagnosis"]
            wf = results["workflow"]
            rec = results["recommendations"]

            agent_card(col_a1, "🔬", "Analyst", "Diagnosis done ✓", "done")
            agent_card(col_a2, "🗺", "Mapper", "Built workflow ✓", "done")
            agent_card(col_a3, "💡", "Advisor", "Recommandations ready ✓", "done")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 📊 Diagnosis of the current process")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.metriques_cles.taux_manuel}</div><div class='metric-label'>Tâches manuelles</div></div>", unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.gains_estimes.heures_economisees_par_mois}h</div><div class='metric-label'>Heures économisées/mois</div></div>", unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.gains_estimes.economies_mensuelles_devise} $</div><div class='metric-label'>Économies mensuelles</div></div>", unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"<div class='metric-card'><div class='metric-value'><span class='badge badge-blue'>{diag.potentiel_automatisation.upper()}</span></div><div class='metric-label'>Potentiel</div></div>", unsafe_allow_html=True)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.gains_estimes.projection_annuelle} $</div><div class='metric-label'>Projection annuelle</div></div>", unsafe_allow_html=True)
            with col_r2:
                roi_retour = f"{diag.gains_estimes.mois_retour_investissement} mois" if diag.gains_estimes.mois_retour_investissement else "N/A"
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{roi_retour}</div><div class='metric-label'>Retour sur investissement</div></div>", unsafe_allow_html=True)

            st.caption(f"📐 {diag.gains_estimes.detail_du_calcul}")
            st.markdown(f"<br>**Processus :** {diag.description}", unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 🗺 Workflow optimised")
            st.markdown(f"*{wf.description_transformation}*")

            try:
                svg_bytes = render_workflow(wf)
                svg_str = svg_bytes.decode("utf-8")
                st.markdown(
                    f"<div style='background:white; border-radius:12px; padding:16px; "
                    f"border:1px solid #E2E8F0; overflow-x:auto; overflow-y:auto; "
                    f"max-height:500px;'>{svg_str}</div>",
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"Viewing unavailable : {e}")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 💡 Technological Recommendations")
            for r in rec.recommandations:
                with st.expander(f"**{r.noeud_label}**"):
                    for outil in r.outils:
                        st.markdown(f"**{outil.nom}** ({outil.priorite}) - *{outil.justification}*")

            st.info(f"💬 **Conseil d'implémentation :** {rec.conseil_implementation}")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.download_button(
                "⬇ Export JSON complet",
                data=json.dumps({"diagnosis": diag.model_dump(), "workflow": wf.model_dump(), "recommendations": rec.model_dump()}, ensure_ascii=False, indent=2),
                file_name=f"export_{source_info['key']}.json",
                mime="application/json",
                use_container_width=True
            )
            try:
                airflow_dag_code = generate_airflow_dag(wf)
                st.download_button(
                    label="🐍 DAG Airflow Export(.py)",
                    data=airflow_dag_code,
                    file_name=f"dag_{source_info['key']}.py",
                    mime="text/x-python",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error compiling Jinja2 Airflow DAG : {str(e)}")

            # Affichage de la télémétrie
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            with st.expander("⏱️ Execution & Consumption Telemetry", expanded=False):
                telemetry = results["telemetry"]
                st.markdown(f"**Total Latency :** {telemetry['total_latency_seconds']}s | **Total Tokens :** {telemetry['total_tokens']}")
                st.json(telemetry["details"])
