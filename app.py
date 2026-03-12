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
from agents.agent_temps import AgentTemps
from business_metrics import compute_roi_from_time_report
from validators.time_validators import validate_time_metrics_report_or_raise, TimeValidationError, audit_time_metrics_report
from validators.roi_validators import validate_roi_metrics_or_raise
from data_engine import DataEngine
from data_refiner import refine_source_data
from workflow_viz import render_workflow
from validators.diagnostic_validators import DiagnosticValidationError, validate_diagnostic_or_raise, audit_diagnostic
from validators.workflow_validators import WorkflowValidationError, validate_workflow_or_raise, audit_workflow
from validators.advisor_validators import validate_advisor_report_or_raise, AdvisorValidationError, audit_advisor_report
from time_resolution import resolve_time_context

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

                # Variables communes pour éviter les UnboundLocal plus loin
                time_report = None
                time_warnings = []
                time_precision_message = None
                roi_metrics = None

                def build_time_precision_message(time_report) -> str:
                    metrics = [
                        time_report.monthly_volume,
                        time_report.activity_duration_minutes,
                        time_report.case_cycle_time_minutes,
                        time_report.manual_time_minutes,
                        time_report.waiting_time_minutes,
                    ]

                    counts = {"provided": 0, "observed": 0, "estimated": 0, "missing": 0}
                    for metric in metrics:
                        counts[metric.confidence_level] += 1

                    if counts["missing"] > 0:
                        return (
                            f"Résultat partiellement quantifié. "
                            f"Temps fournis: {counts['provided']} | "
                            f"observés: {counts['observed']} | "
                            f"estimés: {counts['estimated']} | "
                            f"manquants: {counts['missing']}. "
                            f"Des informations temporelles supplémentaires amélioreraient la précision."
                        )

                    if counts["estimated"] > 0:
                        return (
                            f"Résultat approximatif. "
                            f"Temps fournis: {counts['provided']} | "
                            f"observés: {counts['observed']} | "
                            f"estimés: {counts['estimated']}. "
                            f"Certaines métriques ont été estimées par le pipeline."
                        )

                    return (
                        f"Résultat précis. "
                        f"Temps fournis: {counts['provided']} | "
                        f"observés: {counts['observed']}."
                    )

                def should_offer_time_override(time_report) -> bool:
                    critical = [
                        time_report.monthly_volume,
                        time_report.activity_duration_minutes,
                        time_report.manual_time_minutes,
                    ]
                    return any(metric.confidence_level == "missing" for metric in critical)

                if source_info["key"] == "sop_text":
                    agent_card(col_a1, "🔬", "Analyste", "SOP extraction in progress…", "active")

                    # TODO: remplacer {} par un vrai petit formulaire Streamlit si tu veux autoriser
                    # un chiffrage SOP assisté par l'utilisateur.
                    user_metrics = {}

                    if not user_metrics:
                        time_precision_message = (
                            "Mode SOP : aucun chiffrage temporel déterministe n’a été fourni. "
                            "Le diagnostic est structurel uniquement et les estimations quantitatives doivent être considérées comme non disponibles."
                        )
                        time_warnings = [
                            "Aucune métrique temporelle utilisateur fournie pour la branche SOP."
                        ]
                        roi_metrics = None

                        diagnostic, lat_analyst, meta_analyst = analyze_sop(
                            raw_data,
                            source_info["title"],
                            taux_horaire,
                            roi_metrics=None,
                        )
                        validate_diagnostic_or_raise(diagnostic)
                        diagnostic_warnings = audit_diagnostic(diagnostic)

                    else:
                        agent_temps = AgentTemps(use_business_hours=False)
                        time_report = agent_temps.build_time_metrics_from_user_input(user_metrics)
                        validate_time_metrics_report_or_raise(time_report)
                        time_warnings = audit_time_metrics_report(time_report)
                        time_precision_message = build_time_precision_message(time_report)

                        roi_metrics = compute_roi_from_time_report(
                            time_report=time_report,
                            taux_horaire=taux_horaire,
                            automation_rate=0.8,
                            residual_manual_time_minutes=1.5,
                        )
                        validate_roi_metrics_or_raise(roi_metrics)

                        diagnostic, lat_analyst, meta_analyst = analyze_sop(
                            raw_data,
                            source_info["title"],
                            taux_horaire,
                            roi_metrics=roi_metrics,
                        )
                        validate_diagnostic_or_raise(diagnostic, roi_metrics=roi_metrics)
                        diagnostic_warnings = audit_diagnostic(diagnostic)

                else:
                    if "engine" not in locals():
                        engine = DataEngine()

                    agent_temps = AgentTemps(use_business_hours=False)

                    # TODO: remplacer {} par de vrais inputs utilisateur quand tu ajouteras le formulaire
                    user_metrics = {}

                    resolved_time = resolve_time_context(
                        raw_data=raw_data,
                        user_metrics=user_metrics,
                        source_key=source_info["key"],
                    )

                    if resolved_time.warnings:
                        st.warning("\n".join(f"- {w}" for w in resolved_time.warnings))

                    if resolved_time.requires_user_input:
                        st.warning(resolved_time.user_message or "Informations temporelles insuffisantes.")
                        st.stop()

                    if resolved_time.mode == "provided" and resolved_time.timestamp_col is None:
                        time_report = agent_temps.build_time_metrics_from_user_input(user_metrics)
                    else:
                        time_report = agent_temps.build_time_metrics_report(
                            raw_data=resolved_time.normalized_data,
                            user_metrics=user_metrics,
                            case_id_col=resolved_time.case_id_col or "case_id",
                            activity_col=resolved_time.activity_col or "activity",
                            timestamp_col=resolved_time.timestamp_col or "timestamp",
                            start_timestamp_col=resolved_time.start_timestamp_col,
                            end_timestamp_col=resolved_time.end_timestamp_col,
                        )

                    validate_time_metrics_report_or_raise(time_report)
                    time_warnings = audit_time_metrics_report(time_report)
                    time_precision_message = build_time_precision_message(time_report)

                    st.info(time_precision_message)

                    if time_warnings:
                        st.warning("\n".join(f"- {w}" for w in time_warnings))

                    roi_metrics = compute_roi_from_time_report(
                        time_report=time_report,
                        taux_horaire=taux_horaire,
                        automation_rate=0.8,
                        residual_manual_time_minutes=1.5,
                    )
                    validate_roi_metrics_or_raise(roi_metrics)

                    df_norm = engine.normalize(raw_data)
                    df_clustered = engine.vectorize_and_cluster(df_norm, min_cluster_size, epsilon)
                    payload = engine.generate_payload(df_clustered, source_info["title"])

                    diagnostic, lat_analyst, meta_analyst = analyze(
                        payload,
                        source_info["title"],
                        roi_metrics,
                    )

                    validate_diagnostic_or_raise(diagnostic, roi_metrics=roi_metrics)
                    diagnostic_warnings = audit_diagnostic(diagnostic)

                tracker.record("Analyst", lat_analyst, meta_analyst)
                progress.progress(33, text="Agent Mapper in progress…")

                agent_card(col_a1, "🔬", "Analyst", "Diagnosis done", "done")
                agent_card(col_a2, "🗺", "Mapper", "Construction of the workflow…", "active")

                workflow, resume, lat_mapper, meta_mapper = map_workflow(diagnostic)
                validate_workflow_or_raise(workflow, diagnostic=diagnostic)
                workflow_warnings = audit_workflow(workflow, diagnostic=diagnostic)

                tracker.record("Mapper", lat_mapper, meta_mapper)
                progress.progress(66, text="Agent Advisor en cours…")

                agent_card(col_a2, "🗺", "Mapper", "Built workflow", "done")
                agent_card(col_a3, "💡", "Advisor", "Tools selection…", "active")

                recommendations, lat_advisor, meta_advisor = advise(workflow)

                tool_catalog = load_json("tools_catalog.json")
                validate_advisor_report_or_raise(recommendations, workflow, tool_catalog)
                advisor_warnings = audit_advisor_report(recommendations, workflow, tool_catalog)

                tracker.record("Advisor", lat_advisor, meta_advisor)
                progress.progress(100, text="Analyse complète ✓")

                st.session_state["results"] = {
                    "diagnosis": diagnostic,
                    "workflow": workflow,
                    "transformation_resume": resume,
                    "recommendations": recommendations,
                    "telemetry": tracker.get_summary(),
                    "time_report": time_report,
                    "time_warnings": time_warnings,
                    "time_precision_message": time_precision_message,
                    "diagnostic_warnings": diagnostic_warnings,
                    "workflow_warnings": workflow_warnings,
                    "advisor_warnings": advisor_warnings,
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
            time_report = results.get("time_report")
            time_warnings = results.get("time_warnings", [])
            time_precision_message = results.get("time_precision_message")
            diagnostic_warnings = results.get("diagnostic_warnings", [])
            workflow_warnings = results.get("workflow_warnings", [])
            advisor_warnings = results.get("advisor_warnings", [])



            agent_card(col_a1, "🔬", "Analyst", "Diagnosis done ✓", "done")
            agent_card(col_a2, "🗺", "Mapper", "Built workflow ✓", "done")
            agent_card(col_a3, "💡", "Advisor", "Recommandations ready ✓", "done")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### ⏱️ Temporal Reliability & Precision")

            if time_precision_message:
                st.info(time_precision_message)

            if time_warnings:
                st.warning("\n".join(f"- {w}" for w in time_warnings))

            if time_report is not None:
                time_rows = [
                    {
                        "Metric": "Monthly volume",
                        "Value": time_report.monthly_volume.value,
                        "Unit": time_report.monthly_volume.unit,
                        "Confidence": time_report.monthly_volume.confidence_level,
                        "Source": time_report.monthly_volume.source,
                        "Hypothesis": time_report.monthly_volume.hypothesis or "",
                    },
                    {
                        "Metric": "Activity duration",
                        "Value": time_report.activity_duration_minutes.value,
                        "Unit": time_report.activity_duration_minutes.unit,
                        "Confidence": time_report.activity_duration_minutes.confidence_level,
                        "Source": time_report.activity_duration_minutes.source,
                        "Hypothesis": time_report.activity_duration_minutes.hypothesis or "",
                    },
                    {
                        "Metric": "Case cycle time",
                        "Value": time_report.case_cycle_time_minutes.value,
                        "Unit": time_report.case_cycle_time_minutes.unit,
                        "Confidence": time_report.case_cycle_time_minutes.confidence_level,
                        "Source": time_report.case_cycle_time_minutes.source,
                        "Hypothesis": time_report.case_cycle_time_minutes.hypothesis or "",
                    },
                    {
                        "Metric": "Manual time",
                        "Value": time_report.manual_time_minutes.value,
                        "Unit": time_report.manual_time_minutes.unit,
                        "Confidence": time_report.manual_time_minutes.confidence_level,
                        "Source": time_report.manual_time_minutes.source,
                        "Hypothesis": time_report.manual_time_minutes.hypothesis or "",
                    },
                    {
                        "Metric": "Waiting time",
                        "Value": time_report.waiting_time_minutes.value,
                        "Unit": time_report.waiting_time_minutes.unit,
                        "Confidence": time_report.waiting_time_minutes.confidence_level,
                        "Source": time_report.waiting_time_minutes.source,
                        "Hypothesis": time_report.waiting_time_minutes.hypothesis or "",
                    },
                ]
                st.dataframe(pd.DataFrame(time_rows), use_container_width=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 📊 Diagnosis of the current process")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.metriques_cles.taux_manuel}</div><div class='metric-label'>Tâches manuelles</div></div>", unsafe_allow_html=True)
            heures_mois = diag.gains_estimes.heures_economisees_par_mois
            economies_mois = diag.gains_estimes.economies_mensuelles_devise
            projection_annuelle = diag.gains_estimes.projection_annuelle
            roi_retour = diag.gains_estimes.mois_retour_investissement
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.gains_estimes.projection_annuelle} $</div><div class='metric-label'>Projection annuelle</div></div>", unsafe_allow_html=True)
            with col_r2:
                roi_retour = f"{diag.gains_estimes.mois_retour_investissement} mois" if diag.gains_estimes.mois_retour_investissement else "N/A"
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{roi_retour}</div><div class='metric-label'>Retour sur investissement</div></div>", unsafe_allow_html=True)

            with col_m2:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{heures_mois if heures_mois is not None else 'N/A'}"
                    f"{'h' if heures_mois is not None else ''}</div><div class='metric-label'>Heures économisées/mois</div></div>",
                    unsafe_allow_html=True,
                )
            with col_m3:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{economies_mois if economies_mois is not None else 'N/A'}"
                    f"{' $' if economies_mois is not None else ''}</div><div class='metric-label'>Économies mensuelles</div></div>",
                    unsafe_allow_html=True,
                )
            with col_r1:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{projection_annuelle if projection_annuelle is not None else 'N/A'}"
                    f"{' $' if projection_annuelle is not None else ''}</div><div class='metric-label'>Projection annuelle</div></div>",
                    unsafe_allow_html=True,
                )
            with col_r2:
                roi_retour_txt = f"{roi_retour} mois" if roi_retour else "N/A"
                st.markdown(
                    f"<div class='metric-card'><div class='metric-value'>{roi_retour_txt}</div><div class='metric-label'>Retour sur investissement</div></div>",
                    unsafe_allow_html=True,
                )
            with col_m4:
                st.markdown(f"<div class='metric-card'><div class='metric-value'><span class='badge badge-blue'>{diag.potentiel_automatisation.upper()}</span></div><div class='metric-label'>Potentiel</div></div>", unsafe_allow_html=True)



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

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 🚨 Validation Warnings")

            all_warnings = {
                "Diagnostic": diagnostic_warnings,
                "Workflow": workflow_warnings,
                "Advisor": advisor_warnings,
            }

            has_any_warning = any(values for values in all_warnings.values())

            if has_any_warning:
                for section_name, warnings_list in all_warnings.items():
                    if warnings_list:
                        st.markdown(f"**{section_name}**")
                        st.warning("\n".join(f"- {w}" for w in warnings_list))
            else:
                st.success("No validation warnings detected.")

