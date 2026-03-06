import streamlit as st
import json
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
import instructor

# 1. Initialisation du client natif avec ta clé API (via st.secrets dans Streamlit)
native_client = genai.Client()

# 2. Application du patch Instructor
client = instructor.from_genai(native_client)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_analyst import analyze
from agents.agent_mapper import map_workflow
from agents.agent_advisor import advise
from data_engine import DataEngine
from workflow_viz import render_workflow

st.set_page_config(
    page_title="AI Workflow Discovery Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chargement propre du CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except FileNotFoundError:
    st.warning("Fichier CSS introuvable. L'interface s'affichera sans le style personnalisé.")

load_dotenv()
api_key = os.environ.get("GOOGLE_AI_STUDIO_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_json(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)

SOURCES = [
    {"key": "tickets_support", "icon": "🎫", "title": "Support Client", "subtitle": "Export Zendesk — 20 tickets", "file": "tickets_support.json"},
    {"key": "pipeline_recrutement", "icon": "👥", "title": "Recrutement RH", "subtitle": "Pipeline HubSpot — 47 candidatures", "file": "pipeline_recrutement.json"},
    {"key": "pipeline_marketing", "icon": "📝", "title": "Contenu Marketing", "subtitle": "Notion + Slack — 12 contenus", "file": "pipeline_marketing.json"},
    {"key": "custom_upload", "icon": "📂", "title": "Données Personnalisées", "subtitle": "Upload CSV/JSON (DataEngine)", "file": None}
]

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    if api_key:
        st.success("✓ Clé Gemini AI Studio configurée")
    else:
        st.error("CRITIQUE: Clé GOOGLE_AI_STUDIO_KEY manquante.")
        st.stop()

    st.markdown("---")
    st.markdown("### 🏗 Architecture\n**3 agents (Gemini 3.1) + Pydantic:**\n🔬 **Analyste**\n🗺 **Mapper**\n💡 **Advisor**")

st.markdown("""
<div style='padding: 32px 0 16px 0;'>
    <h1 style='font-size: 2rem; margin-bottom: 4px;'>🔍 AI Workflow Discovery Agent</h1>
    <p style='color: #64748B; font-size: 1.05rem;'>Analyse hybride (Moteur Déterministe + Inférence LLM)</p>
</div>
<hr class='section-divider'>
""", unsafe_allow_html=True)

cols = st.columns(4)
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
        if st.button("✓ Sélectionné" if is_selected else "Analyser →", key=f"btn_{source['key']}", use_container_width=True, type="primary" if is_selected else "secondary"):
            st.session_state["selected_source"] = source["key"]
            st.session_state["results"] = None
            st.rerun()

if selected_source:
    source_info = next(s for s in SOURCES if s["key"] == selected_source)
    raw_data = None

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_info, col_launch = st.columns([3, 1])

    with col_info:
        st.markdown(f"### {source_info['icon']} Source : **{source_info['title']}**")

        # Routage de l'ingestion de données
        if source_info["key"] == "custom_upload" and isinstance(raw_data, list):
            st.text("Exécution du DataEngine (Clustering HDBSCAN)...")
            engine = DataEngine(embedding_model='all-MiniLM-L6-v2')
            # Configuration par défaut pour un CSV standard
            mapping_config = {'ticket_id': 'id', 'description': 'content', 'created_at': 'timestamp_start', 'resolved_at': 'timestamp_end'}
            payload = engine.process_pipeline(raw_data, mapping_config, source_name=source_info["title"])
        else:
            st.text("Ingestion du payload JSON hiérarchique...")
            payload = raw_data

    with col_launch:
        st.markdown("<br><br>", unsafe_allow_html=True)
        launch_btn = st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary", disabled=(raw_data is None))

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

            try:
                agent_card(col_a1, "🔬", "Analyste", "Clustering & Diagnostic…", "active")
                agent_card(col_a2, "🗺", "Mapper", "En attente…", "waiting")
                agent_card(col_a3, "💡", "Advisor", "En attente…", "waiting")


                payload = raw_data

                diagnostic = analyze(payload, source_info["title"])
                progress.progress(33, text="Agent Mapper en cours…")

                agent_card(col_a1, "🔬", "Analyste", "Diagnostic terminé", "done")
                agent_card(col_a2, "🗺", "Mapper", "Construction du workflow…", "active")

                workflow = map_workflow(diagnostic)
                progress.progress(66, text="Agent Advisor en cours…")

                agent_card(col_a2, "🗺", "Mapper", "Workflow construit", "done")
                agent_card(col_a3, "💡", "Advisor", "Sélection des outils…", "active")

                recommendations = advise(workflow)
                progress.progress(100, text="Analyse complète ✓")

                st.session_state["results"] = {
                    "diagnostic": diagnostic,
                    "workflow": workflow,
                    "recommendations": recommendations
                }
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Erreur d'ingénierie : {str(e)}")

        if st.session_state.get("results"):
            results = st.session_state["results"]
            diag = results["diagnostic"]
            wf = results["workflow"]
            rec = results["recommendations"]

            agent_card(col_a1, "🔬", "Analyste", "Diagnostic terminé ✓", "done")
            agent_card(col_a2, "🗺", "Mapper", "Workflow construit ✓", "done")
            agent_card(col_a3, "💡", "Advisor", "Recommandations prêtes ✓", "done")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 📊 Diagnostic du processus actuel")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.metriques_cles.taux_manuel}</div><div class='metric-label'>Tâches manuelles</div></div>", unsafe_allow_html=True)
            with col_m2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{diag.gains_estimes.temps_economise_pct}</div><div class='metric-label'>Temps économisé</div></div>", unsafe_allow_html=True)
            with col_m3: st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(wf.noeuds)}</div><div class='metric-label'>Étapes cibles</div></div>", unsafe_allow_html=True)
            with col_m4: st.markdown(f"<div class='metric-card'><div class='metric-value'><span class='badge badge-blue'>{diag.potentiel_automatisation.upper()}</span></div><div class='metric-label'>Potentiel</div></div>", unsafe_allow_html=True)

            st.markdown(f"<br>**Processus :** {diag.description}", unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 🗺 Workflow optimisé")
            st.markdown(f"*{wf.description_transformation}*")
            st.markdown("### 🗺 Workflow optimisé")
            st.markdown(f"*{wf.description_transformation}*")

            # Injection des octets SVG dans l'interface
            svg_bytes = render_workflow(wf)
            st.markdown(svg_bytes.decode("utf-8"), unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 💡 Recommandations Technologiques")
            for r in rec.recommandations:
                with st.expander(f"**{r.noeud_label}**"):
                    for outil in r.outils:
                        st.markdown(f"**{outil.nom}** ({outil.priorite}) - *{outil.justification}*")

            st.info(f"💬 **Conseil d'implémentation :** {rec.conseil_implementation}")

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.download_button(
                "⬇ Export JSON complet",
                data=json.dumps({"diagnostic": diag.model_dump(), "workflow": wf.model_dump(), "recommendations": rec.model_dump()}, ensure_ascii=False, indent=2),
                file_name=f"export_{source_info['key']}.json",
                mime="application/json",
                use_container_width=True
            )
