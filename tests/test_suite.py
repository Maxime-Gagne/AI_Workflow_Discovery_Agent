"""
=============================================================================
test_suite.py — AI Workflow Discovery Agent
=============================================================================
Couverture : schemas.py | data_engine.py | workflow_viz.py |
             code_generator.py | agent_mapper.py (compute_transformation)

Exécution :
    pip install pytest pydantic graphviz sentence-transformers scikit-learn
    pytest tests/test_suite.py -v

Conventions :
    - Aucun appel réseau réel : les agents LLM sont mockés via unittest.mock
    - Chaque classe de test isole une couche architecturale
    - Les fixtures partagées sont définies en tête de fichier
=============================================================================
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
import ast
from collect_telemetry import MetricsTracker
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — permet d'importer les modules depuis la racine du projet
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from schemas import (
    ResumeTransformation,
    NoeudWorkflow,
    LienWorkflow,
    WorkflowOptimise,
    DiagnosticAnalyste,
    EtapeActuelle,
    MetriquesCles,
    GainsEstimes,
    OutilRecommande,
)
from data_engine import DataEngine
from workflow_viz import _sanitize, render_workflow
from code_generator import _sanitize_id, generate_airflow_dag
from agents.agent_mapper import compute_transformation


# =============================================================================
# FIXTURES PARTAGÉES
# =============================================================================

@pytest.fixture
def etape_factory():
    """Fabrique une EtapeActuelle valide avec des valeurs par défaut surchargeables."""
    def _make(id="E1", nom="Action test", type_etape="manuelle",
              acteur="RH", duree_estimee="30min", volume="10/jour", friction=None):
        return EtapeActuelle(
            id=id, nom=nom, type_etape=type_etape,
            acteur=acteur, duree_estimee=duree_estimee,
            volume=volume, friction=friction,
        )
    return _make


@pytest.fixture
def diagnostic_minimal(etape_factory):
    """DiagnosticAnalyste minimal valide avec 3 étapes pour les tests de transformation."""
    return DiagnosticAnalyste(
        titre_processus="Pipeline Test",
        description="Processus de test. Deux phrases.",
        etapes_actuelles=[
            etape_factory(id="E1", nom="Réception CV", type_etape="manuelle"),
            etape_factory(id="E2", nom="Scoring CV", type_etape="automatisable"),
            etape_factory(id="E3", nom="Entretien RH", type_etape="manuelle"),
        ],
        metriques_cles=MetriquesCles(
            volume_total="47",
            duree_totale_estimee="27 jours",
            taux_manuel="80%",
            principale_friction="Scheduling manuel",
        ),
        potentiel_automatisation="élevé",
        gains_estimes=GainsEstimes(
            heures_economisees_par_mois=20.0,
            economies_mensuelles_devise=500.0,
            projection_annuelle=6000.0,
            mois_retour_investissement=3.0,
            detail_du_calcul="20h × 25$/h = 500$/mois → 6000$/an",
            principales_opportunites=["Scoring auto", "Scheduling auto", "Feedback auto"],
        ),
    )


@pytest.fixture
def workflow_minimal():
    """WorkflowOptimise minimal valide pour les tests de rendu et génération."""
    return WorkflowOptimise(
        titre_workflow="Workflow Test",
        description_transformation="Remplacement des étapes manuelles. Architecture événementielle.",
        noeuds=[
            NoeudWorkflow(id="N1", label="Trigger", type_noeud="trigger",
                          description="Déclencheur du pipeline", etape_originale=None),
            NoeudWorkflow(id="N2", label="Score CV", type_noeud="automatique",
                          description="Scoring automatique du CV", etape_originale="E2"),
            NoeudWorkflow(id="N3", label="Entretien RH", type_noeud="humain",
                          description="Entretien conduit par RH", etape_originale="E3"),
            NoeudWorkflow(id="N4", label="Fin", type_noeud="fin",
                          description="Fin du pipeline", etape_originale=None),
        ],
        liens=[
            LienWorkflow(de="N1", vers="N2", condition=None),
            LienWorkflow(de="N2", vers="N3", condition=None),
            LienWorkflow(de="N3", vers="N4", condition=None),
        ],
    )


# =============================================================================
# T1 — schemas.py : ResumeTransformation lève ValueError si tout est vide
# =============================================================================

class TestResumeTransformation:

    def test_toutes_listes_vides_leve_erreur(self):
        """
        RÈGLE MÉTIER : Au moins une étape doit être classifiée.
        Le model_validator doit rejeter un objet avec 3 listes vides.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResumeTransformation(
                etapes_eliminees=[],
                etapes_automatisees=[],
                etapes_conservees_humaines=[],
            )
        assert "Au moins une étape" in str(exc_info.value)

    def test_une_seule_liste_non_vide_est_valide(self):
        """Une seule étape dans une seule liste suffit à satisfaire le validator."""
        rt = ResumeTransformation(
            etapes_eliminees=["E1"],
            etapes_automatisees=[],
            etapes_conservees_humaines=[],
        )
        assert rt.etapes_eliminees == ["E1"]

    def test_toutes_listes_remplies_est_valide(self):
        """Cas nominal : distribution complète sur les 3 listes."""
        rt = ResumeTransformation(
            etapes_eliminees=["E1"],
            etapes_automatisees=["E2"],
            etapes_conservees_humaines=["E3"],
        )
        assert len(rt.etapes_eliminees) + len(rt.etapes_automatisees) + len(rt.etapes_conservees_humaines) == 3


# =============================================================================
# T2 — schemas.py : Les champs Literal rejettent des valeurs hors-domaine
# =============================================================================

class TestLiteralValidation:

    def test_type_etape_invalide(self):
        """'semi-manuelle' n'est pas dans le Literal autorisé."""
        with pytest.raises(ValidationError):
            EtapeActuelle(
                id="E1", nom="Test", type_etape="semi-manuelle",
                acteur="RH", duree_estimee="1h", volume="10/j",
            )

    def test_type_noeud_invalide(self):
        """'conditionnel' n'est pas un type_noeud valide."""
        with pytest.raises(ValidationError):
            NoeudWorkflow(
                id="N1", label="Test", type_noeud="conditionnel",
                description="Noeud test",
            )

    def test_potentiel_automatisation_invalide(self):
        """'maximum' n'est pas dans les valeurs de potentiel_automatisation."""
        with pytest.raises(ValidationError):
            DiagnosticAnalyste(
                titre_processus="Test",
                description="Test desc.",
                etapes_actuelles=[],
                metriques_cles=MetriquesCles(
                    volume_total="1", duree_totale_estimee="1j",
                    taux_manuel="50%", principale_friction="aucune",
                ),
                potentiel_automatisation="maximum",
                gains_estimes=GainsEstimes(
                    heures_economisees_par_mois=0.0,
                    economies_mensuelles_devise=0.0,
                    projection_annuelle=0.0,
                    detail_du_calcul="0h × 0$/h = 0$",
                    principales_opportunites=[],
                ),
            )

    def test_complexite_outil_invalide(self):
        """'expert' n'est pas dans les Literal de complexite."""
        with pytest.raises(ValidationError):
            OutilRecommande(
                nom="Zapier", justification="Automatise.",
                complexite="expert", priorite="essentiel",
            )

    def test_priorite_outil_invalide(self):
        """'critique' n'est pas dans le Literal de priorite."""
        with pytest.raises(ValidationError):
            OutilRecommande(
                nom="Zapier", justification="Automatise.",
                complexite="faible", priorite="critique",
            )


# =============================================================================
# T3 — data_engine.py : _extract_list() détecte la liste dans diverses structures
# =============================================================================

class TestDataEngineExtractList:

    def test_input_est_une_liste_directe(self):
        """JSON racine = liste → retourné tel quel."""
        data = [{"action": "a"}, {"action": "b"}]
        result = DataEngine._extract_list(data)
        assert result == data

    def test_input_est_un_dict_avec_liste_imbriquee(self):
        """JSON racine = dict contenant une clé dont la valeur est une liste de dicts."""
        data = {"source": "HubSpot", "pipeline": [{"etape": "CV reçu"}, {"etape": "Screening"}]}
        result = DataEngine._extract_list(data)
        assert result == [{"etape": "CV reçu"}, {"etape": "Screening"}]

    def test_input_vide_retourne_liste_vide(self):
        """Un dict sans liste imbriquée retourne une liste vide (pas d'exception)."""
        result = DataEngine._extract_list({"key": "value"})
        assert result == []

    def test_input_dict_avec_liste_de_scalaires_ignoree(self):
        """Une liste de scalaires ne doit pas être retournée (seules les listes de dicts comptent)."""
        data = {"tags": ["python", "ai"], "records": [{"id": 1}]}
        result = DataEngine._extract_list(data)
        # Doit retourner la liste de dicts, pas la liste de scalaires
        assert result == [{"id": 1}]


# =============================================================================
# T4 — data_engine.py : normalize() mappe correctement les colonnes candidates
# =============================================================================

class TestDataEngineNormalize:

    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = DataEngine()

    def test_colonne_action_mappee_vers_content(self):
        """La colonne 'action' doit être renommée 'content' automatiquement."""
        data = [{"action": "lecture CV"}, {"action": "envoi email"}]
        df = self.engine.normalize(data)
        assert "content" in df.columns
        assert df["content"].tolist() == ["lecture CV", "envoi email"]

    def test_colonne_category_mappee_vers_content(self):
        """La colonne 'category' doit être renommée 'content' si 'action' est absent."""
        data = [{"category": "Bug technique"}, {"category": "Facturation"}]
        df = self.engine.normalize(data)
        assert "content" in df.columns

    def test_colonne_duree_moyenne_jours_mappee(self):
        """'duree_moyenne_jours' doit être converti en 'duration_minutes' (float)."""
        data = [{"action": "Screening", "duree_moyenne_jours": 3.2}]
        df = self.engine.normalize(data)
        assert "duration_minutes" in df.columns
        assert df["duration_minutes"].iloc[0] == pytest.approx(3.2)

    def test_calcul_duree_depuis_timestamps(self):
        """Si created/resolved sont présents, duration_minutes doit être calculé."""
        data = [{"action": "TK-1042", "created": "2024-01-15 09:03", "resolved": "2024-01-15 11:03"}]
        df = self.engine.normalize(data)
        assert "duration_minutes" in df.columns
        assert df["duration_minutes"].iloc[0] == pytest.approx(120.0, abs=1.0)

    def test_erreur_si_aucune_colonne_texte(self):
        """normalize() doit lever ValueError si aucune colonne texte candidate n'existe."""
        data = [{"score": 42, "priority": 1}]
        with pytest.raises(ValueError, match="Aucune colonne texte"):
            self.engine.normalize(data)

    def test_lignes_content_null_supprimees(self):
        """Les lignes avec content=NaN doivent être droppées silencieusement."""
        data = [{"action": "valide"}, {"action": None}]
        df = self.engine.normalize(data)
        assert len(df) == 1
        assert df["content"].iloc[0] == "valide"


# =============================================================================
# T5 — data_engine.py : compute_roi() est déterministe et mathématiquement correct
# =============================================================================

class TestDataEngineComputeROI:

    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = DataEngine()

    def _make_payload(self, clusters):
        return {"clusters_repetitifs": clusters}

    def test_calcul_nominal(self):
        """
        Scénario : 2 clusters totalisant 300 minutes perdues, taux=50$/h
        Attendu  : 5h/mois × 50$/h = 250$/mois → 3000$/an
        """
        payload = self._make_payload([
            {"cluster_id": "C0", "frequence": 5, "temps_total_perdu_minutes": 180},
            {"cluster_id": "C1", "frequence": 3, "temps_total_perdu_minutes": 120},
        ])
        roi = self.engine.compute_roi(payload, taux_horaire=50.0)
        assert roi["heures_perdues_par_mois"] == pytest.approx(5.0)
        assert roi["economies_mensuelles"] == pytest.approx(250.0)
        assert roi["projection_annuelle"] == pytest.approx(3000.0)

    def test_payload_sans_clusters_retourne_zeros(self):
        """Aucun cluster répétitif → toutes les métriques financières à zéro."""
        roi = self.engine.compute_roi(self._make_payload([]), taux_horaire=25.0)
        assert roi["heures_perdues_par_mois"] == 0.0
        assert roi["economies_mensuelles"] == 0.0
        assert roi["projection_annuelle"] == 0.0

    def test_temps_null_est_traite_comme_zero(self):
        """Un cluster avec temps_total_perdu_minutes=None ne doit pas planter."""
        payload = self._make_payload([
            {"cluster_id": "C0", "frequence": 2, "temps_total_perdu_minutes": None},
        ])
        roi = self.engine.compute_roi(payload, taux_horaire=25.0)
        assert roi["economies_mensuelles"] == 0.0

    def test_detail_calcul_presente(self):
        """Le champ detail_calcul doit être une chaîne non-vide."""
        payload = self._make_payload([{"cluster_id": "C0", "frequence": 1,
                                       "temps_total_perdu_minutes": 60}])
        roi = self.engine.compute_roi(payload, taux_horaire=25.0)
        assert isinstance(roi["detail_calcul"], str)
        assert len(roi["detail_calcul"]) > 0


# =============================================================================
# T6 — data_engine.py : normalize() sur les 3 datasets réels du projet
# =============================================================================

class TestDataEngineRealDatasets:
    """Vérifie que le DataEngine ingère sans erreur les 3 datasets du projet."""

    @pytest.fixture(autouse=True)
    def engine(self):
        self.engine = DataEngine()

    def _load(self, filename):
        import json
        path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_tickets_support(self):
        data = self._load("tickets_support.json")
        df = self.engine.normalize(data)
        assert len(df) > 0
        assert "content" in df.columns

    def test_pipeline_recrutement(self):
        data = self._load("pipeline_recrutement.json")
        df = self.engine.normalize(data)
        assert len(df) > 0

    def test_pipeline_marketing(self):
        data = self._load("pipeline_marketing.json")
        df = self.engine.normalize(data)
        assert len(df) > 0

# =============================================================================
# T6.5 — data_engine.py : vectorize_and_cluster() et heuristique HDBSCAN
# =============================================================================

class TestDataEngineVectorize:
    @pytest.fixture(autouse=True)
    def engine(self):
        # On mock SentenceTransformer pour éviter le chargement du modèle de 384 dimensions
        with patch('data_engine.SentenceTransformer'):
            self.engine = DataEngine()

    @patch('data_engine.HDBSCAN')
    def test_vectorize_heuristique_logarithmique(self, mock_hdbscan):
        """Vérifie que la taille minimale du cluster s'ajuste dynamiquement selon n."""
        # Setup d'un DataFrame de 100 lignes
        df_mock = pd.DataFrame({"content": ["A"] * 100})
        mock_instance = MagicMock()
        mock_instance.fit_predict.return_value = [0] * 100
        mock_hdbscan.return_value = mock_instance

        # Formule : ln(100) * 1.5 ≈ 6.9 -> cast en int -> 6
        self.engine.vectorize_and_cluster(df_mock)
        mock_hdbscan.assert_called_once()
        args, kwargs = mock_hdbscan.call_args
        assert kwargs['min_cluster_size'] == 6

# =============================================================================
# T7 — code_generator.py : _sanitize_id() couvre tous les cas limites
# =============================================================================

class TestSanitizeId:

    def test_minuscules_uniquement(self):
        assert _sanitize_id("Hello World") == "hello_world"

    def test_caracteres_speciaux_remplaces(self):
        assert _sanitize_id("N1 — étape@critique!") == "n1____tape_critique_"

    def test_chiffres_conserves(self):
        assert _sanitize_id("step_42") == "step_42"

    def test_string_vide(self):
        assert _sanitize_id("") == ""

    def test_tirets_remplaces_par_underscore(self):
        assert _sanitize_id("my-workflow-dag") == "my_workflow_dag"

    def test_majuscules_converties(self):
        assert _sanitize_id("DAG_PIPELINE") == "dag_pipeline"


# =============================================================================
# T8 — code_generator.py : Mapping type_noeud → opérateur Airflow exhaustif
# =============================================================================

class TestAirflowOperatorMapping:
    """
    Vérifie que chaque type_noeud Pydantic produit l'opérateur Airflow attendu
    en inspectant le code DAG généré.
    """

    MAPPING = {
        "trigger":     "EmptyOperator",
        "automatique": "PythonOperator",
        "humain":      "BashOperator",
        "decision":    "BranchPythonOperator",
        "fin":         "EmptyOperator",
    }

    def _make_workflow(self, type_noeud: str) -> WorkflowOptimise:
        return WorkflowOptimise(
            titre_workflow=f"test_{type_noeud}",
            description_transformation="Test workflow. Validation opérateurs Airflow.",
            noeuds=[
                NoeudWorkflow(id="N1", label="Start", type_noeud="trigger",
                              description="Déclencheur"),
                NoeudWorkflow(id="N2", label="Node Test", type_noeud=type_noeud,
                              description=f"Noeud de type {type_noeud}"),
            ],
            liens=[LienWorkflow(de="N1", vers="N2")],
        )

    @pytest.mark.parametrize("type_noeud,expected_operator", [
        ("trigger",     "EmptyOperator"),
        ("automatique", "PythonOperator"),
        ("humain",      "BashOperator"),
        ("decision",    "BranchPythonOperator"),
        ("fin",         "EmptyOperator"),
    ])
    def test_operateur_genere(self, type_noeud, expected_operator):
        workflow = self._make_workflow(type_noeud)
        dag_code = generate_airflow_dag(workflow)
        assert expected_operator in dag_code, (
            f"Opérateur '{expected_operator}' attendu pour type_noeud='{type_noeud}'"
        )

    def test_dag_id_base_sur_titre(self):
        """Le dag_id doit être dérivé du titre_workflow après sanitization."""
        workflow = self._make_workflow("fin")
        dag_code = generate_airflow_dag(workflow)
        assert "dag_test_fin" in dag_code

    def test_dependances_generees(self):
        """Les liens du workflow doivent produire des dépendances Airflow >> ."""
        workflow = self._make_workflow("automatique")
        dag_code = generate_airflow_dag(workflow)
        assert ">>" in dag_code


# =============================================================================
# T9 — workflow_viz.py : _sanitize() respecte la règle de troncature
# =============================================================================

class TestSanitizeViz:

    def test_string_vide_retourne_vide(self):
        assert _sanitize("") == ""

    def test_none_retourne_vide(self):
        assert _sanitize(None) == ""

    def test_string_courte_inchangee(self):
        label = "Scoring CV"
        assert _sanitize(label) == label

    def test_troncature_a_40_caracteres(self):
        long_label = "A" * 50
        result = _sanitize(long_label)
        assert len(result) == 40
        assert result.endswith("..")

    def test_exactement_40_chars_inchange(self):
        label = "B" * 40
        result = _sanitize(label)
        assert result == label

    def test_41_chars_tronque(self):
        label = "C" * 41
        result = _sanitize(label)
        assert len(result) == 40
        assert result.endswith("..")


# =============================================================================
# T10 — workflow_viz.py : Rendu SVG (Mocké pour isolation système)
# =============================================================================

class TestRenderWorkflow:

    @patch('workflow_viz.graphviz.Digraph.pipe')
    def test_svg_produit(self, mock_pipe, workflow_minimal):
        """Simule le retour binaire du binaire dot.exe."""
        mock_pipe.return_value = b"<svg>Mocked</svg>"
        result = render_workflow(workflow_minimal)
        assert isinstance(result, bytes)
        assert b"<svg" in result

    @patch('workflow_viz.graphviz.Digraph.pipe')
    def test_labels_presents_dans_svg(self, mock_pipe, workflow_minimal):
        mock_pipe.return_value = b"<svg>Score CV Entretien RH</svg>"
        result = render_workflow(workflow_minimal)
        svg_str = result.decode("utf-8")
        assert "Score CV" in svg_str
        assert "Entretien RH" in svg_str

    @patch('workflow_viz.graphviz.Digraph.pipe')
    def test_workflow_avec_un_seul_noeud(self, mock_pipe):
        mock_pipe.return_value = b"<svg>Seul</svg>"
        wf = WorkflowOptimise(
            titre_workflow="Minimal",
            description_transformation="Un seul noeud. Pas de liens.",
            noeuds=[NoeudWorkflow(id="N1", label="Seul", type_noeud="fin",
                                  description="Unique noeud")],
            liens=[],
        )
        result = render_workflow(wf)
        assert b"<svg" in result

    @patch('workflow_viz.graphviz.Digraph.pipe')
    def test_workflow_avec_condition_sur_lien(self, mock_pipe):
        mock_pipe.return_value = b"<svg>Decision</svg>"
        wf = WorkflowOptimise(
            titre_workflow="Decision Test",
            description_transformation="Test avec condition. Workflow bifurqué.",
            noeuds=[
                NoeudWorkflow(id="N1", label="Decision", type_noeud="decision",
                              description="Branchement"),
                NoeudWorkflow(id="N2", label="Fin OK", type_noeud="fin",
                              description="Succès"),
            ],
            liens=[LienWorkflow(de="N1", vers="N2", condition="si priorité = critique")],
        )
        result = render_workflow(wf)
        assert b"<svg" in result


# =============================================================================
# T11 — agent_analyst.py : analyze() retourne un DiagnosticAnalyste valide (mock)
# =============================================================================

class TestAgentAnalystMocked:
    """
    Teste la couche d'intégration sans appel réseau réel.
    instructor est mocké pour retourner un DiagnosticAnalyste synthétique.
    """

    @pytest.fixture
    def mock_diagnostic(self):
        return DiagnosticAnalyste(
            titre_processus="Support Client Zendesk",
            description="Pipeline de traitement des tickets. Majoritairement manuel.",
            etapes_actuelles=[
                EtapeActuelle(id="E1", nom="Réception ticket", type_etape="automatisable",
                              acteur="Système", duree_estimee="0min", volume="20/semaine"),
            ],
            metriques_cles=MetriquesCles(
                volume_total="20", duree_totale_estimee="7.2h",
                taux_manuel="75%", principale_friction="Tickets non assignés",
            ),
            potentiel_automatisation="très élevé",
            gains_estimes=GainsEstimes(
                heures_economisees_par_mois=14.4,
                economies_mensuelles_devise=360.0,
                projection_annuelle=4320.0,
                mois_retour_investissement=None,
                detail_du_calcul="14.4h × 25$/h = 360$/mois → 4320$/an",
                principales_opportunites=["Triage auto", "Assignation auto", "Réponse auto"],
            ),
        )

    def test_analyze_retourne_tuple_valide(self, mock_diagnostic):
        """
        Vérifie que la fonction analyze() retourne bien le tuple (diagnostic, latence, meta)
        quand instructor est mocké avec create_with_completion.
        """
        with patch("agents.agent_analyst.instructor") as mock_instructor, \
             patch("agents.agent_analyst.genai") as mock_genai:

            # Setup du mock API GenAI pour la télémétrie
            mock_usage = MagicMock(prompt_token_count=100, candidates_token_count=50, total_token_count=150)
            mock_raw_response = MagicMock(usage_metadata=mock_usage)

            mock_client = MagicMock()
            # On mocke désormais create_with_completion au lieu de create
            mock_client.chat.completions.create_with_completion.return_value = (mock_diagnostic, mock_raw_response)
            mock_instructor.from_genai.return_value = mock_client
            mock_genai.Client.return_value = MagicMock()

            from agents.agent_analyst import analyze

            payload = {"source": "Zendesk", "clusters_repetitifs": []}
            roi = {"heures_perdues_par_mois": 14.4, "economies_mensuelles": 360.0,
                   "projection_annuelle": 4320.0, "detail_calcul": "14.4h × 25$/h"}

            # Dépaquetage du tuple
            result, latence, meta = analyze(payload, "Support Client", roi)

            assert isinstance(result, DiagnosticAnalyste)
            assert result.potentiel_automatisation == "très élevé"
            assert isinstance(latence, float)
            assert meta.total_token_count == 150

# =============================================================================
# T12 — agent_mapper.py : compute_transformation() règles de non-chevauchement
# =============================================================================

class TestComputeTransformation:

    def test_pas_de_chevauchement_entre_listes(self, diagnostic_minimal, workflow_minimal):
        """
        INVARIANT CRITIQUE : une étape ne peut pas apparaître dans deux listes à la fois.
        """
        result = compute_transformation(diagnostic_minimal, workflow_minimal)

        auto = set(result.etapes_automatisees)
        humaines = set(result.etapes_conservees_humaines)
        eliminees = set(result.etapes_eliminees)

        assert auto & humaines == set(), "Chevauchement auto ∩ humaines détecté"
        assert auto & eliminees == set(), "Chevauchement auto ∩ eliminees détecté"
        assert humaines & eliminees == set(), "Chevauchement humaines ∩ eliminees détecté"

    def test_couverture_totale_des_etapes(self, diagnostic_minimal, workflow_minimal):
        """
        Toutes les étapes du diagnostic doivent apparaître dans exactement une liste.
        """
        result = compute_transformation(diagnostic_minimal, workflow_minimal)

        all_classified = (
            set(result.etapes_automatisees)
            | set(result.etapes_conservees_humaines)
            | set(result.etapes_eliminees)
        )
        all_original = {e.id for e in diagnostic_minimal.etapes_actuelles}
        assert all_classified == all_original

    def test_tiebreak_humain_prime_sur_automatique(self, diagnostic_minimal):
        """
        Si E1 a un noeud automatique ET un noeud humain, il doit être classé humain.
        """
        workflow_conflit = WorkflowOptimise(
            titre_workflow="Conflit Test",
            description_transformation="Test tiebreak. E1 mappé sur deux noeuds.",
            noeuds=[
                NoeudWorkflow(id="N1", label="Auto E1", type_noeud="automatique",
                              description="Auto", etape_originale="E1"),
                NoeudWorkflow(id="N2", label="Humain E1", type_noeud="humain",
                              description="Humain", etape_originale="E1"),
            ],
            liens=[LienWorkflow(de="N1", vers="N2")],
        )
        result = compute_transformation(diagnostic_minimal, workflow_conflit)
        assert "E1" in result.etapes_conservees_humaines
        assert "E1" not in result.etapes_automatisees

    def test_noeuds_sans_etape_originale_ignores(self, diagnostic_minimal):
        """
        Les noeuds trigger/fin (etape_originale=None) ne doivent pas polluer
        les listes de transformation — leurs étapes doivent finir en 'eliminees'.
        """
        workflow_sans_ref = WorkflowOptimise(
            titre_workflow="Sans ref",
            description_transformation="Noeuds sans référence aux étapes. Test isolation.",
            noeuds=[
                NoeudWorkflow(id="N1", label="Trigger", type_noeud="trigger",
                              description="Trigger", etape_originale=None),
            ],
            liens=[],
        )
        result = compute_transformation(diagnostic_minimal, workflow_sans_ref)
        # Toutes les étapes E1, E2, E3 doivent être en eliminees
        assert set(result.etapes_eliminees) == {"E1", "E2", "E3"}
        assert result.etapes_automatisees == []
        assert result.etapes_conservees_humaines == []

    def test_resume_transformation_est_un_objet_valide(self, diagnostic_minimal, workflow_minimal):
        """Le retour de compute_transformation doit être un ResumeTransformation Pydantic valide."""
        result = compute_transformation(diagnostic_minimal, workflow_minimal)
        assert isinstance(result, ResumeTransformation)

# =============================================================================
# T13 — code_generator.py : Validation de l'Abstract Syntax Tree (AST)
# =============================================================================

class TestASTValidation:
    def test_generate_airflow_dag_ast_validity(self, workflow_minimal):
        """Parse le script généré avec l'AST Python pour garantir l'absence d'erreurs de syntaxe."""
        code_str = generate_airflow_dag(workflow_minimal)
        try:
            ast.parse(code_str)
            ast_is_valid = True
        except SyntaxError as e:
            ast_is_valid = False
            pytest.fail(f"Syntaxe Python invalide générée : {e}")
        assert ast_is_valid

# =============================================================================
# T14 — telemetry.py : Suivi de la consommation d'inférence
# =============================================================================

class TestMetricsTracker:
    def test_record_and_summary(self):
        """Valide l'agrégation mathématique des latences et des tokens."""
        tracker = MetricsTracker()

        # Mock de l'objet usage_metadata retourné par l'API
        mock_meta_1 = MagicMock(prompt_token_count=100, candidates_token_count=50, total_token_count=150)
        mock_meta_2 = MagicMock(prompt_token_count=200, candidates_token_count=100, total_token_count=300)

        tracker.record("AgentA", 1.5, mock_meta_1)
        tracker.record("AgentB", 2.0, mock_meta_2)

        summary = tracker.get_summary()

        assert summary["total_latency_seconds"] == 3.5
        assert summary["total_prompt_tokens"] == 300
        assert summary["total_completion_tokens"] == 150
        assert summary["total_tokens"] == 450
        assert len(summary["details"]) == 2
        assert summary["details"][0]["agent_name"] == "AgentA"
