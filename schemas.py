from __future__ import annotations
from typing import List, Optional, Literal, Dict, Union
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal, Optional, Union


ConfidenceLevel = Literal["provided", "observed", "estimated", "missing"]


class TimeMetric(BaseModel):
    """
    Représente une métrique temporelle avec son niveau de confiance.
    """
    name: str = Field(..., description="Nom canonique de la métrique")
    unit: str = Field(..., description="Unité de mesure, ex: minutes, hours, cases/month")
    value: Optional[Union[float, int, str]] = Field(
        default=None,
        description="Valeur de la métrique si disponible"
    )
    confidence_level: ConfidenceLevel = Field(
        ...,
        description="Niveau de confiance de la métrique"
    )
    source: str = Field(
        ...,
        description="Origine de la métrique: user_input, event_log, pm4py, rule_engine, etc."
    )
    hypothesis: Optional[str] = Field(
        default=None,
        description="Hypothèse explicite si la métrique est dérivée"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Remarques, limites, précisions"
    )

    @field_validator("name", "unit", "source")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Champ texte vide ou invalide.")
        return v.strip()

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, v: list[str]) -> list[str]:
        cleaned = []
        for note in v:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("Une note est vide ou invalide.")
            cleaned.append(note.strip())
        return cleaned

    @model_validator(mode="after")
    def validate_consistency(self) -> "TimeMetric":
        if self.confidence_level == "missing":
            if self.value is not None:
                raise ValueError("Une métrique 'missing' ne doit pas avoir de value.")
        else:
            if self.value is None:
                raise ValueError("Une métrique non 'missing' doit avoir une value.")

        if self.confidence_level == "estimated":
            if self.hypothesis is None or not self.hypothesis.strip():
                raise ValueError("Une métrique 'estimated' doit documenter une hypothèse.")

        return self


class TimeMetricsReport(BaseModel):
    """
    Rapport structuré sur les métriques temporelles critiques du pipeline.
    """
    monthly_volume: TimeMetric
    activity_duration_minutes: TimeMetric
    case_cycle_time_minutes: TimeMetric
    manual_time_minutes: TimeMetric
    waiting_time_minutes: TimeMetric

    warnings: list[str] = Field(default_factory=list)
    can_compute_full_roi: bool = Field(
        ...,
        description="True si les métriques critiques sont suffisamment fiables pour un ROI complet"
    )
    can_compute_partial_roi: bool = Field(
        ...,
        description="True si un ROI partiel est possible avec avertissements"
    )

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, v: list[str]) -> list[str]:
        cleaned = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Un warning est vide ou invalide.")
            cleaned.append(item.strip())
        return cleaned

    @model_validator(mode="after")
    def validate_roi_flags(self) -> "TimeMetricsReport":
        critical_metrics = [
            self.monthly_volume,
            self.activity_duration_minutes,
            self.manual_time_minutes,
        ]

        levels = [m.confidence_level for m in critical_metrics]

        strong_ok = all(level in {"provided", "observed"} for level in levels)
        partial_ok = all(level in {"provided", "observed", "estimated"} for level in levels) and not any(
            level == "missing" for level in levels
        )

        if self.can_compute_full_roi and not strong_ok:
            raise ValueError(
                "can_compute_full_roi=True incohérent avec les niveaux de confiance critiques."
            )

        if self.can_compute_partial_roi and not partial_ok:
            raise ValueError(
                "can_compute_partial_roi=True incohérent avec les niveaux de confiance critiques."
            )

        if self.can_compute_full_roi and not self.can_compute_partial_roi:
            raise ValueError(
                "Un ROI complet implique nécessairement qu'un ROI partiel soit possible."
            )

        return self

# ==========================================
# 0. Analyse d'une source de données
# ==========================================

class SourceQualityReport(BaseModel):
    is_processable: bool = Field(description="Si les données peuvent être converties en workflow.")
    detected_format: str = Field(description="Ex: 'Log de chat', 'Export CRM', 'SOP textuelle'.")
    transformation_applied: bool = Field(description="Indique si une restructuration lourde a été nécessaire.")
    structured_data: List[Dict] = Field(description="La donnée finale normalisée (1 ligne = 1 action atomique).")
    explanation: str = Field(description="Message clair pour l'utilisateur sur le traitement effectué.")

class DataRefinementOutput(BaseModel):
    is_structurable: bool = Field(description="Si faux, le processus s'arrête ici avec une explication.")
    structured_data: List[Dict] = Field(description="La donnée convertie au format atomique (1 ligne = 1 action).")
    transformation_log: str = Field(description="Explication de ce qui a été modifié pour l'utilisateur.")
    confidence_score: float = Field(description="Niveau de certitude de la conversion.")
# ==========================================
# 1. Contrats de l'Agent Analyste
# ==========================================

class EtapeActuelle(BaseModel):
    id: str = Field(..., description="Identifiant unique de l'étape (ex: E1)")
    nom: str = Field(..., description="Nom court de l'action")
    type_etape: Literal["manuelle", "automatisable", "semi-automatisable"] = Field(..., description="Classification de l'étape")
    acteur: str = Field(..., description="Rôle ou système exécutant la tâche")
    duree_estimee: str = Field(..., description="Durée estimée (ex: 30min, 2h)")
    volume: str = Field(..., description="Fréquence d'exécution")
    friction: Optional[str] = Field(None, description="Problème principal ou goulot d'étranglement détecté")

class MetriquesCles(BaseModel):
    volume_total: str
    duree_totale_estimee: str
    taux_manuel: str = Field(..., description="Pourcentage d'étapes manuelles")
    principale_friction: str

class GainsEstimes(BaseModel):
    heures_economisees_par_mois: float = Field(..., description="Estimation mathématique stricte des heures économisées mensuellement")
    economies_mensuelles_devise: float = Field(..., description="Économies brutes (heures_economisees_par_mois * taux_horaire fourni)")
    projection_annuelle: float = Field(..., description="Projection sur 12 mois : economies_mensuelles_devise * 12")
    mois_retour_investissement: Optional[float] = Field(None, description="Coût estimé implémentation / economies_mensuelles_devise. Null si non calculable.")
    detail_du_calcul: str = Field(..., description="Formule mathématique courte justifiant le calcul du ROI")
    principales_opportunites: List[str] = Field(..., description="Liste de 3 opportunités majeures")

class DiagnosticAnalyste(BaseModel):
    titre_processus: str = Field(..., description="Nom court du processus détecté")
    description: str = Field(..., description="Résumé en 2 phrases du processus actuel")
    etapes_actuelles: List[EtapeActuelle]
    metriques_cles: MetriquesCles
    potentiel_automatisation: Literal["faible", "moyen", "élevé", "très élevé"]
    gains_estimes: GainsEstimes

# ==========================================
# 2. Contrats de l'Agent Mapper
# ==========================================

class NoeudWorkflow(BaseModel):
    id: str = Field(..., description="Identifiant du noeud (ex: N1)")
    label: str = Field(..., description="Nom court du noeud (max 5 mots)")
    type_noeud: Literal["trigger", "automatique", "humain", "decision", "fin", "ia"]
    description: str = Field(..., description="Description de l'action exécutée dans ce noeud")
    etape_originale: Optional[str] = Field(None, description="Référence à l'ID de l'étape manuelle remplacée, si applicable")

class LienWorkflow(BaseModel):
    de: str = Field(..., description="ID du noeud source")
    vers: str = Field(..., description="ID du noeud de destination")
    condition: Optional[str] = Field(None, description="Condition logique de transition (ex: si priorité = critique). Null si toujours vrai.")

class ResumeTransformation(BaseModel):
    etapes_eliminees: List[str]
    etapes_automatisees: List[str]
    etapes_conservees_humaines: List[str]

    @model_validator(mode='after')
    def check_not_all_empty(self) -> 'ResumeTransformation':
        total = (len(self.etapes_eliminees) +
                 len(self.etapes_automatisees) +
                 len(self.etapes_conservees_humaines))
        if total == 0:
            raise ValueError("Au moins une étape doit être classifiée dans la transformation.")
        return self
class WorkflowOptimise(BaseModel):
    titre_workflow: str
    description_transformation: str = Field(..., description="Résumé en 2 phrases du changement de paradigme architectural")
    noeuds: List[NoeudWorkflow]
    liens: List[LienWorkflow]

# ==========================================
# 3. Contrats de l'Agent Advisor
# ==========================================

class OutilRecommande(BaseModel):
    nom: str = Field(..., description="Nom exact de l'outil dans le catalogue")
    justification: str = Field(..., description="Pourquoi cet outil est optimal pour ce noeud (1 phrase)")
    complexite: Literal["faible", "moyen", "élevé"]
    priorite: Literal["essentiel", "recommandé", "optionnel"]

class RecommandationNoeud(BaseModel):
    noeud_id: str
    noeud_label: str
    outils: List[OutilRecommande]

class RapportAdvisor(BaseModel):
    recommandations: List[RecommandationNoeud]
    stack_minimale: List[str] = Field(..., description="Liste des outils strictement essentiels pour un MVP")
    stack_complete: List[str] = Field(..., description="Liste de tous les outils pour la solution finale")
    conseil_implementation: str = Field(..., description="Conseil stratégique par où commencer (2-3 phrases)")

# ==========================================

