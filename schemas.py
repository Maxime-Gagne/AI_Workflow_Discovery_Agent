from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator

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
    temps_economise_pct: str = Field(..., description="Estimation du temps économisé (ex: 40-60%)")
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
    type_noeud: Literal["trigger", "automatique", "humain", "decision", "fin"]
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
