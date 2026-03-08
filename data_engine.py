import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any, Tuple, Optional

TEXT_COLUMN_CANDIDATES = ['content', 'action', 'etape', 'category', 'description', 'titre', 'nom', 'label']
DURATION_COLUMN_CANDIDATES = ['duration_minutes', 'duree_moyenne_jours', 'resolution_time_hours', 'duree_estimee']
START_COLUMN_CANDIDATES = ['timestamp', 'timestamp_start', 'created', 'date_debut']
END_COLUMN_CANDIDATES = ['timestamp_end', 'resolved', 'date_fin']

class DataEngine:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialise le moteur avec un modèle d'embedding léger exécutable sur CPU.
        all-MiniLM-L6-v2 génère des vecteurs de 384 dimensions.
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.current_embeddings = None

    @staticmethod
    def _extract_list(raw_data: Any) -> List[Dict[str, Any]]:
        """
        Détecte automatiquement la liste principale dans un JSON hétérogène.
        Parcourt les valeurs racine et retourne la première liste de dicts trouvée.
        """
        if isinstance(raw_data, list):
            return raw_data
        if isinstance(raw_data, dict):
            for value in raw_data.values():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    return value
        return []

    def normalize(self, raw_data: Any, mapping_config: Dict[str, str] = {}) -> pd.DataFrame:
        """
        Transforme les données hétérogènes en DataFrame selon le Schéma Pivot.
        Détecte automatiquement les colonnes texte et durée.
        """
        if mapping_config is None:
            mapping_config = {}

        records = self._extract_list(raw_data)
        if not records:
            raise ValueError("Aucune liste de données détectable dans le JSON fourni.")

        df = pd.DataFrame(records)

        # Renommage optionnel si mapping fourni
        if mapping_config:
            df = df.rename(columns=mapping_config)

        # Détection automatique colonne texte → renommée 'content'
        if 'content' not in df.columns:
            for candidate in TEXT_COLUMN_CANDIDATES:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: 'content'})
                    break

        if 'content' not in df.columns:
            raise ValueError(f"Aucune colonne texte détectée. Colonnes disponibles : {list(df.columns)}")

        # Détection automatique durée
        if 'duration_minutes' not in df.columns:
            for candidate in DURATION_COLUMN_CANDIDATES:
                if candidate in df.columns:
                    df['duration_minutes'] = pd.to_numeric(df[candidate], errors='coerce')
                    break

        # Calcul durée depuis timestamps si disponibles
        start_col = next((c for c in START_COLUMN_CANDIDATES if c in df.columns), None)
        end_col = next((c for c in END_COLUMN_CANDIDATES if c in df.columns), None)
        if start_col and end_col:
            df['timestamp_start'] = pd.to_datetime(df[start_col], errors='coerce')
            df['timestamp_end'] = pd.to_datetime(df[end_col], errors='coerce')
            mask = df['timestamp_start'].notna() & df['timestamp_end'].notna()
            df.loc[mask, 'duration_minutes'] = (
                df.loc[mask, 'timestamp_end'] - df.loc[mask, 'timestamp_start']
            ).dt.total_seconds() / 60.0

        if 'timestamp' in df.columns and 'duration_minutes' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.sort_values('timestamp')
            # Calcul de la différence entre l'action actuelle et la suivante
            df['duration_minutes'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60.0

        df = df.dropna(subset=['content']).reset_index(drop=True)
        return df

    def vectorize_and_cluster(self, df: pd.DataFrame, min_cluster_size: Optional[int] = None, epsilon: float = 0.5) -> pd.DataFrame:
        textes = df['content'].tolist()
        embeddings = self.encoder.encode(textes, show_progress_bar=False)

        n_samples = len(df)
        if min_cluster_size is None:
            # Heuristique logarithmique pour ajustement dynamique de la densité
            # Prévient la fragmentation excessive sur de larges datasets
            min_cluster_size = max(2, int(np.log(n_samples) * 1.5)) if n_samples > 0 else 2

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_epsilon=epsilon,
            copy=True  # <-- AJOUT : Rend le code compatible avec Scikit-Learn 1.10+
        )
        df['cluster_id'] = clusterer.fit_predict(embeddings)
        self.current_embeddings = embeddings

        return df

    def extract_top_representatives(self, df_cluster: pd.DataFrame, cluster_idx: int) -> List[str]:
        """
        Trouve les 3 textes les plus proches du centre mathématique du cluster.
        """
        idx_in_cluster = df_cluster.index.tolist()
        cluster_embeddings = self.current_embeddings[idx_in_cluster]

        if len(cluster_embeddings) <= 3:
            return df_cluster['content'].tolist()

        # Calcul du centroïde (moyenne vectorielle du cluster)
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

        # Distances cosinus entre les points du cluster et le centroïde
        distances = cosine_distances(cluster_embeddings, centroid).flatten()

        # Récupère les index des 3 textes les plus proches (distance minimale)
        top_3_idx_local = np.argsort(distances)[:3]

        representatives = [df_cluster.iloc[i]['content'] for i in top_3_idx_local]
        return representatives

    def generate_payload(self, df: pd.DataFrame, source_name: str) -> Dict[str, Any]:
        """
        Agrège les métriques mathématiques et génère le payload minimaliste pour le LLM.
        """
        payload = {
            "source": source_name,
            "metriques_globales": {
                "volume_total_lignes": len(df),
                "temps_total_analyse_minutes": df['duration_minutes'].sum() if 'duration_minutes' in df.columns else 0
            },
            "clusters_repetitifs": []
        }

        # Ignorer le cluster -1 qui représente le bruit non répétitif selon HDBSCAN
        valid_clusters = df[df['cluster_id'] != -1]
        grouped = valid_clusters.groupby('cluster_id')

        for cluster_id, group in grouped:
            frequence = len(group)
            temps_perdu = group['duration_minutes'].sum() if 'duration_minutes' in df.columns else None
            exemples = self.extract_top_representatives(group, cluster_id)

            cluster_data = {
                "cluster_id": f"C{cluster_id}",
                "frequence": frequence,
                "temps_total_perdu_minutes": temps_perdu,
                "exemples_representatifs": exemples
            }
            payload["clusters_repetitifs"].append(cluster_data)

        return payload

    def compute_roi(self, payload: dict, taux_horaire: float) -> dict:
        """
        Calcule les métriques ROI de manière déterministe à partir du payload clusterisé.
        Ce calcul est effectué AVANT l'inférence LLM pour éviter toute hallucination de chiffres.
        """
        clusters = payload.get("clusters_repetitifs", [])

        # Somme des minutes perdues sur tous les clusters répétitifs
        total_minutes_perdues = sum(
            c.get("temps_total_perdu_minutes") or 0
            for c in clusters
        )

        # Conversion en heures mensuelles (les données couvrent une période variable)
        heures_perdues_mois = round(total_minutes_perdues / 60, 2)

        # Calculs financiers déterministes
        economies_mensuelles = round(heures_perdues_mois * taux_horaire, 2)
        projection_annuelle = round(economies_mensuelles * 12, 2)

        return {
            "taux_horaire": taux_horaire,
            "heures_perdues_par_mois": heures_perdues_mois,
            "economies_mensuelles": economies_mensuelles,
            "projection_annuelle": projection_annuelle,
            "detail_calcul": f"{heures_perdues_mois}h × {taux_horaire}$/h = {economies_mensuelles}$/mois → {projection_annuelle}$/an"
        }

    def process_pipeline(
        self,
        raw_data: Any,
        mapping_config: Dict[str, str] = None,
        source_name: str = "",
        taux_horaire: float = 25.0,
        min_cluster_size: Optional[int] = None,
        epsilon: float = 0.5
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        if mapping_config is None:
            mapping_config = {}

        df_norm = self.normalize(raw_data, mapping_config)
        df_clustered = self.vectorize_and_cluster(df_norm, min_cluster_size, epsilon)
        payload = self.generate_payload(df_clustered, source_name)
        roi = self.compute_roi(payload, taux_horaire)

        return payload, roi
