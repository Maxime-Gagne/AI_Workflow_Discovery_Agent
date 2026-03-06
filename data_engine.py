import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Dict, Any

class DataEngine:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialise le moteur avec un modèle d'embedding léger exécutable sur CPU.
        all-MiniLM-L6-v2 génère des vecteurs de 384 dimensions.
        """
        self.encoder = SentenceTransformer(embedding_model)

    def normalize(self, raw_data: List[Dict[str, Any]], mapping_config: Dict[str, str]) -> pd.DataFrame:
        """
        Transforme les données hétérogènes en DataFrame selon le Schéma Pivot.
        Calcule la durée de manière déterministe.
        """
        df = pd.DataFrame(raw_data)

        # Renommage des colonnes selon le mapping fourni par l'interface
        df = df.rename(columns=mapping_config)

        # Vérification des colonnes requises
        required_cols = ['id', 'content', 'timestamp_start', 'timestamp_end']
        for col in required_cols:
            if col not in df.columns and col != 'duration_minutes':
                # Si les timestamps n'existent pas, on cherche une colonne durée pré-existante
                pass

        # Calcul déterministe de la durée si les timestamps sont présents
        if 'timestamp_start' in df.columns and 'timestamp_end' in df.columns:
            df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])
            df['timestamp_end'] = pd.to_datetime(df['timestamp_end'])
            df['duration_minutes'] = (df['timestamp_end'] - df['timestamp_start']).dt.total_seconds() / 60.0

        # Nettoyage des valeurs nulles pour le texte
        df = df.dropna(subset=['content']).reset_index(drop=True)
        return df

    def vectorize_and_cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les embeddings et applique HDBSCAN pour identifier les densités sémantiques.
        """
        # 1. Vectorisation
        textes = df['content'].tolist()
        embeddings = self.encoder.encode(textes, show_progress_bar=False)

        # 2. Clustering HDBSCAN (isole le bruit dans le cluster -1)
        # min_cluster_size est ajustable selon le volume de données (ici configuré pour de petits datasets)
        clusterer = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5)
        df['cluster_id'] = clusterer.fit_predict(embeddings)

        # Sauvegarde des embeddings pour le calcul des centroïdes plus tard
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

    def process_pipeline(self, raw_data: List[Dict[str, Any]], mapping_config: Dict[str, str], source_name: str) -> Dict[str, Any]:
        """
        Orchestre l'ensemble du pipeline de données.
        """
        df_norm = self.normalize(raw_data, mapping_config)
        df_clustered = self.vectorize_and_cluster(df_norm)
        return self.generate_payload(df_clustered, source_name)
