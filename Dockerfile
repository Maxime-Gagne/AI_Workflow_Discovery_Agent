FROM python:3.11-slim

# Désactivation du bytecode et du buffer de sortie standard
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configuration stricte des répertoires de cache pour prévenir les exceptions PermissionError (Errno 13)
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache

# Installation de la dépendance système pour workflow_viz.py
RUN apt-get update && apt-get install -y \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Création du groupe et de l'utilisateur non-privilégié (UID/GID 1000)
RUN addgroup --system --gid 1000 appgroup && \
    adduser --system --uid 1000 --gid 1000 --home /app --no-create-home appuser

WORKDIR /app

# Initialisation de la structure de cache et attribution de la propriété
RUN mkdir -p /app/.cache && chown -R appuser:appgroup /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source avec attribution immédiate des droits
COPY --chown=appuser:appgroup . .

# 1. Bascule sur l'utilisateur restreint AVANT de télécharger le modèle
USER appuser

# 2. Le pré-téléchargement s'exécute maintenant avec les bons droits (UID 1000)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
