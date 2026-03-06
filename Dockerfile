# Utilise une image Python légère
FROM python:3.11-slim

# Évite la création de fichiers .pyc et force l'affichage des logs en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système (Graphviz est critique pour workflow_viz.py)
RUN apt-get update && apt-get install -y \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Définit le dossier de travail
WORKDIR /app

# Copie et installe les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du code
COPY . .

# Hugging Face utilise le port 7860 pour les containers Docker
EXPOSE 7860

# Commande pour lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
