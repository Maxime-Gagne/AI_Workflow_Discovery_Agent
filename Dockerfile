# Utilise une image Python légère
FROM python:3.10-slim

# Installe Graphviz (requis pour tes schémas de workflow)
RUN apt-get update && apt-get install -y \
    graphviz \
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
