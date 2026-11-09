# Image de base : Python 3.13 allégé (la version utilisée en développement et en CI)
FROM python:3.13-slim-bookworm

# Bibliothèques système exigées par OpenCV (cv2 ne s'importe pas sans elles)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Les dépendances d'abord : cette couche, la plus lourde (PyTorch), est mise en cache
# par Docker tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le code ensuite (le dataset et les modèles sont exclus par .dockerignore)
COPY . .

# Pas d'écran dans un conteneur : matplotlib n'ouvre pas de fenêtre
ENV MPLBACKEND=Agg

# Par défaut : l'évaluation complète (le dossier data/ doit être monté, voir le README)
CMD ["python", "evaluate.py"]
