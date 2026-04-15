# NYC Taxi Trip Duration Prediction

Projet de ML pour la prédiction de la durée des trajets en taxi à New York City.

## 📋 Description

Ce projet vise à construire un modèle de machine learning capable de prédire la durée d'un trajet en taxi à NYC en fonction de différentes caractéristiques comme la localisation de départ/arrivée, la date/heure, etc.



## 🚀 Installation

### Créer et activer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

## 📊 Utilisation

### 1. Prétraitement des données

```bash
python src/data/preprocessing.py
```

### 2. Entraînement du modèle

```bash
python src/model/training.py
```

### 3. Inférence

```bash
python src/model/inference.py
```

## 🌐 Lancer l'API et le Dashboard

### Prérequis

Assure-toi d'avoir activé le venv et installé les dépendances (voir section Installation).
Le modèle `models/taxi_model.pkl` doit exister (sinon lance l'entraînement d'abord).

### 1. Lancer l'API FastAPI

**Option A — via le script bash :**
```bash
bash run_api.sh
```

**Option B — manuellement :**
```bash
source venv/bin/activate
cd api
python main.py
```

L'API est disponible sur :
- Endpoints : http://localhost:8000
- Documentation interactive (Swagger) : http://localhost:8000/docs

### 2. Lancer le Dashboard Streamlit

Dans un **nouveau terminal** (l'API doit rester en marche) :

```bash
source venv/bin/activate
streamlit run app.py
```

Le dashboard s'ouvre automatiquement sur http://localhost:8501

### Ordre de lancement

```
Terminal 1                        Terminal 2
──────────────────────────────    ──────────────────────────────
bash run_api.sh                   streamlit run app.py
→ API sur localhost:8000          → Dashboard sur localhost:8501
```

> Le dashboard appelle l'API pour les prédictions. Si l'API n'est pas lancée, les pages "Prédiction Unique" et "Batch Upload" afficheront une erreur de connexion.

## 📦 Dépendances principales

- pandas
- scikit-learn
- numpy
- pyyaml
- fastapi / uvicorn
- streamlit
- sqlalchemy
