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

## 📦 Dépendances principales

- pandas
- scikit-learn
- numpy
- pyyaml
