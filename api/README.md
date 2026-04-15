# NYC Taxi Trip Duration Prediction API

API FastAPI pour prédire la durée des trajets en taxi à NYC

## Installation

```bash
# Activer le venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Lancer l'API

```bash
# Depuis le répertoire racine du projet
cd api
python main.py
```

L'API sera disponible à: `http://localhost:8000`

## Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### 1. Health Check

```
GET /
```

Vérifie que l'API fonctionne

### 2. Prédiction unique

```
POST /predict
```

**Request body:**

```json
{
   "pickup_datetime": "2016-03-14 17:24:55",
   "pickup_latitude": 40.7614,
   "pickup_longitude": -73.9776,
   "dropoff_latitude": 40.7489,
   "dropoff_longitude": -73.968,
   "passenger_count": 1,
   "vendor_id": 1
}
```

**Response:**

```json
{
  "prediction_id": 1,
  "predicted_duration_seconds": 900.5,
  "predicted_duration_minutes": 15.0,
  "input_data": {...},
  "timestamp": "2024-04-15T10:30:00"
}
```

### 3. Prédictions par lot

```
POST /predict-batch
```

Envoyer une liste de demandes de prédiction

### 4. Statistiques

```
GET /stats
```

Retourne les statistiques globales des prédictions

### 5. Historique

```
GET /history?limit=10
```

Retourne les N dernières prédictions

## Base de données

Les prédictions sont stockées dans `predictions.db` (SQLite)

Structure:

- `id`: ID unique de la prédiction
- `timestamp`: Heure de la prédiction
- `pickup_datetime`: Heure de départ du trajet
- `pickup_latitude`, `pickup_longitude`: Coordonnées de départ
- `dropoff_latitude`, `dropoff_longitude`: Coordonnées d'arrivée
- `passenger_count`: Nombre de passagers
- `vendor_id`: ID du vendeur
- `predicted_duration_seconds`: Durée prédite en secondes

## Testing

### Avec curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2016-03-14 17:24:55",
    "pickup_latitude": 40.7614,
    "pickup_longitude": -73.9776,
    "dropoff_latitude": 40.7489,
    "dropoff_longitude": -73.9680,
    "passenger_count": 1,
    "vendor_id": 1
  }'
```

### Avec Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "pickup_datetime": "2016-03-14 17:24:55",
    "pickup_latitude": 40.7614,
    "pickup_longitude": -73.9776,
    "dropoff_latitude": 40.7489,
    "dropoff_longitude": -73.9680,
    "passenger_count": 1,
    "vendor_id": 1
}

response = requests.post(url, json=payload)
print(response.json())
```

## Architecture

```
api/
├── main.py          # Application FastAPI principale
├── schemas.py       # Schémas Pydantic (input/output)
├── database.py      # Configuration SQLAlchemy
└── __init__.py      # Package init
```

## Fonctionnalités

✅ Prédiction simple ou par lot
✅ Validation des entrées avec Pydantic
✅ Persistance des prédictions en SQLite
✅ Statistiques et historique
✅ Documentation Swagger/ReDoc
✅ CORS middleware pour cross-origin requests
