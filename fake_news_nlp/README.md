# fake news nlp - detection automatique de desinformation

Pipeline NLP complet: TF-IDF + TensorFlow + FastAPI.

---

## Prerequis

- Python 3.10+
- Le fichier `data/news.csv`
- (Recommande) environnement virtuel `.venv`

---

## Installation

Depuis `fake_news_nlp/`:

```bash
# 1) creer l'environnement virtuel
python -m venv .venv

# 2) activer
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

# 3) installer les dependances
pip install -r requirements.txt

# 4) modele spaCy
python -m spacy download en_core_web_sm
```

---

## Entrainement via notebook

```bash
jupyter notebook notebook/ecf_fake_news.ipynb
```

Executer les cellules dans l'ordre pour generer:

- `models/best_model.keras`
- `models/vectorizer.pkl`
- `models/best_model_lstm.keras`
- `data/titles_clean.csv`

---

## Lancer l'API en local

Deux options valides:

### Option A (depuis `fake_news_nlp/`)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option B (depuis `fake_news_nlp/api/`)

```bash
python main.py
```

Ensuite:

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Endpoints

| Methode | Route | Description |
|---|---|---|
| GET | `/health` | verification de disponibilite API/modele |
| POST | `/predict` | prediction d'un titre |
| POST | `/predict/batch` | prediction d'une liste de titres |

---

## Test detaille dans Swagger (`/docs`)

### 1) Sante API

- Ouvrir `GET /health`
- Cliquer `Try it out` puis `Execute`
- Attendu: code `200` et `{"status":"ok", ...}`

### 2) Prediction unitaire

`POST /predict` -> body:

```json
{
  "title": "Scientists discover new treatment for common disease"
}
```

Attendu:

- code `200`
- `label` dans `REAL` ou `FAKE`
- `confidence` entre `0` et `1`

### 3) Prediction batch

`POST /predict/batch` -> body:

```json
{
  "titles": [
    "Central bank raises interest rates by 0.25 points",
    "You wont believe this miracle cure",
    "Parliament votes on new environmental law"
  ]
}
```

Attendu: code `200` + une reponse par titre.

### 4) Validation des erreurs (important)

- `/predict` avec titre vide -> `422`
- `/predict` avec titre > 300 caracteres -> `400`
- `/predict/batch` avec liste vide -> `400`
- `/predict/batch` avec plus de 50 titres -> `400`

---

## Exemple cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"title":"Scientists discover new treatment for common disease"}'
```

---

## Variables d'environnement (optionnel)

Le backend lit:

- `MODEL_PATH` (defaut: `../models/best_model.keras`)
- `VECTORIZER_PATH` (defaut: `../models/vectorizer.pkl`)

Tu peux les definir dans `.env` si tu changes l'arborescence.

---

## Structure projet

```text
fake_news_nlp/
├── api/
│   └── main.py
├── data/
│   ├── news.csv
│   └── titles_clean.csv
├── models/
│   ├── best_model.keras
│   ├── best_model_lstm.keras
│   └── vectorizer.pkl
├── notebook/
│   └── ecf_fake_news.ipynb
├── requirements.txt
├── README.md
└── skills.md
```