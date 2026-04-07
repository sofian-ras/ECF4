# fake news nlp — détection automatique de désinformation

pipeline nlp complet : tf-idf · tensorflow · fastapi

---

## prérequis

- python 3.10+
- le fichier `news.csv` doit se trouver dans `data/news.csv`

---

## installation

```bash
# 1. créer et activer l'environnement virtuel
python -m venv .venv
# windows
.venv\Scripts\activate
# mac/linux
source .venv/bin/activate

# 2. installer les dépendances
pip install -r requirements.txt

# 3. télécharger le modèle spacy
python -m spacy download en_core_web_sm

# 4. télécharger les ressources nltk (à la première exécution du notebook)
# les cellules du notebook s'en chargent automatiquement
```

---

## lancer le notebook

```bash
jupyter notebook notebook/ecf_fake_news.ipynb
```

exécuter toutes les cellules dans l'ordre. à la fin, les fichiers suivants seront générés :
- `models/best_model.keras`
- `models/vectorizer.pkl`
- `data/titles_clean.csv`

---

## lancer l'api

après avoir exécuté le notebook :

```bash
uvicorn api.main:app --reload
```

l'api est accessible sur `http://localhost:8000`  
la documentation interactive est sur `http://localhost:8000/docs`

---

## endpoints disponibles

| méthode | route | description |
|---|---|---|
| GET | `/health` | vérifie que l'api est opérationnelle |
| POST | `/predict` | prédit la classe d'un titre |
| POST | `/predict/batch` | prédit la classe d'une liste de titres |

### exemple `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Scientists discover new treatment for common disease"}'
```

réponse :
```json
{
  "title": "Scientists discover new treatment for common disease",
  "label": "REAL",
  "confidence": 0.87
}
```

---

## structure du projet

```
fake_news_nlp/
├── notebook/
│   └── ecf_fake_news.ipynb
├── api/
│   └── main.py
├── models/
│   ├── best_model.keras
│   └── vectorizer.pkl
├── data/
│   └── titles_clean.csv
├── .env
├── .gitignore
├── requirements.txt
├── README.md
└── skills.md
```
