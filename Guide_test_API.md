# Guide de test - ECF4 Fake News NLP

Ce guide permet de valider rapidement l'API en local via Swagger.

## 1) Demarrage

Depuis la racine du projet `ECF4`:

```bash
.\.venv\Scripts\Activate.ps1
uvicorn fake_news_nlp.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Ouvrir ensuite:
- Swagger: http://127.0.0.1:8000/docs

---

## 2) Endpoints a verifier

### A. GET /health
- Action: `Try it out` -> `Execute`
- Attendu:
  - HTTP `200`
  - Body: `{"status":"ok","model":"fake_news_detector"}`

### B. POST /predict
- Body exemple:

```json
{
  "title": "Scientists discover new treatment for common disease"
}
```

- Attendu:
  - HTTP `200`
  - Reponse contenant `title`, `label`, `confidence`

### C. POST /predict/batch
- Body exemple:

```json
{
  "titles": [
    "Central bank raises interest rates by 0.25 points",
    "You won't believe what this politician did last night",
    "Doctors don't want you to know this secret remedy"
  ]
}
```

- Attendu:
  - HTTP `200`
  - Liste de predictions (1 resultat par titre)

---

## 3) Cas limites (validation)

### /predict - titre vide
```json
{"title":"   "}
```
Attendu: HTTP `422`

### /predict - titre trop long (>300)
- Utiliser un texte de plus de 300 caracteres
- Attendu: HTTP `400`

### /predict/batch - liste vide
```json
{"titles":[]}
```
Attendu: HTTP `400`

### /predict/batch - plus de 50 titres
- Envoyer une liste de 51 titres
- Attendu: HTTP `400`

---

## 4) Livrables a controler

- Notebook principal: `fake_news_nlp/notebook/ecf_fake_news.ipynb`
- API: `fake_news_nlp/api/main.py`
- Modele: `fake_news_nlp/models/best_model.keras`
- Vectoriseur: `fake_news_nlp/models/vectorizer.pkl`
- Donnees nettoyees: `fake_news_nlp/data/titles_clean.csv`
