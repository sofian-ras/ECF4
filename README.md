# ECF4 - Fake News NLP

Projet principal: `fake_news_nlp` (NLP + TensorFlow + FastAPI).

## Setup rapide

Depuis la racine `ECF4`:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r fake_news_nlp/requirements.txt
python -m spacy download en_core_web_sm
```

## Notebook (entrainement)

```bash
jupyter notebook fake_news_nlp/notebook/ecf_fake_news.ipynb
```

Fichiers generes:
- `fake_news_nlp/models/best_model.keras`
- `fake_news_nlp/models/vectorizer.pkl`
- `fake_news_nlp/models/best_model_lstm.keras`
- `fake_news_nlp/data/titles_clean.csv`

## API locale

Option 1 (depuis `ECF4`):

```bash
uvicorn fake_news_nlp.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Option 2 (depuis `fake_news_nlp/api`):

```bash
python main.py
```

URLs:
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`
- `http://localhost:8000/health`

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /predict/batch`

## Notes Git

Le repo ignore maintenant globalement:
- `.venv/`
- `.claude/`
- caches Python/Jupyter
- artefacts `fake_news_nlp/models/*` et `fake_news_nlp/data/*.csv`
