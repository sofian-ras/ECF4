"""
api de détection de fake news.
charge le modèle et le vectoriseur une seule fois au démarrage.
endpoints : /health, /predict, /predict/batch
"""

import os
import re
from pathlib import Path

import joblib
import numpy as np
import spacy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from pydantic import BaseModel

import nltk
nltk.download("stopwords", quiet=True)

load_dotenv()

API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent


def resolve_artifact_path(raw_value: str | None, default_rel_path: str) -> Path:
    value = raw_value or default_rel_path
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [
        (PROJECT_DIR / path).resolve(),
        (API_DIR / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


MODEL_PATH = resolve_artifact_path(os.getenv("MODEL_PATH"), "models/best_model.keras")
VECTORIZER_PATH = resolve_artifact_path(os.getenv("VECTORIZER_PATH"), "models/vectorizer.pkl")

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(MODEL_PATH))
    vectorizer = joblib.load(str(VECTORIZER_PATH))
    nlp = spacy.load("en_core_web_sm")
    print("modèle et vectoriseur chargés avec succès.")
except Exception as e:
    print(f"erreur au chargement : {e}")
    model = None
    vectorizer = None
    nlp = None

# pipeline de nettoyage (identique au notebook)

negation_words = {"not", "no", "never", "neither"}
stop_words = set(stopwords.words("english")) - negation_words

contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "i've": "i have",
    "i'll": "i will",
    "i'd": "i would",
    "they've": "they have",
    "we've": "we have",
    "you've": "you have",
    "he's": "he is",
    "she's": "she is",
}


def expand_contractions(text: str) -> str:
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    return text


def clean_title(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = expand_contractions(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    lemmas = [t for t in lemmas if len(t) >= 2]
    return " ".join(lemmas)


def predict_one(title: str) -> dict:
    """nettoie, vectorise et prédit la classe d'un titre."""
    cleaned = clean_title(title)
    vector = vectorizer.transform([cleaned]).toarray()
    score = float(model.predict(vector, verbose=0).flatten()[0])
    label = "REAL" if score >= 0.5 else "FAKE"
    confidence = round(score if score >= 0.5 else 1 - score, 4)
    return {"title": title, "label": label, "confidence": confidence}

# schémas pydantic


class PredictRequest(BaseModel):
    title: str

    model_config = {
        "json_schema_extra": {
            "example": {"title": "scientists discover new treatment for common disease"}
        }
    }


class PredictResponse(BaseModel):
    title: str
    label: str
    confidence: float


class BatchRequest(BaseModel):
    titles: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "titles": [
                    "scientists discover new treatment",
                    "shocking: government hiding truth about water supply",
                ]
            }
        }
    }


# application fastapi

app = FastAPI(
    title="fake news detector",
    description="api de détection de désinformation dans les titres de presse.",
    version="1.0.0",
)


@app.get("/health", tags=["health"])
def health():
    """vérifie que l'api et le modèle sont opérationnels."""
    return {"status": "ok", "model": "fake_news_detector"}


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(payload: PredictRequest):
    """
    prédit si un titre est fake ou real.

    - retourne 422 si le titre est vide ou composé uniquement d'espaces
    - retourne 400 si le titre dépasse 300 caractères
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="modèle non disponible")

    title = payload.title

    if not title or not title.strip():
        raise HTTPException(
            status_code=422,
            detail="le titre ne peut pas être vide.",
        )

    if len(title) > 300:
        raise HTTPException(
            status_code=400,
            detail=f"le titre dépasse 300 caractères ({len(title)} caractères). veuillez le raccourcir.",
        )

    return predict_one(title)


@app.post("/predict/batch", response_model=list[PredictResponse], tags=["predict"])
def predict_batch(payload: BatchRequest):
    """
    prédit la classe d'une liste de titres.

    - retourne 400 si la liste est vide ou dépasse 50 titres
    - chaque titre est soumis aux mêmes validations que /predict
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="modèle non disponible")

    titles = payload.titles

    if len(titles) == 0:
        raise HTTPException(
            status_code=400,
            detail="la liste de titres est vide.",
        )

    if len(titles) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"la liste dépasse 50 titres ({len(titles)} reçus). maximum autorisé : 50.",
        )

    results = []
    for title in titles:
        if not title or not title.strip():
            raise HTTPException(
                status_code=422,
                detail=f"un titre de la liste est vide.",
            )
        if len(title) > 300:
            raise HTTPException(
                status_code=400,
                detail=f"un titre dépasse 300 caractères : '{title[:50]}...'",
            )
        results.append(predict_one(title))

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
