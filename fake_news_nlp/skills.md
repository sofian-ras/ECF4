# skills — guide de modification du projet fake news nlp

ce fichier décrit comment modifier les parties clés du projet sans tout casser.

---

## 1. ajouter ou modifier des contractions

**où :** `notebook/ecf_fake_news.ipynb` — cellule "2.1 pipeline de nettoyage"  
**et :** `api/main.py` — dictionnaire `contractions` (ligne ~60)

les deux fichiers partagent le même dictionnaire. si tu en modifies un, modifie l'autre pour rester cohérent.

```python
# exemple : ajouter une contraction
contractions = {
    ...
    "they'll": "they will",   # ← nouvelle entrée
}
```

**attention :** les contractions sont cherchées en minuscules après l'étape 1 (lowercase). inutile d'ajouter des variantes en majuscules.

---

## 2. changer les paramètres tf-idf

**où :** `notebook/ecf_fake_news.ipynb` — section 3.1 "vectorisation tf-idf"

```python
vectorizer = TfidfVectorizer(
    max_features=3000,    # ↑ plus de features = plus de mémoire
    min_df=2,             # ↓ moins = plus de mots rares inclus
    max_df=0.85,          # ↑ plus = mots fréquents inclus
    ngram_range=(1, 2),   # (1,3) pour inclure les trigrammes
    sublinear_tf=True,    # laisser True (atténue les mots très fréquents)
)
```

après modification :
1. relancer la cellule de vectorisation
2. relancer l'entraînement du modèle dense
3. re-sauvegarder le vectoriseur (`joblib.dump(...)`)
4. relancer l'api pour qu'elle charge le nouveau vectoriseur

---

## 3. changer l'architecture du modèle dense

**où :** `notebook/ecf_fake_news.ipynb` — section 4.1 — fonction `build_dense_model`

```python
def build_dense_model(input_dim: int) -> keras.Sequential:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),   # ← changer 256 par autre chose
        layers.Dropout(0.4),                    # ← entre 0.2 et 0.5
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # ← ne pas toucher (binaire)
    ])
    return model
```

règles :
- la dernière couche doit toujours être `Dense(1, activation="sigmoid")`
- la loss doit toujours être `binary_crossentropy`
- après modification, relancer l'entraînement et le modelcheckpoint

---

## 4. changer l'architecture bilstm

**où :** `notebook/ecf_fake_news.ipynb` — section 4.2 — fonction `build_lstm_model`

```python
def build_lstm_model(text_vec_layer) -> keras.Sequential:
    model = keras.Sequential([
        text_vec_layer,                          # ← ne pas toucher
        layers.Embedding(5000, 64, mask_zero=True),  # output_dim: 32/64/128
        layers.Bidirectional(
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
        ),                                       # units: 32/64/128
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model
```

si tu changes `max_tokens` dans `TextVectorization`, mets à jour `input_dim` dans `Embedding` avec la même valeur.

---

## 5. changer les hyperparamètres d'entraînement

**où :** cellules d'entraînement dans les sections 4.1 et 4.2

```python
history = model.fit(
    X_train_tfidf,
    y_train,
    epochs=30,           # ← nombre max d'epochs (earlystopping peut couper avant)
    validation_split=0.15,  # ← 15% du train utilisé pour la validation
    callbacks=callbacks,
    verbose=1,
)
```

pour `EarlyStopping` :
```python
EarlyStopping(
    patience=5,               # ← nb d'epochs sans amélioration avant arrêt
    restore_best_weights=True,  # ← toujours True
    monitor="val_loss",       # ← peut aussi monitorer "val_accuracy"
)
```

---

## 6. ajouter un endpoint à l'api

**où :** `api/main.py`

pattern à suivre (calqué sur les demos 30_API) :

```python
class NouveauRequest(BaseModel):
    champ: str

class NouveauResponse(BaseModel):
    champ: str
    resultat: str

@app.post("/nouveau-endpoint", response_model=NouveauResponse, tags=["predict"])
def nouveau_endpoint(payload: NouveauRequest):
    # validation
    if not payload.champ.strip():
        raise HTTPException(status_code=422, detail="champ vide")
    # logique
    return NouveauResponse(champ=payload.champ, resultat="...")
```

---

## 7. changer le modèle chargé dans l'api

**où :** `api/main.py` — bloc de chargement au démarrage (lignes ~30-40)  
**et :** `.env` — variables `MODEL_PATH` et `VECTORIZER_PATH`

option 1 — via `.env` :
```
MODEL_PATH=../models/mon_nouveau_modele.keras
VECTORIZER_PATH=../models/mon_nouveau_vectorizer.pkl
```

option 2 — directement dans le code :
```python
model = tf.keras.models.load_model("../models/mon_nouveau_modele.keras")
vectorizer = joblib.load("../models/mon_nouveau_vectorizer.pkl")
```

après modification : redémarrer l'api (`ctrl+c` puis `uvicorn api.main:app --reload`).

---

## 8. réentraîner depuis zéro

1. vérifier que `news.csv` est bien dans `ECF4/` (dossier parent)
2. activer le venv : `.venv\Scripts\activate`
3. ouvrir le notebook : `jupyter notebook notebook/ecf_fake_news.ipynb`
4. **kernel → restart & run all**
5. attendre la fin de l'exécution (~5-15 min selon la machine)
6. les fichiers `models/best_model.keras` et `models/vectorizer.pkl` sont mis à jour automatiquement
7. redémarrer l'api

---

## 9. tester l'api rapidement

```bash
# health check
curl http://localhost:8000/health

# prédiction simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "scientists discover new treatment for common disease"}'

# prédiction en lot
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"titles": ["scientists discover cure", "shocking government secret revealed"]}'

# tester les cas limites
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "   "}'   # → 422

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "'"$(python -c "print('a'*301)")"'"}'  # → 400
```

ou utiliser la documentation interactive : `http://localhost:8000/docs`

---

## 10. dépendances — que fait chaque package

| package | rôle |
|---|---|
| `pandas` | manipulation des données csv |
| `numpy` | calculs numériques |
| `matplotlib` / `seaborn` | visualisations |
| `scikit-learn` | tfidf, split, métriques |
| `nltk` | stopwords anglais |
| `spacy` | lemmatisation (en_core_web_sm) |
| `tensorflow` | modèles keras (dense + bilstm) |
| `joblib` | sauvegarde/chargement du vectoriseur |
| `fastapi` | framework api rest |
| `uvicorn` | serveur asgi pour fastapi |
| `python-dotenv` | chargement du fichier .env |
| `pydantic` | validation des données api |
| `ipykernel` | kernel jupyter pour le notebook |
