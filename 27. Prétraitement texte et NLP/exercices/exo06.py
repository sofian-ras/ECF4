# %% [markdown]
# # Exercice 06 : Analyse de sentiments - approche classique
#
# Version corrigée dans le style de `demo04` :
# - code simple et direct
# - peu de fonctions
# - commentaires courts
# - une progression cellule par cellule

# %%
# Imports
import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# %%
# Donnees
reviews = [
    "Excellent produit, très satisfait! Livraison rapide, je recommande!",
    "Pourquoi ce produit est-il si cher pour une si mauvaise qualité?",
    "Parfait! Exactement ce que j'attendais. Très bon rapport qualité-prix.",
    "Horrible expérience. Le produit est cassé à la réception.",
    "Magnifique! Dépassé mes attentes. Un incontournable.",
    "Décevant. Les photos du site ne correspondaient pas au produit réel.",
    "Super ! Très content de mon achat. Je rachèterai!",
    "Pire achat de ma vie. Service client inexistant.",
    "Bon produit pour le prix. Pas extraordinaire mais correct.",
    "Absolument terrible. Qualité déplorable et défaut de fabrication.",
    "Sympa ! Livré rapidement et bien emballé.",
    "Vraiment nul. Ça ne fonctionne pas du tout.",
    "Très bien! Produit conforme à la description.",
    "Complètement inutile, j'ai demandé le remboursement.",
    "Excellent service et produit de qualité!",
    "Déception totale. N'achetez pas ce produit."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

df = pd.DataFrame({
    "review": reviews,
    "label": labels
})

df["sentiment"] = df["label"].map({1: "positif", 0: "negatif"})
df

# %%
# Nettoyage simple du texte
def enlever_accents(texte):
    texte_normalise = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in texte_normalise if not unicodedata.combining(c))

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = enlever_accents(texte)
    texte = re.sub(r"[’']", " ", texte)
    texte = re.sub(r"[^a-z0-9\s]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte

df["review_clean"] = df["review"].apply(nettoyer_texte)

df[["review", "review_clean", "sentiment"]]

# %%
# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = vectorizer.fit_transform(df["review_clean"])
y = df["label"]

print("Shape TF-IDF :", X.shape)
print("Nombre de termes :", len(vectorizer.get_feature_names_out()))

# %%
# Split train / test
X_train, X_test, y_train, y_test, review_train, review_test = train_test_split(
    X,
    y,
    df["review"],
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)

print("\nEquilibre des classes - train")
print(y_train.value_counts())

print("\nEquilibre des classes - test")
print(y_test.value_counts())

# %%
# Logistic Regression
modele_lr = LogisticRegression(max_iter=1000, random_state=42)
modele_lr.fit(X_train, y_train)

y_pred_lr = modele_lr.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_lr, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_lr, zero_division=0))
print()
print(classification_report(y_test, y_pred_lr, zero_division=0))

cm_lr = confusion_matrix(y_test, y_pred_lr)
ConfusionMatrixDisplay(cm_lr, display_labels=["negatif", "positif"]).plot()
plt.show()

# %%
# SVM kernel linear
modele_svm_linear = SVC(kernel="linear", probability=True, random_state=42)
modele_svm_linear.fit(X_train, y_train)

y_pred_svm_linear = modele_svm_linear.predict(X_test)

print("=== SVM linear ===")
print("Accuracy :", accuracy_score(y_test, y_pred_svm_linear))
print("Precision:", precision_score(y_test, y_pred_svm_linear, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_svm_linear, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_svm_linear, zero_division=0))

# %%
# Test de plusieurs kernels SVM
resultats_svm = []

for kernel in ["linear", "rbf", "poly"]:
    modele = SVC(kernel=kernel, probability=True, random_state=42)
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    resultats_svm.append([kernel, acc, prec, rec, f1])

df_svm = pd.DataFrame(
    resultats_svm,
    columns=["kernel", "accuracy", "precision", "recall", "f1_score"]
)

df_svm = df_svm.sort_values(by="f1_score", ascending=False)
df_svm

# %%
# Naive Bayes
modele_nb = MultinomialNB()
modele_nb.fit(X_train, y_train)

y_pred_nb = modele_nb.predict(X_test)

print("=== Naive Bayes ===")
print("Accuracy :", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_nb, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_nb, zero_division=0))
print()
print(classification_report(y_test, y_pred_nb, zero_division=0))

# %%
# Probabilites de classe pour 3 avis de test avec Naive Bayes
X_test_dense_reviews = review_test.reset_index(drop=True)
proba_nb = modele_nb.predict_proba(X_test)

for i in range(min(3, len(X_test_dense_reviews))):
    print("Avis :", X_test_dense_reviews[i])
    print("Probabilites [negatif, positif] :", proba_nb[i])
    print("Prediction :", "positif" if modele_nb.predict(X_test[i])[0] == 1 else "negatif")
    print("-" * 60)

# %% [markdown]
# ### Naive Bayes : avantages / inconvénients
#
# **Avantages**
# - rapide
# - simple
# - efficace sur petits corpus texte
#
# **Inconvénients**
# - hypothèse d'indépendance des mots
# - moins fin qu'un SVM ou une Logistic Regression
# - peut mal gérer les formulations ambiguës

# %%
# Comparaison globale des modeles
comparaison = pd.DataFrame([
    ["Logistic Regression",
     accuracy_score(y_test, y_pred_lr),
     precision_score(y_test, y_pred_lr, zero_division=0),
     recall_score(y_test, y_pred_lr, zero_division=0),
     f1_score(y_test, y_pred_lr, zero_division=0)],
    ["SVM linear",
     accuracy_score(y_test, y_pred_svm_linear),
     precision_score(y_test, y_pred_svm_linear, zero_division=0),
     recall_score(y_test, y_pred_svm_linear, zero_division=0),
     f1_score(y_test, y_pred_svm_linear, zero_division=0)],
    ["Naive Bayes",
     accuracy_score(y_test, y_pred_nb),
     precision_score(y_test, y_pred_nb, zero_division=0),
     recall_score(y_test, y_pred_nb, zero_division=0),
     f1_score(y_test, y_pred_nb, zero_division=0)],
], columns=["modele", "accuracy", "precision", "recall", "f1_score"])

comparaison.sort_values(by="f1_score", ascending=False)

# %%
# Feature importance de la Logistic Regression
feature_names = vectorizer.get_feature_names_out()
coefficients = modele_lr.coef_[0]

df_coef = pd.DataFrame({
    "terme": feature_names,
    "coefficient": coefficients
})

top_negatifs = df_coef.sort_values(by="coefficient").head(10)
top_positifs = df_coef.sort_values(by="coefficient", ascending=False).head(10)

print("Top 10 termes negatifs")
display(top_negatifs)

print("Top 10 termes positifs")
display(top_positifs)

# %%
# Visualisation des termes les plus importants
plt.figure(figsize=(10, 5))
plt.barh(top_negatifs["terme"], top_negatifs["coefficient"])
plt.title("Top 10 termes negatifs - Logistic Regression")
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(top_positifs["terme"], top_positifs["coefficient"])
plt.title("Top 10 termes positifs - Logistic Regression")
plt.show()

# %% [markdown]
# Analyse rapide :
# - les termes négatifs devraient être liés à *horrible*, *nul*, *remboursement*, *terrible*...
# - les termes positifs devraient être liés à *excellent*, *parfait*, *magnifique*, *recommande*...
# - sur un petit corpus, certains termes peuvent être surévalués

# %%
# GridSearchCV pour optimiser le SVM
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring="f1"
)

grid.fit(X_train, y_train)

print("Meilleurs parametres :", grid.best_params_)
print("Meilleur score CV    :", grid.best_score_)

best_svm = grid.best_estimator_
y_pred_best_svm = best_svm.predict(X_test)

print("\n=== SVM apres tuning ===")
print("Accuracy :", accuracy_score(y_test, y_pred_best_svm))
print("Precision:", precision_score(y_test, y_pred_best_svm, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred_best_svm, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred_best_svm, zero_division=0))

# %%
# Comparaison avant / apres tuning
avant_apres = pd.DataFrame([
    ["SVM linear avant tuning",
     accuracy_score(y_test, y_pred_svm_linear),
     f1_score(y_test, y_pred_svm_linear, zero_division=0)],
    ["SVM apres tuning",
     accuracy_score(y_test, y_pred_best_svm),
     f1_score(y_test, y_pred_best_svm, zero_division=0)]
], columns=["modele", "accuracy", "f1_score"])

avant_apres

# %%
# Evaluation sur 5 nouveaux avis
nouveaux_avis = [
    "Excellent achat, je suis tres satisfait.",
    "Produit inutilisable, tres mauvaise qualite.",
    "Livraison rapide mais article moyen.",
    "Je ne recommande pas du tout ce produit.",
    "Tres bon service client et produit conforme."
]

df_nouveaux = pd.DataFrame({"texte_brut": nouveaux_avis})
df_nouveaux["texte_clean"] = df_nouveaux["texte_brut"].apply(nettoyer_texte)

X_new = vectorizer.transform(df_nouveaux["texte_clean"])

# On utilise le meilleur modele SVM
predictions = best_svm.predict(X_new)
probabilites = best_svm.predict_proba(X_new)

df_nouveaux["prediction"] = ["positif" if p == 1 else "negatif" for p in predictions]
df_nouveaux["confiance"] = probabilites.max(axis=1)

df_nouveaux[["texte_brut", "prediction", "confiance"]]

# %% [markdown]
# ### Analyse des erreurs possibles
#
# - **Faux positifs** : avis mitigé avec quelques mots positifs dominants
# - **Faux négatifs** : négation ou ironie mal captée
# - **Pourquoi ?** : corpus trop petit, peu de vocabulaire, peu de contexte
#
# ### Comment améliorer le modèle ?
#
# - ajouter beaucoup plus d'avis
# - mieux nettoyer le texte
# - tester d'autres `ngram_range`
# - utiliser de la validation croisée
# - passer ensuite à une approche deep learning