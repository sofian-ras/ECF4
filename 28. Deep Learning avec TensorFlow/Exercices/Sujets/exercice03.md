# Exercice 3 : CNN pour la Vision par Ordinateur

## Objectif

Construire et optimiser un réseau de convolution (CNN) pour la classification d'images CIFAR-10.

## Contexte

Vous développez un système de classification d'objets. Vous travaillez avec CIFAR-10 (voitures, chevaux, chats, etc.). Vous devez concevoir et optimiser une architecture CNN efficace.

## Dataset

CIFAR-10 depuis Keras :

- 50,000 images entraînement (32x32 RGB)
- 10,000 images test
- 10 classes : avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion

## Tâches

### Tâche 1 : Chargement et Exploration

1. Chargez CIFAR-10 depuis `keras.datasets.cifar10`
2. Visualisez 20 images aléatoires avec labels
3. Analysez les dimensions et range de pixels
4. Calculez la distribution de classes
5. Normalisez les pixels dans [0, 1]

Attendus : Dataset exploré et normalisé

### Tâche 2 : Architecture CNN Simple

1. Construisez un CNN :
   - Conv2D(32, 3x3, relu) + MaxPooling2D(2x2)
   - Conv2D(64, 3x3, relu) + MaxPooling2D(2x2)
   - Flatten
   - Dense(128, relu) + Dropout(0.3)
   - Dense(10, softmax)
2. Compilez avec Adam et categorical_crossentropy
3. Entraînez 15 epochs

Attendus : Modèle entraîné, accuracy noté

### Tâche 3 : CNN Profond

1. Construisez un CNN plus profond :
   - Conv2D(32, 3x3) × 2 + MaxPooling
   - Conv2D(64, 3x3) × 2 + MaxPooling
   - Conv2D(128, 3x3) × 2 + MaxPooling
   - Flatten + Dense(256) + Dropout(0.5)
   - Dense(128) + Dropout(0.3)
   - Dense(10, softmax)
2. Utilisez Batch Normalization après chaque Conv2D
3. Entraînez 20 epochs
4. Comparez avec le modèle simple

Attendus : CNN profond entraîné, métriques comparées

### Tâche 4 : Évaluation Complète

1. Évaluez le meilleur modèle sur test set
2. Calculez accuracy, loss
3. Prédisez sur 20 images de test
4. Visualisez 10 prédictions (image + label + confiance)
5. Identifiez les erreurs (images mal classifiées)

Attendus : Évaluation précise, visualisations claires

### Tâche 5 : Optimisation d'Hyperparamètres

1. Testez différentes configurations :
   - Learning rates : 0.001, 0.0005, 0.0001
   - Batch sizes : 16, 32, 64
   - Dropout : 0.2, 0.3, 0.5
2. Comparez les performances
3. Identifiez la meilleure configuration

## Critères d'évaluation

- Dataset correctement exploré et préparé
- Architecture CNN implémentée correctement
- Batch normalization utilisée
- Évaluation précise et visualisée
- Hyperparamètres optimisés systématiquement

## Conseils

- Utilisez `Conv2D`, `MaxPooling2D`, `BatchNormalization` de Keras
- Visualisez les images avec `plt.imshow()`
- Monitorez le rapport train/val loss pour détecter l'overfitting
- Sauvegardez le meilleur modèle avec callbacks
- Utilisez `padding='same'` dans les couches de convolution pour le CNN profond
