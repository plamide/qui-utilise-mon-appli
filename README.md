# Qui utilise mon appli ?

## Présentation

L'objectif est d'identifier automatiquement l'utilisateur d'un logiciel à partir de ses traces d'utilisation, c'est-à-dire des différentes actions réalisées au cours d'une session.

---

## Objectif Machine Learning

Le problème est formulé comme un problème de **classification multiclasse** :

- **Entrées** : traces d'utilisation correspondant aux actions réalisées au cours d'une session ;
- **Sortie** : identité de l'utilisateur ayant réalisé la session.

Le pipeline doit donc transformer les traces brutes en caractéristiques exploitables par un modèle de Machine Learning, entraîner un classifieur sur le jeu d'entraînement puis prédire l'utilisateur associé aux sessions du jeu de test.

---

## Métrique d'évaluation

La métrique principale utilisée pour évaluer les modèles est le **F1-score moyen**.

Le F1-score combine la précision et le rappel :

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

avec :

\[
Precision = \frac{TP}{TP + FP}
\]

et :

\[
Recall = \frac{TP}{TP + FN}
\]

Cette métrique permet de prendre simultanément en compte les faux positifs et les faux négatifs.

L'objectif est donc de maximiser le F1-score obtenu par le modèle sur les utilisateurs à prédire.

---

## Architecture du projet

Le projet est organisé de manière modulaire afin de séparer la préparation des données, l'entraînement, l'optimisation et l'évaluation des modèles.

    .
    ├── data/
    │   ├── train.csv
    │   ├── test.csv
    │   └── train_data.py
    │
    ├── data_explore/
    │   └── *.ipynb
    │
    ├── main/
    │   └── main.py
    │
    ├── model/
    │   ├── train_model.py
    │   └── optimize.py
    │
    ├── utils/
    │   └── reporter.py
    │
    ├── .gitignore
    └── README.md

### `data/`

Contient les données utilisées par le projet ainsi que les fonctions de préparation des données.

- `train.csv` : données utilisées pour l'entraînement des modèles ;
- `test.csv` : données sur lesquelles les prédictions finales doivent être réalisées ;
- `train_data.py` : chargement, nettoyage et transformation des données brutes en caractéristiques exploitables par les modèles.

### `data_explore/`

Contient les notebooks Jupyter utilisés pour l'exploration et l'analyse des données.

Ces notebooks permettent notamment d'étudier la structure des traces avant leur utilisation dans le pipeline d'apprentissage.

### `main/`

Contient le point d'entrée principal du projet.

`main.py` orchestre les différentes étapes du pipeline :

1. préparation des données ;
2. création et entraînement du modèle ;
3. évaluation du modèle ;
4. génération éventuelle des résultats et rapports.

### `model/`

Contient la partie consacrée à la modélisation.

- `train_model.py` : création, entraînement et évaluation des modèles de classification ;
- `optimize.py` : optimisation des hyperparamètres à l'aide d'une recherche bayésienne avec Optuna.

### `utils/`

Contient les fonctions utilitaires utilisées par les autres modules.

`reporter.py` a pour objectif de générer automatiquement un rapport HTML présentant les performances d'un modèle. Cette fonctionnalité est actuellement en cours de développement.

---

## Feature Engineering

Baseline réalisée avec une régression logistique (f1-score = 0.72) puis optimisée via optuna pour le jeu d'hyperparamètres C (régularisation) et tol (tolérance). 

La branche **navigator_optimize** a pour but d'améliorer ce score en tenant compte du fait que l'ensemble des utilisateurs sont mononavigateurs afin de réaliser un prétraitement des données en fonction du navigateur. 

La piste d'un meilleur traitement des **actions** est à envisager (vocabulaire des écrans, modules ERP $...$, capture des routines : ex : Saisie -> Raccourci -> Bouton)

---


## Optimisation des hyperparamètres

L'optimisation des hyperparamètres est réalisée à l'aide de **Optuna**.

Le module `model/optimize.py` permet d'effectuer une optimisation bayésienne afin de rechercher automatiquement des configurations d'hyperparamètres performantes selon la métrique d'évaluation retenue.

---

## Résultats

Les performances des différents modèles sont évaluées principalement à l'aide du F1-score.

| Modèle | F1-score 


---

## Exécution du projet

Le pipeline principal peut être lancé depuis la racine du projet avec :

    python main/main.py

Le programme assure ensuite le chargement des données, leur préparation et l'entraînement du modèle configuré.

---

## Librairies utilisées

- Python
- pandas
- plotly
- NumPy
- SciPy
- scikit-learn
- Optuna
- Jupyter Notebook

---
