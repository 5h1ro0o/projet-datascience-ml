# Projet Data Science & Machine Learning - Prédiction Prix Voitures d'Occasion

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

## 📋 Description du Projet

Ce projet vise à prédire le prix de voitures d'occasion en fonction de leurs caractéristiques techniques (marque, année, kilométrage, puissance, type de carburant, etc.) en utilisant différentes techniques de Machine Learning.

**Objectif** : Construire un modèle de régression avec une erreur de prédiction inférieure à 15% du prix réel.

**Résultat** : Objectif atteint avec 4 modèles sur 5, le meilleur atteignant 9-10% d'erreur.

---

## 📁 Structure des Fichiers

Le projet est organisé en **3 documents distincts et complémentaires** :

### 1. `projet_automobile.ipynb` - **Notebook Jupyter Principal**
**Contenu** : Tout le code exécutable du projet
- Chargement et exploration des données
- Nettoyage et prétraitement
- Visualisations (10+ graphiques)
- Modélisation (5 algorithmes)
- Validation et évaluation
- Clustering
- Conclusions

**Comment l'utiliser** :
```bash
jupyter notebook projet_automobile.ipynb
```
Exécutez toutes les cellules dans l'ordre (Cell → Run All).

### 2. `questions_reponses.md` - **Documentation des Décisions**
**Contenu** : Toutes les questions posées dans l'énoncé et nos réponses justifiées
- Définition du problème
- Exploration des données (valeurs manquantes, outliers, corrélations)
- Prétraitement (imputation, encodage, standardisation)
- Méthodologies (CRISP-DM, KDD, SEMMA)
- Choix des modèles et hyperparamètres
- Interprétation des résultats
- Limites et perspectives

**Format** : Markdown, lisible directement sur GitHub ou avec n'importe quel éditeur.

### 3. `visualisations_interpretations.md` - **Guide des Visualisations**
**Contenu** : Catalogue de toutes les visualisations avec interprétations détaillées
- Distribution des prix et exploration initiale
- Matrice de corrélation
- Scatter plots (relations prix-features)
- Boxplots (détection d'outliers)
- Résultats des modèles
- Courbes d'apprentissage
- Clustering (coude, dendrogramme, segments)
- Comparaisons finales

**Format** : Markdown, contient les références aux images générées par le notebook.

---

## 🚀 Installation et Exécution

### Prérequis
- Python 3.8+
- Jupyter Notebook

### Installation des dépendances
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Exécution
1. Ouvrir le notebook Jupyter :
```bash
jupyter notebook projet_automobile.ipynb
```

2. Exécuter toutes les cellules (Cell → Run All)

3. Les visualisations seront sauvegardées automatiquement en PNG

4. Consulter les fichiers MD pour les explications détaillées

---

## 🎯 Dataset Utilisé

**Source** : UCI Machine Learning Repository - Automobile Dataset
**URL** : https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data

**Caractéristiques** :
- 205 véhicules
- 26 variables (16 numériques, 10 catégorielles)
- Période : années 1980
- Variable cible : Prix (en dollars)

**Variables clés** :
- `make` : Marque (Toyota, Honda, BMW, etc.)
- `fuel_type` : Type de carburant (gas, diesel)
- `engine_size` : Taille du moteur (en cubic inches)
- `horsepower` : Puissance (en HP)
- `curb_weight` : Poids du véhicule (en lbs)
- `price` : Prix (en $) - **variable à prédire**

---

## 🔬 Méthodologie

Le projet suit la méthodologie **CRISP-DM** (Cross-Industry Standard Process for Data Mining) :

1. **Compréhension métier** : Définition du problème de prédiction de prix
2. **Compréhension des données** : Exploration statistique et visuelle
3. **Préparation des données** : Nettoyage, imputation, encodage, standardisation
4. **Modélisation** : Test de 5 algorithmes de régression
5. **Évaluation** : Validation croisée, métriques multiples (RMSE, MAE, R²)
6. **Déploiement** : Rapport final et recommandations

---

## 🤖 Modèles Testés

| Modèle | R² Test | Erreur % | Objectif Atteint |
|---|---|---|---|
| **Gradient Boosting** | 0.93 | ~9% | ✅ |
| **Random Forest** | 0.92 | ~10% | ✅ |
| **KNN** | 0.89 | ~12% | ✅ |
| **Régression Linéaire** | 0.88 | ~13% | ✅ |
| **Arbre de Décision** | 0.84 | ~16% | ⚠️ |

**Meilleur modèle** : **Gradient Boosting** (R²=0.93, erreur=9%)
**Meilleur compromis** : **Random Forest** (performance proche, plus stable et interprétable)

---

## 📊 Résultats Clés

### Variables les plus importantes
1. **curb_weight** (poids) - 25-30%
2. **engine_size** (moteur) - 15-20%
3. **horsepower** (puissance) - 10-15%
4. **width** (largeur) - 8-12%
5. **highway_mpg** (consommation) - 6-10%

### Clustering - 4 Segments Identifiés
1. **Économiques** : petits moteurs, faible prix (~7k$)
2. **Familiaux** : caractéristiques moyennes, prix moyens (~13k$)
3. **Sportifs** : puissance élevée, prix élevés (~20k$)
4. **Premium/Luxe** : gros moteurs, prix très élevés (>25k$)

---

## 📈 Visualisations Générées

Le notebook génère automatiquement 10+ visualisations :
- `visualisations_exploration.png` - Distribution prix, marques, fuel type, cylindres
- `matrice_correlation.png` - Heatmap des corrélations
- `scatter_plots.png` - Relations prix-features (6 graphiques)
- `boxplots_outliers.png` - Détection des valeurs aberrantes
- `regression_lineaire_predictions.png` - Prédictions vs réalité
- `knn_optimisation_k.png` - Choix du paramètre k optimal
- `arbre_decision.png` - Structure de l'arbre
- `random_forest_importance.png` - Importance des features
- `courbes_apprentissage.png` - Validation du modèle
- `kmeans_coude.png` - Méthode du coude
- `clusters_visualisation.png` - Segments visualisés
- `dendrogramme.png` - Classification hiérarchique
- `comparaison_modeles.png` - Comparaison finale (4 métriques)
- `predictions_tous_modeles.png` - Tous les modèles comparés

---

## 🎓 Concepts Data Science Abordés

### Partie I - Projet Complet
- Exploration de données (EDA)
- Statistiques descriptives
- Visualisations (matplotlib, seaborn)
- Gestion des valeurs manquantes (imputation)
- Détection des outliers
- Encodage de variables catégorielles
- Standardisation (Z-score)
- Méthodologies (CRISP-DM, KDD, SEMMA)

### Partie II - Machine Learning
- **Régression** : Linéaire, KNN, Arbres, Ensembles
- **Validation croisée** : 5-fold
- **Métriques** : RMSE, MAE, R²
- **Optimisation** : GridSearch, hyperparamètres
- **Diagnostics** : Courbes d'apprentissage, surapprentissage
- **Clustering** : K-means, classification hiérarchique, méthode du coude
- **Feature importance** : Interprétabilité des modèles

---

## 💡 Points d'Attention

### Code
- **Sans commentaires** (volontaire pour la lisibilité)
- **Simple et direct** : adapté au niveau initiation
- **Bibliothèques standard** : pandas, numpy, sklearn, matplotlib, seaborn

### Données
- Dataset ancien (1985) mais pédagogiquement excellent
- Valeurs manquantes gérées de manière standard (médiane/mode)
- Outliers conservés (représentent des segments réels)

### Modélisation
- Pas d'over-engineering : focus sur la méthodologie complète
- Hyperparamètres choisis manuellement (pas de GridSearch systématique)
- Validation simple mais rigoureuse

---

## 📚 Références

### Dataset
- [UCI Machine Learning Repository - Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile)

### Méthodologies
- CRISP-DM : Standard de l'industrie pour les projets Data Mining
- KDD : Knowledge Discovery in Databases (académique)
- SEMMA : Sample, Explore, Modify, Model, Assess (SAS)

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib/Seaborn Gallery](https://matplotlib.org/stable/gallery/)

---

## ✅ Checklist de Vérification

- [x] Notebook Jupyter complet et exécutable
- [x] Fichier Questions/Réponses exhaustif
- [x] Fichier Visualisations avec interprétations
- [x] 5 modèles de régression implémentés
- [x] Validation croisée et métriques multiples
- [x] Clustering (K-means + hiérarchique)
- [x] 10+ visualisations générées
- [x] Objectif <15% d'erreur atteint
- [x] Code simple et sans commentaires
- [x] Documentation claire et authentique (pas de bullshit)

---

## 📧 Contact

**Auteurs** : Matteo Robin, Florian Huguet
**Projet** : Module d'initiation Data Science et Machine Learning

---

## 🔄 Évolutions Possibles

Pour un projet avancé, nous pourrions :
1. Utiliser un dataset plus récent et volumineux (Kaggle)
2. Feature engineering : ratio puissance/poids, âge du véhicule
3. One-Hot Encoding au lieu de Label Encoding
4. GridSearchCV systématique pour tous les modèles
5. Stacking/Blending d'ensembles de modèles
6. Analyse des résidus approfondie
7. Déploiement avec API (Flask/FastAPI)

---

**Dernière mise à jour** : Décembre 2025
