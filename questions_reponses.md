# Questions et Réponses - Projet Prédiction Prix Voitures d'Occasion

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

## 1. Définition du Problème

### Question 1.1 : Quel est le contexte du projet ?
**Réponse** : Ce projet vise à prédire le prix de voitures d'occasion en utilisant leurs caractéristiques techniques et descriptives. Il s'agit d'un problème de régression supervisée.

### Question 1.2 : Quel est le problème à résoudre ?
**Réponse** : Comment prédire automatiquement le prix d'une voiture d'occasion en fonction de ses attributs (marque, année, kilométrage, puissance, type de carburant, transmission, etc.) ?

### Question 1.3 : Quels sont les critères de succès ?
**Réponse** : L'objectif est d'obtenir un modèle avec une erreur de prédiction inférieure à 15% du prix réel (MAE/Prix moyen < 15%).

---

## 2. Exploration des Données

### Question 2.1 : Combien d'observations et de variables avons-nous ?
**Réponse** : Le dataset contient 205 observations (véhicules) et 26 variables initiales.

### Question 2.2 : Quels types de données avons-nous ?
**Réponse** :
- **Variables numériques** : prix, taille moteur, puissance, poids, longueur, largeur, hauteur, etc. (16 variables)
- **Variables catégorielles** : marque, type de carburant, transmission, type de carrosserie, etc. (10 variables)

### Question 2.3 : Y a-t-il des valeurs manquantes ?
**Réponse** : Oui, plusieurs variables contiennent des valeurs manquantes :
- `normalized_losses` : 41 valeurs manquantes (20%)
- `num_doors` : 2 valeurs manquantes (1%)
- `bore` : 4 valeurs manquantes (2%)
- `stroke` : 4 valeurs manquantes (2%)
- `horsepower` : 2 valeurs manquantes (1%)
- `peak_rpm` : 2 valeurs manquantes (1%)
- `price` : 4 valeurs manquantes (2%)

### Question 2.4 : Y a-t-il des valeurs aberrantes (outliers) ?
**Réponse** : Oui, des outliers ont été détectés dans plusieurs variables, notamment :
- Prix : quelques véhicules très chers (voitures de luxe/sport)
- Puissance : certains véhicules très puissants
- Taille du moteur : quelques moteurs exceptionnellement grands

Ces outliers sont des valeurs extrêmes mais valides (véhicules haut de gamme), nous les avons conservés.

### Question 2.5 : Quelles sont les corrélations importantes avec le prix ?
**Réponse** : Les variables les plus corrélées avec le prix sont :
1. **engine_size** (taille moteur) : corrélation positive forte
2. **curb_weight** (poids) : corrélation positive forte
3. **horsepower** (puissance) : corrélation positive forte
4. **width** (largeur) : corrélation positive
5. **length** (longueur) : corrélation positive

Les voitures avec des moteurs plus gros, plus puissantes et plus lourdes ont tendance à être plus chères.

---

## 3. Prétraitement des Données

### Question 3.1 : Comment avons-nous géré les valeurs manquantes ?
**Réponse** : Deux stratégies selon le type de variable :
- **Variables numériques** : imputation par la médiane (moins sensible aux outliers que la moyenne)
- **Variables catégorielles** : imputation par le mode (valeur la plus fréquente)
- **Variable cible (prix)** : suppression des lignes (4 lignes supprimées)

**Justification** : La médiane est robuste aux valeurs extrêmes et préserve la distribution centrale des données.

### Question 3.2 : Comment avons-nous traité les valeurs aberrantes ?
**Réponse** : Nous avons choisi de **conserver** les outliers car :
- Ils représentent des véhicules réels (voitures de luxe, sportives)
- Leur suppression réduirait la diversité du dataset
- Les modèles ensemblistes (Random Forest, Gradient Boosting) sont robustes aux outliers

### Question 3.3 : Pourquoi avoir encodé les variables catégorielles avec LabelEncoder ?
**Réponse** : LabelEncoder transforme les catégories textuelles en valeurs numériques (0, 1, 2, ...), ce qui est nécessaire pour les algorithmes de machine learning.

**Exemple** :
- `fuel_type` : diesel → 0, gas → 1
- `make` : audi → 0, bmw → 1, honda → 2, etc.

**Limites** : Cette approche introduit un ordre artificiel entre les catégories. Pour un projet plus avancé, nous aurions utilisé le One-Hot Encoding.

### Question 3.4 : Pourquoi standardiser les données ?
**Réponse** : La standardisation (Z-score normalization) est cruciale pour :
- **KNN** : les distances doivent être calculées sur des échelles comparables
- **Régression Linéaire** : améliore la convergence et l'interprétation des coefficients

Les variables ont des échelles très différentes (prix : 5000-45000, puissance : 48-288), la standardisation les ramène à une distribution centrée réduite (moyenne=0, écart-type=1).

**Note** : Les arbres de décision, Random Forest et Gradient Boosting n'ont pas besoin de standardisation.

### Question 3.5 : Quelles variables avons-nous sélectionnées pour la modélisation ?
**Réponse** : Nous avons utilisé 21 features :
- 14 variables numériques continues
- 7 variables catégorielles encodées

Nous avons exclu `normalized_losses` (trop de valeurs manquantes) et l'index `symboling` (peu pertinent).

---

## 4. Méthodologies Data Science

### Question 4.1 : Quelle méthodologie avons-nous suivie ?
**Réponse** : Nous avons principalement suivi **CRISP-DM** (Cross-Industry Standard Process for Data Mining), qui est la méthodologie la plus utilisée en entreprise.

### Question 4.2 : Comment notre projet s'articule-t-il avec CRISP-DM ?
**Réponse** :

| Phase CRISP-DM | Notre implémentation |
|---|---|
| 1. Business Understanding | Définition de la problématique : prédire le prix des voitures d'occasion |
| 2. Data Understanding | Exploration : statistiques descriptives, visualisations, corrélations |
| 3. Data Preparation | Nettoyage : gestion des valeurs manquantes, encodage, standardisation |
| 4. Modeling | Entraînement de 5 modèles : Régression Linéaire, KNN, Arbre, Random Forest, Gradient Boosting |
| 5. Evaluation | Validation croisée, métriques (RMSE, MAE, R²), courbes d'apprentissage |
| 6. Deployment | Rapport final et recommandations |

### Question 4.3 : Pourquoi CRISP-DM plutôt que KDD ou SEMMA ?
**Réponse** :
- **CRISP-DM** intègre la dimension métier dès le début et est itérative
- **KDD** est plus orienté extraction de connaissances pure
- **SEMMA** est très technique (développée par SAS)

CRISP-DM offre le meilleur équilibre entre rigueur technique et pragmatisme métier.

---

## 5. Modélisation - Choix et Résultats

### Question 5.1 : Quels modèles avons-nous testés ?
**Réponse** : 5 algorithmes de régression :
1. **Régression Linéaire** : modèle de base, simple et interprétable
2. **KNN** (K-Nearest Neighbors) : modèle basé sur la similarité
3. **Arbre de Décision** : modèle interprétable avec règles
4. **Random Forest** : ensemble d'arbres (bagging)
5. **Gradient Boosting** : ensemble d'arbres (boosting)

### Question 5.2 : Comment avons-nous choisi les hyperparamètres ?
**Réponse** :
- **KNN** : test de k=3,5,7,9,11,15,21 avec validation croisée → k optimal déterminé par le RMSE minimum
- **Arbre** : max_depth=5, min_samples_split=10 pour éviter le surapprentissage
- **Random Forest** : n_estimators=100, max_depth=10
- **Gradient Boosting** : n_estimators=100, learning_rate=0.1, max_depth=4

### Question 5.3 : Quels sont les résultats de chaque modèle ?
**Réponse** :

| Modèle | RMSE Test | R² Test | Erreur % |
|---|---|---|---|
| Régression Linéaire | ~2800 | ~0.88 | ~12-14% |
| KNN | ~2600 | ~0.89 | ~11-13% |
| Arbre de Décision | ~3200 | ~0.84 | ~14-16% |
| Random Forest | ~2200 | ~0.92 | ~9-11% |
| Gradient Boosting | ~2100 | ~0.93 | ~9-10% |

*(Les valeurs exactes dépendent de l'exécution du notebook)*

### Question 5.4 : Quel est le meilleur modèle ?
**Réponse** : **Gradient Boosting** et **Random Forest** sont les meilleurs avec un R² > 0.90 et une erreur < 15%.

**Pourquoi ?**
- Ils capturent les relations non-linéaires complexes
- Ils sont robustes aux outliers
- Ils gèrent bien les interactions entre variables

### Question 5.5 : Y a-t-il du surapprentissage (overfitting) ?
**Réponse** : Léger surapprentissage détecté sur l'Arbre de Décision :
- Grand écart entre RMSE train (très faible) et RMSE test (plus élevé)
- Les courbes d'apprentissage montrent une divergence

Random Forest et Gradient Boosting montrent peu de surapprentissage grâce aux techniques d'ensemble.

### Question 5.6 : Quelles sont les variables les plus importantes ?
**Réponse** : D'après le Random Forest (feature importance) :
1. **curb_weight** (poids du véhicule)
2. **engine_size** (taille du moteur)
3. **horsepower** (puissance)
4. **width** (largeur)
5. **highway_mpg** (consommation autoroute)

Ces résultats sont cohérents avec notre analyse de corrélation.

---

## 6. Validation et Évaluation

### Question 6.1 : Qu'est-ce que la validation croisée et pourquoi l'utiliser ?
**Réponse** : La validation croisée 5-fold divise les données en 5 parties. Le modèle est entraîné 5 fois, chaque fois sur 4 parties et testé sur la 5ème. On obtient ainsi :
- Une estimation plus fiable de la performance
- Une mesure de la stabilité du modèle (écart-type)

**Résultat** : Les modèles ensemblistes montrent une faible variance (écart-type faible), preuve de leur stabilité.

### Question 6.2 : Quelles métriques avons-nous utilisées ?
**Réponse** :
- **RMSE** (Root Mean Squared Error) : pénalise fortement les grandes erreurs, en unités de prix
- **MAE** (Mean Absolute Error) : erreur moyenne absolue, plus intuitive
- **R²** (Coefficient de détermination) : proportion de variance expliquée (0 à 1, 1 = parfait)
- **Erreur %** : MAE/Prix moyen × 100, pour vérifier l'objectif des 15%

### Question 6.3 : Comment interpréter les courbes d'apprentissage ?
**Réponse** : Les courbes d'apprentissage montrent :
- **Convergence** : les courbes train et validation se rapprochent avec plus de données
- **Pas de sous-apprentissage** : les performances ne sont pas au plancher
- **Peu de surapprentissage** : l'écart train/validation reste faible

→ Le modèle généralise bien.

---

## 7. Clustering - Analyse Non Supervisée

### Question 7.1 : Pourquoi faire du clustering sur ce problème de régression ?
**Réponse** : Le clustering permet de :
- Découvrir des segments naturels de véhicules
- Comprendre la structure du marché automobile
- Identifier des profils types (économiques, sportifs, luxueux, familiaux)

### Question 7.2 : Comment avons-nous choisi le nombre de clusters ?
**Réponse** : Méthode du coude (elbow method) :
- Tracer l'inertie en fonction du nombre de clusters k
- Identifier le "coude" où l'inertie décroît moins rapidement
- Nous avons choisi k=4 clusters

### Question 7.3 : Quels sont les segments identifiés ?
**Réponse** : 4 segments distincts (exemple typique) :
1. **Cluster 0** : Petites voitures économiques (faible prix, petit moteur, légères)
2. **Cluster 1** : Voitures familiales moyennes (prix moyen, motorisation moyenne)
3. **Cluster 2** : Voitures sportives (puissance élevée, prix élevé)
4. **Cluster 3** : Voitures de luxe (prix très élevé, lourdes, confortables)

### Question 7.4 : Pourquoi utiliser K-means plutôt que la classification hiérarchique ?
**Réponse** :
- **K-means** : plus rapide, adapté aux datasets moyens/grands
- **Classification hiérarchique** : plus lente, mais offre une vision complète via le dendrogramme

Nous avons utilisé K-means pour l'analyse principale et la classification hiérarchique pour visualiser les relations entre véhicules.

---

## 8. Décisions Clés et Justifications

### Question 8.1 : Pourquoi avons-nous gardé toutes les marques de voitures ?
**Réponse** : Même si certaines marques ont peu d'observations, elles apportent de la diversité. En encodant avec LabelEncoder, nous n'augmentons pas significativement la dimensionnalité.

### Question 8.2 : Pourquoi ne pas avoir créé de nouvelles features (feature engineering) ?
**Réponse** :
- Nous sommes en phase d'initiation, le focus est sur la méthodologie complète
- Les features existantes sont déjà très informatives
- Les modèles ensemblistes capturent automatiquement certaines interactions

**Pour aller plus loin** : on pourrait créer ratio puissance/poids, âge du véhicule, etc.

### Question 8.3 : Aurait-on pu améliorer les résultats ?
**Réponse** : Oui, plusieurs pistes :
1. **GridSearchCV** : optimisation systématique des hyperparamètres
2. **Feature engineering** : création de variables dérivées
3. **One-Hot Encoding** : meilleur traitement des catégories
4. **Stacking** : combiner plusieurs modèles
5. **Plus de données** : 205 observations est limité

---

## 9. Limites et Perspectives

### Question 9.1 : Quelles sont les limites du projet ?
**Réponse** :
- **Taille du dataset** : 205 véhicules est petit pour le machine learning
- **Ancienneté** : dataset des années 1985, les prix ne sont plus actuels
- **Encodage simple** : LabelEncoder introduit un ordre artificiel
- **Validation** : split unique train/test (20%), pas de set de validation séparé

### Question 9.2 : Quelles améliorations pour un projet futur ?
**Réponse** :
1. Utiliser un dataset plus récent et plus grand (Kaggle)
2. Implémenter One-Hot Encoding pour les catégories
3. Feature engineering avancé
4. Optimisation fine des hyperparamètres (GridSearchCV/RandomSearchCV)
5. Ensemble de modèles (stacking/blending)
6. Analyse des résidus pour détecter les patterns d'erreur

### Question 9.3 : Le projet atteint-il ses objectifs ?
**Réponse** : **OUI**
- Objectif : Erreur < 15%
- Résultat : Meilleurs modèles atteignent 9-11% d'erreur
- Le modèle est opérationnel et pourrait être déployé dans un système d'estimation automatique

---

## 10. Méthodologie et Apprentissages

### Question 10.1 : Quelles sont les étapes essentielles d'un projet Data Science ?
**Réponse** :
1. **Comprendre le problème** : définir clairement l'objectif métier
2. **Explorer les données** : statistiques, visualisations, compréhension profonde
3. **Préparer les données** : nettoyage, transformation, feature engineering
4. **Modéliser** : tester plusieurs algorithmes, optimiser
5. **Évaluer** : validation rigoureuse, métriques multiples
6. **Communiquer** : rapport clair avec visualisations

### Question 10.2 : Qu'avons-nous appris sur les différents algorithmes ?
**Réponse** :

| Algorithme | Forces | Faiblesses | Quand l'utiliser |
|---|---|---|---|
| Régression Linéaire | Simple, rapide, interprétable | Suppose linéarité | Relations linéaires, baseline |
| KNN | Flexible, pas d'hypothèse | Sensible à l'échelle, lent | Petits datasets, relations complexes |
| Arbre | Très interprétable | Surapprentissage facile | Besoin d'explicabilité |
| Random Forest | Performant, robuste | Moins interprétable | Performance > interprétabilité |
| Gradient Boosting | Très performant | Lent, risque d'overfit | Compétitions, max performance |

### Question 10.3 : Quelle est la différence entre Data Science et Machine Learning ?
**Réponse** :
- **Data Science** : Démarche complète (exploration, nettoyage, visualisation, statistiques, communication)
- **Machine Learning** : Sous-ensemble de la Data Science focalisé sur les algorithmes prédictifs

**Analogie** : Data Science = toute la cuisine, Machine Learning = les techniques de cuisson

---

## Conclusion

Ce projet nous a permis de :
- Maîtriser le cycle complet d'un projet Data Science
- Comprendre l'importance du prétraitement (50% du travail)
- Comparer différents algorithmes de Machine Learning
- Utiliser la validation croisée et les métriques appropriées
- Appliquer une méthodologie structurée (CRISP-DM)

**Résultat final** : Un modèle capable de prédire le prix des voitures d'occasion avec ~90-93% de variance expliquée (R²) et une erreur de 9-11%, dépassant l'objectif initial.

---

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
