# Questions et Réponses - Projet Prédiction Prix Voitures d'Occasion

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

# PARTIE I - VOTRE PREMIER PROJET DATA SCIENCE

## Étape 1 : Choix du Problème

### Question 1.1 : Quel est le contexte du projet ?
**Réponse** : Ce projet vise à prédire le prix de voitures d'occasion en utilisant leurs caractéristiques techniques et descriptives. Il s'agit d'un problème de régression supervisée dans le cadre d'un module d'initiation.

### Question 1.2 : Quel est le problème à résoudre ?
**Réponse** : Comment prédire automatiquement le prix d'une voiture d'occasion en fonction de ses attributs (marque, taille moteur, puissance, poids, type de carburant, etc.) ?

### Question 1.3 : Quels sont les critères de succès ?
**Réponse** : L'objectif est d'obtenir un modèle avec une erreur de prédiction inférieure à 15% du prix réel (MAE/Prix moyen < 15%).

---

## Étape 2 : Explorer les Données

### Question 2.1 : Combien d'observations et de variables avons-nous ?
**Réponse** : Le dataset contient initialement 205 observations (véhicules) et 26 variables. Après suppression des lignes sans prix, nous avons 201 observations exploitables.

### Question 2.2 : Quels types de données avons-nous ?
**Réponse** :
- **Variables numériques** : prix, taille moteur, puissance, poids, longueur, largeur, hauteur, consommation, etc. (16 variables)
- **Variables catégorielles** : marque, type de carburant, transmission, type de carrosserie, nombre de portes, etc. (10 variables)

### Question 2.3 : Y a-t-il des valeurs manquantes ?
**Réponse** : Oui, plusieurs variables contiennent des valeurs manquantes :
- `normalized_losses` : 41 valeurs manquantes (20%)
- `price` : 4 valeurs manquantes (2%)
- `bore`, `stroke`, `horsepower`, `peak_rpm` : 2-4 valeurs manquantes chacune

### Question 2.4 : Y a-t-il des valeurs aberrantes (outliers) ?
**Réponse** : Oui, des outliers ont été détectés dans plusieurs variables :
- Prix : quelques véhicules très chers (voitures de luxe/sport jusqu'à 45 000$)
- Puissance : certains véhicules très puissants (>200 HP)
- Taille du moteur : quelques moteurs exceptionnellement grands (>250 cu.in)

**Décision** : Ces outliers sont des valeurs extrêmes mais valides (véhicules haut de gamme), nous les avons conservés.

### Question 2.5 : Quelles sont les corrélations importantes avec le prix ?
**Réponse** : Les variables les plus corrélées avec le prix (valeurs exactes du notebook) :
1. **engine_size** (taille moteur) : 0.872
2. **curb_weight** (poids) : 0.834
3. **horsepower** (puissance) : 0.811
4. **width** (largeur) : 0.751
5. **length** (longueur) : 0.691

Les voitures avec des moteurs plus gros, plus puissantes et plus lourdes ont tendance à être plus chères.

---

## Étape 3 : Nettoyer et Préparer les Données

### Question 3.1 : Comment avons-nous géré les valeurs manquantes ?
**Réponse** : Deux stratégies selon le type de variable :
- **Variables numériques** : imputation par la médiane (robuste aux outliers)
- **Variables catégorielles** : imputation par le mode (valeur la plus fréquente)
- **Variable cible (prix)** : suppression des lignes (4 lignes supprimées)

**Justification** : La médiane est robuste aux valeurs extrêmes et préserve la distribution centrale des données.

### Question 3.2 : Comment avons-nous traité les valeurs aberrantes ?
**Réponse** : Nous avons choisi de **conserver** les outliers car :
- Ils représentent des véhicules réels (voitures de luxe, sportives)
- Leur suppression réduirait la diversité du dataset
- Ce sont des segments légitimes du marché

### Question 3.3 : Pourquoi avoir encodé les variables catégorielles avec LabelEncoder ?
**Réponse** : LabelEncoder transforme les catégories textuelles en valeurs numériques (0, 1, 2, ...), ce qui est nécessaire pour les algorithmes de machine learning.

**Exemple** :
- `fuel_type` : diesel → 0, gas → 1
- `make` : audi → 0, bmw → 1, honda → 2, etc.

**Limites pour niveau initiation** : Cette approche introduit un ordre artificiel entre les catégories. Pour un projet plus avancé, on utiliserait le One-Hot Encoding.

### Question 3.4 : Pourquoi standardiser les données ?
**Réponse** : La standardisation (Z-score normalization) est cruciale pour :
- **KNN** : les distances doivent être calculées sur des échelles comparables
- **Régression Linéaire** : améliore la convergence et l'interprétation des coefficients

Les variables ont des échelles très différentes (prix : 5000-45000, puissance : 48-288), la standardisation les ramène à une distribution centrée réduite (moyenne=0, écart-type=1).

### Question 3.5 : Quelles variables avons-nous sélectionnées pour la modélisation ?
**Réponse** : Nous avons utilisé 21 features :
- 14 variables numériques continues
- 7 variables catégorielles encodées

Nous avons exclu `normalized_losses` (trop de valeurs manquantes, 20%).

---

## Étape 4 : Comprendre les Méthodologies

### Question 4.1 : Quelle méthodologie avons-nous suivie ?
**Réponse** : Nous avons suivi **CRISP-DM** (Cross-Industry Standard Process for Data Mining), la méthodologie la plus utilisée en entreprise.

### Question 4.2 : Comment notre projet s'articule-t-il avec CRISP-DM ?
**Réponse** :

| Phase CRISP-DM | Notre implémentation |
|---|---|
| 1. Business Understanding | Définition de la problématique : prédire le prix des voitures d'occasion avec erreur < 15% |
| 2. Data Understanding | Exploration : statistiques descriptives, visualisations, corrélations sur 201 véhicules |
| 3. Data Preparation | Nettoyage : gestion des valeurs manquantes, encodage label, standardisation |
| 4. Modeling | Entraînement de 2 modèles simples : Régression Linéaire et KNN |
| 5. Evaluation | Validation croisée 5-fold, métriques (RMSE, MAE, R²) |
| 6. Deployment | Documentation et rapport final |

### Question 4.3 : Pourquoi CRISP-DM plutôt que KDD ou SEMMA ?
**Réponse** :
- **CRISP-DM** intègre la dimension métier dès le début et est itérative
- **KDD** est plus orienté extraction de connaissances pure
- **SEMMA** est très technique (développée par SAS)

Pour un projet d'initiation, CRISP-DM offre le meilleur équilibre entre rigueur technique et pragmatisme métier.

---

# PARTIE II - MACHINE LEARNING EN PRATIQUE

## Module 1 : Cycle de Vie d'un Projet ML

### Question 5.1 : Quels sont les acteurs impliqués dans un projet ML ?
**Réponse** : Dans un projet ML complet :
- **Chef de projet / Expert métier** : définit le problème et valide les résultats
- **Data engineer** : collecte et prépare les données
- **Data scientist** : construit et entraîne le modèle
- **Équipe métier** : valide la pertinence des prédictions

**Pour notre projet d'initiation** : Nous (Florian et Matteo) avons joué tous ces rôles !

### Question 5.2 : Quels sont les livrables d'un projet ML ?
**Réponse** :
- Cahier des charges (définition du problème)
- Rapport d'exploration des données
- Données nettoyées et préparées
- Modèles entraînés
- Documentation technique
- Rapport final avec recommandations

---

## Module 2 : Data Science vs Machine Learning

### Question 6.1 : Quelle est la différence entre Data Science et Machine Learning ?
**Réponse** :
- **Data Science** : Démarche complète (exploration, nettoyage, visualisation, statistiques, communication, storytelling)
- **Machine Learning** : Sous-ensemble de la Data Science focalisé sur les algorithmes prédictifs

**Analogie** : Data Science = toute la cuisine, Machine Learning = les techniques de cuisson

### Question 6.2 : Dans notre projet, quelles tâches relèvent de chaque domaine ?
**Réponse** :

| Tâche | Data Science | Machine Learning |
|---|---|---|
| Exploration des données | ✓ | |
| Visualisations | ✓ | |
| Calcul de statistiques | ✓ | |
| Nettoyage des données | ✓ | |
| Entraînement de modèles | | ✓ |
| Optimisation des hyperparamètres | | ✓ |
| Validation croisée | | ✓ |

---

## Module 3 : Premier Modèle Prédictif

### Question 7.1 : Quels modèles avons-nous testés ?
**Réponse** : Pour un niveau initiation, nous avons testé **2 algorithmes simples** de régression :
1. **Régression Linéaire** : modèle de base, simple et interprétable
2. **KNN** (K-Nearest Neighbors) : modèle basé sur la similarité

### Question 7.2 : Comment avons-nous choisi les hyperparamètres pour KNN ?
**Réponse** : Test de plusieurs valeurs de k (3, 5, 7, 9, 11, 15, 21) avec validation croisée 5-fold.
- **k optimal** : k=3 (RMSE = 3 612.82$ en validation croisée)
- Principe : k petit = surapprentissage, k grand = sous-apprentissage

### Question 7.3 : Quels sont les résultats de chaque modèle ?
**Réponse** :

| Modèle | RMSE Test | R² Test | Erreur % | Objectif atteint ? |
|---|---|---|---|---|
| Régression Linéaire | 4 660$ | 0.8225 | 20.17% | ❌ Non |
| KNN (k=3) | 5 407$ | 0.7611 | 20.20% | ❌ Non |

### Question 7.4 : Pourquoi les deux modèles n'atteignent-ils pas l'objectif de 15% ?
**Réponse** : **Pour un niveau initiation, c'est normal !**
- Les modèles simples ont des limites :
  - Régression Linéaire suppose des relations linéaires
  - KNN souffre de la dimensionnalité élevée (21 features)
- Le dataset est petit (201 véhicules)
- Pour atteindre 15%, il faudrait tester des modèles plus complexes (arbres, ensembles) - ce qui est au programme des modules avancés

### Question 7.5 : Y a-t-il du surapprentissage (overfitting) ?
**Réponse** : Oui, KNN montre un surapprentissage significatif :
- Grand écart entre RMSE train (2 026$) et RMSE test (5 407$)
- Gap de 3 380$ indique que le modèle mémorise les données d'entraînement

La Régression Linéaire est plus stable avec un écart moindre.

### Question 7.6 : Quelles sont les variables les plus importantes ?
**Réponse** : D'après l'analyse de corrélation :
1. **curb_weight** (poids du véhicule) : 0.834
2. **engine_size** (taille du moteur) : 0.872
3. **horsepower** (puissance) : 0.811
4. **width** (largeur) : 0.751

Ces 4 variables expliquent la majeure partie de la variance du prix.

---

## Module 4 : Clustering

### Question 8.1 : Pourquoi faire du clustering sur ce problème de régression ?
**Réponse** : Le clustering (analyse non supervisée) permet de :
- Découvrir des segments naturels de véhicules
- Comprendre la structure du marché automobile
- Identifier des profils types (économiques, sportifs, luxueux, familiaux)

Ce n'est pas pour prédire le prix, mais pour **explorer** les données !

### Question 8.2 : Comment avons-nous choisi le nombre de clusters ?
**Réponse** : Méthode du coude (elbow method) :
- Tracer l'inertie en fonction du nombre de clusters k
- Identifier le "coude" où l'inertie décroît moins rapidement
- **Choix** : k=4 clusters

### Question 8.3 : Quels sont les 4 segments identifiés ?
**Réponse** :

| Cluster | Caractéristiques | Exemples |
|---|---|---|
| 0 (Économique) | Petit moteur, faible puissance, léger, prix bas | Honda Civic, Toyota Corolla |
| 1 (Familial) | Moteur moyen, puissance moyenne, prix moyen | Volkswagen, Nissan, Honda Accord |
| 2 (Sportif/Premium) | Moteur moyen-gros, haute puissance, prix élevé | Mazda RX-7, Alfa Romeo, BMW |
| 3 (Luxe) | Gros moteur, très haute puissance, prix très élevé | Mercedes, Jaguar, Porsche |

### Question 8.4 : Pourquoi utiliser K-means plutôt que la classification hiérarchique ?
**Réponse** :
- **K-means** : plus rapide, adapté aux datasets moyens, facile à comprendre pour une initiation
- **Classification hiérarchique** : plus lente, mais offre une vision complète via le dendrogramme

Pour un projet d'initiation, K-means est plus accessible. Le dendrogramme sert de **complément visuel** pour vérifier la cohérence.

---

## Module 5 : Ajuster et Diagnostiquer les Modèles

### Question 9.1 : Qu'est-ce que la validation croisée et pourquoi l'utiliser ?
**Réponse** : La validation croisée 5-fold divise les données en 5 parties. Le modèle est entraîné 5 fois :
- Chaque fois sur 4 parties (80%)
- Testé sur la 5ème partie (20%)

**Avantages** :
- Estimation plus fiable de la performance
- Mesure de la stabilité du modèle (écart-type)
- Réduit le risque de biais dû à un seul découpage train/test

**Nos résultats** :
- Régression Linéaire : RMSE moyen = 3 097$ (±430) → Stable
- KNN : RMSE moyen = 3 524$ (±798) → Plus variable

### Question 9.2 : Quelles métriques avons-nous utilisées ?
**Réponse** :
- **RMSE** (Root Mean Squared Error) : pénalise fortement les grandes erreurs, en dollars
- **MAE** (Mean Absolute Error) : erreur moyenne absolue, plus intuitive
- **R²** (Coefficient de détermination) : proportion de variance expliquée (0 à 1, 1 = parfait)
- **Erreur %** : MAE/Prix moyen × 100, pour vérifier l'objectif des 15%

### Question 9.3 : Comment interpréter les courbes d'apprentissage ?
**Réponse** : Les courbes d'apprentissage montrent l'évolution du RMSE en fonction de la taille du dataset :
- **Début** : Grand écart train/test → surapprentissage (peu de données)
- **Milieu** : Les courbes se rapprochent progressivement
- **Fin** : Convergence → le modèle généralise correctement

**Notre diagnostic** : Le modèle généralise bien avec le dataset complet de 201 véhicules. Plus de données n'améliorerait que marginalement les performances.

---

## Limites et Perspectives

### Question 10.1 : Quelles sont les limites du projet ?
**Réponse** :
- **Taille du dataset** : 201 véhicules est petit pour le machine learning
- **Ancienneté** : dataset de 1985, les prix ne sont plus actuels
- **Encodage simple** : LabelEncoder introduit un ordre artificiel
- **Modèles simples** : Régression Linéaire et KNN ont des limites intrinsèques
- **Validation** : split unique train/test (80/20), pas de set de validation séparé

### Question 10.2 : Quelles améliorations pour progresser ?
**Réponse** :
1. **Tester des modèles plus complexes** : arbres de décision, Random Forest, Gradient Boosting (modules avancés)
2. **Feature engineering** : créer de nouvelles variables (ratio puissance/poids, âge du véhicule)
3. **One-Hot Encoding** : meilleur traitement des variables catégorielles
4. **GridSearchCV** : optimisation systématique des hyperparamètres
5. **Dataset plus récent et plus grand** : données actuelles avec plus de véhicules

### Question 10.3 : Le projet atteint-il ses objectifs pour une initiation ?
**Réponse** : **OUI** ✅

**Objectifs pédagogiques atteints** :
- ✅ Comprendre le cycle complet d'un projet Data Science
- ✅ Maîtriser l'exploration et le prétraitement des données
- ✅ Tester 2 modèles simples et les comparer
- ✅ Utiliser la validation croisée
- ✅ Appliquer une méthodologie structurée (CRISP-DM)
- ✅ Découvrir le clustering non supervisé

**Objectif de performance** :
- ❌ Erreur < 15% : non atteint avec les modèles simples (20% d'erreur)
- **Mais c'est normal pour une initiation !** Les modèles complexes sont au programme des modules avancés

### Question 10.4 : Qu'avons-nous appris ?
**Réponse** :

**Apprentissages techniques** :
- L'exploration est cruciale avant toute modélisation
- La qualité des données impacte directement les résultats
- Les modèles simples ont des limites - c'est normal !
- La validation croisée est essentielle pour estimer la vraie performance
- Le clustering révèle des structures cachées dans les données

**Apprentissages méthodologiques** :
- CRISP-DM structure efficacement un projet
- La documentation est aussi importante que le code
- Itérer est normal : on n'obtient pas le meilleur modèle du premier coup
- Les visualisations aident à comprendre et à communiquer

**Apprentissages réalistes** :
- 201 véhicules est un dataset petit pour du ML
- Les modèles simples (Régression Linéaire, KNN) ne suffisent pas toujours
- Pour atteindre des objectifs ambitieux (15% d'erreur), il faut des algorithmes plus sophistiqués
- C'est OK de ne pas atteindre l'objectif avec des méthodes basiques !

---

## Conclusion

Ce projet d'initiation nous a permis de découvrir concrètement la Data Science et le Machine Learning en travaillant sur un problème réel de prédiction de prix.

**Bilan** :
- ✅ Méthodologie CRISP-DM appliquée de bout en bout
- ✅ Exploration rigoureuse des données
- ✅ Prétraitement soigné (imputation, encodage, standardisation)
- ✅ 2 modèles simples testés et comparés
- ✅ Clustering réussi (4 segments de marché)
- ✅ Validation croisée et courbes d'apprentissage maîtrisées
- ❌ Objectif de 15% d'erreur non atteint (20% avec les modèles basiques)

**Enseignement principal** : Les modèles simples (Régression Linéaire, KNN) ne permettent pas toujours d'atteindre des objectifs ambitieux. Pour progresser, il faudra explorer des algorithmes plus sophistiqués (arbres, ensembles) lors de modules avancés - et c'est exactement ce qui est prévu dans la progression pédagogique !

---

**Projet réalisé par Matteo Robin et Florian Huguet**
**Cours d'initiation au Data Science et Machine Learning - Décembre 2025**
