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
| 4. Modeling | Test de 5 modèles (2 simples en Partie II, 3 avancés en Partie III) |
| 5. Evaluation | Validation croisée 5-fold, métriques (RMSE, MAE, R²), comparaison finale |
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

## Module 3 : Votre Premier Modèle Prédictif

### Question 7.1 : Quels modèles simples avons-nous testés en premier ?
**Réponse** : Nous avons commencé par tester **2 algorithmes de régression simples** :
1. **Régression Linéaire** : modèle de base, simple et interprétable
2. **KNN** (K-Nearest Neighbors) : modèle basé sur la similarité

Ces modèles permettent de comprendre les fondamentaux du machine learning avant de passer aux algorithmes plus avancés.

### Question 7.2 : Quels sont les résultats des modèles simples ?
**Réponse** :

| Modèle | RMSE Test | R² Test | Erreur % | Objectif atteint ? |
|---|---|---|---|---|
| Régression Linéaire | 4 660$ | 0.8225 | 20.17% | ❌ Non |
| KNN (k=3) | 5 407$ | 0.7611 | 20.20% | ❌ Non |

### Question 7.3 : Comment avons-nous choisi les hyperparamètres pour KNN ?
**Réponse** : Test de plusieurs valeurs de k (3, 5, 7, 9, 11, 15, 21) avec validation croisée 5-fold.
- **k optimal** : k=3 (RMSE = 3 612.82$ en validation croisée)
- Principe : k petit = surapprentissage, k grand = sous-apprentissage

### Question 7.4 : Pourquoi les modèles simples n'atteignent-ils pas l'objectif de 15% ?
**Réponse** : Les modèles simples ont des limites intrinsèques :
- **Régression Linéaire** suppose des relations linéaires, alors que la relation prix-features est non-linéaire
- **KNN** souffre de la dimensionnalité élevée (21 features) et du nombre limité de voisins pertinents
- Le dataset est relativement petit (201 véhicules)
- Ces modèles ne capturent pas les interactions complexes entre variables

### Question 7.5 : Quelle est la conclusion de ce premier test ?
**Réponse** : ❌ **Les 2 modèles simples n'atteignent pas l'objectif de 15% d'erreur**

Avec des erreurs autour de 20%, les modèles simples ne suffisent pas pour ce problème. Il est donc nécessaire de tester des algorithmes plus sophistiqués (Partie III - Module 8).

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

# PARTIE III - MACHINE LEARNING SUPERVISÉ AVANCÉ

## Module 8 : Méthodes de Classification Avancées

### Question 11.1 : Pourquoi tester des modèles avancés ?
**Réponse** : Suite aux résultats insuffisants des modèles simples (20% d'erreur), il est nécessaire d'explorer des algorithmes plus sophistiqués pour atteindre l'objectif de 15% d'erreur.

Les modèles à base d'arbres et les méthodes ensemblistes sont particulièrement adaptés pour :
- Capturer les relations non-linéaires
- Gérer les interactions entre variables
- Traiter la multicolinéarité sans problème

### Question 11.2 : Quels modèles avancés avons-nous testés ?
**Réponse** : Nous avons testé **3 algorithmes avancés** basés sur les arbres de décision :
1. **Arbre de Décision** : modèle non-linéaire très interprétable
2. **Random Forest** : ensemble de 100 arbres de décision
3. **Gradient Boosting** : ensemble séquentiel optimisant les erreurs

### Question 11.3 : Quels sont les résultats des modèles avancés ?
**Réponse** :

| Modèle | RMSE Test | R² Test | Erreur % | Objectif atteint ? |
|---|---|---|---|---|
| Gradient Boosting | 2 383$ | 0.9536 | 9.39% | ✅ Oui |
| Random Forest | 2 195$ | 0.9606 | 9.65% | ✅ Oui |
| Arbre de Décision | 2 856$ | 0.9333 | 11.49% | ✅ Oui |

**Constat** : **Les 3 modèles avancés atteignent l'objectif de 15%** ✅

### Question 11.4 : Pourquoi les modèles ensemblistes atteignent-ils l'objectif ?
**Réponse** : Les modèles à base d'arbres ont plusieurs avantages :
- **Arbre de Décision** : capture les relations non-linéaires avec des règles simples
- **Random Forest** : combine 100 arbres pour réduire la variance et améliorer la généralisation
- **Gradient Boosting** : construit séquentiellement des arbres qui corrigent les erreurs des précédents
- Ces modèles gèrent naturellement les interactions entre variables et la multicolinéarité

### Question 11.5 : Quel est le meilleur modèle et pourquoi ?
**Réponse** : **Random Forest** est le meilleur modèle pour ce projet :
- **Performance** : 9.65% d'erreur (largement < 15%), R² = 0.9606
- **Robustesse** : Gap train/test de seulement 663$ (pas de surapprentissage)
- **Stabilité** : Validation croisée confirme la performance
- **Interprétabilité** : Importance des variables accessible

Gradient Boosting a une meilleure erreur (9.39%) mais montre un surapprentissage massif (gap train/test de 2 041$).

### Question 11.6 : Y a-t-il du surapprentissage avec les modèles avancés ?
**Réponse** : Analyse du gap train/test pour chaque modèle avancé :

**Random Forest** : Gap de 663$ ✅ (Excellent équilibre)
- Meilleur compromis généralisation/performance

**Arbre de Décision** : Gap de 936$ ✅ (Léger surapprentissage acceptable)
- La profondeur max=5 limite efficacement le surapprentissage

**Gradient Boosting** : Gap de 2 041$ ⚠️ (Surapprentissage important)
- Performance train exceptionnelle mais généralisation limitée
- Nécessiterait une régularisation plus forte (learning_rate plus faible)

### Question 11.7 : Quelles sont les variables les plus importantes ?
**Réponse** : D'après l'importance des variables du Random Forest :
1. **curb_weight** (poids du véhicule) : 43.2%
2. **engine_size** (taille du moteur) : 31.7%
3. **horsepower** (puissance) : 6.5%
4. **highway_mpg** (consommation autoroute) : 6.4%

Le poids et la taille du moteur représentent **74.9% de l'importance combinée**. Ces résultats confirment l'analyse de corrélation initiale.

### Question 11.8 : Quelle est la comparaison finale de tous les modèles ?
**Réponse** : Comparaison des 5 modèles testés (2 simples + 3 avancés) :

| Modèle | RMSE Test | R² Test | Erreur % | Objectif |
|---|---|---|---|---|
| **Gradient Boosting** | 2 383$ | 0.9536 | **9.39%** | ✅ |
| **Random Forest** | **2 195$** | **0.9606** | 9.65% | ✅ |
| **Arbre de Décision** | 2 856$ | 0.9333 | 11.49% | ✅ |
| Régression Linéaire | 4 660$ | 0.8225 | 20.17% | ❌ |
| KNN (k=3) | 5 407$ | 0.7611 | 20.20% | ❌ |

**Conclusion** :
- **3 modèles sur 5** atteignent l'objectif de 15% d'erreur
- Les modèles ensemblistes surpassent largement les modèles simples
- Random Forest offre le meilleur équilibre performance/robustesse

---

## Limites et Perspectives

### Question 10.1 : Quelles sont les limites du projet ?
**Réponse** :
- **Taille du dataset** : 201 véhicules est petit pour le machine learning
- **Ancienneté** : dataset de 1985, les prix ne sont plus actuels
- **Encodage simple** : LabelEncoder introduit un ordre artificiel (One-Hot serait mieux)
- **Surapprentissage** : Gradient Boosting montre un surapprentissage important
- **Validation** : split unique train/test (80/20), pas de set de validation séparé

### Question 10.2 : Quelles améliorations pour progresser encore ?
**Réponse** :
1. **Feature engineering** : créer de nouvelles variables (ratio puissance/poids, âge du véhicule)
2. **One-Hot Encoding** : meilleur traitement des variables catégorielles que LabelEncoder
3. **GridSearchCV** : optimisation systématique des hyperparamètres
4. **Autres modèles ensemblistes** : tester XGBoost, LightGBM, CatBoost
5. **Dataset plus récent et plus grand** : données actuelles avec plus de véhicules

### Question 10.3 : Le projet atteint-il ses objectifs ?
**Réponse** : **OUI, COMPLÈTEMENT** ✅

**Objectifs pédagogiques atteints** :
- ✅ Comprendre le cycle complet d'un projet Data Science
- ✅ Maîtriser l'exploration et le prétraitement des données
- ✅ Tester et comparer plusieurs modèles (5 au total)
- ✅ Utiliser la validation croisée et les courbes d'apprentissage
- ✅ Appliquer une méthodologie structurée (CRISP-DM)
- ✅ Découvrir le clustering non supervisé

**Objectif de performance** :
- ✅ **Erreur < 15% : OBJECTIF DÉPASSÉ avec 9.65% (Random Forest)**
- ✅ **3 modèles sur 5 atteignent l'objectif** (Arbre, Random Forest, Gradient Boosting)
- ✅ Le meilleur modèle (Random Forest) prédit avec 96.06% de variance expliquée

### Question 10.4 : Qu'avons-nous appris ?
**Réponse** :

**Apprentissages techniques** :
- L'exploration est cruciale avant toute modélisation
- La qualité des données impacte directement les résultats
- Les modèles simples ont des limites intrinsèques pour les problèmes non-linéaires
- Les modèles ensemblistes (Random Forest, Gradient Boosting) surpassent largement les modèles simples
- La validation croisée est essentielle pour estimer la vraie performance
- Le clustering révèle des structures cachées dans les données
- **L'importance de tester plusieurs types de modèles pour trouver le meilleur**

**Apprentissages méthodologiques** :
- CRISP-DM structure efficacement un projet
- La documentation est aussi importante que le code
- Itérer est normal : on n'obtient pas le meilleur modèle du premier coup
- Les visualisations aident à comprendre et à communiquer
- Comparer systématiquement les modèles permet de justifier les choix

**Apprentissages réalistes** :
- 201 véhicules est un dataset petit mais suffisant pour obtenir de bons résultats
- **Progression pédagogique naturelle** :
  - Module 3 (modèles simples) → 20% d'erreur → objectif non atteint
  - Module 8 (modèles avancés) → 9-11% d'erreur → objectif dépassé
- Les modèles simples (Régression Linéaire, KNN) ne suffisent pas pour des objectifs ambitieux
- Les algorithmes plus sophistiqués permettent d'atteindre et dépasser les objectifs
- **Le poids et la taille du moteur expliquent 75% du prix - ces deux variables sont cruciales**
- **Avec les bons algorithmes, l'objectif de 15% EST atteignable (9.65% obtenu) !**

---

## Conclusion

Ce projet nous a permis de découvrir concrètement la Data Science et le Machine Learning en travaillant sur un problème réel de prédiction de prix, de l'exploration initiale jusqu'à la sélection d'un modèle performant.

**Bilan final** :
- ✅ Méthodologie CRISP-DM appliquée de bout en bout
- ✅ Exploration rigoureuse des données (201 véhicules, 26 variables)
- ✅ Prétraitement soigné (imputation, encodage, standardisation)
- ✅ **5 modèles testés et comparés systématiquement**
- ✅ Clustering réussi (4 segments de marché identifiés)
- ✅ Validation croisée et courbes d'apprentissage maîtrisées
- ✅ **Objectif de 15% d'erreur DÉPASSÉ : 9.65% avec Random Forest**

**Résultats clés** :
- **Random Forest** sélectionné comme modèle final (9.65% d'erreur, R² = 0.9606)
- **3 modèles sur 5** atteignent l'objectif (Arbre de Décision, Random Forest, Gradient Boosting)
- Les variables poids (43.2%) et taille moteur (31.7%) dominent l'importance
- 4 segments de marché clairement identifiés (économique, familial, sportif/premium, luxe)

**Enseignement principal** :

Ce projet illustre une **progression pédagogique naturelle en Data Science** :
1. **Partie II - Module 3** : Les modèles simples (Régression Linéaire, KNN) atteignent ~20% d'erreur → objectif non atteint
2. **Partie III - Module 8** : Les modèles avancés (Arbres, Random Forest, Gradient Boosting) atteignent 9-11% d'erreur → objectif largement dépassé

Cette progression démontre qu'avec une méthodologie rigoureuse (CRISP-DM) et le choix des bons algorithmes, même un petit dataset de 201 véhicules permet d'obtenir d'excellentes performances prédictives. L'importance de ne pas se limiter aux approches simples et de tester progressivement des modèles plus sophistiqués est clairement démontrée.

---

**Projet réalisé par Matteo Robin et Florian Huguet**
**Cours d'initiation au Data Science et Machine Learning - Décembre 2025**
