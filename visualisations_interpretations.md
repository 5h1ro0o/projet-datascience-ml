# Visualisations et Interprétations - Projet Prédiction Prix Voitures d'Occasion

**Auteurs** : Florian Huguet, Matteo Robin
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

## Introduction

Ce document présente l'analyse détaillée des visualisations générées dans le cadre du projet de prédiction du prix de voitures d'occasion. L'objectif est de construire un modèle de régression capable de prédire le prix avec une erreur inférieure à 15% du prix réel.

**Structure du document** : Ce projet suit la méthodologie de l'énoncé avec la Partie I (exploration et préparation) et la Partie II (modélisation avec 2 modèles simples adaptés à une initiation).

Toutes les interprétations ci-dessous sont basées exclusivement sur les visualisations produites par notre analyse.

---

# PARTIE I - VOTRE PREMIER PROJET DATA SCIENCE

## Étape 2 : Exploration des Données

**Fichier source** : `visualisations_exploration.png` (4 subplots)

### Visualisation 1.1 : Distribution des Prix

**Observation** : L'histogramme montre une distribution asymétrique à droite (right-skewed) des prix.

**Interprétations** :
- La majorité des véhicules se concentre dans la fourchette 5 000$ - 15 000$
- Un pic principal visible autour de 10 000$, représentant le segment des voitures standards
- Une longue queue vers la droite s'étend jusqu'à 45 000$, correspondant aux véhicules haut de gamme
- Présence de plusieurs véhicules dans la tranche 30 000$ - 45 000$ (véhicules de luxe et sportifs)

**Implications pour la modélisation** :
- La distribution non-normale peut affecter les performances de la régression linéaire classique
- Les modèles ensemblistes seront probablement plus adaptés à cette distribution
- Pas de transformation logarithmique nécessaire vu la distribution relativement équilibrée

---

### Visualisation 1.2 : Top 10 des Marques

**Observation** : Le diagramme en barres horizontales révèle la répartition des marques dans le dataset.

**Interprétations** :
- Toyota domine largement avec environ 30 véhicules
- Les marques japonaises sont fortement représentées : Nissan, Mazda, Honda, Mitsubishi, Subaru
- Présence significative de marques européennes : Volkswagen, Peugeot, Volvo
- Marque américaine Dodge également présente dans le top 10

**Implications** :
- Dataset orienté vers le marché automobile des années 80 avec prédominance japonaise
- L'encodage de la variable "marque" sera important car certaines marques sont associées à des segments premium
- Possible biais géographique/temporel dans les données

---

### Visualisation 1.3 : Prix par Type de Carburant

**Observation** : Les boxplots comparent les distributions de prix entre véhicules diesel et essence.

**Interprétations** :
- Les véhicules diesel présentent une médiane de prix nettement supérieure aux véhicules essence
- La boîte interquartile du diesel est plus haute et plus large, indiquant une variabilité importante
- Plusieurs outliers visibles dans les deux catégories
- Le diesel atteint des prix maximums plus élevés (environ 32 000$) contre environ 45 000$ pour l'essence

**Analyse** :
- Dans les années 80, le diesel était associé aux véhicules plus lourds et haut de gamme
- Le type de carburant est un prédicteur significatif du prix
- Les véhicules essence couvrent un spectre plus large (économique à luxe)

---

### Visualisation 1.4 : Prix par Nombre de Cylindres

**Observation** : Les boxplots montrent une relation croissante entre nombre de cylindres et prix.

**Interprétations** :
- Relation progressive : plus de cylindres = prix plus élevé
- Véhicules 4 cylindres (four) : segment économique, médiane environ 9 000$
- Véhicules 5 cylindres (five) : segment intermédiaire, médiane environ 13 000$
- Véhicules 6 cylindres (six) : segment moyen-premium, médiane environ 15 000$
- Véhicules 8 cylindres (eight) : segment premium, médiane environ 21 000$
- Catégories "twelve" et "two" : outliers avec prix élevés mais peu de représentants

**Analyse** :
- Le nombre de cylindres est un proxy direct de la puissance et du segment de marché
- Variable catégorielle hautement informative pour la prédiction
- Les véhicules 3 cylindres ("three") existent mais sont très rares

---

## Étape 2 (suite) : Analyse des Corrélations

**Fichier source** : `matrice_correlation.png`

### Visualisation 2.1 : Matrice de Corrélation Complète

**Observation** : La heatmap révèle les relations entre toutes les variables numériques du dataset.

**Corrélations fortes avec le prix (> 0.75)** :
1. engine_size (0.872) : La taille du moteur est le prédicteur le plus corrélé
2. curb_weight (0.834) : Le poids du véhicule est fortement corrélé au prix
3. horsepower (0.811) : La puissance est directement liée au prix
4. width (0.751) : La largeur du véhicule indique le segment

**Corrélations moyennes (0.50 - 0.75)** :
- highway_mpg (0.705) : Consommation autoroutière
- length (0.691) : Longueur du véhicule
- city_mpg (0.687) : Consommation urbaine
- wheel_base (0.585) : Empattement
- bore (0.543) : Alésage du cylindre

**Note sur les corrélations de consommation** :
Les valeurs positives de city_mpg et highway_mpg dans la matrice indiquent une corrélation avec les autres variables. La relation inverse avec le prix (plus économique = moins cher) est visible dans les scatter plots.

**Multicolinéarité visible** :
- Les dimensions du véhicule sont fortement corrélées entre elles
- La taille du moteur, le poids et la puissance sont interdépendants
- Les consommations ville/autoroute sont très liées
- Cette multicolinéarité est attendue et normale pour des véhicules

**Implications pour la modélisation** :
- La multicolinéarité peut déstabiliser les coefficients de la régression linéaire
- Les modèles à base d'arbres (Random Forest) ne seront pas affectés
- Possibilité d'éliminer highway_mpg (redondant avec city_mpg)
- Les 3 variables majeures (curb_weight, engine_size, horsepower) capturent l'essentiel

---

## Étape 2 (suite) : Relations Variables-Prix

**Fichier source** : `scatter_plots.png` (6 subplots)

### Visualisation 3.1 : Scatter Plots - Prix vs Variables Clés

Six nuages de points révèlent les relations entre variables explicatives et prix.

#### Plot 3.1.1 : Prix vs Engine Size (Taille du moteur)

**Interprétations** :
- Relation positive quasi-linéaire très nette
- Corrélation visuelle très forte, confirmant le coefficient de 0.872
- Peu de dispersion, la tendance est claire
- Quelques points au-dessus de la ligne de tendance : marques premium avec petit moteur mais prix élevé (ex: voitures de luxe compactes)

#### Plot 3.1.2 : Prix vs Horsepower (Puissance)

**Interprétations** :
- Relation positive avec dispersion modérée
- Certains véhicules peu puissants (<100 HP) atteignent 15 000-20 000$ : véhicules axés luxe/confort plutôt que performance
- Certains véhicules puissants (>150 HP) restent sous 15 000$ : sportives d'entrée de gamme
- La puissance seule ne détermine pas tout le prix

#### Plot 3.1.3 : Prix vs Curb Weight (Poids)

**Interprétations** :
- Relation positive et linéaire, très nette
- Le poids est un excellent prédicteur du prix
- Les véhicules lourds (>3 500 lbs) sont systématiquement chers
- Le poids reflète à la fois la taille, les matériaux et le segment

#### Plot 3.1.4 : Prix vs Length (Longueur)

**Interprétations** :
- Relation positive mais avec dispersion importante
- Les grandes berlines sont chères, mais pas systématiquement
- Certains breaks familiaux sont longs mais abordables
- La longueur seule est un prédicteur moins fiable

#### Plot 3.1.5 : Prix vs Width (Largeur)

**Interprétations** :
- Tendance similaire à la longueur
- Largeur = indicateur de confort et standing
- Relation positive mais avec variabilité

#### Plot 3.1.6 : Prix vs City MPG (Consommation ville)

**Interprétations** :
- Relation négative claire : plus le MPG est élevé (= consommation faible), plus le prix est bas
- Les véhicules économiques (>30 MPG) sont concentrés sous 15 000$
- Les véhicules de luxe/sport consomment beaucoup (15-25 MPG) et coûtent cher
- Logique économique : performance et luxe vs économie de carburant

---

## Étape 3 : Détection des Outliers (Prétraitement)

**Fichier source** : `boxplots_outliers.png` (6 subplots)

### Visualisation 4.1 : Boxplots des Variables Clés

**Objectif** : Identifier les valeurs aberrantes sur 6 variables importantes.

#### Price (Prix)

**Outliers identifiés** :
- Plusieurs véhicules au-dessus de 30 000$
- Maximum à environ 45 000$ (très probablement Porsche, Jaguar)

**Décision** : Conservés - ce sont des segments légitimes (luxe/sport)

#### Engine Size (Taille du moteur)

**Outliers identifiés** :
- 2-3 moteurs > 250 cubic inches (>4.1L)
- Probablement des grosses cylindrées américaines ou allemandes

**Décision** : Conservés - valeurs réelles

#### Horsepower (Puissance)

**Outliers identifiés** :
- Quelques véhicules au-dessus de 200 HP
- Maximum vers 260 HP (sportives)

**Décision** : Conservés - segment légitime

#### Curb Weight (Poids)

**Outliers identifiés** :
- Un véhicule dépassant 4 000 lbs
- Probablement SUV ou grosse berline

**Décision** : Conservé - valeur réaliste

#### Length & Width

**Outliers identifiés** :
- Quelques véhicules particulièrement grands
- Length > 200 inches, Width > 72 inches

**Décision** : Conservés - limousines ou breaks

**Conclusion générale** : Tous les outliers détectés représentent des segments réels du marché automobile et ont été conservés dans le dataset.

## Étape 4 : Méthodologies Data Science

**Méthodologie suivie** : CRISP-DM (Cross-Industry Standard Process for Data Mining)

**Pourquoi CRISP-DM** :
- La plus utilisée en entreprise
- Intègre la dimension métier dès le début
- Approche itérative et pratique
- Mieux adaptée qu'un projet d'initiation que KDD ou SEMMA

**Application à notre projet** :
1. **Business Understanding** : Prédire le prix de voitures d'occasion avec erreur < 15%
2. **Data Understanding** : Exploration de 201 véhicules, 26 variables, identification des corrélations
3. **Data Preparation** : Gestion des valeurs manquantes, encodage, standardisation
4. **Modeling** : Test de 2 modèles simples (Régression Linéaire, KNN)
5. **Evaluation** : Validation croisée, métriques RMSE/R²/Erreur%
6. **Deployment** : Documentation et rapport final

---

# PARTIE II - MACHINE LEARNING EN PRATIQUE

## Module 3 : Votre Premier Modèle Prédictif

### A) Régression Linéaire

**Fichier source** : `regression_lineaire_predictions.png`

**Observation** : Scatter plot des prédictions vs valeurs réelles avec ligne de référence y=x.

**Interprétations** :
- Les points sont globalement proches de la diagonale (bonnes prédictions)
- Alignement correct pour les prix 5 000$ - 25 000$
- Dispersion visible pour les prix > 30 000$ : le modèle sous-estime plusieurs véhicules haut de gamme
- Pas de biais systématique majeur
- Quelques prédictions autour de 25 000-28 000$ pour des véhicules à 31 000$+

**Résultats du modèle** :
- RMSE Test : 4 660.42$
- R² Test : 0.8225 (82.25% de variance expliquée)
- Erreur moyenne : 20.17%

**Analyse pour niveau initiation** :
- Le modèle linéaire est simple et interprétable
- Performance correcte mais ne parvient pas à l'objectif de 15% d'erreur
- Limitation : suppose une relation linéaire entre les variables et le prix

---

### B) KNN (K-Nearest Neighbors)

**Fichier source** : `knn_optimisation_k.png`

**Fichier source** : `knn_optimisation_k.png`

**Observation** : Courbe du RMSE en fonction du nombre de voisins k.

**Interprétations** :

**k = 3** : RMSE = 3 612.82$
- Meilleur RMSE en validation croisée
- Mais risque de surapprentissage avec peu de voisins

**k = 5** : RMSE = 4 000.78$
- Légère augmentation du RMSE

**k = 7 à k = 11** : RMSE = 4 112$ - 4 165$
- Zone de stabilité

**k = 15 et k = 21** : RMSE = 4 231$ - 4 331$
- Modèle trop simple, sous-apprentissage
- Les prédictions deviennent trop générales

**Résultats du modèle** :
- k optimal : 3 (RMSE = 3 612.82$ en validation croisée)
- RMSE Test avec k=3 : 5 406.69$
- R² Test : 0.7611 (76.11% de variance expliquée)
- Erreur moyenne : 20.20%

**Enseignement pour niveau initiation** :
- k = 3 donne le meilleur RMSE en validation croisée
- La courbe montre une augmentation progressive au-delà de k=3
- Confirmation du principe biais-variance : k petit = surapprentissage, k grand = sous-apprentissage
- KNN ne parvient pas non plus à l'objectif de 15% d'erreur sur ce dataset

**Comparaison Régression Linéaire vs KNN** :
- Les deux modèles ont des performances similaires (autour de 20% d'erreur)
- Régression Linéaire : plus simple, plus rapide, plus interprétable
- KNN : plus flexible mais nécessite standardisation et choix de k

---

## Module 5 : Validation et Diagnostics

### Validation Croisée 5-fold

**Résultats obtenus** :
- Régression Linéaire : RMSE moyen = 3 097.20$ (±429.72)
- KNN (k=3) : RMSE moyen = 3 523.63$ (±797.80)

**Interprétation** :
- La validation croisée confirme que les deux modèles ont des performances similaires
- L'écart-type indique la stabilité du modèle
- Régression Linéaire est légèrement plus stable (écart-type plus faible)

### Courbes d'Apprentissage

**Fichier source** : `courbes_apprentissage.png`

**Observation** : Évolution du RMSE train et validation en fonction de la taille du dataset (pour la Régression Linéaire).

**Observations** :
- **Début de courbe** (peu d'exemples) : Grand écart entre train et validation → surapprentissage
- **Milieu de courbe** : Les courbes se rapprochent progressivement
- **Fin de courbe** : Les courbes convergent vers une performance stable

**Diagnostic pour niveau initiation** :
- Pas de sous-apprentissage : la performance s'améliore avec plus de données
- Surapprentissage initial se réduit avec plus d'exemples
- Le modèle généralise correctement sur le dataset complet
- Plus de données n'améliorerait que marginalement les performances

**Conclusion** : Le modèle est correctement calibré pour ce dataset de 201 véhicules.

---

## Module 4 : Clustering - Découvrir des Groupes Cachés

### Visualisation 7.1 : Méthode du Coude - K-means

**Fichier source** : `kmeans_coude.png`

**Observation** : Courbe de l'inertie (somme des distances intra-cluster) en fonction de k.

**Analyse** :

**k = 2** : Inertie très élevée (environ 335)
- Seulement 2 clusters, trop large, perte d'information

**k = 3** : Inertie descend à environ 193
- Amélioration significative

**k = 4** : Inertie environ 147
- Coude visible : point d'inflexion de la courbe
- Amélioration notable mais la décroissance ralentit

**k = 5 à k = 10** : Inertie continue de baisser progressivement
- Amélioration marginale décroissante
- Complexité accrue sans gain proportionnel

**Choix optimal** : k = 4 clusters
- Point où le rapport amélioration/complexité est optimal
- Interprétation métier : 4 segments de marché distincts

---

### Visualisation 7.2 : Visualisation des Clusters

**Fichier source** : `clusters_visualisation.png` (2 subplots)

**Description** : Deux scatter plots colorés par cluster.

#### Plot gauche : Taille moteur vs Puissance

**Cluster 0 (jaune)** :
- Petits moteurs (50-80 cu.in), faible puissance (45-80 HP)
- Segment économique : citadines, compactes
- Exemples : Honda Civic, Toyota Corolla, petites Mazda

**Cluster 1 (bleu clair)** :
- Moteurs moyens (90-130 cu.in), puissance moyenne (80-110 HP)
- Segment familial : berlines moyennes, véhicules polyvalents
- Exemples : Volkswagen, Nissan Maxima, Honda Accord

**Cluster 2 (violet/mauve)** :
- Moteurs moyens/gros (140-180 cu.in), puissance élevée (110-160 HP)
- Segment sportif/premium : coupés sportifs, berlines dynamiques
- Exemples : Mazda RX-7, Alfa Romeo, BMW entrée de gamme

**Cluster 3 (vert/turquoise)** :
- Gros moteurs (>180 cu.in), très haute puissance (>160 HP)
- Segment luxe/sport : grosses berlines, sportives haut de gamme
- Exemples : BMW Série 7, Mercedes, Jaguar, Porsche

**Séparation** : Les clusters sont bien distincts avec peu de chevauchement.

#### Plot droit : Poids vs Prix

**Relation cluster-prix** :

**Cluster 0 (jaune)** :
- Poids : 1 500-2 300 lbs
- Prix : 5 000-10 000$
- Véhicules économiques

**Cluster 1 (bleu)** :
- Poids : 2 300-2 800 lbs
- Prix : 8 000-17 000$
- Véhicules familiaux

**Cluster 2 (violet)** :
- Poids : 2 500-3 200 lbs
- Prix : 12 000-25 000$
- Véhicules sportifs/premium

**Cluster 3 (vert)** :
- Poids : 3 000-4 000 lbs
- Prix : 20 000-45 000$
- Véhicules luxe/sport haut de gamme

**Analyse marketing** :
- Segmentation claire et exploitable commercialement
- Chaque cluster correspond à un persona client différent
- Stratégies marketing et pricing adaptables par cluster
- Les 4 segments reflètent la réalité du marché automobile

---

### Visualisation 7.3 : Dendrogramme - Classification Hiérarchique

**Fichier source** : `dendrogramme.png`

**Observation** : Arbre hiérarchique montrant les fusions successives de 50 véhicules.

**Lecture** :
- Axe horizontal : index des véhicules (0-50)
- Axe vertical : distance de fusion (mesure de dissimilarité)
- Hauteur de fusion : plus elle est élevée, plus les groupes sont différents

**Analyse** :

**Niveau bas (distance 0-5)** :
- Fusions rapides de véhicules très similaires
- Plusieurs petits groupes de 2-4 véhicules (même modèle, motorisation différente)
- Exemples : véhicules 46-44-45 forment un groupe homogène

**Niveau moyen (distance 5-10)** :
- Regroupements de sous-segments
- Formation de clusters plus larges

**Niveau haut (distance 10-20)** :
- Fusions majeures entre grands groupes
- Si on coupe à hauteur 10-15, on obtient 4 grands clusters
- Cohérence avec la méthode K-means

**Véhicules isolés** :
- Certaines branches s'isolent tardivement (ex: indices 46, 22)
- Ce sont des véhicules atypiques ou outliers

**Avantages du dendrogramme** :
- Vision hiérarchique complète de la structure des données
- Permet de choisir le niveau de granularité souhaité
- Identifie les véhicules similaires et les outliers
- Complément visuel à K-means

---

## Synthèse et Apprentissages (Niveau Initiation)

### Ce que nous avons réalisé

**Partie I - Exploration et Préparation** :
- Exploration complète d'un dataset de 201 véhicules avec 26 variables
- Identification des corrélations clés : engine_size (0.872), curb_weight (0.834), horsepower (0.811)
- Gestion rigoureuse des valeurs manquantes (imputation médiane/mode)
- Conservation justifiée des outliers (segments légitimes du marché)
- Application de la méthodologie CRISP-DM

**Partie II - Modélisation Basique** :
- Test de 2 modèles simples adaptés à une initiation :
  - **Régression Linéaire** : Simple, rapide, interprétable (erreur 20.17%)
  - **KNN** : Flexible mais nécessite optimisation de k (erreur 20.20%)
- Validation croisée 5-fold pour estimer la stabilité
- Analyse des courbes d'apprentissage
- Segmentation du marché en 4 clusters distincts avec K-means

### Résultats obtenus

**Performance des modèles testés** :
| Modèle | RMSE Test | R² Test | Erreur % | Objectif atteint? |
|--------|-----------|---------|----------|-------------------|
| Régression Linéaire | 4 660$ | 0.8225 | 20.17% | ❌ Non |
| KNN (k=3) | 5 407$ | 0.7611 | 20.20% | ❌ Non |

**Constat pour niveau initiation** :
- Aucun des 2 modèles simples n'atteint l'objectif de 15% d'erreur
- Les deux modèles ont des performances similaires (environ 20% d'erreur)
- La Régression Linéaire est légèrement plus stable et plus simple à interpréter

**Segmentation marché** :
- 4 clusters identifiés avec K-means : économique, familial, sportif/premium, luxe
- Segmentation cohérente confirmée par le dendrogramme
- Variables clés : engine_size, horsepower, curb_weight, price

### Variables les plus importantes

D'après l'analyse de corrélation :
1. **engine_size** (0.872) - Taille du moteur
2. **curb_weight** (0.834) - Poids du véhicule
3. **horsepower** (0.811) - Puissance
4. **width** (0.751) - Largeur

Ces 4 variables expliquent la majeure partie de la variance du prix.

### Limites identifiées

**Limites des modèles simples** :
- La régression linéaire suppose des relations linéaires (limitation forte)
- KNN souffre de la dimensionnalité élevée (21 features)
- Les deux modèles peinent sur les véhicules haut de gamme (>30 000$)

**Limites du projet** :
- Dataset petit (201 véhicules) pour le machine learning
- Données anciennes (1985) - prix non actuels
- Encodage label simple (pourrait être amélioré avec one-hot encoding)

### Perspectives d'amélioration

**Pour atteindre l'objectif de 15% d'erreur, il faudrait** :
- Tester des modèles plus complexes (arbres de décision, ensembles de modèles)
- Faire du feature engineering (créer de nouvelles variables)
- Optimiser les hyperparamètres avec GridSearch
- Utiliser un dataset plus grand et plus récent

**Apprentissages pour deux étudiants en initiation** :
- Importance de l'exploration des données avant toute modélisation
- La méthodologie CRISP-DM structure bien un projet data science
- Les modèles simples ne suffisent pas toujours - c'est normal !
- La validation croisée est essentielle pour estimer la performance réelle
- Le clustering permet de découvrir des structures cachées dans les données

---

## Conclusion

Ce projet d'initiation nous a permis de mettre en pratique l'ensemble du cycle de vie d'un projet de Data Science, de l'exploration des données à la modélisation en passant par le prétraitement rigoureux.

**Bilan réaliste** :
- ✅ Méthodologie complète appliquée (CRISP-DM)
- ✅ Exploration et préparation rigoureuses
- ✅ 2 modèles simples testés et validés
- ✅ Segmentation marché réussie (4 clusters)
- ❌ Objectif de 15% d'erreur non atteint avec les modèles basiques

**Enseignement principal** : Les modèles simples (Régression Linéaire, KNN) ne permettent pas d'atteindre l'objectif ambitieux de 15% d'erreur sur ce dataset. Pour progresser, il faudrait explorer des algorithmes plus sophistiqués lors de modules avancés.

---

---

**Projet réalisé par Florian Huguet et Matteo Robin**
**Cours d'initiation au Data Science et Machine Learning - Décembre 2025**
