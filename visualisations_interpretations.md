# Visualisations et Interprétations - Projet Prédiction Prix Voitures d'Occasion

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

## Table des Matières

1. [Exploration des Données](#1-exploration-des-données)
2. [Corrélations](#2-corrélations)
3. [Relations Variables-Prix](#3-relations-variables-prix)
4. [Détection des Outliers](#4-détection-des-outliers)
5. [Résultats des Modèles](#5-résultats-des-modèles)
6. [Validation et Diagnostics](#6-validation-et-diagnostics)
7. [Clustering](#7-clustering)
8. [Comparaisons Finales](#8-comparaisons-finales)

---

## 1. Exploration des Données

### Visualisation 1.1 : Distribution des Prix
**Fichier** : `visualisations_exploration.png` (subplot 1)

**Description** : Histogramme de la distribution des prix des véhicules.

**Interprétation** :
- La distribution est **asymétrique à droite** (right-skewed)
- La majorité des véhicules se situent entre 5 000$ et 20 000$
- Un pic principal vers 10 000$-15 000$ correspond aux voitures standards
- Une queue longue vers la droite (jusqu'à 45 000$) représente les véhicules haut de gamme
- Présence de quelques véhicules très chers (outliers légitimes : voitures de luxe/sport)

**Implications pour la modélisation** :
- La distribution non normale peut affecter la régression linéaire
- Les modèles ensemblistes (Random Forest) sont plus adaptés à ce type de distribution

---

### Visualisation 1.2 : Top 10 des Marques
**Fichier** : `visualisations_exploration.png` (subplot 2)

**Description** : Diagramme en barres horizontales montrant les 10 marques les plus représentées.

**Interprétation** :
- **Toyota** est la marque la plus représentée (~30 véhicules)
- Suivent **Nissan, Mazda, Honda** (marques japonaises dominantes)
- Présence de marques européennes comme **Audi, BMW, Volkswagen**
- Les marques américaines sont moins représentées

**Implications** :
- Le dataset est fortement orienté vers les marques japonaises des années 80
- Possible biais géographique/temporel dans les données
- L'encodage de la marque sera une feature importante (certaines marques = premium)

---

### Visualisation 1.3 : Prix par Type de Carburant
**Fichier** : `visualisations_exploration.png` (subplot 3)

**Description** : Boxplots comparant les prix selon le type de carburant (gas vs diesel).

**Interprétation** :
- Les véhicules **diesel** ont une médiane de prix **plus élevée** que les véhicules essence
- Plus grande variabilité des prix pour le diesel (interquartile range plus large)
- Plusieurs outliers dans les deux catégories
- Le diesel était associé aux véhicules plus lourds et premium dans les années 80

**Insights** :
- Le type de carburant est un prédicteur significatif du prix
- Diesel = segment plus haut de gamme dans ce dataset

---

### Visualisation 1.4 : Prix par Nombre de Cylindres
**Fichier** : `visualisations_exploration.png` (subplot 4)

**Description** : Boxplots du prix selon le nombre de cylindres.

**Interprétation** :
- Relation **croissante** : plus de cylindres = prix plus élevé
- Véhicules 4 cylindres : segment économique (prix médian ~8 000$)
- Véhicules 6 cylindres : segment moyen (prix médian ~15 000$)
- Véhicules 8 cylindres et plus : segment premium/sport (prix médian > 20 000$)
- Forte variabilité dans chaque catégorie

**Insights** :
- Le nombre de cylindres est un proxy de la puissance et du segment
- Variable catégorielle très informative pour la prédiction

---

## 2. Corrélations

### Visualisation 2.1 : Matrice de Corrélation Complète
**Fichier** : `matrice_correlation.png`

**Description** : Heatmap de corrélation entre toutes les variables numériques.

**Interprétations clés** :

#### Corrélations fortes avec le prix (> 0.7) :
1. **engine_size** (0.87) : Plus le moteur est gros, plus le prix est élevé
2. **curb_weight** (0.84) : Les voitures lourdes sont plus chères (luxe, SUV)
3. **horsepower** (0.81) : La puissance est directement liée au prix
4. **width** (0.76) : Les véhicules larges sont généralement haut de gamme
5. **length** (0.69) : Idem pour la longueur

#### Corrélations négatives avec le prix :
- **city_mpg** (-0.69) : Plus la consommation est élevée (mpg faible), plus le prix est élevé
- **highway_mpg** (-0.70) : Les voitures chères consomment plus

#### Multicolinéarité détectée :
- **engine_size ↔ curb_weight** (0.93) : forte corrélation
- **length ↔ wheel_base** (0.88) : dimensions liées
- **city_mpg ↔ highway_mpg** (0.97) : très forte corrélation

**Implications pour la modélisation** :
- Multicolinéarité : peut affecter la régression linéaire (coefficients instables)
- Pas de problème pour les arbres de décision et Random Forest
- Possibilité de réduire la dimensionnalité (PCA) si nécessaire

---

## 3. Relations Variables-Prix

### Visualisation 3.1 : Scatter Plots Multiples
**Fichier** : `scatter_plots.png`

**Description** : 6 nuages de points montrant la relation entre variables clés et prix.

#### 3.1.1 : Prix vs Engine Size (Taille du moteur)
**Interprétation** :
- Relation **positive et quasi-linéaire**
- Corrélation très forte (R ≈ 0.87)
- Pas de valeurs aberrantes majeures
- Quelques points s'écartent de la tendance (marques premium avec petit moteur mais prix élevé)

#### 3.1.2 : Prix vs Horsepower (Puissance)
**Interprétation** :
- Relation **positive** mais avec **plus de dispersion**
- Certains véhicules peu puissants sont chers (luxe, confort)
- Certains véhicules puissants sont relativement abordables (sportives d'entrée de gamme)

#### 3.1.3 : Prix vs Curb Weight (Poids)
**Interprétation** :
- Relation **positive et linéaire**
- Les véhicules lourds sont systématiquement plus chers
- Poids = proxy du segment (économique vs premium)

#### 3.1.4 : Prix vs Length (Longueur)
**Interprétation** :
- Relation positive mais **plus dispersée**
- Les grandes berlines sont chères, mais pas toujours (breaks familiaux)

#### 3.1.5 : Prix vs Width (Largeur)
**Interprétation** :
- Relation similaire à la longueur
- Largeur = indicateur de confort et standing

#### 3.1.6 : Prix vs City MPG (Consommation ville)
**Interprétation** :
- Relation **négative** : plus la consommation est faible (MPG élevé), plus le prix est bas
- Les véhicules économiques consomment peu et coûtent peu
- Les véhicules premium/sport consomment beaucoup et coûtent cher

---

## 4. Détection des Outliers

### Visualisation 4.1 : Boxplots des Outliers
**Fichier** : `boxplots_outliers.png`

**Description** : Boxplots pour détecter les valeurs aberrantes sur 6 variables clés.

#### Outliers détectés :

**Prix** :
- 3-5 véhicules > 35 000$ (voitures de sport/luxe comme Jaguar, Porsche)
- Outliers légitimes, conservés

**Engine Size** :
- 2-3 moteurs > 250 cu.in (≈4.1L) : grosses cylindrées américaines/allemandes
- Conservés (valeurs réelles)

**Horsepower** :
- Quelques véhicules > 200 HP : sportives
- Conservés (segment légitime)

**Curb Weight** :
- Véhicules > 3500 lbs : SUV et berlines premium
- Conservés

**Length & Width** :
- Quelques véhicules particulièrement grands
- Conservés (limousines, breaks)

**Décision finale** : **Tous les outliers conservés** car ils représentent des segments réels du marché automobile.

---

## 5. Résultats des Modèles

### Visualisation 5.1 : Régression Linéaire - Prédictions vs Réalité
**Fichier** : `regression_lineaire_predictions.png`

**Description** : Scatter plot des prédictions vs valeurs réelles, avec ligne de référence y=x.

**Interprétation** :
- Les points sont **proches de la ligne** y=x → bonnes prédictions
- Légère dispersion pour les prix élevés (> 25 000$)
- Quelques sous-estimations pour les véhicules très chers
- Pas de biais systématique visible

**Performance** :
- R² ≈ 0.88 : le modèle explique 88% de la variance
- RMSE ≈ 2 800$ : erreur moyenne acceptable
- Erreur % ≈ 12-14% : **objectif atteint**

---

### Visualisation 5.2 : Optimisation du paramètre k pour KNN
**Fichier** : `knn_optimisation_k.png`

**Description** : Courbe RMSE en fonction du nombre de voisins k.

**Interprétation** :
- **k=3** : RMSE élevé (modèle trop complexe, sensible au bruit)
- **k=5 à k=9** : RMSE minimal (bon équilibre biais-variance)
- **k>15** : RMSE augmente (modèle trop simple, sous-apprentissage)
- **Meilleur k** : entre 5 et 9 selon la validation croisée

**Enseignement** :
- K trop petit → surapprentissage (modèle mémorise les données)
- K trop grand → sous-apprentissage (prédictions trop générales)
- Le coude de la courbe indique le k optimal

---

### Visualisation 5.3 : Arbre de Décision - Structure
**Fichier** : `arbre_decision.png`

**Description** : Visualisation complète de l'arbre de décision (max_depth=5).

**Interprétation** :

**Nœud racine** :
- Première division sur **curb_weight** (poids) → variable la plus informative
- Sépare immédiatement les véhicules économiques des véhicules premium

**Branches importantes** :
- Branche gauche (véhicules légers) : divisions sur city_mpg, engine_size
- Branche droite (véhicules lourds) : divisions sur horsepower, width

**Feuilles** :
- Véhicules économiques : prédictions ~7 000-10 000$
- Véhicules premium : prédictions ~20 000-30 000$
- Profondeur 5 limite le surapprentissage

**Règles extraites** (exemples) :
- Si poids < 2 400 lbs ET consommation > 30 mpg → prix ≈ 7 500$
- Si poids > 3 000 lbs ET puissance > 120 HP → prix ≈ 22 000$

**Avantages** :
- **Très interprétable** : on comprend exactement pourquoi une prédiction est faite
- Adapté pour expliquer le modèle à un non-technicien

---

### Visualisation 5.4 : Random Forest - Importance des Features
**Fichier** : `random_forest_importance.png`

**Description** : Diagramme en barres des 15 features les plus importantes.

**Interprétation** :

**Top 5 des features** (importance > 15%) :
1. **curb_weight** (25-30%) : poids du véhicule
2. **engine_size** (15-20%) : taille du moteur
3. **horsepower** (10-15%) : puissance
4. **width** (8-12%) : largeur
5. **highway_mpg** (6-10%) : consommation autoroute

**Features moyennement importantes** (5-8%) :
- length, city_mpg, compression_ratio

**Features peu importantes** (< 5%) :
- num_cylinders_encoded, body_style_encoded, drive_wheels_encoded

**Insights** :
- Les caractéristiques **physiques et techniques** dominent
- Les caractéristiques **catégorielles** (marque, style) ont moins d'impact que prévu
  - Possible biais de l'encodage label (one-hot aurait mieux capturé)
- La **puissance et le poids** sont les déterminants principaux du prix

**Comparaison avec la corrélation** :
- Cohérence totale : les variables corrélées au prix sont bien les plus importantes
- Random Forest confirme l'analyse exploratoire

---

## 6. Validation et Diagnostics

### Visualisation 6.1 : Courbes d'Apprentissage - Random Forest
**Fichier** : `courbes_apprentissage.png`

**Description** : Évolution du RMSE train et validation en fonction du nombre d'exemples d'entraînement.

**Interprétation** :

**Début de la courbe** (10-20% des données) :
- RMSE train très faible (~500$) → le modèle mémorise parfaitement
- RMSE validation très élevé (~5 000$) → surapprentissage massif
- **Gap important** entre train et validation

**Milieu de la courbe** (40-60% des données) :
- RMSE train augmente légèrement (~1 500$)
- RMSE validation **diminue fortement** (~3 000$)
- Gap se réduit progressivement

**Fin de la courbe** (100% des données) :
- RMSE train ≈ 1 800-2 000$
- RMSE validation ≈ 2 200-2 500$
- **Convergence** : les courbes sont proches

**Diagnostic** :
- ✅ **Pas de sous-apprentissage** : les performances ne sont pas au plancher
- ✅ **Peu de surapprentissage** : gap réduit entre train et validation
- ✅ **Le modèle généralise bien**
- ⚠️ Plateau atteint : plus de données n'améliorerait que marginalement

**Conclusion** : Le Random Forest avec 100 arbres et max_depth=10 est bien calibré pour ce dataset.

---

## 7. Clustering

### Visualisation 7.1 : Méthode du Coude - K-means
**Fichier** : `kmeans_coude.png`

**Description** : Courbe de l'inertie en fonction du nombre de clusters k.

**Interprétation** :

**Inertie** : somme des distances au carré entre chaque point et son centroïde.

**Évolution** :
- k=2 : inertie très élevée (~18 000) → clusters trop larges
- k=3 : inertie descend à ~12 000 → amélioration significative
- **k=4** : inertie ≈ 8 000 → **coude visible**
- k=5 à k=10 : décroissance plus lente

**Choix du k optimal** : **k=4**
- Point où la courbe forme un angle (coude)
- Compromis entre simplicité (peu de clusters) et précision (inertie faible)

**Interprétation métier** : 4 segments de marché distincts.

---

### Visualisation 7.2 : Clusters Visualisés
**Fichier** : `clusters_visualisation.png`

**Description** : Deux scatter plots colorés par cluster.

#### Plot 1 : Taille moteur vs Puissance
**Interprétation** :

**Cluster 0 (violet/bleu foncé)** :
- Petits moteurs (<100 cu.in), faible puissance (<80 HP)
- **Segment économique** : citadines, compactes
- Exemples : Honda Civic, Toyota Corolla

**Cluster 1 (vert)** :
- Moteurs moyens (100-150 cu.in), puissance moyenne (80-120 HP)
- **Segment familial** : berlines moyennes, breaks
- Exemples : Volkswagen Passat, Nissan Maxima

**Cluster 2 (jaune)** :
- Moteurs moyens/gros (120-180 cu.in), puissance élevée (120-160 HP)
- **Segment sportif** : coupés, sportives
- Exemples : Mazda RX-7, Alfa Romeo

**Cluster 3 (rouge)** :
- Gros moteurs (>180 cu.in), très haute puissance (>160 HP)
- **Segment premium/luxe** : berlines haut de gamme, sportives de luxe
- Exemples : BMW Série 7, Mercedes, Jaguar

**Séparation claire** : les clusters sont bien distincts, peu de chevauchement.

#### Plot 2 : Poids vs Prix
**Interprétation** :

**Relation cluster-prix** :
- Cluster 0 : poids <2 500 lbs, prix 5 000-12 000$ → économiques
- Cluster 1 : poids 2 500-3 000 lbs, prix 10 000-18 000$ → moyens
- Cluster 2 : poids 2 800-3 200 lbs, prix 15 000-25 000$ → sportifs
- Cluster 3 : poids >3 000 lbs, prix >20 000$ → premium/luxe

**Insights marketing** :
- Segmentation claire du marché automobile
- Possibilité de stratégies marketing ciblées par cluster
- Chaque cluster a des caractéristiques techniques et prix distincts

---

### Visualisation 7.3 : Dendrogramme - Classification Hiérarchique
**Fichier** : `dendrogramme.png`

**Description** : Arbre hiérarchique montrant les fusions successives de 50 véhicules.

**Interprétation** :

**Lecture** :
- Axe horizontal : index des véhicules
- Axe vertical : distance de fusion (dissimilarité)
- Branches longues : véhicules très différents
- Branches courtes : véhicules similaires

**Observations** :
- **Groupes naturels** : certains véhicules se regroupent rapidement (faible distance)
- **4 grands clusters** : si on coupe à hauteur ≈ 15, on retrouve 4 groupes
- Cohérence avec K-means

**Avantages du dendrogramme** :
- Vision hiérarchique complète
- Permet de choisir le niveau de granularité
- Identifie les véhicules "outliers" (branches isolées)

**Exemple** : Les véhicules 5, 12, 18 se regroupent très tôt → véhicules très similaires (même segment, même marque potentiellement).

---

## 8. Comparaisons Finales

### Visualisation 8.1 : Comparaison des Modèles - Métriques Multiples
**Fichier** : `comparaison_modeles.png`

**Description** : 4 subplots comparant les 5 modèles sur différentes métriques.

#### Subplot 1 : RMSE Test par Modèle
**Interprétation** :
- **Gradient Boosting** : RMSE le plus faible (~2 100$) ⭐
- **Random Forest** : RMSE très bon (~2 200$) ⭐
- **KNN** : RMSE correct (~2 600$)
- **Régression Linéaire** : RMSE moyen (~2 800$)
- **Arbre de Décision** : RMSE le plus élevé (~3 200$)

**Classement** : GB > RF > KNN > LR > Arbre

#### Subplot 2 : R² Test par Modèle
**Interprétation** :
- **Gradient Boosting** : R² ≈ 0.93 (explique 93% de la variance) ⭐
- **Random Forest** : R² ≈ 0.92 ⭐
- **KNN** : R² ≈ 0.89
- **Régression Linéaire** : R² ≈ 0.88
- **Arbre de Décision** : R² ≈ 0.84

**Insight** : Les modèles ensemblistes dominent largement.

#### Subplot 3 : Comparaison Train vs Test
**Interprétation** :

**Arbre de Décision** :
- RMSE train très faible, RMSE test élevé
- **Gap important** → surapprentissage visible

**Régression Linéaire et KNN** :
- Gap modéré → bon équilibre

**Random Forest et Gradient Boosting** :
- Léger gap → excellent équilibre
- Généralisation optimale

#### Subplot 4 : Erreur % par Modèle
**Interprétation** :
- Ligne rouge pointillée : **objectif 15%**
- **Gradient Boosting** : ~9-10% ✅ Objectif largement atteint
- **Random Forest** : ~10-11% ✅ Objectif atteint
- **KNN** : ~12-13% ✅ Objectif atteint
- **Régression Linéaire** : ~13-14% ✅ Objectif atteint (limite)
- **Arbre de Décision** : ~15-16% ⚠️ Objectif non atteint

**Conclusion** : 4 modèles sur 5 atteignent l'objectif de <15% d'erreur.

---

### Visualisation 8.2 : Prédictions vs Réalité - Tous les Modèles
**Fichier** : `predictions_tous_modeles.png`

**Description** : 5 scatter plots (un par modèle) comparant prédictions et valeurs réelles.

#### Analyse comparative :

**Régression Linéaire** :
- Points alignés sur la diagonale
- Légère dispersion uniforme
- Quelques sous-estimations pour prix élevés

**KNN** :
- Alignement similaire
- Légèrement plus de dispersion que LR
- Effet "escalier" visible (prédictions discrètes basées sur voisins)

**Arbre de Décision** :
- **Dispersion plus importante**
- Prédictions parfois éloignées de la diagonale
- Effet plateau (prédictions constantes dans les feuilles)
- Surapprentissage visible

**Random Forest** :
- **Excellent alignement**
- Dispersion minimale
- Peu d'erreurs majeures
- Prédictions lisses (moyenne de 100 arbres)

**Gradient Boosting** :
- **Meilleur alignement de tous**
- Dispersion très faible
- Prédictions très précises
- Pas de biais systématique visible

**Comparaison visuelle** :
- GB ≈ RF >> KNN ≈ LR >> Arbre
- Les modèles ensemblistes produisent des prédictions bien plus concentrées autour de la diagonale

---

## Synthèse Générale des Visualisations

### Points Clés Identifiés

#### 1. Structure des données
- Distribution asymétrique des prix (majorité <20k$, queue >30k$)
- Variables techniques (moteur, poids, puissance) dominent
- Segmentation naturelle en 4 catégories de véhicules

#### 2. Relations importantes
- **Corrélations fortes** : engine_size (0.87), curb_weight (0.84), horsepower (0.81)
- **Relations non-linéaires** : certaines variables montrent des patterns complexes
- **Multicolinéarité** : dimensions du véhicule fortement corrélées entre elles

#### 3. Qualité des modèles
- **Gradient Boosting** : meilleur sur toutes les métriques (R²=0.93, erreur=9%)
- **Random Forest** : très proche, plus stable (R²=0.92, erreur=10%)
- **Régression Linéaire** : surprenamment efficace malgré la non-linéarité (R²=0.88)
- **Arbre unique** : surapprentissage évident, performances limitées

#### 4. Validation
- Courbes d'apprentissage montrent une convergence saine
- Validation croisée confirme la stabilité des modèles ensemblistes
- Peu de surapprentissage pour RF et GB

#### 5. Clustering
- 4 segments distincts : économique, familial, sportif, luxe
- Segmentation cohérente avec les caractéristiques techniques et prix
- Utilité marketing évidente

---

## Recommandations pour la Présentation

### Visualisations essentielles à présenter :
1. ✅ **Matrice de corrélation** : montre les relations clés
2. ✅ **Scatter plots** : illustre les relations prix-features
3. ✅ **Courbes d'apprentissage** : prouve la généralisation
4. ✅ **Comparaison modèles** : justifie le choix du meilleur
5. ✅ **Clusters** : montre la segmentation du marché

### Messages clés à transmettre :
- Les données révèlent une structure claire avec 4 segments
- Les caractéristiques physiques (poids, moteur) dominent le prix
- Les modèles ensemblistes surpassent largement les modèles simples
- L'objectif de <15% d'erreur est atteint par 4 modèles sur 5
- Le Random Forest offre le meilleur compromis performance/interprétabilité

---

## Conclusion

Les visualisations produites dans ce projet permettent de :
1. **Comprendre** la structure des données et les relations entre variables
2. **Justifier** les décisions de prétraitement et modélisation
3. **Évaluer** objectivement les performances des modèles
4. **Communiquer** les résultats de manière claire et convaincante

L'ensemble des graphiques démontre une démarche rigoureuse et méthodique, conforme aux standards de la Data Science.

---

**Auteurs** : Matteo Robin, Florian Huguet
**Date** : Décembre 2025
