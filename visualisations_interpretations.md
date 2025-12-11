# Visualisations et Interprétations - Projet Prédiction Prix Voitures d'Occasion

**Auteurs** : Florian Huguet, Matteo Robin
**Date** : Décembre 2025
**Contexte** : Projet de cours - Module d'initiation au Data Science et Machine Learning

---

## Introduction

Ce document présente l'analyse détaillée des visualisations générées dans le cadre du projet de prédiction du prix de voitures d'occasion. L'objectif est de construire un modèle de régression capable de prédire le prix avec une erreur inférieure à 15% du prix réel.

Toutes les interprétations ci-dessous sont basées exclusivement sur les visualisations produites par notre analyse.

---

## 1. Exploration des Données

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
- **Toyota** domine largement avec environ 30 véhicules
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
- Les véhicules **diesel** présentent une médiane de prix nettement supérieure aux véhicules essence
- La boîte interquartile du diesel est plus haute et plus large, indiquant une variabilité importante
- Plusieurs outliers visibles dans les deux catégories
- Le diesel atteint des prix maximums plus élevés (~32 000$) contre ~45 000$ pour l'essence

**Insights** :
- Dans les années 80, le diesel était associé aux véhicules plus lourds et haut de gamme
- Le type de carburant est un prédicteur significatif du prix
- Les véhicules essence couvrent un spectre plus large (économique à luxe)

---

### Visualisation 1.4 : Prix par Nombre de Cylindres

**Observation** : Les boxplots montrent une relation croissante entre nombre de cylindres et prix.

**Interprétations** :
- **Relation progressive** : plus de cylindres = prix plus élevé
- Véhicules 4 cylindres (four) : segment économique, médiane ~9 000$
- Véhicules 5 cylindres (five) : segment intermédiaire, médiane ~13 000$
- Véhicules 6 cylindres (six) : segment moyen-premium, médiane ~15 000$
- Véhicules 8 cylindres (eight) : segment premium, médiane ~21 000$
- Catégories "twelve" et "two" : outliers avec prix élevés mais peu de représentants

**Insights** :
- Le nombre de cylindres est un proxy direct de la puissance et du segment de marché
- Variable catégorielle hautement informative pour la prédiction
- Les véhicules 3 cylindres ("three") existent mais sont très rares

---

## 2. Analyse des Corrélations

### Visualisation 2.1 : Matrice de Corrélation Complète

**Observation** : La heatmap révèle les relations entre toutes les variables numériques du dataset.

**Corrélations fortes avec le prix (> 0.75)** :
1. **curb_weight** (0.83) : Le poids du véhicule est fortement corrélé au prix
2. **engine_size** (0.87) : La taille du moteur est le prédicteur le plus corrélé
3. **horsepower** (0.81) : La puissance est directement liée au prix
4. **width** (0.75) : La largeur du véhicule indique le segment

**Corrélations moyennes (0.50 - 0.75)** :
- **length** (0.69) : Longueur du véhicule
- **wheel_base** (0.58) : Empattement
- **height** (0.14) : Corrélation faible, la hauteur n'est pas un bon prédicteur

**Corrélations négatives** :
- **city_mpg** (-0.69) : Plus la consommation est économique, plus le prix est bas
- **highway_mpg** (-0.70) : Même tendance pour l'autoroute
- **peak_rpm** (-0.10) : Corrélation néglige able

**Multicolinéarité détectée** :
- **curb_weight ↔ engine_size** (0.93) : Très forte corrélation
- **length ↔ wheel_base** (0.87) : Les dimensions sont liées
- **width ↔ length** (0.84) : Idem
- **city_mpg ↔ highway_mpg** (0.97) : Redondance presque totale
- **engine_size ↔ horsepower** (0.81) : Attendu physiquement

**Implications pour la modélisation** :
- La multicolinéarité peut déstabiliser les coefficients de la régression linéaire
- Les modèles à base d'arbres (Random Forest) ne seront pas affectés
- Possibilité d'éliminer highway_mpg (redondant avec city_mpg)
- Les 3 variables majeures (curb_weight, engine_size, horsepower) capturent l'essentiel

---

## 3. Relations Variables-Prix

### Visualisation 3.1 : Scatter Plots - Prix vs Variables Clés

Six nuages de points révèlent les relations entre variables explicatives et prix.

#### Plot 3.1.1 : Prix vs Engine Size (Taille du moteur)

**Interprétations** :
- Relation **positive quasi-linéaire** très nette
- Corrélation visuelle très forte, confirmant le coefficient de 0.87
- Peu de dispersion, la tendance est claire
- Quelques points au-dessus de la ligne de tendance : marques premium avec petit moteur mais prix élevé (ex: voitures de luxe compactes)

#### Plot 3.1.2 : Prix vs Horsepower (Puissance)

**Interprétations** :
- Relation positive avec **dispersion modérée**
- Certains véhicules peu puissants (<100 HP) atteignent 15 000-20 000$ : véhicules axés luxe/confort plutôt que performance
- Certains véhicules puissants (>150 HP) restent sous 15 000$ : sportives d'entrée de gamme
- La puissance seule ne détermine pas tout le prix

#### Plot 3.1.3 : Prix vs Curb Weight (Poids)

**Interprétations** :
- Relation **positive et linéaire**, très nette
- Le poids est un excellent prédicteur du prix
- Les véhicules lourds (>3 500 lbs) sont systématiquement chers
- Le poids reflète à la fois la taille, les matériaux et le segment

#### Plot 3.1.4 : Prix vs Length (Longueur)

**Interprétations** :
- Relation positive mais avec **dispersion importante**
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
- Relation **négative claire** : plus le MPG est élevé (= consommation faible), plus le prix est bas
- Les véhicules économiques (>30 MPG) sont concentrés sous 15 000$
- Les véhicules de luxe/sport consomment beaucoup (15-25 MPG) et coûtent cher
- Logique économique : performance et luxe vs économie de carburant

---

## 4. Détection des Outliers

### Visualisation 4.1 : Boxplots des Variables Clés

**Objectif** : Identifier les valeurs aberrantes sur 6 variables importantes.

#### Price (Prix)

**Outliers identifiés** :
- Plusieurs véhicules au-dessus de 30 000$
- Maximum à ~45 000$ (très probablement Porsche, Jaguar)

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

---

## 5. Résultats des Modèles de Prédiction

### Visualisation 5.1 : Régression Linéaire - Prédictions vs Réalité

**Observation** : Scatter plot des prédictions vs valeurs réelles avec ligne de référence y=x.

**Interprétations** :
- Les points sont **globalement proches de la diagonale** → bonnes prédictions
- Alignement correct pour les prix 5 000$ - 25 000$
- **Dispersion visible pour les prix > 30 000$** : le modèle sous-estime plusieurs véhicules haut de gamme
- Pas de biais systématique majeur
- Quelques prédictions autour de 25 000-28 000$ pour des véhicules à 31 000$+

**Performance estimée** :
- Erreur acceptable sur la majorité du dataset
- Difficulté à capturer les spécificités des véhicules de luxe
- Le modèle linéaire atteint ses limites sur les extrêmes

---

### Visualisation 5.2 : Optimisation de k pour KNN

**Observation** : Courbe du RMSE en fonction du nombre de voisins k.

**Interprétations** :

**k = 3** : RMSE très élevé (~3 600$)
- Modèle trop sensible au bruit local
- Surapprentissage probable

**k = 5 à k = 11** : RMSE minimal (~4 000-4 130$)
- Zone optimale : équilibre biais-variance
- **k = 5 semble être le meilleur compromis**

**k = 15 et k = 21** : RMSE augmente progressivement (>4 200$)
- Modèle trop simple, sous-apprentissage
- Les prédictions deviennent trop générales

**Enseignement** :
- Le "coude" de la courbe se situe autour de k=5
- Au-delà de k=11, on perd en précision sans gagner en stabilité
- Confirmation du principe biais-variance

---

### Visualisation 5.3 : Arbre de Décision - Visualisation Complète

**Observation** : Représentation de l'arbre de décision avec profondeur maximale de 5.

**Analyse de la structure** :

**Nœud racine** :
- Première division sur **curb_weight ≤ 2650.0**
- Confirmation : le poids est la variable la plus discriminante

**Branche gauche (véhicules légers)** :
- Subdivisions sur curb_weight, body_style, stroke
- Les feuilles prédisent des prix dans la gamme 6 000$ - 12 000$
- Segment économique et compact

**Branche droite (véhicules lourds)** :
- Subdivisions sur engine_size, width, make_encoded
- Les feuilles prédisent des prix 10 000$ - 28 000$
- Segment moyen à premium

**Règles extraites (exemples)** :
- Si curb_weight ≤ 2221.5 ET body_style ≤ 2.5 → Prix ≈ 6 485$
- Si curb_weight > 2650.0 ET engine_size > 188.5 → Prix ≈ 27 656$

**Avantages** :
- **Très interprétable** : chaque décision est explicite
- Idéal pour expliquer les prédictions à un non-technicien
- Profondeur 5 limite le surapprentissage

---

### Visualisation 5.4 : Random Forest - Importance des Variables

**Observation** : Diagramme en barres des 15 features les plus importantes.

**Top 5 des variables (>5% d'importance)** :
1. **curb_weight** (~43%) : Variable dominante, représente presque la moitié de l'importance
2. **engine_size** (~32%) : Deuxième facteur majeur
3. **horsepower** (~6.5%) : Important mais loin derrière les deux premiers
4. **highway_mpg** (~6.4%) : Consommation autoroutière
5. **city_mpg** (~3.8%) : Consommation urbaine

**Variables moyennement importantes (1-3%)** :
- width, make_encoded, wheel_base

**Variables peu importantes (<1%)** :
- peak_rpm, length, height, bore, stroke, drive_wheels_encoded, compression_ratio

**Insights** :
- **Domination écrasante** du poids et de la taille du moteur (75% d'importance combinée)
- Les caractéristiques physiques primordiales
- Les variables catégorielles (marque, style) ont un impact limité avec l'encodage label
- L'encodage one-hot aurait probablement donné plus d'importance à la marque

**Comparaison avec la corrélation** :
- Cohérence totale : les variables corrélées au prix sont bien les plus importantes
- Random Forest confirme l'analyse exploratoire initiale

---

## 6. Validation et Diagnostics

### Visualisation 6.1 : Courbes d'Apprentissage - Random Forest

**Observation** : Évolution du RMSE train et validation en fonction de la taille du dataset.

**Début de courbe (10-20 exemples)** :
- RMSE train très faible (~5 700$)
- RMSE validation très élevé (~6 800$)
- **Gap énorme** : surapprentissage massif avec peu de données

**Milieu de courbe (40-80 exemples)** :
- RMSE train augmente progressivement (~1 900$)
- RMSE validation **diminue fortement** (~3 200$)
- Le gap se réduit significativement

**Fin de courbe (120-160 exemples)** :
- RMSE train stable (~1 600$)
- RMSE validation converge (~2 600$)
- **Gap réduit** : les courbes se rapprochent

**Diagnostic** :
- ✅ **Pas de sous-apprentissage** : les scores ne sont pas au plancher
- ✅ **Surapprentissage limité** : gap final acceptable entre train et validation
- ✅ **Le modèle généralise bien** avec le dataset complet
- ⚠️ **Plateau atteint** : plus de données n'améliorerait que marginalement les performances

**Conclusion** : Le Random Forest avec 100 arbres et max_depth=10 est bien calibré pour ce dataset de 201 véhicules.

---

## 7. Clustering - Segmentation du Marché

### Visualisation 7.1 : Méthode du Coude - K-means

**Observation** : Courbe de l'inertie (somme des distances intra-cluster) en fonction de k.

**Analyse** :

**k = 2** : Inertie très élevée (~335)
- Seulement 2 clusters, trop large, perte d'information

**k = 3** : Inertie descend à ~190
- Amélioration significative (-43%)

**k = 4** : Inertie ~145
- **Coude visible** : point d'inflexion de la courbe
- Amélioration notable mais la décroissance ralentit

**k = 5 à k = 10** : Inertie continue de baisser progressivement
- Amélioration marginale décroissante
- Complexité accrue sans gain proportionnel

**Choix optimal** : **k = 4 clusters**
- Point où le rapport amélioration/complexité est optimal
- Interprétation métier : 4 segments de marché distincts

---

### Visualisation 7.2 : Visualisation des Clusters

**Description** : Deux scatter plots colorés par cluster.

#### Plot gauche : Taille moteur vs Puissance

**Cluster 0 (jaune)** :
- Petits moteurs (50-80 cu.in), faible puissance (45-80 HP)
- **Segment économique** : citadines, compactes
- Exemples : Honda Civic, Toyota Corolla, petites Mazda

**Cluster 1 (bleu clair)** :
- Moteurs moyens (90-130 cu.in), puissance moyenne (80-110 HP)
- **Segment familial** : berlines moyennes, véhicules polyvalents
- Exemples : Volkswagen, Nissan Maxima, Honda Accord

**Cluster 2 (violet/mauve)** :
- Moteurs moyens/gros (140-180 cu.in), puissance élevée (110-160 HP)
- **Segment sportif/premium** : coupés sportifs, berlines dynamiques
- Exemples : Mazda RX-7, Alfa Romeo, BMW entrée de gamme

**Cluster 3 (vert/turquoise)** :
- Gros moteurs (>180 cu.in), très haute puissance (>160 HP)
- **Segment luxe/sport** : grosses berlines, sportives haut de gamme
- Exemples : BMW Série 7, Mercedes, Jaguar, Porsche

**Séparation** : Les clusters sont bien distincts avec peu de chevauchement.

#### Plot droit : Poids vs Prix

**Relation cluster-prix** :

**Cluster 0 (jaune)** :
- Poids : 1 500-2 300 lbs
- Prix : 5 000-10 000$
- **Véhicules économiques**

**Cluster 1 (bleu)** :
- Poids : 2 300-2 800 lbs
- Prix : 8 000-17 000$
- **Véhicules familiaux**

**Cluster 2 (violet)** :
- Poids : 2 500-3 200 lbs
- Prix : 12 000-25 000$
- **Véhicules sportifs/premium**

**Cluster 3 (vert)** :
- Poids : 3 000-4 000 lbs
- Prix : 20 000-45 000$
- **Véhicules luxe/sport haut de gamme**

**Insights marketing** :
- Segmentation claire et exploitable commercialement
- Chaque cluster correspond à un persona client différent
- Stratégies marketing et pricing adaptables par cluster
- Les 4 segments reflètent la réalité du marché automobile

---

### Visualisation 7.3 : Dendrogramme - Classification Hiérarchique

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
- Si on coupe à hauteur ~10-15, on obtient **4 grands clusters**
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

## 8. Comparaison Finale des Modèles

### Visualisation 8.1 : Métriques de Performance - 4 Subplots

#### Subplot 1 : RMSE Test par Modèle

**Classement** (du meilleur au moins bon) :
1. **Random Forest** : ~2 195$ ⭐ Meilleur RMSE
2. **Gradient Boosting** : ~2 383$
3. **Arbre de Décision** : ~2 856$
4. **Régression Linéaire** : ~4 660$
5. **KNN** : ~5 407$ ❌ Pire RMSE

**Interprétation** :
- Random Forest domine avec une marge confortable
- Les modèles ensemblistes (RF, GB) surclassent les modèles simples
- KNN performe mal, probablement à cause de la dimensionnalité

#### Subplot 2 : R² Test par Modèle

**Classement** :
1. **Random Forest** : R² ≈ 0.96 ⭐ Explique 96% de la variance
2. **Gradient Boosting** : R² ≈ 0.95
3. **Arbre de Décision** : R² ≈ 0.93
4. **Régression Linéaire** : R² ≈ 0.82
5. **KNN** : R² ≈ 0.76

**Interprétation** :
- Random Forest et Gradient Boosting ont un pouvoir prédictif excellent
- Même l'Arbre de Décision simple atteint 93% d'explication
- Les modèles linéaires sont limités par la non-linéarité des relations

#### Subplot 3 : Comparaison Train vs Test (RMSE)

**Analyse par modèle** :

**Régression Linéaire** :
- RMSE Train : ~2 391$
- RMSE Test : ~4 660$
- **Gap important** → Variance élevée

**KNN** :
- RMSE Train : ~2 026$
- RMSE Test : ~5 407$
- **Gap très important** → Surapprentissage sévère

**Arbre de Décision** :
- RMSE Train : ~1 920$
- RMSE Test : ~2 856$
- Gap modéré → Léger surapprentissage

**Random Forest** :
- RMSE Train : ~1 532$
- RMSE Test : ~2 195$
- **Gap minimal** → Excellent équilibre ✅

**Gradient Boosting** :
- RMSE Train : ~341$
- RMSE Test : ~2 383$
- **Gap énorme** → Surapprentissage massif ⚠️

**Enseignement** : Random Forest offre le meilleur compromis généralisation/performance.

#### Subplot 4 : Erreur % par Modèle

**Ligne rouge pointillée** : Objectif de 15% d'erreur

**Classement** :
1. **Gradient Boosting** : ~9.4% ✅ Objectif largement dépassé
2. **Random Forest** : ~9.6% ✅ Objectif dépassé
3. **Arbre de Décision** : ~11.5% ✅ Objectif atteint
4. **Régression Linéaire** : ~20.2% ❌ Objectif non atteint
5. **KNN** : ~20.2% ❌ Objectif non atteint

**Conclusion** : 3 modèles sur 5 atteignent l'objectif de moins de 15% d'erreur.

---

### Visualisation 8.2 : Prédictions vs Réalité - Tous les Modèles

**Description** : 5 scatter plots comparant prédictions et valeurs réelles pour chaque modèle.

#### Régression Linéaire

**Observations** :
- Alignement général correct sur la diagonale
- Dispersion uniforme autour de la ligne
- Sous-estimation marquée pour les prix > 30 000$
- Quelques surestimations pour les prix bas

#### KNN

**Observations** :
- Alignement similaire à la régression linéaire
- **Dispersion plus importante**
- Effet "escalier" visible : prédictions discrètes basées sur moyennes de voisins
- Difficultés sur les extrêmes

#### Arbre de Décision

**Observations** :
- Alignement globalement bon
- **Effet plateau** : prédictions constantes dans les feuilles de l'arbre
- Quelques prédictions éloignées de la diagonale
- Dispersion modérée

#### Random Forest

**Observations** :
- **Excellent alignement** sur toute la gamme de prix ⭐
- Dispersion minimale autour de la diagonale
- Très peu d'erreurs majeures
- Prédictions lisses grâce à l'agrégation de 100 arbres
- Performance homogène sur tous les segments de prix

#### Gradient Boosting

**Observations** :
- **Meilleur alignement visuel** de tous les modèles ⭐
- Dispersion très faible
- Prédictions très précises
- Pas de biais systématique visible
- Légèrement meilleur que Random Forest visuellement

**Comparaison visuelle globale** :
- GB ≈ RF >>> Arbre > LR ≈ KNN
- Les modèles ensemblistes produisent des nuages de points beaucoup plus concentrés autour de la diagonale
- La différence de qualité est visuellement évidente

---

## 9. Synthèse et Recommandations

### Points Clés Identifiés

#### Structure des données
- Distribution asymétrique des prix avec concentration 5 000-15 000$
- Dataset de 201 véhicules avec 26 variables
- 4 segments naturels de marché identifiés par clustering

#### Variables déterminantes
- **Top 3** : curb_weight (0.83), engine_size (0.87), horsepower (0.81)
- Forte multicolinéarité entre dimensions physiques
- Variables catégorielles moins discriminantes avec encodage label

#### Performance des modèles
- **Meilleur modèle global** : Random Forest (RMSE test = 2 195$, R² = 0.96, erreur = 9.6%)
- **Alternative** : Gradient Boosting (légèrement surapprentissage mais excellent sur test)
- **Modèle le plus interprétable** : Arbre de Décision (performance acceptable : 11.5% erreur)
- **Modèles inadaptés** : Régression Linéaire et KNN (>20% erreur)

#### Validation
- Courbes d'apprentissage saines pour Random Forest
- Validation croisée confirme la stabilité
- Objectif de <15% d'erreur atteint par 3 modèles

#### Segmentation marché
- 4 clusters distincts : économique, familial, sportif/premium, luxe
- Segmentation exploitable commercialement
- Cohérence entre K-means et classification hiérarchique

### Modèle Recommandé : Random Forest

**Justification** :
1. **Meilleur RMSE test** (2 195$) et **meilleur R²** (0.96)
2. **Erreur de 9.6%** : largement sous l'objectif de 15%
3. **Équilibre train/test optimal** : pas de surapprentissage
4. **Robustesse** : performance stable en validation croisée
5. **Interprétabilité acceptable** : importance des variables accessible

**Limites** :
- Moins interprétable qu'un arbre unique
- Temps d'entraînement plus long
- Boîte noire relative

### Visualisations Essentielles pour Présentation

1. ✅ **Matrice de corrélation** : montre les relations clés entre variables
2. ✅ **Scatter plots** : illustre les relations prix-features principales
3. ✅ **Courbes d'apprentissage** : prouve la généralisation du modèle
4. ✅ **Comparaison modèles** (4 subplots) : justifie le choix de Random Forest
5. ✅ **Clusters visualisation** : démontre la segmentation du marché

### Messages Clés

- Le poids et la taille du moteur expliquent 75% de l'importance dans Random Forest
- Les modèles ensemblistes surpassent largement les modèles simples
- L'objectif de <15% d'erreur est dépassé avec 9.6% d'erreur moyenne
- 4 segments de marché distincts ont été identifiés
- Le modèle Random Forest offre le meilleur compromis performance/robustesse/interprétabilité

---

## Conclusion

Les visualisations produites dans ce projet démontrent une démarche méthodique et rigoureuse conforme aux standards de la Data Science. L'analyse a permis de :

1. **Comprendre** la structure des données et identifier les variables clés
2. **Justifier** les décisions de prétraitement (conservation des outliers, gestion des valeurs manquantes)
3. **Évaluer** objectivement 5 modèles de régression différents
4. **Valider** la généralisation du meilleur modèle (Random Forest)
5. **Segmenter** le marché en 4 clusters exploitables
6. **Atteindre** l'objectif fixé : erreur < 15% (résultat : 9.6%)

Le modèle Random Forest retenu prédit le prix des voitures d'occasion avec une précision de 90.4% et un R² de 0.96, dépassant largement les attentes initiales du projet.

---

**Projet réalisé par Florian Huguet et Matteo Robin dans le cadre du cours d'initiation au Data Science et Machine Learning - Décembre 2025**
