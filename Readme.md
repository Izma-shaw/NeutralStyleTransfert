# Classification d’Images par Transfert d’Apprentissage

## Contexte

Ce projet a été réalisé dans le cadre d’un module d’**Apprentissage Profond Avancé** de Master 2. Il vise à explorer et comparer différentes architectures de **réseaux de neurones convolutifs** (CNN) modernes et classiques pour la classification d’images, avec une attention particulière sur le **transfert d’apprentissage**.

## Objectifs

- Comparer la performance de plusieurs modèles pré-entraînés : **VGG16**, **ResNet50**, **DenseNet**, **MobileNet**
- Appliquer des techniques de **data augmentation** pour améliorer la robustesse du modèle
- Utiliser des **callbacks** avancés (early stopping, réduction de learning rate)
- Générer des **matrices de confusion** et courbes de précision/perte
- Évaluer les performances en fonction du temps de calcul et des métriques

## Modèles testés

- **VGG16** (réseau classique, profond et dense)
- **ResNet50** (avec connexions résiduelles)
- **DenseNet121**
- **MobileNetV2** (modèle léger pour les environnements contraints)

## Pipeline général

1. Téléchargement et préparation du jeu de données (via `gdown`, `zipfile`, `cv2`)
2. Prétraitement des images et mise en forme avec `ImageDataGenerator`
3. Chargement des modèles pré-entraînés avec `include_top=False` et `fine-tuning`
4. Compilation, entraînement avec `EarlyStopping`, `ModelCheckpoint`, etc.
5. Évaluation sur les ensembles de test
6. Visualisation des performances (matrice de confusion, courbes, temps)

## Librairies utilisées

- `TensorFlow`, `Keras`, `OpenCV`
- `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn` (confusion matrix)
- `gdown` pour le téléchargement depuis Google Drive

## Résultats attendus

- Comparaison objective des modèles en termes de :
  - **Précision** sur l’ensemble de test
  - **Temps d’entraînement**
  - **Complexité du modèle**
- Visualisation claire des performances via des graphiques et matrices

## Exécution

1. Télécharger ou cloner ce dépôt
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
