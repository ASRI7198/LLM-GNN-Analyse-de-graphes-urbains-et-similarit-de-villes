# 🌆 LLM-GNN : Analyse de graphes urbains et similarité de villes

## 📖 Description du projet

LLM-GNN combine **Graph Neural Networks (GNN)** et **Large Language Models (LLM)** pour analyser les relations et similarités entre villes. Le projet utilise un graphe construit à partir de **OpenStreetMap (OSM)** pour :
- Prédire les liens entre villes
- Générer des explications textuelles sur la similarité des villes selon la population, la localisation et le pays

## 📂 Structure du projet
```
LLM-GNN/
│
├── Architectures/
│   ├── Encoder.py
│   ├── GAT.py               # Graph Attention Network
│   └── GCN.py               # Graph Convolutional Network
│
├── Data/
│   ├── OSM.xml              # Dataset OpenStreetMap
│   ├── Preparation_data.ipynb
│   └── processed_data.pt    # Données transformées pour PyG
│
├── Embeddings/
│   ├── GAT_embeddings.pt
│   ├── GCN_embeddings.pt
│   └── VGAE_embeddings.pt
│
├── LLM/
│   ├── Function.py
│   └── Preparation.ipynb
│
├── main/
│   ├── Evaluation.py
│   ├── Main.py
│   ├── Test.py
│   └── Train.py
│
├── Modules/
│   ├── GAT_model.pth
│   ├── GCN_model.pth
│   └── VGAE_model.pth
│
├── config.py
└── README.md
```

### Détails des dossiers :
- **Architectures/** : Contient les fichiers pour différents modèles de GNN (GCN, GAT, VGAE)
- **Data/** : Contient le dataset brut OSM.xml, les notebooks pour préparer les données et le fichier .pt prêt pour PyG
- **Embeddings/** : Contient les embeddings générés par les modèles GNN pour les villes
- **LLM/** : Contient les fonctions et notebooks pour l'analyse avec le LLM (explications de similarité)
- **main/** : Contient le code principal pour l'entraînement, l'évaluation et les tests des modèles
- **Modules/** : Contient les modèles sauvegardés au format .pth
- **config.py** : Paramètres globaux du projet

## 📊 Dataset

Le dataset utilisé est extrait de **OpenStreetMap** sous format GraphML/XML.

### Exemple de structure XML :
```xml
<node id="0">
  <data key="d0">-145.509722</data>    <!-- Longitude -->
  <data key="d1">-17.353889</data>     <!-- Latitude -->
  <data key="d2">10000</data>          <!-- Population -->
  <data key="d3">FRENCH_POLYNESIA</data> <!-- Country -->
  <data key="d4">Anaa</data>           <!-- City Name -->
</node>
<node id="1">
  <data key="d0">-140.95</data>
  <data key="d1">-18.066667</data>
  <data key="d2">10000</data>
  <data key="d3">FRENCH_POLYNESIA</data>
  <data key="d4">Hao Island</data>
</node>
```

## Caractéristiques du graphe

- **3363 nœuds (villes)**  
- **13547 arêtes (relations)**

### Attributs des nœuds

Chaque nœud représente une ville avec :  

- **Features numériques pour les GNN :**  
  - longitude  
  - latitude  
  - population  

- **Informations textuelles pour les explications via LLM :**  
  - pays  
  - nom de la ville  

## 🧰 Dépendances

- Python 3.10  
- PyTorch ≥ 2.0  
- PyTorch Geometric (PyG)  
- Transformers (HuggingFace)  
- scikit-learn  
- Matplotlib, Seaborn, NumPy, tqdm  


