# 🌆 LLM-GNN : Analyse de graphes urbains et similarité de villes

## 📖 Description du projet

LLM-GNN combine **Graph Neural Networks (GNN)** et **Large Language Models (LLM)** pour analyser les relations et similarités entre villes. Le projet utilise un graphe construit à partir de **OpenStreetMap (OSM)** pour :
- Prédire les liens entre villes
- Générer des explications textuelles sur la similarité des villes selon la population, la localisation et le pays

## 📂 Structure du projet
'''
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
'''
