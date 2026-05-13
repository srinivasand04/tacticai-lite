# TacticAI-Lite 🏟️

> GCN & GAT applied to real football corner kick data  
> Built as part of AI Coach research internship preparation — DVRC / ESILV Paris

---

## What This Project Demonstrates

- **Graph Neural Network (GCN)** implementation using PyTorch Geometric — Kipf & Welling 2017
- **Graph Attention Network (GAT)** with learnable per-edge attention weights — Veličković et al. 2018
- Real football data processing using **StatsBomb Open Data** (no API key required)
- **Corner kick receiver prediction** — a node classification task on 22-player graphs
- **What-If tactical simulation** — perturb player positions, observe prediction shifts
- Side-by-side **model comparison** of two GNN architectures on the same task

---

## Results

| Model | Val Accuracy | Baseline | Improvement |
|-------|-------------|----------|-------------|
| GCN   | ~35–40%     | ~9%      | ~4× over random  |
| GAT   | ~38–44%     | ~9%      | ~4.5× over random|

The baseline is `1 / avg_players_per_graph ≈ 9%` (random guess from 11 attackers).  
Both models significantly exceed random — showing the graph structure encodes real tactical information.

---

## Quick Start

### 1. Create virtual environment

```bash
python -m venv tacticai_env

# Mac / Linux
source tacticai_env/bin/activate

# Windows
tacticai_env\Scripts\activate
```

### 2. Install dependencies

```bash
# PyTorch (CPU) — adjust URL for GPU if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric

# Remaining packages
pip install statsbombpy mplsoccer pandas numpy matplotlib scikit-learn jupyter tqdm
```

### 3. Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "from statsbombpy import sb; print('StatsBomb OK:', sb.competitions().shape)"
```

### 4. Run in order (step by step)

```bash
# Step 1: Download & cache corner kick data (~1 min first time)
python -m data.loader

# Step 2: Build and inspect graphs
python -m graph.builder

# Step 3: Verify GCN forward pass
python -m models.gcn

# Step 4: Verify GAT forward pass
python -m models.gat

# Step 5: 5-epoch smoke test of training loop
python -m train.trainer

# Step 6: Save a sample pitch visualization
python -m viz.pitch_viz

# Step 7: Run What-If simulation
python -m whatif.simulator

# Step 8: Launch full demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## Project Structure

```
tacticai-lite/
├── requirements.txt          ← All dependencies with versions
├── data/
│   ├── loader.py             ← StatsBomb data loading + freeze frame extraction
│   └── corners_cache.pkl     ← Auto-generated disk cache (first run only)
├── graph/
│   └── builder.py            ← Freeze frame → PyTorch Geometric Data graph
├── models/
│   ├── gcn.py                ← GCN model (GCNConv, 3 layers)
│   └── gat.py                ← GAT model (GATConv, multi-head, 3 layers)
├── train/
│   └── trainer.py            ← Shared training loop + model comparison table
├── viz/
│   └── pitch_viz.py          ← Pitch + graph + attention visualizations
├── whatif/
│   └── simulator.py          ← What-If perturbation engine
├── notebooks/
│   └── demo.ipynb            ← End-to-end walkthrough (main showcase)
└── outputs/
    ├── figures/              ← Saved pitch PNGs
    └── results/              ← Training logs, comparison CSVs
```

---

## Key Concepts

### What is a GCN?

A **Graph Convolutional Network** (Kipf & Welling, 2017) applies a learned linear transformation to each node's features, then aggregates information from neighbouring nodes with **equal weights**.

```
New embedding of player i = W · (average of { features of i and all nearby players })
```

The normalised adjacency formulation `D^{-1/2} A D^{-1/2}` prevents feature explosion in high-degree nodes. After 3 layers, each player's embedding encodes the tactical structure of their 3-hop neighbourhood.

### What is a GAT?

A **Graph Attention Network** (Veličković et al., 2018) extends GCN by replacing equal-weight averaging with **learned attention coefficients**:

```
α_{ij} = softmax( LeakyReLU( aᵀ · [W·h_i ‖ W·h_j] ) )
h_i'   = Σ_{j ∈ N(i)}  α_{ij} · W · h_j
```

The model learns that "the near-post attacker matters more than the goalkeeper" when predicting corner receivers — without being told this explicitly.

### Why football graphs?

A football team is a natural graph: players are nodes, their interactions (passing lanes, proximity, blocking) are edges. Corner kicks are ideal because:
- Every player has a fixed starting position (freeze frame)
- There is a clear discrete outcome (who receives?)
- The task is hard enough that random baseline is only ~9%

### What is the What-If simulator?

The simulator perturbs a player's position by `(dx, dy)` metres, rebuilds the proximity graph, and re-runs model inference. The probability delta table shows how much a positional change shifts receiver predictions — testing whether the model has learned tactical reasoning.

---

## Research Context

This project is inspired by **TacticAI** (Wang et al., 2024, *Nature Communications*):
- DeepMind + Liverpool FC collaboration
- Uses GNN on StatsBomb corner kick data (same dataset)
- Models player interactions as graphs, predicts tactical outcomes
- Extended with tactical advice generation using a retrieval system

**TacticAI-Lite** replicates the core GNN architecture on open data as a learning exercise and interview preparation piece.

---

## Next Steps (Full Internship Project)

| Extension | Description | Difficulty |
|---|---|---|
| Temporal GNN | Replace static snapshot with frame sequences using T-GCN or EvolveGCN | High |
| Real event edges | Use actual passing/pressing events instead of proximity threshold | Medium |
| Velocity features | Add `[vx, vy]` to node features for motion context | Low |
| Open play extension | Apply from set-pieces to open-play 11v11 | High |
| YOLO + DeepSORT | Real-time player detection + graph construction from video | Very High |
| Larger dataset | Use all StatsBomb 360 freeze-frame matches | Low |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | ≥2.0.0 | Deep learning framework |
| torch-geometric | ≥2.3.0 | Graph neural network layers |
| statsbombpy | ≥1.1.0 | StatsBomb open data access |
| mplsoccer | ≥1.2.0 | Football pitch visualization |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | Train/val split |
| matplotlib | ≥3.7.0 | Plotting |

---

## References

1. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. https://arxiv.org/abs/1609.02907

2. Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR. https://arxiv.org/abs/1710.10903

3. Wang, R. et al. (2024). *TacticAI: an AI assistant for football tactics*. Nature Communications. https://doi.org/10.1038/s41467-024-45965-x

4. StatsBomb Open Data. https://github.com/statsbomb/open-data
