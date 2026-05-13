# TacticAI-Lite 🏟️

> GCN & GAT applied to real football corner kick data  
> Built as interview preparation for AI Coach research internship — DVRC / ESILV Paris

---

## Results (v2 — Full Dataset, 815 Sequences)

| Model | Top-1 Accuracy | Top-3 Accuracy | Parameters | Inference | Stopped at |
|-------|---------------|---------------|------------|-----------|------------|
| GCN   | 12.5%         | **26.6%**     | 6,913      | 0.18 ms   | Epoch 45   |
| GAT   | **13.4%**     | 23.3%         | 77,697     | 0.46 ms   | Epoch 19   |
| Random baseline | 5.7% | 17.1%     | —          | —         | —          |

**Key finding:** GCN outperforms GAT on Top-3 accuracy (26.6% vs 23.3%), which is the tactically meaningful metric — narrowing the field from ~17 candidates to 3. GAT has more parameters (77k) than training signal (652 graphs), so it overfits fast and early-stops at epoch 19. GCN generalises better: simpler architecture, 2.6× faster, 11× fewer parameters.

**v1 → v2 improvements:** 228 → 815 sequences (3.6×), BatchNorm after every GNN layer, Top-3 accuracy metric, early stopping with patience=15, 6 competitions instead of 1.

---

## What This Project Demonstrates

- **Graph Convolutional Network (GCN)** — Kipf & Welling 2017, normalised aggregation
- **Graph Attention Network (GAT)** — Veličković et al. 2018, learned per-edge attention
- Real football data processing from **StatsBomb Open Data** (no API key required)
- **Corner kick receiver prediction** — node classification on 17-player tactical graphs
- **What-If tactical simulation** — move a player, see how predictions shift
- **Bias-variance tradeoff** in GNNs: expressiveness vs. dataset size
- **Top-3 accuracy** as a tactically useful metric (narrows 17 players to 3)

---

## Branches

| Branch | Dataset | Highlights |
|--------|---------|------------|
| `main` | WC 2018 only (228 sequences) | v1 baseline — simple loader, no BatchNorm |
| `v2-extended` | 6 competitions (815 sequences) | BatchNorm, Top-3 metric, early stopping, multi-comp loader |

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/tacticai-lite.git
cd tacticai-lite
git checkout v2-extended     # for the full v2 version

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

### 3. Run in order

```bash
# Step 1: Download & cache corner kick data from 6 competitions (~10 min first time)
python -c "from data.loader import load_corners; load_corners(use_cache=False)"

# Step 2: Verify graph builder
python -m graph.builder

# Step 3: Verify GCN forward pass
python -m models.gcn

# Step 4: Verify GAT forward pass
python -m models.gat

# Step 5: Full 100-epoch training with early stopping
python -m train.trainer

# Step 6: Pitch visualizations
python -m viz.pitch_viz

# Step 7: What-If simulation
python -m whatif.simulator

# Step 8: Full demo notebook
jupyter notebook notebooks/demo.ipynb
```

---

## Project Structure

```
tacticai-lite/
├── requirements.txt          ← All dependencies with versions
├── data/
│   ├── loader.py             ← Multi-competition StatsBomb loader (v2: 6 competitions)
│   └── corners_cache_v2.pkl  ← Auto-generated disk cache (815 sequences)
├── graph/
│   └── builder.py            ← Freeze frame → PyTorch Geometric Data graph
│                                Node features: [x, y, is_attacker, is_gk, dist_goal]
│                                Edges: proximity < 10m, fallback 3-NN
├── models/
│   ├── gcn.py                ← GCN: 5→64→64→32→1, BatchNorm, Dropout(0.3)
│   └── gat.py                ← GAT: 4-head attention, 5→256→256→32→1, BatchNorm
├── train/
│   └── trainer.py            ← Training loop, Top-1/Top-3 accuracy, early stopping
├── viz/
│   └── pitch_viz.py          ← Pitch + graph + attention + training curve plots
├── whatif/
│   └── simulator.py          ← Perturb player position → measure Δprobability
├── notebooks/
│   └── demo.ipynb            ← End-to-end walkthrough (main showcase)
└── outputs/
    ├── figures/              ← Saved pitch PNGs (committed for GitHub preview)
    └── results/              ← Training logs, comparison CSVs
```

---

## Key Concepts

### Graph representation of a corner kick

At the moment a shot is taken off a corner, every player on the pitch is at a known (x, y) position (the StatsBomb freeze frame). TacticAI-Lite converts this into a graph:

- **Nodes** — all players present (~17 per graph after filtering), each with 5 features:
  `[x_norm, y_norm, is_attacker, is_goalkeeper, distance_to_goal]`
- **Edges** — players within 10 metres are connected; isolated players get edges to their 3 nearest neighbours
- **Label** — the node corresponding to the player who actually shot (distance to shot location = 0)

### What is a GCN?

A **Graph Convolutional Network** (Kipf & Welling, 2017) aggregates neighbourhood information with **equal weights**:

```
h_i' = W · mean({ h_j : j ∈ neighbours(i) ∪ {i} })
```

After 3 layers, each player's embedding has absorbed the tactical context of their 3-hop neighbourhood — nearby attackers, blocking defenders, goalkeeper position.

Normalised adjacency `D^{-1/2} A D^{-1/2}` prevents feature explosion in dense clusters.

### What is a GAT?

A **Graph Attention Network** (Veličković et al., 2018) replaces equal-weight averaging with **learned attention**:

```
e_ij   = LeakyReLU( aᵀ · [W·h_i ‖ W·h_j] )
α_ij   = softmax_j( e_ij )
h_i'   = Σ_{j ∈ N(i)} α_ij · W · h_j
```

The model can learn that "the near-post attacker matters more than a midfielder 15m away" — without being told this explicitly. With enough data, GAT should outperform GCN. With only 815 samples, GCN generalises better.

### Why Top-3 accuracy?

With ~17 players per graph, random Top-1 accuracy is 5.7% and random Top-3 is 17.1%. A model reaching 26.6% Top-3 **narrows the predicted receiver pool from 17 to 3**, which has real tactical value — a coach or analyst can act on "these 3 players are likely to receive" before the corner is taken.

### What-If simulation

The simulator moves a player by `(dx, dy)` metres, rebuilds the proximity graph, and re-runs inference. The probability delta table (in percentage points, pp) shows whether the model has learned position-dependent tactical reasoning — e.g., moving an attacker toward the near post increases their reception probability.

---

## Architecture Summary

### GCN (v2)
```
Input: [N, 5]  — N players, 5 features each
GCNConv(5→64)  + BatchNorm1d(64)  + ReLU + Dropout(0.3)
GCNConv(64→64) + BatchNorm1d(64)  + ReLU + Dropout(0.3)
GCNConv(64→32) + ReLU
Linear(32→1)   → per-graph softmax → receiver probability
Total: 6,913 parameters
```

### GAT (v2)
```
Input: [N, 5]
GATConv(5→64, heads=4,  concat=True)  + BatchNorm1d(256) + ELU + Dropout(0.2)
GATConv(256→64, heads=4, concat=True) + BatchNorm1d(256) + ELU + Dropout(0.2)
GATConv(256→32, heads=1, concat=False) + ELU
Linear(32→1)   → per-graph softmax → receiver probability
Total: 77,697 parameters
```

---

## Research Context

This project is inspired by **TacticAI** (Wang et al., 2024, *Nature Communications*):
- DeepMind + Liverpool FC collaboration
- GNN on StatsBomb corner kick data (same open dataset used here)
- Models player interactions as graphs, predicts tactical outcomes
- Extended with generative tactical advice using a retrieval system

TacticAI-Lite replicates the core GNN node-classification task on open data as a learning exercise and interview preparation piece.

---

## Possible Extensions

| Extension | Description | Difficulty |
|---|---|---|
| Velocity features | Add `[vx, vy]` to node features using consecutive frames | Low |
| Real event edges | Use actual passing/pressing sequences instead of proximity | Medium |
| Temporal GNN | Replace static snapshot with frame sequences (T-GCN / EvolveGCN) | High |
| StatsBomb 360 | Use full 360 freeze-frame dataset for 10× more data | Low |
| Open play | Apply GNN from set-pieces to open-play sequences | High |
| YOLO + DeepSORT | Real-time player detection + graph construction from video | Very High |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| torch | ≥2.0.0 | Deep learning framework |
| torch-geometric | ≥2.3.0 | GCNConv, GATConv layers |
| statsbombpy | ≥1.1.0 | StatsBomb open data access |
| mplsoccer | ≥1.2.0 | Football pitch visualization |
| pandas | ≥2.0.0 | Data manipulation |
| scikit-learn | ≥1.3.0 | Train/val split |
| matplotlib | ≥3.7.0 | Plotting |
| tqdm | ≥4.65.0 | Progress bars during data loading |

---

## References

1. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR. https://arxiv.org/abs/1609.02907

2. Veličković, P. et al. (2018). *Graph Attention Networks*. ICLR. https://arxiv.org/abs/1710.10903

3. Wang, R. et al. (2024). *TacticAI: an AI assistant for football tactics*. Nature Communications. https://doi.org/10.1038/s41467-024-45965-x

4. StatsBomb Open Data. https://github.com/statsbomb/open-data
