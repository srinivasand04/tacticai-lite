"""
generate_figures.py — Generate all TacticAI-Lite v2 visualizations
===================================================================
Run this script once after training to produce four figures:

    outputs/figures/corner_graph.png       — player graph on a football pitch
    outputs/figures/attention_heatmap.png  — GAT attention weights
    outputs/figures/training_curves.png    — loss + Top-1 + Top-3 over epochs
    outputs/figures/model_comparison.png   — Top-1, Top-3, inference bar charts

Usage:
    python generate_figures.py

This script trains both models (up to 100 epochs with early stopping) and
then generates all figures. If you want to skip retraining, save the model
weights first and load them instead.
"""

import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader    import load_corners
from graph.builder  import build_dataset
from train.trainer  import run_full_training, compare_models
from viz.pitch_viz  import (
    plot_corner_graph,
    plot_attention_heatmap,
    plot_training_curves,
    plot_model_comparison,
)
from sklearn.model_selection import train_test_split

print("=" * 62)
print("TacticAI-Lite v2 — Figure Generator")
print("=" * 62)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("\n[1/5] Loading 815-sequence dataset ...")
corners = load_corners(use_cache=True)
graphs  = build_dataset(corners)
print(f"      {len(graphs)} graphs ready.")

# ── Step 2: Train both models ─────────────────────────────────────────────────
print("\n[2/5] Training GCN (up to 100 epochs, early stopping) ...")
gcn_model, gcn_history = run_full_training(graphs, "gcn", epochs=100, verbose=True)

print("\n[3/5] Training GAT (up to 100 epochs, early stopping) ...")
gat_model, gat_history = run_full_training(graphs, "gat", epochs=100, verbose=True)

# Val split (same random_state as trainer for consistency)
_, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

# ── Step 3: Corner graph on pitch ─────────────────────────────────────────────
print("\n[4/5] Generating pitch visualizations ...")

sample_graph = graphs[0]
import torch
with torch.no_grad():
    gcn_logits = gcn_model(sample_graph)
    pred_idx   = int(gcn_logits.squeeze(-1).argmax().item())
true_idx = sample_graph.y.item()

plot_corner_graph(
    sample_graph,
    title         = "Corner Kick Graph — GCN Prediction vs True Receiver",
    prediction    = pred_idx,
    true_receiver = true_idx,
    save_name     = "corner_graph.png",
)

# ── Step 4: GAT attention heatmap ─────────────────────────────────────────────
# Focus on the true receiver (player the model should attend to)
attn_ei, attn_alpha = gat_model.get_attention_weights(sample_graph)
plot_attention_heatmap(
    sample_graph,
    attention_weights    = attn_alpha,
    attention_edge_index = attn_ei,
    focus_player_idx     = true_idx,
    save_name            = "attention_heatmap.png",
)

# ── Step 5: Training curves and model comparison ──────────────────────────────
plot_training_curves(gcn_history, gat_history, save_name="training_curves.png")

df = compare_models(gcn_model, gat_model, val_graphs)
print("\nv2 Model Comparison:")
col_widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}
header = "  ".join(col.ljust(col_widths[col]) for col in df.columns)
print("\n" + header)
print("-" * len(header))
for _, row in df.iterrows():
    print("  ".join(str(row[col]).ljust(col_widths[col]) for col in df.columns))

plot_model_comparison(df, save_name="model_comparison.png")

print("\n[5/5] Done. Figures saved to outputs/figures/")
print("      corner_graph.png")
print("      attention_heatmap.png")
print("      training_curves.png")
print("      model_comparison.png")
