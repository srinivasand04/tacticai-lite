"""
viz/pitch_viz.py — Football pitch visualizations using mplsoccer
=================================================================
Four visualization functions:

1. plot_corner_graph       — draw player graph on a dark pitch
2. plot_attention_heatmap  — GAT attention weights as edge intensity
3. plot_training_curves    — loss + accuracy over epochs (GCN vs GAT)
4. plot_model_comparison   — horizontal bar chart comparing both models

All figures are saved to outputs/figures/ automatically.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

# Output directory
FIG_DIR = Path(__file__).parent.parent / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Corner graph on pitch ─────────────────────────────────────────────────

def plot_corner_graph(
    graph_data,
    title:           str          = "Corner Kick Graph",
    prediction:      int | None   = None,
    true_receiver:   int | None   = None,
    save_name:       str          = "corner_graph.png",
) -> None:
    """
    Draw the player interaction graph on a football pitch.

    Node colours:
        Red   = attacker (same team as corner taker)
        Blue  = defender
        Green = goalkeeper
    Markers:
        Circle = outfield player
        Square = goalkeeper
    Overlays:
        Gold star      = predicted receiver
        Bright green ★ = true receiver (if labels differ from prediction)

    Parameters
    ----------
    graph_data    : torch_geometric.data.Data  — graph with .locations, .x, .edge_index
    title         : figure title
    prediction    : node index of model's top prediction (optional)
    true_receiver : node index of actual receiver (optional)
    save_name     : filename inside outputs/figures/
    """
    try:
        from mplsoccer import Pitch
    except ImportError:
        print("[viz] mplsoccer not installed. Run: pip install mplsoccer")
        return

    # ── Set up pitch ─────────────────────────────────────────────────────────
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a472a",
        line_color="#ffffff",
        line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1b2a")

    # ── Extract node positions (raw pitch coords, not normalised) ─────────────
    if hasattr(graph_data, "locations"):
        locs = graph_data.locations.numpy()      # [N, 2]  in StatsBomb coords
    else:
        # Fallback: de-normalise from x features
        locs = graph_data.x[:, :2].numpy()
        locs[:, 0] *= 120.0
        locs[:, 1] *= 80.0

    N = len(locs)
    features = graph_data.x.numpy()             # [N, 5]

    # ── Draw edges ───────────────────────────────────────────────────────────
    edge_index = graph_data.edge_index.numpy()  # [2, E]
    edge_attr  = (
        graph_data.edge_attr.numpy().flatten()
        if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None
        else np.zeros(edge_index.shape[1])
    )

    seen_edges = set()
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        if (j, i) in seen_edges:
            continue
        seen_edges.add((i, j))

        x_vals = [locs[i, 0], locs[j, 0]]
        y_vals = [locs[i, 1], locs[j, 1]]
        # Closer players → brighter edge
        alpha = max(0.1, 1.0 - edge_attr[k]) if k < len(edge_attr) else 0.3
        ax.plot(x_vals, y_vals, color="white", linewidth=0.8, alpha=alpha, zorder=3)

    # ── Draw nodes ───────────────────────────────────────────────────────────
    for i in range(N):
        x_pos, y_pos   = locs[i, 0], locs[i, 1]
        is_attacker    = features[i, 2] == 1.0
        is_gk          = features[i, 3] == 1.0

        color  = "#e63946" if is_attacker else "#457b9d"  # red / blue
        marker = "s" if is_gk else "o"
        size   = 180 if is_gk else 150

        if is_gk:
            color = "#2dc653"  # bright green for GK

        ax.scatter(x_pos, y_pos, c=color, s=size, marker=marker,
                   zorder=5, edgecolors="white", linewidths=0.8)

        # Player index label
        ax.text(x_pos + 0.8, y_pos + 0.8, str(i),
                color="white", fontsize=7, zorder=6, fontweight="bold")

    # ── Highlight true receiver ───────────────────────────────────────────────
    if true_receiver is not None and true_receiver < N:
        xr, yr = locs[true_receiver, 0], locs[true_receiver, 1]
        ax.scatter(xr, yr, c="#00ff88", s=400, marker="o", zorder=7,
                   edgecolors="white", linewidths=2.0, alpha=0.85,
                   label=f"True receiver (#{true_receiver})")

    # ── Highlight prediction ──────────────────────────────────────────────────
    if prediction is not None and prediction < N:
        xp, yp = locs[prediction, 0], locs[prediction, 1]
        ax.scatter(xp, yp, c="#ffd700", s=600, marker="*", zorder=8,
                   edgecolors="white", linewidths=1.5,
                   label=f"Predicted (#{prediction})")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#e63946", label="Attacker"),
        mpatches.Patch(color="#457b9d", label="Defender"),
        mpatches.Patch(color="#2dc653", label="Goalkeeper"),
    ]
    if prediction is not None:
        legend_handles.append(
            plt.scatter([], [], c="#ffd700", s=200, marker="*", label="Predicted")
        )
    if true_receiver is not None:
        legend_handles.append(
            plt.scatter([], [], c="#00ff88", s=120, marker="o", label="True receiver")
        )

    ax.legend(handles=legend_handles, loc="upper left",
              facecolor="#1a1a2e", edgecolor="white",
              labelcolor="white", fontsize=9)

    ax.set_title(title, color="white", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()

    save_path = FIG_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {save_path}")


# ─── 2. GAT Attention heatmap ─────────────────────────────────────────────────

def plot_attention_heatmap(
    graph_data,
    attention_weights: torch.Tensor,
    attention_edge_index: torch.Tensor,
    focus_player_idx: int = 0,
    save_name: str = "attention_heatmap.png",
) -> None:
    """
    Visualise GAT attention weights from the focus player's perspective.

    Edge thickness and colour intensity reflect how much attention the
    focus player assigns to each of its neighbours.

    Parameters
    ----------
    graph_data           : Data object with .locations
    attention_weights    : Tensor [E, 1] from GAT.get_attention_weights()
    attention_edge_index : Tensor [2, E]
    focus_player_idx     : player index to focus on
    save_name            : filename
    """
    try:
        from mplsoccer import Pitch
    except ImportError:
        print("[viz] mplsoccer not installed.")
        return

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#1a472a",
        line_color="#cccccc",
        line_zorder=2,
    )
    fig, ax = pitch.draw(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1b2a")

    if hasattr(graph_data, "locations"):
        locs = graph_data.locations.numpy()
    else:
        locs = graph_data.x[:, :2].numpy()
        locs[:, 0] *= 120.0
        locs[:, 1] *= 80.0

    N        = len(locs)
    ei       = attention_edge_index.numpy()
    alpha    = attention_weights.detach().numpy().flatten()

    # Draw edges from focus player, coloured by attention
    for k in range(ei.shape[1]):
        src, dst = ei[0, k], ei[1, k]
        if src != focus_player_idx:
            continue
        a_val = float(alpha[k]) if k < len(alpha) else 0.1
        lw    = 1.0 + 8.0 * a_val      # thicker = higher attention
        color = plt.cm.plasma(a_val)    # plasma colormap: dark→yellow
        ax.plot([locs[src, 0], locs[dst, 0]],
                [locs[src, 1], locs[dst, 1]],
                color=color, linewidth=lw, alpha=0.9, zorder=4)

    # Draw all nodes
    features = graph_data.x.numpy()
    for i in range(N):
        is_att = features[i, 2] == 1.0
        is_gk  = features[i, 3] == 1.0
        color  = "#2dc653" if is_gk else ("#e63946" if is_att else "#457b9d")
        ax.scatter(locs[i, 0], locs[i, 1], c=color, s=150,
                   zorder=5, edgecolors="white", linewidths=0.8)
        ax.text(locs[i, 0] + 0.8, locs[i, 1] + 0.8, str(i),
                color="white", fontsize=7, zorder=6, fontweight="bold")

    # Highlight focus player
    ax.scatter(locs[focus_player_idx, 0], locs[focus_player_idx, 1],
               c="#ffd700", s=500, marker="*", zorder=7,
               edgecolors="white", linewidths=2, label=f"Focus: Player {focus_player_idx}")

    ax.legend(facecolor="#1a1a2e", edgecolor="white",
              labelcolor="white", fontsize=10, loc="upper left")
    ax.set_title(
        f"GAT Attention — Player {focus_player_idx} focuses on these teammates",
        color="white", fontsize=13, pad=12, fontweight="bold"
    )

    plt.tight_layout()
    save_path = FIG_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {save_path}")


# ─── 3. Training curves ───────────────────────────────────────────────────────

def plot_training_curves(
    gcn_history: dict,
    gat_history: dict,
    save_name: str = "training_curves.png",
) -> None:
    """
    Side-by-side training loss, Top-1, and Top-3 accuracy curves for GCN and GAT.

    Parameters
    ----------
    gcn_history : dict with keys train_loss, val_loss, train_acc1, val_acc1,
                  train_acc3, val_acc3  (v2 trainer format)
    gat_history : same structure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0d1b2a")

    GCN_COLOR = "#2ec4b6"   # teal
    GAT_COLOR = "#ff9f1c"   # orange

    # Align lengths in case early stopping gave different epoch counts
    gcn_len = len(gcn_history["train_loss"])
    gat_len = len(gat_history["train_loss"])

    gcn_ep = range(1, gcn_len + 1)
    gat_ep = range(1, gat_len + 1)

    # ── Panel 1: Loss ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.plot(gcn_ep, gcn_history["val_loss"],   color=GCN_COLOR, lw=2, label="GCN val")
    ax.plot(gat_ep, gat_history["val_loss"],   color=GAT_COLOR, lw=2, label="GAT val")
    ax.plot(gcn_ep, gcn_history["train_loss"], color=GCN_COLOR, lw=1, ls="--", alpha=0.5, label="GCN train")
    ax.plot(gat_ep, gat_history["train_loss"], color=GAT_COLOR, lw=1, ls="--", alpha=0.5, label="GAT train")
    ax.set_title("NLL Loss", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#0d1b2a", edgecolor="#444", labelcolor="white", fontsize=8)

    # ── Panel 2: Top-1 accuracy ──────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.plot(gcn_ep, [v*100 for v in gcn_history["val_acc1"]],   color=GCN_COLOR, lw=2, label="GCN val")
    ax.plot(gat_ep, [v*100 for v in gat_history["val_acc1"]],   color=GAT_COLOR, lw=2, label="GAT val")
    ax.plot(gcn_ep, [v*100 for v in gcn_history["train_acc1"]], color=GCN_COLOR, lw=1, ls="--", alpha=0.5, label="GCN train")
    ax.plot(gat_ep, [v*100 for v in gat_history["train_acc1"]], color=GAT_COLOR, lw=1, ls="--", alpha=0.5, label="GAT train")
    ax.set_title("Top-1 Accuracy", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#0d1b2a", edgecolor="#444", labelcolor="white", fontsize=8)

    # ── Panel 3: Top-3 accuracy ──────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#1a1a2e")
    ax.plot(gcn_ep, [v*100 for v in gcn_history["val_acc3"]],   color=GCN_COLOR, lw=2, label="GCN val")
    ax.plot(gat_ep, [v*100 for v in gat_history["val_acc3"]],   color=GAT_COLOR, lw=2, label="GAT val")
    ax.plot(gcn_ep, [v*100 for v in gcn_history["train_acc3"]], color=GCN_COLOR, lw=1, ls="--", alpha=0.5, label="GCN train")
    ax.plot(gat_ep, [v*100 for v in gat_history["train_acc3"]], color=GAT_COLOR, lw=1, ls="--", alpha=0.5, label="GAT train")
    ax.set_title("Top-3 Accuracy (tactical metric)", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, alpha=0.2)
    ax.legend(facecolor="#0d1b2a", edgecolor="#444", labelcolor="white", fontsize=8)

    plt.suptitle("GCN vs GAT — Training History (v2, 815 sequences)",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = FIG_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {save_path}")


# ─── 4. Model comparison bar chart ───────────────────────────────────────────

def plot_model_comparison(
    comparison_df,
    save_name: str = "model_comparison.png",
) -> None:
    """
    Horizontal bar chart comparing GCN and GAT.

    Parameters
    ----------
    comparison_df : pandas DataFrame from trainer.compare_models()
                    Columns: Model, Val Accuracy, Parameters, Avg Inference (ms)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0d1b2a")

    models = comparison_df["Model"].tolist()
    colors = ["#2ec4b6", "#ff9f1c"]   # GCN teal, GAT orange

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#0d1b2a")

    def _pct(col):
        return [float(v.replace("%", "")) for v in comparison_df[col].tolist()]

    # ── Top-1 Accuracy ───────────────────────────────────────────────────────
    accs1 = _pct("Top-1 Accuracy")
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    bars = ax.barh(models, accs1, color=colors, edgecolor="white", linewidth=0.6, height=0.5)
    for bar, val in zip(bars, accs1):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=11)
    ax.set_title("Top-1 Accuracy (%)", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(accs1) * 1.4)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    # ── Top-3 Accuracy ───────────────────────────────────────────────────────
    accs3 = _pct("Top-3 Accuracy")
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    bars = ax.barh(models, accs3, color=colors, edgecolor="white", linewidth=0.6, height=0.5)
    for bar, val in zip(bars, accs3):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=11)
    ax.set_title("Top-3 Accuracy (%) — tactical metric", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(accs3) * 1.3)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    # ── Inference time ───────────────────────────────────────────────────────
    times = [float(v) for v in comparison_df["Avg Inference (ms)"].tolist()]
    ax = axes[2]
    ax.set_facecolor("#1a1a2e")
    bars = ax.barh(models, times, color=colors, edgecolor="white", linewidth=0.6, height=0.5)
    for bar, val in zip(bars, times):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f} ms", va="center", color="white", fontsize=11)
    ax.set_title("Avg Inference (ms / graph)", color="white", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(times) * 1.5)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")

    plt.suptitle("GCN vs GAT — Model Comparison (v2, 815 sequences)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = FIG_DIR / save_name
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved → {save_path}")


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from data.loader   import load_corners
    from graph.builder import build_dataset

    corners = load_corners(use_cache=True)
    graphs  = build_dataset(corners, threshold=10.0)

    if graphs:
        print(f"Plotting corner graph for graph[0] ...")
        plot_corner_graph(
            graphs[0],
            title="Sample Corner Kick Graph",
            true_receiver=graphs[0].y.item(),
            save_name="test_corner_graph.png",
        )
        print("Pitch visualization saved. Check outputs/figures/")
    else:
        print("No graphs — run `python -m data.loader` first.")
