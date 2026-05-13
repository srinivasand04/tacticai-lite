"""
train/trainer.py — Shared training loop for GCN and GAT
=========================================================
Both models share the same training logic:
  - Loss      : Negative Log-Likelihood (NLLLoss) on per-graph softmax output
  - Optimizer : Adam (lr=0.001, weight_decay=1e-4)
  - Scheduler : StepLR — halve lr every 20 epochs
  - Epochs    : 60

Why NLL loss?
    After the final linear layer, we apply log_softmax over all player nodes
    in a graph. NLLLoss then penalises the negative log-probability assigned
    to the true receiver node. This is equivalent to cross-entropy loss but
    structured for our per-graph node classification setup.

Why Adam + StepLR?
    Adam adapts learning rates per parameter — robust to scale differences
    between feature dimensions. StepLR prevents overfitting after initial
    fast convergence on a small dataset (~500–800 graphs).

Expected accuracy
    Random baseline: 1 / avg_players ≈ 9 % (if 11 attackers per frame).
    GCN typically reaches 30–40 % on this task.
    GAT typically reaches 35–44 % — better but not always by a huge margin.
    The limited dataset size is the main bottleneck, not architecture choice.
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


# ─── Loss helper ─────────────────────────────────────────────────────────────

def _per_graph_nll_loss(logits: torch.Tensor, y: torch.Tensor, ptr) -> torch.Tensor:
    """
    Compute NLL loss by applying per-graph log_softmax then selecting the
    true receiver node's log-probability.

    Parameters
    ----------
    logits : Tensor [N, 1]  — raw scores from model
    y      : Tensor [B]     — receiver node index (relative to each graph)
    ptr    : Tensor [B+1]   — cumulative node counts (from DataBatch)

    The ptr tensor tells us where each graph starts and ends in the flattened
    node list. For example, ptr = [0, 11, 20, 33] means:
        graph 0: nodes 0–10,  graph 1: nodes 11–19,  graph 2: nodes 20–32
    """
    loss = 0.0
    n_graphs = len(ptr) - 1

    for i in range(n_graphs):
        start, end = ptr[i].item(), ptr[i + 1].item()
        graph_logits = logits[start:end].squeeze(-1)          # [n_i]
        log_probs    = F.log_softmax(graph_logits, dim=0)     # [n_i]
        label        = y[i].item()

        if label < len(log_probs):
            loss += -log_probs[label]
        else:
            # Safety: if label out of range (shouldn't happen), use last node
            loss += -log_probs[-1]

    return loss / max(n_graphs, 1)


def _per_graph_accuracy(logits: torch.Tensor, y: torch.Tensor, ptr) -> float:
    """Compute fraction of graphs where argmax matches true receiver."""
    correct  = 0
    n_graphs = len(ptr) - 1

    for i in range(n_graphs):
        start, end = ptr[i].item(), ptr[i + 1].item()
        graph_logits = logits[start:end].squeeze(-1)
        pred  = graph_logits.argmax().item()
        label = y[i].item()
        if pred == label:
            correct += 1

    return correct / max(n_graphs, 1)


# ─── Epoch-level functions ────────────────────────────────────────────────────

def train_epoch(model, loader: DataLoader, optimizer) -> tuple[float, float]:
    """
    Run one training epoch over `loader`.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss   = _per_graph_nll_loss(logits, batch.y, batch.ptr)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = _per_graph_accuracy(logits, batch.y, batch.ptr)

        total_loss += loss.item()
        total_acc  += acc
        n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def evaluate(model, loader: DataLoader) -> tuple[float, float]:
    """
    Evaluate model on `loader` — no gradient computation.

    Returns
    -------
    (avg_loss, avg_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            loss   = _per_graph_nll_loss(logits, batch.y, batch.ptr)
            acc    = _per_graph_accuracy(logits, batch.y, batch.ptr)

            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


# ─── Full training run ────────────────────────────────────────────────────────

def run_full_training(
    graphs:      list,
    model_name:  str   = "gcn",
    epochs:      int   = 60,
    lr:          float = 0.001,
    weight_decay:float = 1e-4,
    batch_size:  int   = 32,
    val_frac:    float = 0.2,
    step_size:   int   = 20,
    gamma:       float = 0.5,
    verbose:     bool  = True,
) -> tuple:
    """
    Train GCN or GAT on the corner kick dataset.

    Parameters
    ----------
    graphs      : list of torch_geometric.data.Data objects
    model_name  : "gcn" or "gat"
    epochs      : number of training epochs
    lr          : Adam initial learning rate
    weight_decay: Adam L2 regularisation
    batch_size  : graphs per mini-batch
    val_frac    : fraction of data held out for validation
    step_size   : StepLR period (epochs)
    gamma       : StepLR decay factor

    Returns
    -------
    (model, history_dict)
        history_dict keys: train_loss, val_loss, train_acc, val_acc
    """
    # ── Import model ────────────────────────────────────────────────────────
    if model_name.lower() == "gcn":
        from models.gcn import GCN
        model = GCN()
    elif model_name.lower() == "gat":
        from models.gat import GAT
        model = GAT()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'gcn' or 'gat'.")

    # ── Split ───────────────────────────────────────────────────────────────
    train_graphs, val_graphs = train_test_split(
        graphs, test_size=val_frac, random_state=42
    )

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

    # ── Optimiser + scheduler ───────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # ── Baseline ────────────────────────────────────────────────────────────
    avg_nodes    = np.mean([g.num_nodes for g in graphs])
    baseline_acc = 1.0 / avg_nodes
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}  |  {model.count_parameters():,} parameters")
        print(f"{'='*60}")
        print(f"  Dataset   : {len(train_graphs)} train / {len(val_graphs)} val")
        print(f"  Baseline  : {baseline_acc*100:.1f}%  (random — 1/{avg_nodes:.0f} players)")
        print()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : [],
    }

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"  ].append(vl_loss)
        history["train_acc" ].append(tr_acc)
        history["val_acc"   ].append(vl_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Loss {tr_loss:.3f}/{vl_loss:.3f} | "
                f"Acc  {tr_acc*100:.1f}%/{vl_acc*100:.1f}%  "
                f"(train/val)"
            )

    best_val = max(history["val_acc"])
    if verbose:
        print(f"\n  Best val accuracy : {best_val*100:.1f}%")
        print(f"  Baseline          : {baseline_acc*100:.1f}%")
        improvement = best_val / baseline_acc if baseline_acc > 0 else 0
        print(f"  Improvement       : {improvement:.1f}× over random")
        print(f"{'='*60}\n")

    return model, history


# ─── Model comparison ─────────────────────────────────────────────────────────

def compare_models(
    gcn_model,
    gat_model,
    val_graphs: list,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Build a comparison table for GCN vs GAT.

    Measures:
      - Validation accuracy
      - Parameter count
      - Average inference time per graph (ms)

    Returns
    -------
    pandas.DataFrame  with columns: Model, Val Accuracy, Parameters,
                                    Avg Inference (ms)
    """
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    rows = []

    for name, model in [("GCN", gcn_model), ("GAT", gat_model)]:
        # Accuracy
        _, acc = evaluate(model, val_loader)

        # Inference time (average per graph)
        model.eval()
        times = []
        with torch.no_grad():
            for batch in val_loader:
                t0 = time.perf_counter()
                model(batch)
                t1 = time.perf_counter()
                n_graphs_in_batch = (len(batch.ptr) - 1)
                times.append((t1 - t0) * 1000 / n_graphs_in_batch)  # ms per graph

        avg_ms = np.mean(times)

        rows.append({
            "Model"             : name,
            "Val Accuracy"      : f"{acc*100:.1f}%",
            "Parameters"        : f"{model.count_parameters():,}",
            "Avg Inference (ms)": f"{avg_ms:.2f}",
        })

    df = pd.DataFrame(rows)
    return df


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.loader  import load_corners
    from graph.builder import build_dataset

    print("Loading data ...")
    corners = load_corners(use_cache=True)
    graphs  = build_dataset(corners)

    if len(graphs) < 10:
        print("Not enough graphs. Run `python -m data.loader` first.")
        sys.exit(1)

    # Quick 5-epoch smoke test
    gcn_model, gcn_history = run_full_training(graphs, model_name="gcn", epochs=5, verbose=True)
    gat_model, gat_history = run_full_training(graphs, model_name="gat", epochs=5, verbose=True)

    from sklearn.model_selection import train_test_split
    _, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)

    df = compare_models(gcn_model, gat_model, val_graphs)
    print("\nModel Comparison:")
    print(df.to_string(index=False))
