"""
train/trainer.py — Training loop v2
=====================================
v2 changes vs v1:
    + Top-3 accuracy metric (was the true shooter in the top 3 predictions?)
    + Early stopping: stop if val accuracy doesn't improve for 15 epochs
    + Best-model checkpointing: saves the weights at the best val epoch
    + Longer training: 100 epochs max (early stopping kicks in earlier usually)
    + compare_models() now reports Top-1 AND Top-3 accuracy

Why Top-3 accuracy?
    With ~17 players per graph, Top-1 baseline = 5.8%, Top-3 baseline = 17.6%.
    A model that puts the true shooter in its top 3 is tactically useful —
    it narrows the field from 17 to 3. This is the metric TacticAI reports.
"""

import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split


# ── Loss and accuracy helpers ──────────────────────────────────────────────────

def _per_graph_nll_loss(logits, y, ptr):
    """NLL loss: apply log_softmax per graph, index the true receiver."""
    loss = 0.0
    n = len(ptr) - 1
    for i in range(n):
        s, e = ptr[i].item(), ptr[i+1].item()
        lp    = F.log_softmax(logits[s:e].squeeze(-1), dim=0)
        label = min(y[i].item(), e - s - 1)
        loss += -lp[label]
    return loss / max(n, 1)


def _per_graph_topk_accuracy(logits, y, ptr, k=1):
    """
    Top-k accuracy: fraction of graphs where the true receiver
    appears in the model's top-k predictions.

    k=1  → exact match (standard accuracy)
    k=3  → true receiver is one of the 3 highest-probability players
    """
    correct = 0
    n = len(ptr) - 1
    for i in range(n):
        s, e  = ptr[i].item(), ptr[i+1].item()
        scores = logits[s:e].squeeze(-1)
        topk   = torch.topk(scores, min(k, e - s)).indices.tolist()
        label  = y[i].item()
        if label in topk:
            correct += 1
    return correct / max(n, 1)


# ── Epoch-level train/eval ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, total_acc1, total_acc3, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch)
        loss   = _per_graph_nll_loss(logits, batch.y, batch.ptr)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item()
            total_acc1 += _per_graph_topk_accuracy(logits, batch.y, batch.ptr, k=1)
            total_acc3 += _per_graph_topk_accuracy(logits, batch.y, batch.ptr, k=3)
            n += 1
    return total_loss/max(n,1), total_acc1/max(n,1), total_acc3/max(n,1)


def evaluate(model, loader):
    model.eval()
    total_loss, total_acc1, total_acc3, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            total_loss += _per_graph_nll_loss(logits, batch.y, batch.ptr).item()
            total_acc1 += _per_graph_topk_accuracy(logits, batch.y, batch.ptr, k=1)
            total_acc3 += _per_graph_topk_accuracy(logits, batch.y, batch.ptr, k=3)
            n += 1
    return total_loss/max(n,1), total_acc1/max(n,1), total_acc3/max(n,1)


# ── Full training run ──────────────────────────────────────────────────────────

def run_full_training(
    graphs,
    model_name   = "gcn",
    epochs       = 100,
    lr           = 0.001,
    weight_decay = 1e-4,
    batch_size   = 32,
    val_frac     = 0.2,
    step_size    = 25,
    gamma        = 0.5,
    patience     = 15,      # NEW: early stopping patience
    verbose      = True,
):
    """
    Train GCN or GAT with early stopping and best-model checkpointing.

    Early stopping:
        If val Top-1 accuracy doesn't improve for `patience` epochs,
        training stops and the best weights are restored.
        This prevents overfitting on the training set.

    Returns (model_with_best_weights, history_dict)
    """
    if model_name.lower() == "gcn":
        from models.gcn import GCN
        model = GCN()
    elif model_name.lower() == "gat":
        from models.gat import GAT
        model = GAT()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_graphs, val_graphs = train_test_split(graphs, test_size=val_frac, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    avg_nodes    = np.mean([g.num_nodes for g in graphs])
    baseline_1   = 1.0 / avg_nodes
    baseline_3   = min(3.0 / avg_nodes, 1.0)

    if verbose:
        print(f"\n{'='*62}")
        print(f"Training {model_name.upper()} v2  |  {model.count_parameters():,} parameters")
        print(f"{'='*62}")
        print(f"  Dataset   : {len(train_graphs)} train / {len(val_graphs)} val")
        print(f"  Baseline  : Top-1 {baseline_1*100:.1f}%  |  Top-3 {baseline_3*100:.1f}%")
        print(f"  Patience  : {patience} epochs early stopping")
        print()

    history = {"train_loss":[], "val_loss":[], "train_acc1":[], "val_acc1":[],
               "train_acc3":[], "val_acc3":[]}

    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_1, tr_3 = train_epoch(model, train_loader, optimizer)
        vl_loss, vl_1, vl_3 = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"  ].append(vl_loss)
        history["train_acc1"].append(tr_1)
        history["val_acc1"  ].append(vl_1)
        history["train_acc3"].append(tr_3)
        history["val_acc3"  ].append(vl_3)

        # Checkpoint best model
        if vl_1 > best_val_acc:
            best_val_acc = vl_1
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss {tr_loss:.3f}/{vl_loss:.3f} | "
                  f"Top1 {tr_1*100:.1f}%/{vl_1*100:.1f}% | "
                  f"Top3 {tr_3*100:.1f}%/{vl_3*100:.1f}%")

        # Early stopping
        if no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
            break

    # Restore best weights
    model.load_state_dict(best_weights)

    if verbose:
        _, final_1, final_3 = evaluate(model, val_loader)
        print(f"\n  Best val Top-1 : {best_val_acc*100:.1f}%  "
              f"(baseline {baseline_1*100:.1f}% → {best_val_acc/baseline_1:.1f}×)")
        print(f"  Best val Top-3 : {final_3*100:.1f}%  "
              f"(baseline {baseline_3*100:.1f}% → {final_3/baseline_3:.1f}×)")
        print(f"{'='*62}\n")

    return model, history


# ── Model comparison table ──────────────────────────────────────────────────────

def compare_models(gcn_model, gat_model, val_graphs, batch_size=32):
    """
    Build a comparison DataFrame including Top-1, Top-3, parameters, speed.
    """
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    rows = []

    for name, model in [("GCN", gcn_model), ("GAT", gat_model)]:
        _, acc1, acc3 = evaluate(model, val_loader)

        model.eval()
        times = []
        with torch.no_grad():
            for batch in val_loader:
                t0 = time.perf_counter()
                model(batch)
                t1 = time.perf_counter()
                n_graphs = len(batch.ptr) - 1
                times.append((t1 - t0) * 1000 / n_graphs)

        rows.append({
            "Model"             : name,
            "Top-1 Accuracy"    : f"{acc1*100:.1f}%",
            "Top-3 Accuracy"    : f"{acc3*100:.1f}%",
            "Parameters"        : f"{model.count_parameters():,}",
            "Avg Inference (ms)": f"{np.mean(times):.2f}",
        })

    return pd.DataFrame(rows)


# ── Standalone smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from data.loader   import load_corners
    from graph.builder import build_dataset

    print("Loading v2 data ...")
    corners = load_corners(use_cache=True)
    graphs  = build_dataset(corners)

    if len(graphs) < 10:
        print("Not enough graphs. Run `python -m data.loader` first.")
        sys.exit(1)

    # Full training run: 100 epochs max, early stopping at patience=15
    gcn, gcn_h = run_full_training(graphs, "gcn", epochs=100, verbose=True)
    gat, gat_h = run_full_training(graphs, "gat", epochs=100, verbose=True)

    from sklearn.model_selection import train_test_split
    _, val_g = train_test_split(graphs, test_size=0.2, random_state=42)
    df = compare_models(gcn, gat, val_g)
    print("\nv2 Model Comparison:")
    # Use tabulate-style formatting so all columns align cleanly
    col_widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}
    header = "  ".join(col.ljust(col_widths[col]) for col in df.columns)
    print("\n" + header)
    print("-" * len(header))
    for _, row in df.iterrows():
        print("  ".join(str(row[col]).ljust(col_widths[col]) for col in df.columns))