"""
graph/builder.py — Convert corner kick freeze frames into PyG graphs
=====================================================================
Each corner kick becomes a `torch_geometric.data.Data` object:

    Data(
        x          = node feature matrix  [N, 5]   (N = players in frame)
        edge_index = adjacency in COO format [2, E]
        edge_attr  = edge features          [E, 1]  (normalized distance)
        y          = receiver node index    scalar
    )

Node features (5D)
------------------
    [0] x_norm      — x position / 120  → [0, 1]   (pitch length = 120 m)
    [1] y_norm      — y position / 80   → [0, 1]   (pitch width  = 80 m)
    [2] is_attacker — 1.0 if teammate of corner taker, else 0.0
    [3] is_gk       — 1.0 if goalkeeper, else 0.0
    [4] dist_goal   — normalized Euclidean distance to attacking goal (120, 40)

Why these features?
    x_norm / y_norm : spatial position is the primary signal for set-pieces
    is_attacker     : the model must learn that attackers are more likely
                      receivers — this gives it a strong prior
    is_gk           : goalkeeper almost never receives a corner — a negative signal
    dist_goal       : proximity to goal is a key tactical driver for corners;
                      players near goal are often targeted

Edge construction
-----------------
    Two players are connected if their Euclidean distance < threshold metres.
    We use real StatsBomb coordinates (1 unit ≈ 1 metre).
    If a node has no edges (isolated), we connect it to its 3 nearest neighbours
    so the graph is always connected enough for message passing.

Label
-----
    The player whose position is closest to `pass_end_location` is treated as
    the receiver. This is a proxy — StatsBomb gives the landing spot, not the
    exact receiver ID in freeze frames — but it is accurate enough for learning.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

# ─── Constants ────────────────────────────────────────────────────────────────
PITCH_LENGTH = 120.0    # StatsBomb pitch x-axis extent
PITCH_WIDTH  = 80.0     # StatsBomb pitch y-axis extent
GOAL_X       = 120.0    # attacking goal x coordinate
GOAL_Y       = 40.0     # attacking goal centre y coordinate
MAX_DIST_GOAL = math.sqrt(PITCH_LENGTH**2 + PITCH_WIDTH**2)  # diagonal ≈ 144 m

GK_POSITIONS = {"Goalkeeper"}  # StatsBomb position names for GK


# ─── Node feature extraction ──────────────────────────────────────────────────

def _node_features(players: list[dict]) -> torch.Tensor:
    """
    Build the [N, 5] node feature matrix from a list of freeze-frame player dicts.

    Each player dict (from StatsBomb) looks like:
        {
          "location"  : [x, y],
          "teammate"  : True | False,
          "position"  : {"name": "Centre Forward"},
          "actor"     : True | False   # True = the corner taker themselves
        }
    """
    rows = []
    for p in players:
        loc = p.get("location", [60.0, 40.0])  # fallback to pitch centre
        x, y = float(loc[0]), float(loc[1])

        x_norm      = x / PITCH_LENGTH
        y_norm      = y / PITCH_WIDTH
        is_attacker = 1.0 if p.get("teammate", False) else 0.0
        pos_name    = p.get("position", {}).get("name", "") or ""
        is_gk       = 1.0 if pos_name in GK_POSITIONS else 0.0
        dist_goal   = math.sqrt((x - GOAL_X)**2 + (y - GOAL_Y)**2) / MAX_DIST_GOAL

        rows.append([x_norm, y_norm, is_attacker, is_gk, dist_goal])

    return torch.tensor(rows, dtype=torch.float)


# ─── Edge construction ────────────────────────────────────────────────────────

def _build_edges(
    players: list[dict],
    threshold: float = 10.0,
    k_fallback: int  = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index and edge_attr using proximity threshold.

    Parameters
    ----------
    players   : list of freeze-frame player dicts
    threshold : connect two players if distance < threshold metres
    k_fallback: if a node is isolated, connect it to its k nearest neighbours

    Returns
    -------
    edge_index : LongTensor [2, E]  — (source, target) pairs (undirected → 2×)
    edge_attr  : FloatTensor [E, 1] — normalized distance for each edge
    """
    n = len(players)
    locs = []
    for p in players:
        loc = p.get("location", [60.0, 40.0])
        locs.append((float(loc[0]), float(loc[1])))

    # Compute pairwise distances
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = locs[i][0] - locs[j][0]
            dy = locs[i][1] - locs[j][1]
            dist_mat[i, j] = math.sqrt(dx * dx + dy * dy)

    max_dist = dist_mat.max() if dist_mat.max() > 0 else 1.0

    src_list, dst_list, attr_list = [], [], []

    # Primary: connect if within threshold
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if dist_mat[i, j] < threshold:
                adj[i].add(j)
                adj[j].add(i)
                d_norm = dist_mat[i, j] / max_dist
                # Undirected: add both directions
                src_list += [i, j]
                dst_list += [j, i]
                attr_list += [d_norm, d_norm]

    # Fallback: isolated nodes → k nearest neighbours
    for i in range(n):
        if len(adj[i]) == 0:
            # Sort all others by distance, take k closest
            sorted_j = sorted(range(n), key=lambda j: dist_mat[i, j])
            for j in sorted_j[1: k_fallback + 1]:  # skip self (index 0)
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    d_norm = dist_mat[i, j] / max_dist
                    src_list += [i, j]
                    dst_list += [j, i]
                    attr_list += [d_norm, d_norm]

    if len(src_list) == 0:
        # Absolute fallback: fully connect first 2 players
        src_list = [0, 1]
        dst_list = [1, 0]
        attr_list = [0.0, 0.0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor([[a] for a in attr_list], dtype=torch.float)

    return edge_index, edge_attr


# ─── Label extraction ─────────────────────────────────────────────────────────

def _get_label(players: list[dict], pass_end_location: list) -> int:
    """
    Find which player is closest to pass_end_location.
    That player's index in the list becomes the receiver label.

    We intentionally label from all players (not just attackers) so the model
    must learn by itself who is likely to be targeted.
    """
    ex, ey = float(pass_end_location[0]), float(pass_end_location[1])
    best_idx  = 0
    best_dist = float("inf")

    for i, p in enumerate(players):
        loc = p.get("location", [60.0, 40.0])
        dx  = float(loc[0]) - ex
        dy  = float(loc[1]) - ey
        d   = math.sqrt(dx * dx + dy * dy)
        if d < best_dist:
            best_dist = d
            best_idx  = i

    return best_idx


# ─── Main builder functions ───────────────────────────────────────────────────

def build_graph(corner: dict, threshold: float = 10.0) -> Optional[Data]:
    """
    Convert one corner kick dict (from data/loader.py) into a PyG Data object.

    Parameters
    ----------
    corner    : dict with keys location, pass_end_location, freeze_frame, team
    threshold : proximity threshold in metres for connecting players

    Returns
    -------
    torch_geometric.data.Data | None
        Returns None if the freeze frame is too small to be useful.
    """
    ff = corner.get("freeze_frame", [])
    if len(ff) < 4:
        return None

    end_loc = corner.get("pass_end_location", [GOAL_X, GOAL_Y])

    x          = _node_features(ff)
    edge_index, edge_attr = _build_edges(ff, threshold=threshold)
    y          = torch.tensor(_get_label(ff, end_loc), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    # Store raw locations for visualization later
    data.locations = torch.tensor(
        [[float(p.get("location", [0, 0])[0]),
          float(p.get("location", [0, 0])[1])] for p in ff],
        dtype=torch.float
    )
    data.is_attacker = x[:, 2]   # slice for convenience
    data.num_nodes   = len(ff)

    return data


def build_dataset(corners: list[dict], threshold: float = 10.0) -> list[Data]:
    """
    Convert the full list of corner dicts into a list of PyG Data objects.

    Skips any corner that produces None from build_graph.
    """
    graphs = []
    skipped = 0
    for c in corners:
        g = build_graph(c, threshold=threshold)
        if g is not None:
            graphs.append(g)
        else:
            skipped += 1

    print(f"[builder] Built {len(graphs)} graphs ({skipped} skipped — too few players).")
    return graphs


# ─── Dataset statistics ───────────────────────────────────────────────────────

def visualize_graph_stats(dataset: list[Data]) -> None:
    """
    Print basic statistics about the dataset.
    Helps verify that graphs are reasonable before training.
    """
    if not dataset:
        print("[builder] Empty dataset — nothing to show.")
        return

    node_counts = [g.num_nodes for g in dataset]
    edge_counts = [g.edge_index.shape[1] // 2 for g in dataset]  # undirected
    labels      = [g.y.item() for g in dataset]

    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"  Total graphs      : {len(dataset)}")
    print(f"  Avg nodes/graph   : {np.mean(node_counts):.1f}  "
          f"(min {min(node_counts)}, max {max(node_counts)})")
    print(f"  Avg edges/graph   : {np.mean(edge_counts):.1f}  "
          f"(min {min(edge_counts)}, max {max(edge_counts)})")
    print(f"  Label range       : {min(labels)} – {max(labels)}")

    # Class balance: how many unique receiver positions?
    # Since labels = node indices, we want to know if any node dominates
    from collections import Counter
    label_counts = Counter(labels)
    n_classes    = len(label_counts)
    print(f"  Unique receiver idx: {n_classes}  (ideal: spread across all players)")

    # Fraction of receiver that is an attacker vs defender
    attacker_hits = sum(
        1 for g in dataset
        if g.is_attacker[g.y.item()].item() == 1.0
    )
    print(f"  Receiver is attacker: {attacker_hits}/{len(dataset)} "
          f"({100*attacker_hits/len(dataset):.1f}%)")
    print("=" * 50)


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.loader import load_corners

    print("Loading corners from cache (or StatsBomb) ...")
    corners = load_corners(use_cache=True)

    if not corners:
        print("No corners found. Run `python -m data.loader` first.")
        sys.exit(1)

    print(f"\nBuilding graphs from {len(corners)} corners ...")
    graphs = build_dataset(corners, threshold=10.0)

    # Show the first graph in detail
    g = graphs[0]
    print(f"\nFirst graph:")
    print(f"  node features x : {g.x.shape}  (N players × 5 features)")
    print(f"  edge_index      : {g.edge_index.shape}")
    print(f"  edge_attr       : {g.edge_attr.shape}")
    print(f"  label y         : {g.y.item()}  (receiver = player #{g.y.item()})")
    print(f"  receiver pos    : {g.locations[g.y.item()].tolist()}")

    visualize_graph_stats(graphs)
