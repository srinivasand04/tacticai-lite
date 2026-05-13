"""
whatif/simulator.py — What-If tactical perturbation engine
============================================================
Answers questions like:
    "What happens if we push the near-post attacker 5 m closer to goal?"
    "What if we pull the central midfielder back 8 m?"
    "What if we move a defender out of position?"

The simulator:
    1. Deep-copies the graph
    2. Moves a player by (dx_meters, dy_meters)
    3. Re-builds edges based on new positions
    4. Runs both original and modified graphs through the model
    5. Prints a formatted probability delta table
    6. Optionally saves side-by-side pitch plots

This is the most interview-friendly part of the project — it shows that
the model has learned real tactical structure, not just memorised positions.

StatsBomb coordinate system:
    x: 0 (own goal line) → 120 (opponent goal line)
    y: 0 (left touchline) → 80 (right touchline)
    1 unit ≈ 1 metre (StatsBomb uses metric coordinates)
"""

from __future__ import annotations
import copy
import math

import torch
import numpy as np
from torch_geometric.data import Data


# ─── Constants (must match builder.py) ───────────────────────────────────────
PITCH_LENGTH  = 120.0
PITCH_WIDTH   = 80.0
GOAL_X        = 120.0
GOAL_Y        = 40.0
MAX_DIST_GOAL = math.sqrt(PITCH_LENGTH**2 + PITCH_WIDTH**2)
DEFAULT_THRESHOLD = 10.0     # edge proximity threshold in metres


# ─── Core simulation ─────────────────────────────────────────────────────────

def what_if_simulation(
    model,
    graph_data: Data,
    player_idx:  int,
    dx_meters:   float = 0.0,
    dy_meters:   float = 0.0,
    description: str   = "",
    plot:        bool  = False,
    threshold:   float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Perturb a player's position and measure how predictions change.

    Parameters
    ----------
    model        : trained GCN or GAT model
    graph_data   : a single-graph Data object (not a batch)
    player_idx   : index of the player to move (0-indexed in the node list)
    dx_meters    : horizontal displacement in metres (+ = toward opponent goal)
    dy_meters    : vertical displacement in metres   (+ = toward right touchline)
    description  : text label for the printed table
    plot         : if True, save before/after pitch images
    threshold    : proximity threshold for edge re-building (metres)

    Returns
    -------
    dict with keys:
        original_probs  : np.ndarray [N] — original receiver probabilities
        modified_probs  : np.ndarray [N] — after perturbation
        delta           : np.ndarray [N] — probability change (pp)
        original_graph  : Data           — original graph
        modified_graph  : Data           — perturbed graph
        player_idx      : int
        original_pos    : [x, y]         — original StatsBomb coords
        new_pos         : [x, y]         — new StatsBomb coords
    """
    N = graph_data.num_nodes

    if player_idx >= N:
        raise ValueError(f"player_idx={player_idx} out of range (graph has {N} nodes).")

    # ── 1. Deep-copy the graph ────────────────────────────────────────────────
    modified = _deep_copy_graph(graph_data)

    # ── 2. Get original position (from .locations or de-normalise x) ─────────
    if hasattr(graph_data, "locations") and graph_data.locations is not None:
        orig_x = float(graph_data.locations[player_idx, 0].item())
        orig_y = float(graph_data.locations[player_idx, 1].item())
    else:
        orig_x = float(graph_data.x[player_idx, 0].item()) * PITCH_LENGTH
        orig_y = float(graph_data.x[player_idx, 1].item()) * PITCH_WIDTH

    # ── 3. Compute new position and clamp to pitch bounds ────────────────────
    new_x = float(np.clip(orig_x + dx_meters, 0.0, PITCH_LENGTH))
    new_y = float(np.clip(orig_y + dy_meters, 0.0, PITCH_WIDTH))

    # ── 4. Update node features in modified graph ─────────────────────────────
    with torch.no_grad():
        modified.x[player_idx, 0] = new_x / PITCH_LENGTH
        modified.x[player_idx, 1] = new_y / PITCH_WIDTH
        dist_goal = math.sqrt((new_x - GOAL_X)**2 + (new_y - GOAL_Y)**2) / MAX_DIST_GOAL
        modified.x[player_idx, 4] = dist_goal

    # Update .locations if present
    if hasattr(modified, "locations") and modified.locations is not None:
        modified.locations[player_idx, 0] = new_x
        modified.locations[player_idx, 1] = new_y

    # ── 5. Re-build edges based on new positions ──────────────────────────────
    modified.edge_index, modified.edge_attr = _rebuild_edges(modified, threshold)

    # ── 6. Run inference on both graphs ──────────────────────────────────────
    orig_probs = _get_probs(model, graph_data)      # [N]
    mod_probs  = _get_probs(model, modified)         # [N]

    delta = (mod_probs - orig_probs) * 100.0         # percentage points

    # ── 7. Print formatted report ─────────────────────────────────────────────
    _print_report(
        player_idx   = player_idx,
        orig_x=orig_x, orig_y=orig_y,
        new_x=new_x,   new_y=new_y,
        dx=dx_meters,  dy=dy_meters,
        orig_probs   = orig_probs,
        mod_probs    = mod_probs,
        delta        = delta,
        description  = description,
    )

    # ── 8. Optional: save pitch plots side by side ───────────────────────────
    if plot:
        try:
            from viz.pitch_viz import plot_corner_graph
            plot_corner_graph(
                graph_data,
                title=f"Original — Player {player_idx}",
                prediction=int(np.argmax(orig_probs)),
                true_receiver=graph_data.y.item() if hasattr(graph_data, "y") else None,
                save_name=f"whatif_original_p{player_idx}.png",
            )
            plot_corner_graph(
                modified,
                title=f"After Move — Player {player_idx} (+{dx_meters}m, +{dy_meters}m)",
                prediction=int(np.argmax(mod_probs)),
                true_receiver=graph_data.y.item() if hasattr(graph_data, "y") else None,
                save_name=f"whatif_modified_p{player_idx}.png",
            )
        except Exception as e:
            print(f"[simulator] Plot skipped: {e}")

    return {
        "original_probs" : orig_probs,
        "modified_probs" : mod_probs,
        "delta"          : delta,
        "original_graph" : graph_data,
        "modified_graph" : modified,
        "player_idx"     : player_idx,
        "original_pos"   : [orig_x, orig_y],
        "new_pos"        : [new_x, new_y],
    }


def batch_what_if(model, graph_data: Data, scenarios: list[dict]) -> list[dict]:
    """
    Run multiple What-If scenarios on the same graph.

    Parameters
    ----------
    scenarios : list of dicts, each with keys:
                    player_idx, dx_meters, dy_meters, description

    Returns
    -------
    List of result dicts from what_if_simulation
    """
    results = []
    for s in scenarios:
        result = what_if_simulation(
            model      = model,
            graph_data = graph_data,
            player_idx = s.get("player_idx", 0),
            dx_meters  = s.get("dx_meters", 0.0),
            dy_meters  = s.get("dy_meters", 0.0),
            description= s.get("description", ""),
            plot       = s.get("plot", False),
            threshold  = s.get("threshold", DEFAULT_THRESHOLD),
        )
        results.append(result)
    return results


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _deep_copy_graph(data: Data) -> Data:
    """Deep copy a Data object, handling all tensor fields."""
    new_data = Data()
    for key in data.keys():
        val = data[key]
        if isinstance(val, torch.Tensor):
            new_data[key] = val.clone()
        else:
            new_data[key] = copy.deepcopy(val)
    return new_data


def _rebuild_edges(data: Data, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Re-compute edges after a player position change.
    Mirrors the logic in graph/builder.py but operates on the tensor directly.
    """
    N    = data.num_nodes
    # De-normalise positions
    locs = data.x[:, :2].numpy()
    locs[:, 0] *= PITCH_LENGTH
    locs[:, 1] *= PITCH_WIDTH

    dist_mat = np.zeros((n := N, n))
    for i in range(N):
        for j in range(N):
            dx = locs[i, 0] - locs[j, 0]
            dy = locs[i, 1] - locs[j, 1]
            dist_mat[i, j] = math.sqrt(dx * dx + dy * dy)

    max_dist = dist_mat.max() if dist_mat.max() > 0 else 1.0

    src_list, dst_list, attr_list = [], [], []
    adj = {i: set() for i in range(N)}

    for i in range(N):
        for j in range(i + 1, N):
            if dist_mat[i, j] < threshold:
                adj[i].add(j)
                adj[j].add(i)
                d_norm = dist_mat[i, j] / max_dist
                src_list += [i, j]
                dst_list += [j, i]
                attr_list += [d_norm, d_norm]

    # Fallback for isolated nodes
    for i in range(N):
        if len(adj[i]) == 0:
            sorted_j = sorted(range(N), key=lambda j: dist_mat[i, j])
            for j in sorted_j[1:4]:
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    d_norm = dist_mat[i, j] / max_dist
                    src_list += [i, j]
                    dst_list += [j, i]
                    attr_list += [d_norm, d_norm]

    if len(src_list) == 0:
        src_list = [0, 1]; dst_list = [1, 0]; attr_list = [0.0, 0.0]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor([[a] for a in attr_list], dtype=torch.float)
    return edge_index, edge_attr


def _get_probs(model, data: Data) -> np.ndarray:
    """Run the model on a single graph and return numpy probability array."""
    model.eval()
    with torch.no_grad():
        logits = model(data).squeeze(-1)   # [N]
        probs  = torch.softmax(logits, dim=0)
    return probs.numpy()


def _print_report(
    player_idx, orig_x, orig_y, new_x, new_y,
    dx, dy, orig_probs, mod_probs, delta, description
) -> None:
    """Print a formatted What-If comparison table."""
    N       = len(orig_probs)
    TOP_K   = min(5, N)

    # Top-k by original probability
    top_orig = np.argsort(-orig_probs)[:TOP_K]

    print("\n" + "═" * 54)
    desc_line = description if description else f"Move Player {player_idx}"
    print(f"  What-If: {desc_line}")
    print(f"  Player {player_idx}: ({orig_x:.1f}, {orig_y:.1f}) → ({new_x:.1f}, {new_y:.1f})")
    print(f"  Δ = ({dx:+.1f} m,  {dy:+.1f} m)")
    print("─" * 54)
    print(f"  {'Player':<10} {'Original':>10} {'After':>10} {'Δ (pp)':>10}")
    print("─" * 54)

    for idx in top_orig:
        arrow = "▲" if delta[idx] > 0.5 else ("▼" if delta[idx] < -0.5 else " ")
        print(
            f"  Player {idx:<4}  "
            f"{orig_probs[idx]*100:>8.1f}%  "
            f"{mod_probs[idx]*100:>8.1f}%  "
            f"{arrow}{abs(delta[idx]):>7.1f}pp"
        )

    # Tactical insight
    target_delta = delta[player_idx]
    print("─" * 54)
    if abs(target_delta) > 1.0:
        direction = "increases" if target_delta > 0 else "decreases"
        print(f"\n  Tactical insight:")
        print(f"  Moving Player {player_idx} changes their predicted receiver")
        print(f"  probability by {target_delta:+.1f} pp ({direction}).")
    else:
        print(f"\n  Tactical insight: Minimal impact from this move.")

    print("═" * 54 + "\n")


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from data.loader   import load_corners
    from graph.builder import build_dataset
    from models.gcn    import GCN

    corners = load_corners(use_cache=True)
    graphs  = build_dataset(corners)

    if not graphs:
        print("No graphs — run `python -m data.loader` first.")
        sys.exit(1)

    # Use a simple untrained model for the smoke test
    model = GCN()

    g = graphs[0]

    # Find an attacker to perturb
    attackers = [i for i in range(g.num_nodes)
                 if g.x[i, 2].item() == 1.0]
    player = attackers[0] if attackers else 0

    # Run three scenarios
    batch_what_if(
        model, g,
        scenarios=[
            {
                "player_idx" : player,
                "dx_meters"  : 5.0,
                "dy_meters"  : 0.0,
                "description": f"Push Player {player} (attacker) 5m toward goal",
            },
            {
                "player_idx" : player,
                "dx_meters"  : -5.0,
                "dy_meters"  : 0.0,
                "description": f"Pull Player {player} (attacker) 5m away from goal",
            },
            {
                "player_idx" : player,
                "dx_meters"  : 0.0,
                "dy_meters"  : 5.0,
                "description": f"Move Player {player} 5m toward near post",
            },
        ]
    )
