"""
models/gcn.py — Graph Convolutional Network (GCN)
==================================================
Implements the architecture from Kipf & Welling (2017):
    "Semi-Supervised Classification with Graph Convolutional Networks"
    https://arxiv.org/abs/1609.02907

How GCN works (plain English)
------------------------------
Imagine each player on the pitch holds a "status card" — a vector of numbers
describing their position, role, and relationship to the goal.

In a GCN layer, every player does three things:
    1. Collect the status cards of all their nearby connected players.
    2. Average those cards together (equal weight to every neighbour).
    3. Multiply the average by a learned weight matrix W, then apply ReLU.

After 3 rounds of this, each player's card encodes not just their own position
but the *tactical structure of their neighbourhood*. A centre-forward's card
will have absorbed information from the near-post attacker and the corner taker.

The normalised adjacency trick (Kipf & Welling):
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    - A = raw adjacency matrix
    - I = self-loops (so each player also keeps their own info)
    - D = diagonal degree matrix
    - The D^{-1/2} terms prevent feature explosion in high-degree nodes
    PyTorch Geometric's GCNConv handles this automatically.

Architecture
------------
    Input:  node features [N, 5]
    Layer 1: GCNConv(5  → 64)  + ReLU + Dropout(0.3)
    Layer 2: GCNConv(64 → 64)  + ReLU + Dropout(0.3)
    Layer 3: GCNConv(64 → 32)  + ReLU
    Output:  Linear(32 → 1) → score per node
             Softmax over all nodes → receiver probability distribution

The per-graph softmax is key: it forces the model to assign probabilities that
sum to 1 over the players in that single corner kick graph. We use the `ptr`
index from a DataLoader batch to split graphs correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Three-layer Graph Convolutional Network for corner kick receiver prediction.

    Parameters
    ----------
    in_channels  : int  — number of input node features (default 5)
    hidden_dim   : int  — hidden layer width (default 64)
    out_dim      : int  — pre-output dimension (default 32)
    dropout      : float — dropout rate applied after first two GCNConv layers
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim:  int = 64,
        out_dim:     int = 32,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Three GCNConv layers ──────────────────────────────────────────────
        # GCNConv(in, out) applies: x_i' = Σ_j (1/sqrt(d_i * d_j)) * W * x_j
        # where the sum is over neighbours j of node i (including i itself).
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim,  hidden_dim)
        self.conv3 = GCNConv(hidden_dim,  out_dim)

        # ── Final linear classifier (one score per node) ──────────────────────
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data or DataBatch
            Expected fields: x, edge_index

        Returns
        -------
        logits : Tensor [N, 1]   — one raw score per node
        """
        x, edge_index = data.x, data.edge_index

        # Layer 1: message passing → ReLU → dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: another round of neighbourhood aggregation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3: compress to 32-dimensional representation
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Linear head → scalar score per node
        logits = self.classifier(x)   # [N, 1]
        return logits

    def predict_proba(self, data) -> torch.Tensor:
        """
        Return softmax receiver probabilities per graph.

        For a single graph (not batched), this is simply softmax over all nodes.
        For a batch, we split by graph using data.ptr and apply softmax per-graph.

        Returns
        -------
        probs : Tensor [N]  — probability that each player is the receiver
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data).squeeze(-1)   # [N]

            if hasattr(data, "ptr"):
                # Batched: apply softmax per individual graph
                probs = torch.zeros_like(logits)
                for i in range(len(data.ptr) - 1):
                    start, end = data.ptr[i].item(), data.ptr[i + 1].item()
                    probs[start:end] = torch.softmax(logits[start:end], dim=0)
            else:
                probs = torch.softmax(logits, dim=0)

        return probs

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data

    # Synthetic graph: 11 players, 5 features each, some edges
    N = 11
    x = torch.randn(N, 5)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 10]],
        dtype=torch.long
    )
    data = Data(x=x, edge_index=edge_index)

    model = GCN(in_channels=5, hidden_dim=64, out_dim=32, dropout=0.3)
    print(f"GCN parameters : {model.count_parameters():,}")

    logits = model(data)
    probs  = model.predict_proba(data)

    print(f"Input  x shape : {x.shape}")
    print(f"Logits shape   : {logits.shape}")
    print(f"Probs shape    : {probs.shape}")
    print(f"Probs sum      : {probs.sum():.4f}  (should be ≈ 1.0)")
    print(f"Predicted node : {probs.argmax().item()}")
    print("GCN forward pass OK ✓")
