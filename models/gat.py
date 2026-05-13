"""
models/gat.py — Graph Attention Network (GAT)
==============================================
Implements the architecture from Veličković et al. (2018):
    "Graph Attention Networks"
    https://arxiv.org/abs/1710.10903

The key difference from GCN — plain English
--------------------------------------------
GCN gives EQUAL weight to all neighbours when aggregating:
    x_i' = mean({ x_j : j ∈ N(i) })    (plus normalisation & learned W)

GAT learns DIFFERENT weights for each neighbour, via an attention mechanism:
    x_i' = Σ_{j ∈ N(i)}  α_{ij} · W · x_j

where α_{ij} is the attention coefficient for the edge (i → j).

How attention coefficients are computed:
    e_{ij}  = LeakyReLU( aᵀ · [ W·h_i ‖ W·h_j ] )
    α_{ij}  = softmax_j( e_{ij} )
             = exp(e_{ij}) / Σ_{k ∈ N(i)} exp(e_{ik})

Breaking it down:
    - W   : shared linear transformation (same for all nodes)
    - h_i : current feature vector of node i
    - ‖   : concatenation of the two transformed features
    - aᵀ  : learned attention vector (a single dense layer)
    - LeakyReLU + softmax normalises so weights sum to 1 over neighbours

In football terms:
    A GAT layer learns that "the near-post attacker matters more than the
    goalkeeper when predicting who will receive this corner."
    The model discovers this just from training data — no hand-coding needed.

Multi-head attention:
    We use 4 attention heads per layer. Each head learns a different
    "view" of who matters. Their outputs are concatenated (concat=True)
    giving richer representations. The final layer uses 1 head for clean output.

Architecture
------------
    Input:  [N, 5]
    Layer 1: GATConv(5   → 64,  heads=4, concat=True)  → [N, 256] + ELU + Dropout
    Layer 2: GATConv(256 → 64,  heads=4, concat=True)  → [N, 256] + ELU + Dropout
    Layer 3: GATConv(256 → 32,  heads=1, concat=False) → [N,  32] + ELU
    Output:  Linear(32 → 1) → score per node → per-graph softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Three-layer Graph Attention Network for corner kick receiver prediction.

    Parameters
    ----------
    in_channels  : int   — input node features (default 5)
    hidden_dim   : int   — per-head hidden dimension (default 64)
    out_dim      : int   — pre-output dimension (default 32)
    heads        : int   — number of attention heads in first two layers (default 4)
    dropout      : float — dropout on node features and attention weights (default 0.3)
    """

    def __init__(
        self,
        in_channels: int   = 5,
        hidden_dim:  int   = 64,
        out_dim:     int   = 32,
        heads:       int   = 4,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # ── Layer 1 ────────────────────────────────────────────────────────────
        # 5 → 64 per head, concat=True → output is 64 × 4 = 256
        self.conv1 = GATConv(
            in_channels, hidden_dim,
            heads=heads, concat=True,
            dropout=dropout,
        )

        # ── Layer 2 ────────────────────────────────────────────────────────────
        # 256 → 64 per head, concat=True → output is 256
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim,
            heads=heads, concat=True,
            dropout=dropout,
        )

        # ── Layer 3 ────────────────────────────────────────────────────────────
        # 256 → 32, single head → output is 32
        self.conv3 = GATConv(
            hidden_dim * heads, out_dim,
            heads=1, concat=False,
            dropout=dropout,
        )

        # ── Final scorer ───────────────────────────────────────────────────────
        self.classifier = nn.Linear(out_dim, 1)

    def forward(
        self,
        data,
        return_attention_weights: bool = False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data or DataBatch
        return_attention_weights : bool
            If True, also return the attention (edge_index, attention_weights)
            tuple from the LAST GATConv layer. Useful for visualization.

        Returns
        -------
        logits : Tensor [N, 1]
        (optionally) attention : tuple (edge_index, alpha)
        """
        x, edge_index = data.x, data.edge_index

        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 — optionally capture attention weights
        if return_attention_weights:
            x, attention = self.conv3(
                x, edge_index,
                return_attention_weights=True,
            )
        else:
            x = self.conv3(x, edge_index)

        x = F.elu(x)

        logits = self.classifier(x)   # [N, 1]

        if return_attention_weights:
            return logits, attention
        return logits

    def predict_proba(self, data) -> torch.Tensor:
        """
        Return per-graph softmax receiver probabilities.
        Identical interface to GCN.predict_proba for interchangeable use.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data).squeeze(-1)

            if hasattr(data, "ptr"):
                probs = torch.zeros_like(logits)
                for i in range(len(data.ptr) - 1):
                    start, end = data.ptr[i].item(), data.ptr[i + 1].item()
                    probs[start:end] = torch.softmax(logits[start:end], dim=0)
            else:
                probs = torch.softmax(logits, dim=0)

        return probs

    def get_attention_weights(self, data):
        """
        Convenience method: return (edge_index, alpha) from the final layer.
        Used in viz/pitch_viz.py for the attention heatmap.

        Returns
        -------
        edge_index : LongTensor [2, E]
        alpha      : FloatTensor [E, 1]   (attention weights, summing to 1 per node)
        """
        self.eval()
        with torch.no_grad():
            logits, (att_edge_index, att_alpha) = self.forward(
                data, return_attention_weights=True
            )
        return att_edge_index, att_alpha

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch_geometric.data import Data

    N = 11
    x = torch.randn(N, 5)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
         [1, 0, 2, 1, 4, 3, 6, 5, 8, 7, 10]],
        dtype=torch.long
    )
    data = Data(x=x, edge_index=edge_index)

    model = GAT(in_channels=5, hidden_dim=64, out_dim=32, heads=4, dropout=0.3)
    print(f"GAT parameters  : {model.count_parameters():,}")

    logits = model(data)
    probs  = model.predict_proba(data)

    print(f"Input  x shape  : {x.shape}")
    print(f"Logits shape    : {logits.shape}")
    print(f"Probs shape     : {probs.shape}")
    print(f"Probs sum       : {probs.sum():.4f}  (should be ≈ 1.0)")
    print(f"Predicted node  : {probs.argmax().item()}")

    # Attention weights
    att_ei, att_alpha = model.get_attention_weights(data)
    print(f"Attention edges : {att_ei.shape}  weights: {att_alpha.shape}")
    print("GAT forward pass OK ✓")
