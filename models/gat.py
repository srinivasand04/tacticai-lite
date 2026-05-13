"""
models/gat.py — Graph Attention Network (v2)
=============================================
v2 changes vs v1:
    + BatchNorm1d after layers 1 and 2 (same reason as GCN v2)
    + dropout reduced from 0.3 → 0.2 on attention
      → v1 had 228 samples; v2 has 1,500+, so we can afford less regularisation
    + add_self_loops=True explicitly set (GAT default, but made visible)

The key concept — attention vs GCN:
    GCN: h'_i = W · MEAN({ h_j : j ∈ neighbours(i) })  ← equal weight
    GAT: h'_i = W · SUM({ α_ij · h_j : j ∈ neighbours(i) }) ← learned weight

Attention coefficient α_ij:
    e_ij   = LeakyReLU( a^T · [W·h_i ‖ W·h_j] )
    α_ij   = softmax_over_j( e_ij )

In football terms: GAT learns that a near-post attacker should receive
more weight than a midfielder 9m away — without being told this explicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=64, out_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Layer 1: 5 → 64 per head, concat → 256
        self.conv1 = GATConv(in_channels, hidden_dim,
                             heads=heads, concat=True,
                             dropout=dropout, add_self_loops=True)

        # Layer 2: 256 → 64 per head, concat → 256
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim,
                             heads=heads, concat=True,
                             dropout=dropout, add_self_loops=True)

        # Layer 3: 256 → 32, single head (output layer)
        self.conv3 = GATConv(hidden_dim * heads, out_dim,
                             heads=1, concat=False,
                             dropout=dropout, add_self_loops=True)

        # BatchNorm on the concatenated multi-head output
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)  # NEW in v2
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)  # NEW in v2

        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)                # NEW in v2
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)                # NEW in v2
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention_weights:
            x, attention = self.conv3(x, edge_index, return_attention_weights=True)
        else:
            x = self.conv3(x, edge_index)

        x = F.elu(x)
        logits = self.classifier(x)

        if return_attention_weights:
            return logits, attention
        return logits

    def predict_proba(self, data):
        self.eval()
        with torch.no_grad():
            logits = self.forward(data).squeeze(-1)
            if hasattr(data, "ptr"):
                probs = torch.zeros_like(logits)
                for i in range(len(data.ptr) - 1):
                    s, e = data.ptr[i].item(), data.ptr[i+1].item()
                    probs[s:e] = torch.softmax(logits[s:e], dim=0)
            else:
                probs = torch.softmax(logits, dim=0)
        return probs

    def get_attention_weights(self, data):
        self.eval()
        with torch.no_grad():
            _, (ei, alpha) = self.forward(data, return_attention_weights=True)
        return ei, alpha

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch_geometric.data import Data
    x = torch.randn(11, 5)
    ei = torch.tensor([[0,1,1,2,3,4,5,6,7,8,9],[1,0,2,1,4,3,6,5,8,7,10]], dtype=torch.long)
    data = Data(x=x, edge_index=ei)
    model = GAT()
    print(f"GAT v2 parameters: {model.count_parameters():,}")
    probs = model.predict_proba(data)
    print(f"Probs sum: {probs.sum():.4f}  ✓")
    att_ei, att_alpha = model.get_attention_weights(data)
    print(f"Attention shape: edges {att_ei.shape}, weights {att_alpha.shape}  ✓")
    print("GAT v2 forward pass OK ✓")