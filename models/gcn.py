"""
models/gcn.py — Graph Convolutional Network (v2)
=================================================
v2 changes vs v1:
    + BatchNorm1d after layers 1 and 2
      → normalises node embeddings across the batch
      → stabilises training on variable-size graphs (9–20 nodes)
      → allows faster convergence and slightly higher LR

Architecture:
    GCNConv(5→64)  + BatchNorm + ReLU + Dropout(0.3)
    GCNConv(64→64) + BatchNorm + ReLU + Dropout(0.3)
    GCNConv(64→32) + ReLU
    Linear(32→1)   → softmax per graph

How GCN works (plain English):
    Each player averages the feature vectors of their connected neighbours,
    multiplies by a learned weight matrix W, and updates their own embedding.
    After 3 rounds, each player's vector encodes their entire local
    tactical neighbourhood — not just their own position.

Normalised adjacency (Kipf & Welling):
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    Self-loops (I) keep each player's own info.
    D^{-1/2} prevents feature explosion in crowded areas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels=5, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)

        # BatchNorm normalises embeddings across all nodes in the batch.
        # This is especially helpful when graphs vary in size (9–20 nodes here).
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Layer 1: aggregate neighbours → normalise → activate → regularise
        x = self.conv1(x, edge_index)
        x = self.bn1(x)               # NEW in v2
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: second round of message passing
        x = self.conv2(x, edge_index)
        x = self.bn2(x)               # NEW in v2
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3: compress to 32d tactical representation
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Score per player — softmax applied later in trainer
        return self.classifier(x)     # [N, 1]

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch_geometric.data import Data
    x = torch.randn(11, 5)
    ei = torch.tensor([[0,1,1,2,3,4,5,6,7,8,9],[1,0,2,1,4,3,6,5,8,7,10]], dtype=torch.long)
    data = Data(x=x, edge_index=ei)
    model = GCN()
    print(f"GCN v2 parameters: {model.count_parameters():,}")
    probs = model.predict_proba(data)
    print(f"Probs sum: {probs.sum():.4f}  ✓")
    print("GCN v2 forward pass OK ✓")