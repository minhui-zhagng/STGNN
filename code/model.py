""" STGNN network"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.nn import NNConv

class StatusGNN_withEdgeAttr(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=128):
        super(StatusGNN_withEdgeAttr, self).__init__()

        # layer1: edge_attr -> 20 * 128
        edge_nn1 = Seq(Lin(1, 64), ReLU(), Lin(64, in_channels * hidden_channels))
        self.conv1 = NNConv(in_channels, hidden_channels, edge_nn1, aggr='mean')

        # layer2: edge_attr -> 128 * 128
        edge_nn2 = Seq(Lin(1, 64), ReLU(), Lin(64, hidden_channels * hidden_channels))
        self.conv2 = NNConv(hidden_channels, hidden_channels, edge_nn2, aggr='mean')

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.decoder(x).squeeze(-1)
        return x

    def extract_features(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x.mean(dim=0).detach().cpu().numpy()
