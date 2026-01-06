import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,hidden_channels2,hidden_channels3,out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels2, heads, dropout=0.3)
        self.conv3 = GATConv(hidden_channels2*heads, hidden_channels3, heads, dropout=0.3)
        self.conv4 = GATConv(hidden_channels3 * heads, out_channels, heads=1, concat=False, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x) 
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x) 
        x = self.conv4(x, edge_index)
        return x

    def encode(self, x, edge_index):
        return self.forward(x, edge_index)
    
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()