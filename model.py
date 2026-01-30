import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, emb_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(emb_dim, out_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

