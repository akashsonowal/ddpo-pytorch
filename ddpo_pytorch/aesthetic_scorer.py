import torch
from torch import nn
import clip

class MLP(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    
    def __repr__(self):
        return """xcol is emb and ycol is avg_rating."""
    
    def forward(self, x):
        return self.layers(x)