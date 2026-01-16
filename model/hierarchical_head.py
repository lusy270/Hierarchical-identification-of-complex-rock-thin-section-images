import torch
import torch.nn as nn


class HierarchicalHead(nn.Module):
    def __init__(self, feature_dim=1024, num_coarse=3, num_fine=47, dropout_prob=0.3):
        super().__init__()
        self.fc_coarse = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_coarse)
        )
        self.fc_fine = nn.Sequential(
            nn.Linear(feature_dim + num_coarse, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_fine)
        )

    def forward(self, features):
        coarse_out = self.fc_coarse(features)
        fine_out = self.fc_fine(torch.cat([features, coarse_out], dim=1))
        return coarse_out, fine_out