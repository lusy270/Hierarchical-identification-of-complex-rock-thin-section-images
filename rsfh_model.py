import torch
import torch.nn as nn
from .backbone import ResNetBackbone, SwinBackbone
from .fusion_module import DualAttentionFusion
from .hierarchical_head import HierarchicalHead


class RSFHModel(nn.Module):
    def __init__(self, num_fine=47, num_coarse=3, dropout_prob=0.3):
        super().__init__()
        self.resnet_backbone = ResNetBackbone()
        self.swin_backbone = SwinBackbone()
        self.fusion = DualAttentionFusion()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = HierarchicalHead(feature_dim=1024, num_coarse=num_coarse, num_fine=num_fine, dropout_prob=dropout_prob)

    def forward(self, x):
        res_feat = self.resnet_backbone(x)
        swin_feat = self.swin_backbone(x)
        fused = self.fusion(res_feat, swin_feat)
        pooled = self.pool(fused).flatten(start_dim=1)
        return self.head(pooled)