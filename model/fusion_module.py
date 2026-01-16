import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        gap = nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)
        gmp = nn.functional.adaptive_max_pool2d(x, 1).view(b, c)
        attn = self.sigmoid(self.mlp(gap) + self.mlp(gmp)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class DualAttentionFusion(nn.Module):
    def __init__(self, in_channels_resnet=1024, in_channels_swin=512, out_channels=1024):
        super().__init__()
        self.swin_align = nn.Sequential(
            nn.Conv2d(in_channels_swin, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.channel_attn = ChannelAttention(out_channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, res_feat, swin_feat):
        swin_aligned = self.swin_align(swin_feat)
        fused = res_feat + swin_aligned
        fused = self.channel_attn(fused)
        return self.spatial_attn(fused)