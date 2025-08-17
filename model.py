import os
import re
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torch.utils.data import Dataset
class HierarchicalDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.path_list = []
        self.prefix_to_coarse = {'metamorphic': 0, 'igneous': 1, 'sedimentary': 2}
        self.fine_label_offset = {'metamorphic': 0, 'igneous': 13, 'sedimentary': 29}
        for f in os.listdir(data_path):
            if re.match(r'^(metamorphic|igneous|sedimentary)\d+', f):
                img_path = os.path.join(data_path, f)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.path_list.append(f)
                except Exception:
                    continue
    def __getitem__(self, idx):
        img_name = self.path_list[idx]
        match = re.match(r'^(metamorphic|igneous|sedimentary)(\d+)', img_name)
        prefix, number = match.groups()
        number = int(number)
        coarse_label = self.prefix_to_coarse[prefix]
        fine_label = self.fine_label_offset[prefix] + number - 1
        img_path = os.path.join(self.data_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
        except Exception:
            img = torch.zeros((3, 224, 224))
        return img, torch.tensor(coarse_label), torch.tensor(fine_label)
    def __len__(self):
        return len(self.path_list)
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(concat))
        return x * attn
class AttentionFusionHierarchicalSwinCNN(nn.Module):
    def __init__(self, num_fine=47, num_coarse=3, dropout_prob=0.3):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.res_layers = nn.Sequential(*list(resnet.children())[:-3])
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=False,
            features_only=True
        )
        state_dict = torch.load('swin_base_patch4_window7_224.pth', map_location='cpu')
        self.swin.load_state_dict(state_dict.get('model', state_dict), strict=False)
        self.swin_channel_align = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        self.channel_attn = ChannelAttention(1024)
        self.spatial_attn = SpatialAttention()
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_coarse = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_coarse)
        )
        self.fc_fine = nn.Sequential(
            nn.Linear(1024 + num_coarse, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_fine)
        )
    def forward(self, x):
        res_feat_map = self.res_layers(x)
        swin_feat_maps = self.swin(x)
        swin_feat_map = swin_feat_maps[2].permute(0, 3, 1, 2)
        swin_feat_aligned = self.swin_channel_align(swin_feat_map)
        fused_map = res_feat_map + swin_feat_aligned
        channel_attended_map = self.channel_attn(fused_map)
        final_map = self.spatial_attn(channel_attended_map)
        pooled_feat = self.final_pool(final_map).flatten(start_dim=1)
        coarse_out = self.fc_coarse(pooled_feat)
        fine_out = self.fc_fine(torch.cat([pooled_feat, coarse_out], dim=1))
        return coarse_out, fine_out
