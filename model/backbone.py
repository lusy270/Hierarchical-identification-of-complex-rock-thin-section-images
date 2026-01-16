import torch
import torch.nn as nn
import timm
import torchvision.models as models


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.layers = nn.Sequential(*list(resnet.children())[:-3])

    def forward(self, x):
        return self.layers(x)


class SwinBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False, features_only=True)
        state_dict = torch.load('swin_base_patch4_window7_224.pth', map_location='cpu')
        self.swin.load_state_dict(state_dict.get('model', state_dict), strict=False)

    def forward(self, x):
        features = self.swin(x)
        return features[2].permute(0, 3, 1, 2)