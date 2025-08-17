import torch
import torch.nn as nn
import timm
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import re
import matplotlib.pyplot as plt
import numpy as np

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
                except Exception as e:
                    file_size = os.path.getsize(img_path)
                    print(f"Skipping corrupted image: {f} | Size: {file_size} bytes | Error: {repr(e)}")
        if not self.path_list:
            raise ValueError(f"No valid images found in {data_path}")
        print(f"Initialized dataset from {data_path} with {len(self.path_list)} images.")
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
        except:
            img = torch.zeros((3, 224, 224))
        return img, torch.tensor(coarse_label), torch.tensor(fine_label)
    def __len__(self):
        return len(self.path_list)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
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
        super(SpatialAttention, self).__init__()
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
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
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
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, preds, labels, c_preds, c_labels = 0, [], [], [], []
    with torch.no_grad():
        for x, cy, fy in dataloader:
            x, cy, fy = x.to(device), cy.to(device), fy.to(device)
            c_out, f_out = model(x)
            loss = loss_fn(c_out, cy) + loss_fn(f_out, fy)
            total_loss += loss.item()
            preds.append(f_out.argmax(1).cpu())
            labels.append(fy.cpu())
            c_preds.append(c_out.argmax(1).cpu())
            c_labels.append(cy.cpu())
    pred = torch.cat(preds)
    true = torch.cat(labels)
    c_pred = torch.cat(c_preds)
    c_true = torch.cat(c_labels)
    coarse_conf_matrix = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(c_true.numpy(), c_pred.numpy()):
        coarse_conf_matrix[t, p] += 1
    return (
        total_loss / len(dataloader),
        accuracy_score(true, pred),
        precision_score(true, pred, average='macro', zero_division=0),
        recall_score(true, pred, average='macro', zero_division=0),
        f1_score(true, pred, average='macro', zero_division=0),
        accuracy_score(c_true, c_pred),
        coarse_conf_matrix
    )
def validate_model(model_path, data_path, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionFusionHierarchicalSwinCNN(num_fine=47).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")
    val_dataset = HierarchicalDataset(data_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    print(f"Validation set loaded with {len(val_dataset)} images.")
    loss_fn = nn.CrossEntropyLoss()
    print("Starting model validation...")
    (val_loss, val_acc, val_prec, val_rec, val_f1, val_cacc,
     coarse_conf_matrix) = evaluate(model, val_loader, loss_fn, device)
    print(f"\nValidation complete!")
    print(f"  - Validation Loss: {val_loss:.4f}")
    print(f"  - Coarse Accuracy: {val_cacc:.4f}")
    print(f"  - Fine-grained Accuracy: {val_acc:.4f}")
    print(f"  - Precision: {val_prec:.4f}")
    print(f"  - Recall: {val_rec:.4f}")
    print(f"  - F1 Score: {val_f1:.4f}")
    plt.figure(figsize=(8, 6))
    plt.imshow(coarse_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(3)
    class_names = ['Metamorphic', 'Igneous', 'Sedimentary']
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, coarse_conf_matrix[i, j], ha="center", va="center",
                     color="white" if coarse_conf_matrix[i, j] > coarse_conf_matrix.max() / 2. else "black")
    plt.tight_layout()
    plt.savefig('coarse_confusion_matrix.png', format='png', dpi=600)
    print("Coarse confusion matrix saved to coarse_confusion_matrix.png")
    plt.close()
if __name__ == '__main__':
    validate_model(
        model_path="model_final.pt",  # Replace with your model weights file path
        data_path="val_balanced",      # Replace with your validation set path
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    )
