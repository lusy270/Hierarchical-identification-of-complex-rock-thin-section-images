import torch
from torch.utils.data import Dataset
import os
import re
from PIL import Image
import torchvision.transforms as transforms


class HierarchicalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.path_list = []
        self.prefix_to_coarse = {'变': 0, '火': 1, '沉': 2}
        self.fine_label_offset = {'变': 0, '火': 14, '沉': 43}

        for f in os.listdir(data_path):
            if re.match(r'^[变火沉]\d+', f):
                img_path = os.path.join(data_path, f)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.path_list.append(f)
                except:
                    pass

        if not self.path_list:
            raise ValueError(f"No valid images in {data_path}")

    def __getitem__(self, idx):
        img_name = self.path_list[idx]
        match = re.match(r'^([变火沉])(\d+)', img_name)
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