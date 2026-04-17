import torch
from PIL import Image
import pandas as pd

class YCBSlideDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['label']
        with open(path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class YCBSlidePairedDataset(torch.utils.data.Dataset):
    def __init__(self, touch_csv_file, vision_csv_file, transform=None):
        self.touch_data = pd.read_csv(touch_csv_file)
        self.vision_data = pd.read_csv(vision_csv_file)
        self.transform = transform

        assert len(self.touch_data) == len(self.vision_data), "Touch and vision datasets must have the same number of samples"

    def __len__(self):
        return len(self.touch_data)

    def __getitem__(self, idx):
        touch_path = self.touch_data.iloc[idx]['path']
        vision_path = self.vision_data.iloc[idx]['path']
        label = self.touch_data.iloc[idx]['label']
        with open(touch_path, "rb") as fopen:
            touch_image = Image.open(fopen).convert("RGB")
        with open(vision_path, "rb") as fopen:
            vision_image = Image.open(fopen).convert("RGB")
        if self.transform:
            touch_image = self.transform(touch_image)
            vision_image = self.transform(vision_image)
        return (touch_image, vision_image), label

class YCBSlidedPairedDataset_precomputed_vision(torch.utils.data.Dataset):
    def __init__(self, touch_csv_file, precomputed_vision_features, transform=None):
        self.touch_data = pd.read_csv(touch_csv_file)
        self.vision_data = torch.load(precomputed_vision_features, map_location=torch.device("cpu"))
        self.transform = transform

        assert len(self.touch_data) == len(self.vision_data), "Touch and vision datasets must have the same number of samples"
    
    def __len__(self):
        return len(self.touch_data)
    
    def __getitem__(self, idx):
        touch_path = self.touch_data.iloc[idx]['path']
        label = self.touch_data.iloc[idx]['label']
        with open(touch_path, "rb") as fopen:
            touch_image = Image.open(fopen).convert("RGB")
        if self.transform:
            touch_image = self.transform(touch_image)
        return (touch_image, self.vision_data[idx]), label

"""
data_transform = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

dataloader = torch.utils.data.DataLoader(YCBSlideDataset("YCB-Slide_touch_training_data.csv", transform=data_transform), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
"""