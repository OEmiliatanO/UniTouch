import torch
from PIL import Image
import pandas as pd

# label.txt format:
# 20220601_182052/0000000053.jpg,3
# 20220601_182052/0000000054.jpg,3
# 20220601_182052/0000000055.jpg,3
# 20220601_182052/0000000056.jpg,3

# only tactile data
class TouchAndGoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, mode, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.data = pd.read_csv(f"{dataset_dir}/{mode}.txt", header=None, names=['path', 'label'])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['path'].strip().split('/')
        label = self.data.iloc[idx]['label']
        instance_path = f"{self.dataset_dir}/dataset/{path[0]}/gelsight_frame/{path[1]}"
        with open(instance_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TouchAndGoPairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, mode, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.data = pd.read_csv(f"{dataset_dir}/{mode}.txt", header=None, names=['path', 'label'])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        touch_path = self.data.iloc[idx]['path'].strip().split('/')
        vision_path = self.data.iloc[idx]['path'].strip().split('/')
        label = self.data.iloc[idx]['label']
        touch_instance_path = f"{self.dataset_dir}/dataset/{touch_path[0]}/gelsight_frame/{touch_path[1]}"
        vision_instance_path = f"{self.dataset_dir}/dataset/{vision_path[0]}/video_frame/{vision_path[1]}"
        with open(touch_instance_path, "rb") as fopen:
            touch_image = Image.open(fopen).convert("RGB")
        with open(vision_instance_path, "rb") as fopen:
            vision_image = Image.open(fopen).convert("RGB")
        if self.transform:
            touch_image = self.transform(touch_image)
            vision_image = self.transform(vision_image)
        return (touch_image, vision_image), label

class TouchAndGoDataset_precomputed_vision(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, precomputed_vision_features_loc, mode, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.data = pd.read_csv(f"{dataset_dir}/{mode}.txt", header=None, names=['path', 'label'])
        self.vision_data = torch.load(precomputed_vision_features_loc, map_location=torch.device("cpu"))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        touch_path = self.data.iloc[idx]['path'].strip().split('/')
        label = self.data.iloc[idx]['label']
        touch_instance_path = f"{self.dataset_dir}/dataset/{touch_path[0]}/gelsight_frame/{touch_path[1]}"
        with open(touch_instance_path, "rb") as fopen:
            touch_image = Image.open(fopen).convert("RGB")
        if self.transform:
            touch_image = self.transform(touch_image)
        return (touch_image, self.vision_data[touch_instance_path]), label

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