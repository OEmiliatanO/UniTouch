import glob
import random
import torch
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
import pandas as pd
from YCB_slide_dataset import YCBSlidePairedDataset, YCBSlideDataset
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

classes = [
    "sugar box", "tomato soup can", "mustard bottle", "bleach cleanser", 
    "mug", "power drill", "scissors", "adjustable wrench", "hammer", "baseball"
]
cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

prompts = [f"This feels like {cls}" for cls in classes]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagebined_model = imagebind_huge(pretrained=True)
text_data = data.load_and_transform_text(prompts, device=device)
with torch.no_grad():
    imagebined_model = imagebined_model.to(device)
    text_features = imagebined_model({ModalityType.TEXT: text_data})[ModalityType.TEXT]  # Shape: [C, seq_len, 1024]
    eos_indices = text_data.argmax(dim=-1)
    text_features = text_features[torch.arange(text_features.shape[0]), eos_indices]
text_features = text_features.cpu()

save_dir = "YCB-Slide_dataset_path"

torch.save(text_features, f"{save_dir}/YCB-Slide_text_features.pt")
root_dir = "/work/hans1010/YCB-Slide/dataset/"
dataset_len = 5
classes_touch_paths = {
    "sugar box" : [glob.glob(f"{root_dir}/real/004_sugar_box/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "tomato soup can" : [glob.glob(f"{root_dir}/real/005_tomato_soup_can/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "mustard bottle" : [glob.glob(f"{root_dir}/real/006_mustard_bottle/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "bleach cleanser" : [glob.glob(f"{root_dir}/real/021_bleach_cleanser/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "mug" : [glob.glob(f"{root_dir}/real/025_mug/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "power drill" : [glob.glob(f"{root_dir}/real/035_power_drill/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "scissors" : [glob.glob(f"{root_dir}/real/037_scissors/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "adjustable wrench" : [glob.glob(f"{root_dir}/real/042_adjustable_wrench/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "hammer" : [glob.glob(f"{root_dir}/real/048_hammer/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "baseball" : [glob.glob(f"{root_dir}/real/055_baseball/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)]
}

vision_root_dir = "/work/hans1010/ycb_vision"
classes_vision_paths = {
    "sugar box" : [sorted(glob.glob(f"{vision_root_dir}/004_sugar_box/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "tomato soup can" : [sorted(glob.glob(f"{vision_root_dir}/005_tomato_soup_can/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "mustard bottle" : [sorted(glob.glob(f"{vision_root_dir}/006_mustard_bottle/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "bleach cleanser" : [sorted(glob.glob(f"{vision_root_dir}/021_bleach_cleanser/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "mug" : [sorted(glob.glob(f"{vision_root_dir}/025_mug/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "power drill" : [sorted(glob.glob(f"{vision_root_dir}/035_power_drill/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "scissors" : [sorted(glob.glob(f"{vision_root_dir}/037_scissors/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "adjustable wrench" : [sorted(glob.glob(f"{vision_root_dir}/042_adjustable_wrench/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "hammer" : [sorted(glob.glob(f"{vision_root_dir}/048_hammer/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)],
    "baseball" : [sorted(glob.glob(f"{vision_root_dir}/055_baseball/dataset_{dataset_idx}/sim_frames/*.png")) for dataset_idx in range(dataset_len)]
}
all_labels = []
for key_idx, key in enumerate(classes_touch_paths):
    classes_touch_paths[key] = [item for sublist in classes_touch_paths[key] for item in sublist] # Flatten the list of lists into a single list
    classes_vision_paths[key] = [item for sublist in classes_vision_paths[key] for item in sublist]
    all_labels += [cls_to_idx[key]] * len(classes_touch_paths[key])
    print(f"Class '{key}': {classes_touch_paths[key][:2]}, {classes_vision_paths[key][:2]}")
print(f"Total label samples: {len(all_labels)}")
touch_paths = [path for paths in classes_touch_paths.values() for path in paths] # Flatten the list of lists into a single list
print(f"Total touch samples: {len(touch_paths)}")
touch_paths = list(zip(touch_paths, all_labels))
vision_paths = [path for paths in classes_vision_paths.values() for path in paths] # Flatten the list of lists into a single list
print(f"Total vision samples: {len(vision_paths)}")
vision_paths = list(zip(vision_paths, all_labels))
rng = np.random.default_rng(seed=42)
rng.shuffle(touch_paths)
rng = np.random.default_rng(seed=42)
rng.shuffle(vision_paths)
# random.shuffle(touch_paths)
touch_paths, all_labels = zip(*touch_paths)
vision_paths, _ = zip(*vision_paths)

training_ratio = 0.8

training_touch_paths = touch_paths[:int(training_ratio * len(touch_paths))]
testing_touch_paths = touch_paths[int(training_ratio * len(touch_paths)):]
training_label = all_labels[:int(training_ratio * len(all_labels))]
testing_label = all_labels[int(training_ratio * len(all_labels)):]
pd.DataFrame({"path": training_touch_paths, "label": training_label}).to_csv(f"{save_dir}/YCB-Slide_touch_training_data.csv", index=False)
pd.DataFrame({"path": testing_touch_paths, "label": testing_label}).to_csv(f"{save_dir}/YCB-Slide_touch_testing_data.csv", index=False)

training_vision_paths = vision_paths[:int(training_ratio * len(vision_paths))]
testing_vision_paths = vision_paths[int(training_ratio * len(vision_paths)):]
pd.DataFrame({"path": training_vision_paths, "label": training_label}).to_csv(f"{save_dir}/YCB-Slide_vision_training_data.csv", index=False)
pd.DataFrame({"path": testing_vision_paths, "label": testing_label}).to_csv(f"{save_dir}/YCB-Slide_vision_testing_data.csv", index=False)

import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
from tqdm import tqdm, trange
from torchvision import transforms

imagebind_model = imagebind_huge(pretrained=True)
imagebind_model.eval()

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

touch_vision_paired_training_dataset = YCBSlidePairedDataset(f"{save_dir}/YCB-Slide_touch_training_data.csv", f"{save_dir}/YCB-Slide_vision_training_data.csv", transform=data_transform)
touch_testing_dataset = YCBSlideDataset(f"{save_dir}/YCB-Slide_touch_testing_data.csv", transform=data_transform)

touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(
    touch_vision_paired_training_dataset, 
    batch_size=64, 
    num_workers=4, 
    pin_memory=True,
)

vision_features_list = []
imagebind_model.to(device)
for batch in touch_vision_paired_training_dataloader:
    (touch_images, vision_images), labels = batch
    # Process the vision images to extract features
    with torch.no_grad():
        vision_features = imagebind_model({ModalityType.VISION: vision_images.to(device)})[ModalityType.VISION].cpu()
    vision_features_list.append(vision_features)
vision_features = torch.cat(vision_features_list, dim=0)
torch.save(vision_features, f"{save_dir}/precomputed_training_vision_features.pt")
