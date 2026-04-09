import glob
from YCB_slide_dataset import YCBSlideDataset
import torch
import torch.nn.functional as F
import numpy as np
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
from tqdm import tqdm
from torchvision import transforms
import json
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

@torch.no_grad()
def transfer(touch_model, imagebind_model, modalities=[ModalityType.TOUCH], seed=0):
    """
    Copy the vision truck weights from the imagebind model to the touch model for the specified modalities.
    """
    set_seed(seed)
    other_modalities = set(imagebind_model.modality_trunks.keys()) - set(modalities)
    for modality in other_modalities:
        imagebind_components = [
            imagebind_model.modality_preprocessors[modality],
            imagebind_model.modality_trunks[modality],
            imagebind_model.modality_heads[modality],
            imagebind_model.modality_postprocessors[modality]
        ]
        touch_components = [
            touch_model.modality_preprocessors[modality],
            touch_model.modality_trunks[modality],
            touch_model.modality_heads[modality],
            touch_model.modality_postprocessors[modality]
        ]
        for touch_component, imagebind_component in zip(touch_components, imagebind_components):
            for name, param in touch_component.named_parameters():
                if param.requires_grad and name in imagebind_component.state_dict():
                    param.copy_(imagebind_component.state_dict()[name])

    vision_components = [
        imagebind_model.modality_preprocessors["vision"],
        imagebind_model.modality_trunks["vision"],
        imagebind_model.modality_heads["vision"],
        imagebind_model.modality_postprocessors["vision"]
    ]
    touch_components = [
        touch_model.modality_preprocessors[ModalityType.TOUCH],
        touch_model.modality_trunks[ModalityType.TOUCH],
        touch_model.modality_heads[ModalityType.TOUCH],
        touch_model.modality_postprocessors[ModalityType.TOUCH]
    ]
    std = 0.0001
    for i, (touch_component, vision_component) in enumerate(zip(touch_components, vision_components)):
        for name, param in touch_component.named_parameters():
            if param.requires_grad:
                # noise = torch.randn_like(param) * std
                noise = torch.zeros_like(param)
                if name in vision_component.state_dict():
                    param.copy_(vision_component.state_dict()[name] + noise)

    return touch_model

imagebined_model = imagebind_huge(pretrained=True)
touch_model = x2touch(pretrained=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

touch_training_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", transform=data_transform)
touch_testing_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_testing_data.csv", transform=data_transform)

touch_training_dataloader = torch.utils.data.DataLoader(touch_training_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
touch_testing_dataloader = torch.utils.data.DataLoader(touch_testing_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

text_features = torch.load("YCB-Slide_dataset_path/YCB-Slide_text_features.pt").to(device) # Shape: [C, 1024]

n = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
task_id = n
accs = []
omnimodel = transfer(touch_model, imagebined_model, modalities=[ModalityType.TOUCH], seed = n)
omnimodel.eval()
omnimodel = omnimodel.to(device)

with torch.no_grad():
    preds = []
    all_labels = []
    for batch in tqdm(touch_training_dataloader, desc=f"Evaluating model {n+1}"):
        touch_data, labels = batch
        touch_data = touch_data.to(device)
        outputs = omnimodel({ModalityType.TOUCH: touch_data}) # Shape: [batch_size, 1024]
        touch_features = outputs[ModalityType.TOUCH] # Shape: [batch_size, 1024]
        # print(f"touch_features shape: {touch_features.shape}")
        batch_preds = (touch_features @ text_features.T).argmax(dim=-1) # Shape: [batch_size]
        preds.append(batch_preds.cpu())
        all_labels.extend(labels)
    preds = torch.cat(preds, dim=0) # Shape: [num_samples]
    acc = (preds == torch.tensor(all_labels)).float().mean().item()

result = {"n": n, "accuracy": acc}
os.makedirs("results", exist_ok=True)
with open(f"results/result_{task_id}.json", "w") as f:
    json.dump(result, f)