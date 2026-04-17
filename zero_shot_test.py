import glob
from YCB_slide_dataset import YCBSlidePairedDataset, YCBSlideDataset
import torch
import torch.nn.functional as F
import numpy as np
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
from tqdm import tqdm, trange
from torchvision import transforms
import copy
import json
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class GatherLayer(torch.autograd.Function):
    """
    跨 GPU 收集 Tensor，同時保持梯度能夠反向傳播回各自的 GPU。
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_features(features):
    if dist.is_initialized():
        gathered_features = GatherLayer.apply(features)
        return torch.cat(gathered_features, dim=0)
    return features

@torch.no_grad()
def initialize_touch_model(imagebind_model, touch_model, init_strategy="random", noise_std=0.002, seed=0):
    """
    init_strategy:
        - "random": 一般的隨機初始權重 (維持 x2touch(pretrained=False) 的狀態)
        - "vision_clean": 完全複製 Vision 權重，不加擾動
        - "vision_noise": 複製 Vision 權重，並加入 N(0, std^2) 的高斯擾動
    """
    set_seed(seed)
    # 複製一份獨立的 touch_model，避免改動到原始物件
    new_touch_model = copy.deepcopy(touch_model)
    
    if init_strategy == "random":
        return new_touch_model

    vision_components = [
        imagebind_model.modality_preprocessors["vision"],
        imagebind_model.modality_trunks["vision"],
        imagebind_model.modality_heads["vision"],
        imagebind_model.modality_postprocessors["vision"]
    ]
    touch_components = [
        new_touch_model.modality_preprocessors[ModalityType.TOUCH],
        new_touch_model.modality_trunks[ModalityType.TOUCH],
        new_touch_model.modality_heads[ModalityType.TOUCH],
        new_touch_model.modality_postprocessors[ModalityType.TOUCH]
    ]

    for touch_component, vision_component in zip(touch_components, vision_components):
        for name, param in touch_component.named_parameters():
            if param.requires_grad and name in vision_component.state_dict():
                base_weight = vision_component.state_dict()[name].clone()
                
                if init_strategy == "vision_noise":
                    # W_touch = W_vision + epsilon, epsilon ~ N(0, sigma^2)
                    noise = torch.randn_like(param) * noise_std
                    param.data.copy_(base_weight + noise)
                elif init_strategy == "vision_clean":
                    param.data.copy_(base_weight)

    return new_touch_model

@torch.no_grad()
def evaluate(model, dataloader, text_features, device):
    model.eval()
    preds = []
    all_labels = []
    
    for touch_data, labels in tqdm(dataloader, desc="Evaluating"):
        touch_data = touch_data.to(device)
        outputs = model({ModalityType.TOUCH: touch_data}) 
        touch_features = outputs[ModalityType.TOUCH] 
        
        # L2 正規化後再做內積，確保 Cosine Similarity 的正確性
        touch_features = F.normalize(touch_features, dim=1)
        text_features_norm = F.normalize(text_features, dim=1)
        
        batch_preds = (touch_features @ text_features_norm.T).argmax(dim=-1)
        preds.append(batch_preds.cpu())
        all_labels.extend(labels)
        
    preds = torch.cat(preds, dim=0)
    acc = (preds == torch.tensor(all_labels)).float().mean().item()
    return acc

def align(vision_model, touch_model, paired_dataloader, device, epochs=5, local_rank=0): # infoNCE
    vision_model.eval()
    touch_model.train()
    optimizer = torch.optim.Adam(touch_model.parameters(), lr=1e-5)

    is_main_process = (local_rank == 0)
    progress_bar = tqdm(tot=epochs, desc="Aligning Models (InfoNCE Training)")

    for epoch in range(epochs):
        paired_dataloader.sampler.set_epoch(epoch)
        tot_loss = 0
        for batch in paired_dataloader:
            (touch_images, vision_images), _ = batch
            touch_images = touch_images.to(device)
            vision_images = vision_images.to(device)

            optimizer.zero_grad()

            batch_touch_features = touch_model({ModalityType.TOUCH: touch_images})[ModalityType.TOUCH]
            with torch.no_grad():
                batch_vision_features = vision_model({ModalityType.VISION: vision_images})[ModalityType.VISION]
            # 計算 infoNCE loss 並對 touch_model 進行反向傳播
            temperature = 0.07
            
            local_vision_features = F.normalize(batch_vision_features, dim=1)
            local_touch_features = F.normalize(batch_touch_features, dim=1)

            global_vision_features = gather_features(local_vision_features)
            global_touch_features = gather_features(local_touch_features)

            logits_T2V = local_touch_features @ global_vision_features.T / temperature 
            logits_V2T = local_vision_features @ global_touch_features.T / temperature

            batch_size = local_touch_features.size(0)
            rank_offset = dist.get_rank() * batch_size
            labels = torch.arange(batch_size, dtype=torch.long, device=device) + rank_offset
            loss = (F.cross_entropy(logits_T2V, labels) + F.cross_entropy(logits_V2T, labels)) / 2
            
            loss.backward()
            optimizer.step()
            
            tot_loss += loss.item()

        avg_loss = torch.tensor(tot_loss / len(paired_dataloader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()

        if is_main_process:
            progress_bar.set_postfix({"Epoch Loss": avg_loss.item()})
            progress_bar.update(1)
    return touch_model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 載入基礎模型與資料 (請替換為實際的 dataset 實例化)
    imagebind_model = imagebind_huge(pretrained=True).to(device)
    imagebind_model.eval()
    imagebind_model.requires_grad_(False)
    base_touch_model = x2touch(pretrained=False).to(device)

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

    text_features = torch.load("YCB-Slide_dataset_path/YCB-Slide_text_features.pt").to(device) # Shape: [C, 1024]

    touch_vision_paired_training_dataset = YCBSlidePairedDataset("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/YCB-Slide_vision_training_data.csv", transform=data_transform)
    touch_vision_paired_training_subdataset = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, len(touch_vision_paired_training_dataset), 10))
    touch_testing_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_testing_data.csv", transform=data_transform)
    touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(touch_vision_paired_training_subdataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    touch_testing_subdataset = torch.utils.data.Subset(touch_testing_dataset, indices=range(0, len(touch_testing_dataset), 100))
    touch_testing_dataloader = torch.utils.data.DataLoader(touch_testing_subdataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    strategies = ["random", "vision_clean", "vision_noise"]
    results = {}

    for strategy in strategies:
        print(f"\n{'='*20} Testing Strategy: {strategy} {'='*20}")
        
        # Step A: 權重初始化
        model = initialize_touch_model(
            imagebind_model, 
            base_touch_model, 
            init_strategy=strategy, 
            noise_std=0.0001, 
            seed=42
        ).to(device)
        
        # Step B: 評估 Initial Performance (Zero-shot)
        print("--- Evaluating Initial Performance ---")
        init_acc = evaluate(model, touch_testing_dataloader, text_features, device)
        
        # Step C: Post-alignment Training (InfoNCE)
        print("--- Running Post-alignment Training ---")
        model = align(imagebind_model, model, touch_vision_paired_training_dataloader, device, epochs=5)
        
        # Step D: 評估 Final Performance
        print("--- Evaluating Final Performance ---")
        final_acc = evaluate(model, touch_testing_dataloader, text_features, device)
        
        results[strategy] = {"Init_Acc": init_acc, "Final_Acc": final_acc}
        print(f"[{strategy}] Init Acc: {init_acc:.4f} -> Final Acc: {final_acc:.4f}")

    print("\n================ Final Summary ================")
    for k, v in results.items():
        print(f"Strategy: {k:<15} | Init Acc: {v['Init_Acc']:.4f} | Final Acc: {v['Final_Acc']:.4f}")
