import glob
from YCB_slide_dataset import YCBSlidePairedDataset, YCBSlidedPairedDataset_precomputed_vision, YCBSlideDataset
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
from torch.amp import autocast, GradScaler
from torch.amp import custom_fwd, custom_bwd

import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as diff_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    """初始化分散式訓練環境"""
    # 讀取 torchrun 傳入的環境變數
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # 初始化 Process Group
    dist.init_process_group(backend="nccl")
    # 設定當前進程使用的 GPU
    torch.cuda.set_device(local_rank)
    return local_rank

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
def initialize_touch_model(imagebind_model, init_strategy="random", noise_std=0.002, seed=0):
    """
    init_strategy:
        - "random": 一般的隨機初始權重 (維持 x2touch(pretrained=False) 的狀態)
        - "vision_clean": 完全複製 Vision 權重，不加擾動
        - "vision_noise": 複製 Vision 權重，並加入 N(0, std^2) 的高斯擾動
    """
    imagebind_model.requires_grad_(True)
    set_seed(seed)
    new_touch_model = x2touch(pretrained=False)
    new_touch_model.requires_grad_(False)
    vision_components = [
        imagebind_model.modality_preprocessors[ModalityType.VISION],
        imagebind_model.modality_trunks[ModalityType.VISION],
        imagebind_model.modality_heads[ModalityType.VISION],
        imagebind_model.modality_postprocessors[ModalityType.VISION]
    ]
    touch_components = [
        new_touch_model.modality_preprocessors[ModalityType.TOUCH],
        new_touch_model.modality_trunks[ModalityType.TOUCH],
        new_touch_model.modality_heads[ModalityType.TOUCH],
        new_touch_model.modality_postprocessors[ModalityType.TOUCH]
    ]

    if init_strategy == "random":
        for touch_component, vision_component in zip(touch_components, vision_components):
            vision_params = dict(vision_component.named_parameters())
            for name, param in touch_component.named_parameters():
                if name in vision_component.state_dict() and vision_params[name].requires_grad:
                    param.requires_grad = True

        imagebind_model.requires_grad_(False)
        return new_touch_model

    for touch_component, vision_component in zip(touch_components, vision_components):
        vision_params = dict(vision_component.named_parameters())
        for name, param in touch_component.named_parameters():
            if name in vision_component.state_dict() and vision_params[name].requires_grad:
                base_weight = vision_component.state_dict()[name].clone()
                
                if init_strategy == "vision_noise":
                    # W_touch = W_vision + epsilon, epsilon ~ N(0, sigma^2)
                    noise = torch.randn_like(param) * noise_std
                    param.data.copy_(base_weight + noise)
                elif init_strategy == "vision_clean":
                    param.data.copy_(base_weight)
            
                param.requires_grad = True

    imagebind_model.requires_grad_(False)

    return new_touch_model

# @torch.no_grad()
# def evaluate(model, dataloader, text_features, device):
#     actual_model = model.module if isinstance(model, DDP) else model
#     actual_model.eval()
#     preds = []
#     all_labels = []
    
#     for touch_data, labels in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
#         touch_data = touch_data.to(device)
#         outputs = actual_model({ModalityType.TOUCH: touch_data}) 
#         touch_features = outputs[ModalityType.TOUCH] 
        
#         touch_features = F.normalize(touch_features, dim=1)
#         text_features_norm = F.normalize(text_features, dim=1)
        
#         batch_preds = (touch_features @ text_features_norm.T).argmax(dim=-1)
#         preds.append(batch_preds.cpu())
#         all_labels.extend(labels.cpu().tolist())
        
#     preds = torch.cat(preds, dim=0)
#     acc = (preds == torch.tensor(all_labels)).float().mean().item()
#     return acc

@torch.no_grad()
def evaluate(model, dataloader, text_features, device):
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.eval()
    
    local_correct = 0
    local_total = 0
    
    # 讓所有 Rank 都參與 tqdm 進度條，但只在 Rank 0 顯示
    for touch_data, labels in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
        touch_data = touch_data.to(device)
        labels = labels.to(device)
        
        outputs = actual_model({ModalityType.TOUCH: touch_data}) 
        touch_features = outputs[ModalityType.TOUCH] 
        
        touch_features = F.normalize(touch_features, dim=1)
        text_features_norm = F.normalize(text_features, dim=1)
        
        batch_preds = (touch_features @ text_features_norm.T).argmax(dim=-1)
        
        # 統計當前 GPU 的局部結果
        local_correct += (batch_preds == labels).sum().item()
        local_total += labels.size(0)
        
    # 將 local 變數轉換為 Tensor 以便進行 NCCL 通訊
    # shape: [2] -> [correct_count, total_count]
    metrics = torch.tensor([local_correct, local_total], dtype=torch.float32, device=device)
    
    # 核心同步操作：將所有 GPU 的 metrics 張量進行加總
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    global_correct = metrics[0].item()
    global_total = metrics[1].item()
    
    # 計算全域準確率
    global_acc = global_correct / global_total if global_total > 0 else 0.0
    
    return global_acc

def align(touch_model, paired_dataloader, device, epochs=5, local_rank=0): # infoNCE
    touch_model.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, touch_model.parameters()), lr=1e-5)

    # scaler = GradScaler()

    is_main_process = (local_rank == 0)
    progress_bar = tqdm(total=epochs, desc="Aligning Models (InfoNCE Training)", disable=not is_main_process)

    for epoch in range(epochs):
        if hasattr(paired_dataloader.sampler, "set_epoch"):
            paired_dataloader.sampler.set_epoch(epoch)
        tot_loss = 0
        i = 0
        for batch in paired_dataloader:
            (touch_images, vision_features), _ = batch
            touch_images = touch_images.to(device)
            vision_features = vision_features.to(device)

            optimizer.zero_grad()
            batch_touch_features = touch_model({ModalityType.TOUCH: touch_images})[ModalityType.TOUCH]
            # 計算 infoNCE loss 並對 touch_model 進行反向傳播
            temperature = 0.07
            
            local_touch_features = F.normalize(batch_touch_features, dim=1)
            local_vision_features = F.normalize(vision_features, dim=1)
            global_touch_list = diff_all_gather(local_touch_features)
            global_touch_features = torch.cat(global_touch_list, dim=0)

            with torch.no_grad():
                global_vision_list = [torch.zeros_like(local_vision_features) for _ in range(dist.get_world_size())]
                dist.all_gather(global_vision_list, local_vision_features)
                global_vision_features = torch.cat(global_vision_list, dim=0)

            logits_T2V = local_touch_features @ global_vision_features.T / temperature 
            logits_V2T = local_vision_features @ global_touch_features.T / temperature

            batch_size = local_touch_features.size(0)
            rank_offset = dist.get_rank() * batch_size
            labels = torch.arange(batch_size, dtype=torch.long, device=device) + rank_offset
            loss = (F.cross_entropy(logits_T2V, labels) + F.cross_entropy(logits_V2T, labels)) / 2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            tot_loss += loss.item()

        avg_loss = torch.tensor(tot_loss / len(paired_dataloader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()

        if is_main_process:
            progress_bar.set_postfix({"Epoch Loss": avg_loss.item()})
            progress_bar.update(1)
    return touch_model

if __name__ == "__main__":
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. 載入基礎模型與資料 (請替換為實際的 dataset 實例化)
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

    text_features = torch.load("YCB-Slide_dataset_path/YCB-Slide_text_features.pt").to(device) # Shape: [C, 1024]

    # touch_vision_paired_training_dataset = YCBSlidePairedDataset("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/YCB-Slide_vision_training_data.csv", transform=data_transform)
    touch_vision_paired_training_dataset = YCBSlidedPairedDataset_precomputed_vision("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/precomputed_training_vision_features.pt", transform=data_transform)
    touch_vision_paired_training_subdataset = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, len(touch_vision_paired_training_dataset), 1))
    touch_testing_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_testing_data.csv", transform=data_transform)
    touch_testing_subdataset = torch.utils.data.Subset(touch_testing_dataset, indices=range(0, len(touch_testing_dataset), 1))

    # touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(touch_vision_paired_training_subdataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    # touch_testing_dataloader = torch.utils.data.DataLoader(touch_testing_subdataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    train_sampler = DistributedSampler(touch_vision_paired_training_subdataset, drop_last=True)
    touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(
        touch_vision_paired_training_subdataset, 
        batch_size=20, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
    )

    # 測試集通常只在 Rank 0 上評估，或者保持原樣讓每張卡跑全部測試集再平均
    test_sampler = DistributedSampler(touch_testing_subdataset, shuffle=False, drop_last=True)
    touch_testing_dataloader = torch.utils.data.DataLoader(
        touch_testing_subdataset, 
        batch_size=64, 
        sampler=test_sampler,
        num_workers=4, 
        pin_memory=True
    )

    strategies = ["vision_noise", "random", "vision_clean"]
    results = {}

    for strategy in strategies:
        if local_rank == 0:
            print(f"\n{'='*20} Testing Strategy: {strategy} {'='*20}")
        
        # Step A: 權重初始化
        model = initialize_touch_model(
            imagebind_model, 
            init_strategy=strategy, 
            noise_std=0.002, 
            seed=42
        ).to(device)
        
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Step B: 評估 Initial Performance (Zero-shot)
        if local_rank == 0:
            print("--- Evaluating Initial Performance ---")

        init_acc = evaluate(model, touch_testing_dataloader, text_features, device)
        
        dist.barrier(local_rank)
        
        # Step C: Post-alignment Training (InfoNCE)
        if local_rank == 0:
            print("--- Running Post-alignment Training ---")

        model = align(model, touch_vision_paired_training_dataloader, device, epochs=1, local_rank=local_rank)
        
        # Step D: 評估 Final Performance
        if local_rank == 0:
            print("--- Evaluating Final Performance ---")
            
        final_acc = evaluate(model, touch_testing_dataloader, text_features, device)

        if local_rank == 0:
            results[strategy] = {"Init_Acc": init_acc, "Final_Acc": final_acc}
            print(f"[{strategy}] Init Acc: {init_acc:.4f} -> Final Acc: {final_acc:.4f}")

        # 再次同步
        dist.barrier(local_rank)

        del model
        torch.cuda.empty_cache()

    if local_rank == 0:
        print("\n================ Final Summary ================")
        for k, v in results.items():
            print(f"Strategy: {k:<15} | Init Acc: {v['Init_Acc']:.4f} | Final Acc: {v['Final_Acc']:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()
