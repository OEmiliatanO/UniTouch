import glob
from YCB_slide_dataset import YCBSlidePairedDataset, YCBSlidedPairedDataset_precomputed_vision, YCBSlideDataset
import torch
import torch.nn.functional as F
import numpy as np
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
from tqdm import tqdm, trange
from torchvision import transforms
from datasets import load_from_disk
import copy
import json
import os
import random
import sys
from torch.amp import autocast, GradScaler
from torch.amp import custom_fwd, custom_bwd
import torch.optim.lr_scheduler as lr_scheduler
import wandb

import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as diff_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import cka, mknn

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

@torch.no_grad()
def calculate_weight_drift(current_model, initial_weights_dict):
    drift_metrics = {}
    total_drift_sq = 0.0
    total_init_sq = 0.0
    for name, param in current_model.named_parameters():
        if name in initial_weights_dict:
            current_w = param.detach().cpu()
            initial_w = initial_weights_dict[name]
            diff_norm = torch.norm(current_w - initial_w, p='fro').item()
            init_norm = torch.norm(initial_w, p='fro').item()
            
            relative_drift_value = diff_norm / (init_norm + 1e-8)
            relative_log_name = f"relative_layer_drift/{name.replace('.', '/')}"
            drift_metrics[relative_log_name] = relative_drift_value

            absolute_drift_value = diff_norm
            absolute_log_name = f"absolute_layer_drift/{name.replace('.', '/')}"
            drift_metrics[absolute_log_name] = absolute_drift_value
            
            total_drift_sq += diff_norm ** 2
            total_init_sq += init_norm ** 2
    
    drift_metrics["absolute_total_drift"] = total_drift_sq ** 0.5
    drift_metrics["relative_total_drift"] = total_drift_sq ** 0.5 / (total_init_sq ** 0.5 + 1e-8)
    return drift_metrics

@torch.no_grad()
def initialize_touch_model(imagebind_model, init_strategy="random", freeze_vision=True, noise_std=0.002, seed=0):
    """
    init_strategy:
        - "random": 一般的隨機初始權重 (維持 x2touch(pretrained=False) 的狀態)
        - "vision_clean": 完全複製 Vision 權重，不加擾動
        - "vision_noise": 複製 Vision 權重，並加入 N(0, std^2) 的高斯擾動
    """
    imagebind_model.cpu()
    imagebind_model.requires_grad_(True)
    g = torch.Generator(device='cpu').manual_seed(seed)
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
    touchs_vision_components = [
        new_touch_model.modality_preprocessors[ModalityType.VISION],
        new_touch_model.modality_trunks[ModalityType.VISION],
        new_touch_model.modality_heads[ModalityType.VISION],
        new_touch_model.modality_postprocessors[ModalityType.VISION]
    ]

    if init_strategy == "random":
        for touch_component, touchs_vision_component, vision_component in zip(touch_components, touchs_vision_components, vision_components):
            vision_params = dict(vision_component.named_parameters())
            for name, param in touch_component.named_parameters():
                if name in vision_component.state_dict() and vision_params[name].requires_grad:
                    param.requires_grad = True
            for name, param in touchs_vision_component.named_parameters():
                if name in vision_component.state_dict() and vision_params[name].requires_grad:
                    param.requires_grad = not freeze_vision

        imagebind_model.requires_grad_(False)
        return new_touch_model

    for touch_component, touchs_vision_component, vision_component in zip(touch_components, touchs_vision_components, vision_components):
        vision_params = dict(vision_component.named_parameters())
        for name, param in touch_component.named_parameters():
            if name in vision_component.state_dict() and vision_params[name].requires_grad:
                base_weight = vision_component.state_dict()[name].clone()
                
                if init_strategy == "vision_noise":
                    # W_touch = W_vision + epsilon, epsilon ~ N(0, sigma^2)
                    # noise = torch.randn_like(param) * noise_std
                    noise = torch.empty(param.shape, dtype=param.dtype).normal_(mean=0, std=noise_std, generator=g)
                    param.data.copy_(base_weight + noise)
                elif init_strategy == "vision_clean":
                    param.data.copy_(base_weight)
            
                param.requires_grad = True
        for name, param in touchs_vision_component.named_parameters():
            if name in vision_component.state_dict() and vision_params[name].requires_grad:
                param.requires_grad = not freeze_vision

    imagebind_model.requires_grad_(False)

    return new_touch_model

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
        
        text_features_norm = text_features
        
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

class LinearProbeModel(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.fc = torch.nn.Linear(1024, num_classes)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, features):
        out = self.fc(features)
        return out

def evaluate_on_imagenet(train_loader, val_loader, model, device):
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.eval()

    epochs = 5

    model = LinearProbeModel(num_classes=1000).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    for epoch in range(epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        local_loss = 0.0
        local_correct = 0
        local_total = 0
        
        # Training
        for images, labels in tqdm(train_loader, desc=f"Imagenet Epoch {epoch+1}/{epochs} [Train]", leave=False, disable=not is_main_process):
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                features = actual_model({ModalityType.TOUCH: images})[ModalityType.TOUCH]
                features = torch.flatten(features, 1).detach()
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            local_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            local_total += labels.size(0)
            local_correct += predicted.eq(labels).sum().item()
            
        scheduler.step()

        train_metrics = torch.tensor([local_correct, local_total, local_loss], dtype=torch.float32, device=device)
        if dist.is_initialized():
            dist.all_reduce(train_metrics, op=dist.ReduceOp.SUM)

        global_train_correct = train_metrics[0].item()
        global_train_total = train_metrics[1].item()
        global_train_loss = train_metrics[2].item()

        train_acc = global_train_correct / global_train_total if global_train_total > 0 else 0
        if is_main_process:
            print(f"Epoch {epoch+1}/{epochs} [Train] Loss: {global_train_loss/global_train_total:.4f}, Accuracy: {train_acc:.2f}")
        
    # Validation
    model.eval()
    local_val_loss = 0.0
    local_val_correct = 0
    local_val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"ImageNet-1k Epoch {epoch+1}/{epochs} [Val]", leave=False, disable=not is_main_process):
            images, labels = images.to(device), labels.to(device)

            features = actual_model({ModalityType.TOUCH: images})[ModalityType.TOUCH]
            features = torch.flatten(features, 1).detach()

            outputs = model(features)
            loss = criterion(outputs, labels)
            
            local_val_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            local_val_total += labels.size(0)
            local_val_correct += predicted.eq(labels).sum().item()
        
    val_metrics = torch.tensor([local_val_correct, local_val_total, local_val_loss], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
        
    global_val_correct = val_metrics[0].item()
    global_val_total = val_metrics[1].item()

    val_acc = global_val_correct / global_val_total if global_val_total > 0 else 0
    
    return val_acc

@torch.no_grad()
def evaluate_with_metrics(model, paired_dataloader, device):
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.eval()

    local_touch_features = []
    local_vision_features = []

    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    for batch in tqdm(paired_dataloader, desc="Extracting Features", disable=not is_main_process):
        (touch_images, vision_images), _ = batch
        touch_images = touch_images.to(device)
        vision_images = vision_images.to(device)

        outputs = actual_model({ModalityType.TOUCH: touch_images, ModalityType.VISION: vision_images})

        touch_outputs = outputs[ModalityType.TOUCH]
        vision_outputs = outputs[ModalityType.VISION]

        local_touch_features.append(touch_outputs)
        local_vision_features.append(vision_outputs)
    
    local_touch_features = torch.cat(local_touch_features, dim=0)
    local_vision_features = torch.cat(local_vision_features, dim=0)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        # 準備接收的容器
        gathered_touch = [torch.zeros_like(local_touch_features) for _ in range(world_size)]
        gathered_vision = [torch.zeros_like(local_vision_features) for _ in range(world_size)]
        
        # 進行通訊同步
        dist.all_gather(gathered_touch, local_touch_features)
        dist.all_gather(gathered_vision, local_vision_features)
        
        # 沿著 batch 維度拼接
        all_touch_features = torch.cat(gathered_touch, dim=0)
        all_vision_features = torch.cat(gathered_vision, dim=0)
    else:
        all_touch_features = local_touch_features
        all_vision_features = local_vision_features

    calibrated_cka, _, _ = cka(all_touch_features, all_vision_features)
    calibrated_mknn, _, _ = mknn(all_touch_features, all_vision_features, k=10)
    return {"cka": calibrated_cka, "mknn": calibrated_mknn}

def align(touch_model, paired_dataloader, device, epochs=5, local_rank=0, 
          eval_dataloader=None, text_features=None, evaluate_fn=None, 
          paired_subdataloader=None, imagenet_train_loader=None, imagenet_val_loader=None, precomputed_imagenet_loader=None, 
          initial_weights_cpu=None,
          logger=None, strategy_name=None, seed=None
         ):
    touch_model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, touch_model.parameters()), lr=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # scaler = GradScaler()

    is_main_process = (local_rank == 0)
    progress_bar = tqdm(total=epochs, desc="Aligning Models (InfoNCE Training)", disable=not is_main_process)

    performance_history = {
        "loss": [],
        "accuracy": [],
        "imagenet_accuracy": [],
        "cka": [],
        "mknn": []
    }

    epoch_acc = 0
    if eval_dataloader is not None and text_features is not None and evaluate_fn is not None:
        epoch_acc = evaluate_fn(touch_model, eval_dataloader, text_features, device)
        touch_model.train()
    
    epoch_imagenet_acc = 0
    if imagenet_train_loader is not None and imagenet_val_loader is not None:
        epoch_imagenet_acc = evaluate_on_imagenet(imagenet_train_loader, imagenet_val_loader, touch_model, device)
        touch_model.train()

    cka, mknn = 0, 0
    if paired_subdataloader is not None:
        sim_metrics = evaluate_with_metrics(touch_model, paired_subdataloader, device)
        cka, mknn = sim_metrics["cka"].item(), sim_metrics["mknn"].item()

    if is_main_process:
        logger.log({"epoch/epoch": 0, "epoch/accuracy": epoch_acc, "epoch/imagenet_accuracy": epoch_imagenet_acc, "epoch/cka": cka, "epoch/mknn": mknn})
        performance_history["loss"].append(0)
        performance_history["accuracy"].append(epoch_acc)
        performance_history["imagenet_accuracy"].append(epoch_imagenet_acc)
        performance_history["cka"].append(cka)
        performance_history["mknn"].append(mknn)

    from itertools import cycle
    precomputed_imagenet_iter = cycle(precomputed_imagenet_loader)

    for epoch in range(1, epochs+1):
        if hasattr(paired_dataloader.sampler, "set_epoch"):
            paired_dataloader.sampler.set_epoch(epoch)
        
        touch_model.train()
        tot_loss = 0
        tot_alignment = 0
        tot_uniformity = 0

        for batch in tqdm(paired_dataloader, disable=not is_main_process, leave=False):
            (touch_images, vision_images), _ = batch
            touch_images = touch_images.to(device)
            vision_images = vision_images.to(device)

            optimizer.zero_grad()
            outputs = touch_model({ModalityType.TOUCH: touch_images, ModalityType.VISION: vision_images})
            batch_touch_features = outputs[ModalityType.TOUCH]
            batch_vision_features = outputs[ModalityType.VISION]
            temperature = 0.07
            
            local_touch_features = batch_touch_features
            local_vision_features = batch_vision_features
            
            alignment_metric = (local_touch_features - local_vision_features).norm(p=2, dim=1).pow(2).mean()
            sq_dist = torch.pdist(local_touch_features, p=2).pow(2)
            uniformity_metric = torch.log(torch.mean(torch.exp(-2.0 * sq_dist)))

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
            alignment_loss = (F.cross_entropy(logits_T2V, labels) + F.cross_entropy(logits_V2T, labels)) / 2

            # imagenet_batch = next(precomputed_imagenet_iter)
            # imagenet_images, precomputed_imagenet_features =  imagenet_batch["image"], imagenet_batch["vision_feature"]
            # imagenet_images = imagenet_images.to(device)
            # imagenet_features = touch_model({ModalityType.TOUCH: imagenet_images})[ModalityType.TOUCH]
            # precomputed_imagenet_features = precomputed_imagenet_features.to(device)
            # imagenet_loss = F.mse_loss(imagenet_features, precomputed_imagenet_features)
            total_loss = alignment_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            tot_loss += total_loss.item()
            tot_alignment += alignment_metric.item()
            tot_uniformity += uniformity_metric.item()
            if is_main_process:
                drift_metrics = calculate_weight_drift(touch_model.module if isinstance(touch_model, DDP) else touch_model, initial_weights_cpu)
                logger.log({"step/loss": total_loss.item(), "step/alignment": alignment_metric.item(), "step/uniformity": uniformity_metric.item()} | drift_metrics)

        scheduler.step()

        avg_loss = torch.tensor(tot_loss / len(paired_dataloader), device=device)
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        global_avg_loss = (avg_loss / dist.get_world_size()).item()

        epoch_acc = 0
        if eval_dataloader is not None and text_features is not None and evaluate_fn is not None:
            epoch_acc = evaluate_fn(touch_model, eval_dataloader, text_features, device)
            touch_model.train()
        
        epoch_imagenet_acc = 0
        if imagenet_train_loader is not None and imagenet_val_loader is not None:
            epoch_imagenet_acc = evaluate_on_imagenet(imagenet_train_loader, imagenet_val_loader, touch_model, device)
            touch_model.train()

        cka, mknn = 0, 0
        if paired_subdataloader is not None:
            sim_metrics = evaluate_with_metrics(touch_model, paired_subdataloader, device)
            cka, mknn = sim_metrics["cka"].item(), sim_metrics["mknn"].item()

        if is_main_process:
            print(f"loss: {global_avg_loss}")
            model_to_save = model.module if isinstance(model, DDP) else model
            os.makedirs("ckpts", exist_ok=True)
            torch.save(model_to_save.state_dict(), f"ckpts/touch_model_{strategy_name}_{seed}.pth")
            logger.log({"epoch/epoch": epoch, "epoch/loss": global_avg_loss, "epoch/accuracy": epoch_acc, "epoch/imagenet_accuracy": epoch_imagenet_acc, "epoch/cka": cka, "epoch/mknn": mknn})
            performance_history["loss"].append(global_avg_loss)
            performance_history["accuracy"].append(epoch_acc)
            performance_history["imagenet_accuracy"].append(epoch_imagenet_acc)
            performance_history["cka"].append(cka)
            performance_history["mknn"].append(mknn)
            progress_bar.set_postfix({"Loss": f"{global_avg_loss:.4f}", "Acc": f"{epoch_acc:.4f}", "Imagenet Acc": f"{epoch_imagenet_acc:.4f}", "CKA": f"{cka:.4f}", "mKNN": f"{mknn:.4f}"})
            progress_bar.update(1)
        dist.barrier(device_ids=[local_rank])
    return touch_model, performance_history

def prepare_imagenet_dataloader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_train(examples):
        examples['pixel_values'] = [train_transform(image.convert("RGB")) for image in examples['image']]
        return examples

    def preprocess_val(examples):
        examples['pixel_values'] = [val_transform(image.convert("RGB")) for image in examples['image']]
        return examples

    dataset = load_from_disk("/tmp3/Hans/data/imagenet-1k-hf/")
    # dataset = load_from_disk("/work/hans1010/data/imagenet-1k-hf/")
    train_dataset = dataset['train'].select(range(5000)).with_transform(preprocess_train)
    val_dataset = dataset['validation'].select(range(5000)).with_transform(preprocess_val)

    def collate_fn(batch):
        images = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return images, labels
    
    batch_size = 16
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader

def prepare_precomputed_imagenet_dataloader():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_batch(batch):
        batch["image"] = [train_transform(img.convert("RGB")) for img in batch["image"]]
        batch["vision_feature"] = [torch.tensor(feat) for feat in batch["vision_feature"]]
        batch["label"] = [torch.tensor(lbl) for lbl in batch["label"]]
        return batch
    
    dataset = load_from_disk("/tmp3/Hans/UniTouch/imagenet_with_features/")
    dataset.set_transform(preprocess_batch)

    batch_size = 16
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return dataloader

if __name__ == "__main__":
    seed = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    local_rank = setup_ddp()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    imagenet_train_loader, imagenet_val_loader = prepare_imagenet_dataloader()
    precomputed_imagenet_loader = prepare_precomputed_imagenet_dataloader()
    
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

    touch_vision_paired_training_dataset = YCBSlidePairedDataset("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/YCB-Slide_vision_training_data.csv", transform=data_transform)
    # touch_vision_paired_training_dataset = YCBSlidedPairedDataset_precomputed_vision("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/precomputed_training_vision_features.pt", transform=data_transform)
    touch_vision_paired_training_subdataset = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, len(touch_vision_paired_training_dataset), 100))
    touch_vision_paired_training_subdataset_for_metrics = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range( 0, min(100, len(touch_vision_paired_training_dataset)) ))
    touch_testing_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_testing_data.csv", transform=data_transform)
    touch_testing_subdataset = torch.utils.data.Subset(touch_testing_dataset, indices=range(0, len(touch_testing_dataset), 100))

    # touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(touch_vision_paired_training_subdataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    # touch_testing_dataloader = torch.utils.data.DataLoader(touch_testing_subdataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    train_sampler = DistributedSampler(touch_vision_paired_training_subdataset, drop_last=True)
    touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(
        touch_vision_paired_training_subdataset, 
        batch_size=1, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
    )

    test_sampler = DistributedSampler(touch_testing_subdataset, shuffle=False, drop_last=False)
    touch_testing_dataloader = torch.utils.data.DataLoader(
        touch_testing_subdataset, 
        batch_size=1, 
        sampler=test_sampler,
        num_workers=4, 
        pin_memory=True
    )

    metrics_sampler = DistributedSampler(touch_vision_paired_training_subdataset_for_metrics, shuffle=False, drop_last=True)
    touch_vision_paired_training_subdataloader_for_metrics = torch.utils.data.DataLoader(
        touch_vision_paired_training_subdataset_for_metrics, 
        batch_size=1, 
        sampler=metrics_sampler,
        num_workers=4, 
        pin_memory=True
    )

    strategy = sys.argv[1]
    results = {}

    if local_rank == 0:
        print(f"\n{'='*20} Testing Strategy: {strategy} {'='*20}")
        logger = wandb.init(project="tactile_zero_shot_test", name=f"strategy_{strategy}_seed_{seed}", reinit=True)
    
    model = initialize_touch_model(
        imagebind_model, 
        init_strategy=strategy, 
        noise_std=0.002, 
        freeze_vision=False, 
        seed=seed
    ).to(device)
    # modality_preprocessors
    # modality_trunks
    # modality_heads
    # modality_postprocessors
    # for component in [model.modality_preprocessors, model.modality_trunks, model.modality_heads, model.modality_postprocessors]:
    #     component[ModalityType.VISION].to(device)
    #     component[ModalityType.TOUCH].to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    initial_weights_cpu = {}
    if is_main_process:
        initial_weights_cpu = {
            name: param.detach().cpu().clone() 
            for name, param in model.module.named_parameters() 
            if param.requires_grad
        }

    dist.barrier(device_ids=[local_rank])
    
    # Step C: Post-alignment Training (InfoNCE)
    if local_rank == 0:
        print("--- Running Post-alignment Training ---")

    model, performance_history = align(model, touch_vision_paired_training_dataloader, device, epochs=10, local_rank=local_rank, 
                                       eval_dataloader=touch_testing_dataloader, text_features=text_features, evaluate_fn=evaluate, 
                                       paired_subdataloader=touch_vision_paired_training_subdataloader_for_metrics, 
                                       imagenet_train_loader=imagenet_train_loader, imagenet_val_loader=imagenet_val_loader, precomputed_imagenet_loader=precomputed_imagenet_loader, 
                                       initial_weights_cpu = initial_weights_cpu,  
                                       logger=logger if local_rank == 0 else None, strategy_name=strategy, seed=seed)
    # Step D: 評估 Final Performance
    if local_rank == 0:
        print("--- Evaluating Final Performance ---")
        
    final_acc = evaluate(model, touch_testing_dataloader, text_features, device)
    final_imagenet_acc = evaluate_on_imagenet(imagenet_train_loader, imagenet_val_loader, model, device)

    if local_rank == 0:
        results[strategy] = {"Final_Acc": final_acc, "Final_Imagenet_Acc": final_imagenet_acc, "Performance_History": performance_history}
        logger.log({"final_accuracy": final_acc, "final_imagenet_accuracy": final_imagenet_acc})

    dist.barrier(device_ids=[local_rank])

    if local_rank == 0:
        os.makedirs("results", exist_ok=True)
        with open(f"results/result_{strategy}_{seed}.json", "w") as f:
            json.dump(results, f)

    dist.barrier(device_ids=[local_rank])

    del model
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()
