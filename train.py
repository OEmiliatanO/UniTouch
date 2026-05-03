import glob
from YCB_slide_dataset import YCBSlidePairedDataset, YCBSlidedPairedDataset_precomputed_vision, YCBSlideDataset
from touch_and_go_dataset import TouchAndGoPairedDataset, TouchAndGoDataset_precomputed_vision, TouchAndGoDataset
import torch
import torch.nn.functional as F
import numpy as np
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import ModalityType
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

from utils import cka, mknn, set_seed

import argparse
import datetime

standard_data_transform = transforms.Compose(
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

def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank

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

def prune_unused_modalities(model, keep_modalities):
    target_attrs = [
        'modality_preprocessors', 
        'modality_trunks', 
        'modality_heads', 
        'modality_postprocessors'
    ]

    for attr in target_attrs:
        if hasattr(model, attr):
            module_dict = getattr(model, attr)
            keys_to_remove = [k for k in module_dict.keys() if k not in keep_modalities]
            for k in keys_to_remove:
                del module_dict[k]
    
    if hasattr(model, 'point_trunk') and ModalityType.POINT not in keep_modalities:
        del model.point_trunk
        model.point_trunk = None
    
    return model

@torch.no_grad()
def initialize_touch_model(init_strategy="random", freeze_vision=True, noise_std=0.002, seed=0):
    def get_components(model, modality):
        return [
            model.modality_preprocessors[modality],
            model.modality_trunks[modality],
            model.modality_heads[modality],
            model.modality_postprocessors[modality]
        ]

    if init_strategy == "random":
        from ImageBind.models.x2touch_model_part import x2touch
        set_seed(seed)
        new_touch_model = x2touch(pretrained=False).cpu()
        new_touch_model.requires_grad_(False)

        touch_components = get_components(new_touch_model, ModalityType.TOUCH)
        touchs_vision_components = get_components(new_touch_model, ModalityType.VISION)

        for touch_component, touchs_vision_component in zip(touch_components, touchs_vision_components):
            for name, param in touch_component.named_parameters():
                param.requires_grad = True
            for name, param in touchs_vision_component.named_parameters():
                param.requires_grad = not freeze_vision

        return new_touch_model

    elif init_strategy in ["vision_clean", "vision_noise"]:
        from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch
        imagebind_model = imagebind_huge(pretrained=True).cpu()
        imagebind_model = prune_unused_modalities(imagebind_model, keep_modalities=[ModalityType.VISION])
        imagebind_model.eval()
        imagebind_model.requires_grad_(True)

        g = torch.Generator(device='cpu').manual_seed(seed)
        set_seed(seed)
        
        new_touch_model = x2touch(pretrained=False)
        new_touch_model.requires_grad_(False)

        touch_components = get_components(new_touch_model, ModalityType.TOUCH)
        touchs_vision_components = get_components(new_touch_model, ModalityType.VISION)
        vision_components = get_components(imagebind_model, ModalityType.VISION)

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

        keep_mods = [ModalityType.VISION, ModalityType.TOUCH]
        new_touch_model = prune_unused_modalities(new_touch_model, keep_mods)

        return new_touch_model

    elif init_strategy == "unitouch":
        from ImageBind.models.x2touch_model_part_original import x2touch
        new_touch_model = x2touch(pretrained=True).cpu()

        touch_components = get_components(new_touch_model, ModalityType.TOUCH)
        touchs_vision_components = get_components(new_touch_model, ModalityType.VISION)
        for touch_component, touchs_vision_component in zip(touch_components, touchs_vision_components):
            for name, param in touch_component.named_parameters():
                param.requires_grad = True
            for name, param in touchs_vision_component.named_parameters():
                param.requires_grad = not freeze_vision

        keep_mods = [ModalityType.VISION, ModalityType.TOUCH]
        new_touch_model = prune_unused_modalities(new_touch_model, keep_mods)
        return new_touch_model

@torch.no_grad()
def material_classification_evaluate(model, dataloader, text_features, device):
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.eval()
    
    local_correct = 0
    local_total = 0
    
    for batch in tqdm(dataloader, desc="Evaluating", disable=dist.get_rank() != 0, leave=False):
        touch_data, labels = batch
        touch_data = touch_data.to(device)
        labels = labels.to(device)
        
        outputs = actual_model({ModalityType.TOUCH: touch_data}) 
        touch_features = outputs[ModalityType.TOUCH] 
        
        text_features_norm = F.normalize(text_features, dim=-1)
        touch_features = F.normalize(touch_features, dim=-1)
        
        batch_preds = (touch_features @ text_features_norm.T).argmax(dim=-1)
        
        local_correct += (batch_preds == labels).sum().item()
        local_total += labels.size(0)
        
    # shape: [2] -> [correct_count, total_count]
    metrics = torch.tensor([local_correct, local_total], dtype=torch.float32, device=device)
    
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    global_correct = metrics[0].item()
    global_total = metrics[1].item()
    
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

    if is_main_process:
        print(f" Final ImageNet-1k Training Loss: {global_train_loss/global_train_total:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f" Final ImageNet-1k Validation Accuracy: {val_acc:.4f} ({global_val_correct}/{global_val_total})")
    
    return val_acc

@torch.no_grad()
def evaluate_with_metrics(model, paired_dataloader, device, args):
    actual_model = model.module if isinstance(model, DDP) else model
    actual_model.eval()

    local_touch_features = []
    local_vision_features = []

    is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)

    for batch in tqdm(paired_dataloader, desc="Extracting Features", disable=not is_main_process, leave=False):
        if args.vision_inference:
            (touch_images, vision_images), _ = batch
            touch_images = touch_images.to(device)
            vision_images = vision_images.to(device)

            outputs = actual_model({ModalityType.TOUCH: touch_images, ModalityType.VISION: vision_images})

            touch_outputs = outputs[ModalityType.TOUCH]
            vision_outputs = outputs[ModalityType.VISION]
        else:
            (touch_images, vision_features), _ = batch
            touch_images = touch_images.to(device)

            outputs = actual_model({ModalityType.TOUCH: touch_images})

            touch_outputs = outputs[ModalityType.TOUCH]
            vision_outputs = vision_features.to(device)

        touch_outputs = F.normalize(touch_outputs, dim=-1)
        vision_outputs = F.normalize(vision_outputs, dim=-1)

        local_touch_features.append(touch_outputs)
        local_vision_features.append(vision_outputs)
    
    local_touch_features = torch.cat(local_touch_features, dim=0)
    local_vision_features = torch.cat(local_vision_features, dim=0)

    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        gathered_touch = [torch.zeros_like(local_touch_features) for _ in range(world_size)]
        gathered_vision = [torch.zeros_like(local_vision_features) for _ in range(world_size)]
        
        dist.all_gather(gathered_touch, local_touch_features)
        dist.all_gather(gathered_vision, local_vision_features)
        
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
          logger=None, strategy_name=None, seed=None, save_dir="./results",
          args=None
         ):

    touch_model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, touch_model.parameters()), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    is_main_process = (local_rank == 0)

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
        sim_metrics = evaluate_with_metrics(touch_model, paired_subdataloader, device, args)
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

    progress_bar = tqdm(total=epochs, desc="Aligning Models (InfoNCE Training)", disable=not is_main_process)
    for epoch in range(1, epochs+1):
        if hasattr(paired_dataloader.sampler, "set_epoch"):
            paired_dataloader.sampler.set_epoch(epoch)
        
        touch_model.train()
        tot_loss = 0
        tot_alignment = 0
        tot_uniformity = 0

        for batch in tqdm(paired_dataloader, disable=not is_main_process, leave=False):
            optimizer.zero_grad()

            if args.vision_inference:
                (touch_images, vision_images), _ = batch
                touch_images = touch_images.to(device)
                vision_images = vision_images.to(device)

                outputs = touch_model({ModalityType.TOUCH: touch_images, ModalityType.VISION: vision_images})
                
                batch_vision_features = outputs[ModalityType.VISION]
                batch_touch_features = outputs[ModalityType.TOUCH]
            else:
                (touch_images, vision_features), _ = batch
                vision_features = vision_features.to(device)

                outputs = touch_model({ModalityType.TOUCH: touch_images})

                batch_vision_features = vision_features
                batch_touch_features = outputs[ModalityType.TOUCH]

            temperature = 0.07
            
            batch_vision_features = F.normalize(batch_vision_features, dim=-1)
            batch_touch_features = F.normalize(batch_touch_features, dim=-1)

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

            imagenet_loss = 0
            if args.preserver_imagenet_features:
                imagenet_batch = next(precomputed_imagenet_iter)
                imagenet_images, precomputed_imagenet_features =  imagenet_batch["image"], imagenet_batch["vision_feature"]
                imagenet_images = imagenet_images.to(device)
                imagenet_features = touch_model({ModalityType.TOUCH: imagenet_images})[ModalityType.TOUCH]
                precomputed_imagenet_features = precomputed_imagenet_features.to(device)
                imagenet_loss = F.mse_loss(imagenet_features, precomputed_imagenet_features)
            
            total_loss = alignment_loss + imagenet_loss

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
            sim_metrics = evaluate_with_metrics(touch_model, paired_subdataloader, device, args)
            cka, mknn = sim_metrics["cka"].item(), sim_metrics["mknn"].item()

        if is_main_process:
            print(f" loss: {global_avg_loss:.4f}, accuracy: {epoch_acc:.4f}, imagenet_acc: {epoch_imagenet_acc:.4f}, cka: {cka:.4f}, mknn: {mknn:.4f}")

            logger.log({"epoch/epoch": epoch, "epoch/loss": global_avg_loss, "epoch/accuracy": epoch_acc, "epoch/imagenet_accuracy": epoch_imagenet_acc, "epoch/cka": cka, "epoch/mknn": mknn})
            
            performance_history["loss"].append(global_avg_loss)
            performance_history["accuracy"].append(epoch_acc)
            performance_history["imagenet_accuracy"].append(epoch_imagenet_acc)
            performance_history["cka"].append(cka)
            performance_history["mknn"].append(mknn)
            
            progress_bar.set_postfix({"Loss": f"{global_avg_loss:.4f}", "Acc": f"{epoch_acc:.4f}", "Imagenet Acc": f"{epoch_imagenet_acc:.4f}", "CKA": f"{cka:.4f}", "mKNN": f"{mknn:.4f}"})
            progress_bar.update(1)

        dist.barrier(device_ids=[local_rank])

    if is_main_process:
        model_to_save = touch_model.module if isinstance(touch_model, DDP) else touch_model
        os.makedirs(f"{save_dir}/ckpts", exist_ok=True)
        torch.save(model_to_save.state_dict(), f"{save_dir}/ckpts/touch_model.pth")

    return touch_model, performance_history

def prepare_imagenet_dataloader(args, batch_size=16):
    imagenet_hf_dir = "imagenet-1k-hf"

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

    dataset = load_from_disk(imagenet_hf_dir)
    train_dataset = dataset['train'].select(range(5000)).with_transform(preprocess_train)
    val_dataset = dataset['validation'].select(range(5000)).with_transform(preprocess_val)

    def collate_fn(batch):
        images = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return images, labels
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, collate_fn=collate_fn)
    return train_loader, val_loader

def prepare_precomputed_imagenet_dataloader(args, batch_size=16):
    def preprocess_batch(batch):
        batch["image"] = [standard_data_transform(img.convert("RGB")) for img in batch["image"]]
        batch["vision_feature"] = [torch.tensor(feat) for feat in batch["vision_feature"]]
        batch["label"] = [torch.tensor(lbl) for lbl in batch["label"]]
        return batch
    
    precomputed_imagenet_dir = "imagenet_with_features"

    dataset = load_from_disk(precomputed_imagenet_dir)
    dataset.set_transform(preprocess_batch)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    return dataloader

def main(args):
    seed = args.seed
    local_rank = setup_ddp()
    is_main_process = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}")

    save_dir = f"results/{args.exp_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_strategy_{args.strategy}_dataset_{args.dataset}_seed_{seed}_freeze_vision_{args.freeze_vision}_preserve_imagenet_{args.preserver_imagenet_features}"

    # Prepare ImageNet dataloaders
    imagenet_train_loader, imagenet_val_loader = prepare_imagenet_dataloader(args, batch_size=args.imagenet_testing_batch_size)
    precomputed_imagenet_loader = prepare_precomputed_imagenet_dataloader(args, batch_size=args.imagenet_testing_batch_size)

    # Prepare tactile dataset and dataloader
    text_features = torch.load("touch_and_go/touch_and_go_text_features.pt").to(device) # Shape: [C, 1024]

    if args.vision_inference:
        if args.dataset == "ycb_slide":
            touch_vision_paired_training_dataset = YCBSlidePairedDataset("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/YCB-Slide_vision_training_data.csv", transform=standard_data_transform)
        elif args.dataset == "touch_and_go":
            touch_vision_paired_training_dataset = TouchAndGoPairedDataset("touch_and_go", mode="train", transform=standard_data_transform)
    else:
        if args.dataset == "ycb_slide":
            touch_vision_paired_training_dataset = YCBSlidedPairedDataset_precomputed_vision("YCB-Slide_dataset_path/YCB-Slide_touch_training_data.csv", "YCB-Slide_dataset_path/precomputed_training_vision_features.pt", transform=standard_data_transform)
        elif args.dataset == "touch_and_go":
            touch_vision_paired_training_dataset = TouchAndGoDataset_precomputed_vision("touch_and_go", "touch_and_go/precomputed_training_vision_features.pt", mode="train", transform=standard_data_transform)
    if args.dataset == "ycb_slide":
        touch_testing_dataset = YCBSlideDataset("YCB-Slide_dataset_path/YCB-Slide_touch_testing_data.csv", transform=standard_data_transform)
    elif args.dataset == "touch_and_go":
        touch_testing_dataset = TouchAndGoDataset("touch_and_go", mode="test", transform=standard_data_transform)

    if args.debug:
        touch_vision_paired_training_subdataset = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, 1000))
        touch_vision_paired_training_subdataset_for_metrics = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, 1000))
        touch_testing_subdataset = torch.utils.data.Subset(touch_testing_dataset, indices=range(0, 1000))
    else:
        touch_vision_paired_training_subdataset = touch_vision_paired_training_dataset
        touch_vision_paired_training_subdataset_for_metrics = torch.utils.data.Subset(touch_vision_paired_training_dataset, indices=range(0, 3000))
        touch_testing_subdataset = touch_testing_dataset

    train_sampler = DistributedSampler(touch_vision_paired_training_subdataset, shuffle=True, drop_last=True)
    touch_vision_paired_training_dataloader = torch.utils.data.DataLoader(
        touch_vision_paired_training_subdataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True,
    )

    test_sampler = DistributedSampler(touch_testing_subdataset, shuffle=False, drop_last=False)
    touch_testing_dataloader = torch.utils.data.DataLoader(
        touch_testing_subdataset, 
        batch_size=args.testing_batch_size, 
        sampler=test_sampler,
        num_workers=4, 
        pin_memory=True
    )

    metrics_sampler = DistributedSampler(touch_vision_paired_training_subdataset_for_metrics, shuffle=False, drop_last=True)
    touch_vision_paired_training_subdataloader_for_metrics = torch.utils.data.DataLoader(
        touch_vision_paired_training_subdataset_for_metrics, 
        batch_size=args.testing_batch_size, 
        sampler=metrics_sampler,
        num_workers=4, 
        pin_memory=True
    )

    strategy = args.strategy
    results = {}

    if local_rank == 0:
        print(f"\n{'='*20} Testing Strategy: {strategy} {'='*20}")
        logger = wandb.init(project="tactile_zero_shot_test", name=f"{args.exp_name}_strategy_{args.strategy}_seed_{seed}", reinit=True)
    
    model = initialize_touch_model(
        init_strategy=strategy, 
        noise_std=0.002, 
        freeze_vision=args.freeze_vision, 
        seed=seed
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # for calucating weight drift
    initial_weights_cpu = {}
    if is_main_process:
        initial_weights_cpu = {
            name: param.detach().cpu().clone() 
            for name, param in model.module.named_parameters() 
            if param.requires_grad
        }

    dist.barrier(device_ids=[local_rank])
    
    # Post-alginment training
    if local_rank == 0:
        print("--- Running Post-alignment Training ---")

    model, performance_history = align(model, touch_vision_paired_training_dataloader, device, epochs=args.epochs, local_rank=local_rank, 
                                       eval_dataloader=touch_testing_dataloader, text_features=text_features, evaluate_fn=material_classification_evaluate, 
                                       paired_subdataloader=touch_vision_paired_training_subdataloader_for_metrics, 
                                       imagenet_train_loader=imagenet_train_loader, imagenet_val_loader=imagenet_val_loader, precomputed_imagenet_loader=precomputed_imagenet_loader, 
                                       initial_weights_cpu = initial_weights_cpu,  
                                       logger=logger if local_rank == 0 else None, strategy_name=strategy, seed=seed, save_dir=save_dir, args=args)
    
    # Evaluating final performance
    if local_rank == 0:
        print("--- Evaluating Final Performance ---")
        
    final_material_acc = material_classification_evaluate(model, touch_testing_dataloader, text_features, device)
    final_imagenet_acc = evaluate_on_imagenet(imagenet_train_loader, imagenet_val_loader, model, device)

    if local_rank == 0:
        results[strategy] = {"Final_Acc": final_material_acc, "Final_Imagenet_Acc": final_imagenet_acc, "Performance_History": performance_history}
        logger.log({"final_accuracy": final_material_acc, "final_imagenet_accuracy": final_imagenet_acc})

    dist.barrier(device_ids=[local_rank])

    if local_rank == 0:
        with open(f"{save_dir}/results.json", "w") as f:
            json.dump(results, f)

    dist.barrier(device_ids=[local_rank])

    del model
    torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Touch Model with Different Initialization Strategies")
    parser.add_argument("--strategy", type=str, required=True, choices=["random", "vision_clean", "vision_noise", "unitouch"], help="Initialization strategy for the touch model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--freeze_vision", action="store_true", help="Whether to freeze vision components during training")
    parser.add_argument("--vision_inference", action="store_true", help="Whether to use the vision branch for inference")
    parser.add_argument("--preserver_imagenet_features", action="store_true", help="Whether to use precomputed ImageNet features for an additional loss term")
    parser.add_argument("--exp_name", type=str, default="tactile_zero_shot_test", help="WandB experiment name")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode with fewer epochs and smaller dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for alignment training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for alignment training")
    parser.add_argument("--testing_batch_size", type=int, default=32, help="Batch size for testing dataloader")
    parser.add_argument("--imagenet_testing_batch_size", type=int, default=64, help="Batch size for ImageNet testing dataloader")
    parser.add_argument("--dataset", type=str, default="touch_and_go", choices=["touch_and_go", "ycb_slide"], help="Which tactile dataset to use for training and evaluation")

    args = parser.parse_args()

    main(args)
