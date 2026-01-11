import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.pyplot as plt

# Models
from models.customs.GarbageCustom_1 import GC1

from config import Config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed impostato: {seed}")

def get_device(device_pref: str = "auto"):
    if device_pref != "auto":
        return torch.device(device_pref)
    
    if torch.cuda.is_available():
        print(f"[INFO] Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[INFO] Device: MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("[INFO] Device: CPU")
        return torch.device("cpu")

def get_model(model_name: str, input_channels: int, num_classes: int):
    """
    Factory per il caricamento dinamico del modello.
    """
    if model_name == "GC1":
        # GC1 ora accetta num_classes
        return GC1(input_size=input_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Modello '{model_name}' non riconosciuto.")

class GarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            # Fallback utile se lanciato da root invece che da Custom Models
            if os.path.exists('images'):
                 self.root_dir = 'images'
            elif os.path.exists(os.path.join('..', 'images')):
                 self.root_dir = os.path.join('..', 'images')
            else:
                 raise FileNotFoundError(f"Directory {root_dir} non trovata.")
            
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        self.data = []
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append((img_path, idx))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            # Load raw image to preserve channels
            image = Image.open(img_path)
            # Ensure it is at least converted to something tensor-able if not RGB specific
            # But for standard models we usually enforce RGB. 
            # If we want automatic, we should check mode.
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception:
            raise Exception(f"Impossibile aprire l'immagine: {img_path}")
            
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_targets(self):
        return [label for _, label in self.data]
    
    def detect_channels(self):
        """
        Detects the number of channels from the first image in the dataset.
        """
        if len(self.data) == 0:
            return 3 # Default
        img_path, _ = self.data[0]
        img = Image.open(img_path)
        # Handle modes: L=1, RGB=3, RGBA=4 (but we usually ignore alpha or convert to RGB)
        if img.mode == 'L':
            return 1
        return 3 # RGB default

def get_dataset_stats(dataset, batch_size=32, num_workers=0, img_size=256):
    """
    Calcola la media e la deviazione standard del dataset.
    """
    print("[INFO] Calcolo media e deviazione standard del dataset (può richiedere tempo)...")
    
    # Dataset temporaneo senza normalizzazione
    # We need to wrap the dataset to apply Resize and ToTensor ONLY for stats calculation
    # Since original dataset might have transform=None initially
    
    temp_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Create a temporary loader using the same data but with temp_transform
    # We can't just change self.transform on the fly safely if it's shared.
    # So we instantiate a lightweight wrapper or reuse GarbageDataset structure.
    # Easiest is to re-instantiate GarbageDataset with temp_transform using same root_dir
    
    stat_dataset = GarbageDataset(dataset.root_dir, transform=temp_transform)
    loader = DataLoader(stat_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    for images, _ in tqdm(loader, desc="Calcolo Stats", leave=False):
        batch_samples = images.size(0) # batch size
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    print(f"[INFO] Calcolato -> Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

def create_dataloaders(config):
    # 1. Instantiate basic dataset to detect config properties
    temp_ds = GarbageDataset(root_dir=config.root_dir, transform=None)
    config.num_classes = len(temp_ds.classes)
    
    # Detect channels (simple approach: check first image)
    config.input_channels = temp_ds.detect_channels()
    print(f"[INFO] Canali Input Rilevati: {config.input_channels}")

    # 2. Stats
    if config.compute_stats:
        mean, std = get_dataset_stats(temp_ds, config.batch_size, config.num_workers, config.img_size)
        config.mean = mean
        config.std = std
    
    # 3. Final Transform
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std) 
    ])

    full_dataset = GarbageDataset(root_dir=config.root_dir, transform=transform)
    
    targets = full_dataset.get_targets()
    dataset_indices = list(range(len(full_dataset)))
    
    # Split 1: Test vs (Train+Val)
    train_val_idx, test_idx = train_test_split(
        dataset_indices, test_size=config.test_split, random_state=config.seed, stratify=targets
    )
    
    # Split 2: Train vs Val
    train_val_targets = [targets[i] for i in train_val_idx]
    
    # Fix potential zero division or empty split if dataset is small
    if len(train_val_idx) == 0:
        raise ValueError("Dataset too small for the requested splits.")

    relative_val_split = config.val_split / (1.0 - config.test_split)
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=relative_val_split, random_state=config.seed, stratify=train_val_targets
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    print(f"[INFO] Dati -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, test_loader

def save_model(model, path: str, optimizer=None, val_acc=None, epoch=None):
    state = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if val_acc is not None:
        state['val_acc'] = val_acc
    if epoch is not None:
        state['epoch'] = epoch
    torch.save(state, path)

def load_model(model, path: str, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
