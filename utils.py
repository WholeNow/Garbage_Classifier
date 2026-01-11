import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Models
from models.customs.GarbageCustom_1 import GC1
from models.customs.GarbageCustom_2 import GC2
from models.Xception import Xception

def set_seed(seed: int):
    """
    Set seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set: {seed}")


def get_device(device_pref: str = "auto"):
    """
    Get device for training.

    Args:
        device_pref (str): Device preference ("auto", "cuda", "mps", "cpu").

    Returns:
        torch.device: Device object.
    """
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


def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Get model instance.

    Args:
        model_name (str): Model name.
        num_classes (int): Number of classes.

    Returns:
        torch.nn.Module: Model instance.
    """
    if model_name is None:
        raise ValueError("Model name must be specified.")
    
    if model_name == "GC1":
        return GC1(num_classes=num_classes)
    if model_name == "GC2":
        return GC2(num_classes=num_classes)
    if model_name == "Xception":
        return Xception(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")


def save_model(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer = None, val_acc: float = None, epoch: int = None):
    """
    Save model state.

    Args:
        model (torch.nn.Module): Model to save.
        path (str): Path to save the model.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save. Defaults to None.
        val_acc (float, optional): Validation accuracy to save. Defaults to None.
        epoch (int, optional): Current epoch. Defaults to None.
    """
    state = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if val_acc is not None:
        state['val_acc'] = val_acc
    if epoch is not None:
        state['epoch'] = epoch
    torch.save(state, path)


def load_model(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """
    Load model state.

    Args:
        model (torch.nn.Module): Model to load.
        path (str): Path to load the model from.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint