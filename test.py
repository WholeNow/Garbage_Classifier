import os
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import test_cfg, TestConfig
from utils import get_device, get_model, load_model
from dataset import create_dataloaders


def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    """
    Plots the confusion matrix.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        classes (list): List of class names.
        output_dir (str): Directory to save the confusion matrix image.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"[INFO] Confusion matrix saved to '{plot_path}'")

    # Check if running in a notebook
    try:
        if 'IPKernelApp' in get_ipython().config:
            plt.show()
    except NameError:
        pass
    except AttributeError:
        pass


def test(config: TestConfig = None):
    if config is None:
        config = test_cfg
    device = get_device(config.device)

    print(f"\n[TEST] Starting model testing of '{config.model_name}'")
    
    # Data loading
    _, _, test_loader = create_dataloaders(config)
    classes = test_loader.dataset.dataset.classes 
    
    # Model loading
    try:
        model = get_model(config.model_name, config.num_classes).to(device)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    # Load Checkpoint    
    ckpt_path = config.checkpoint_path    
    try:
        load_model(model, ckpt_path, device)
        print(f"[INFO] Model loaded from {ckpt_path}")
    except FileNotFoundError:
        print(f"[ERROR] File checkpoint '{ckpt_path}' not found.")
        return
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return

    # Inference
    model.eval()
    correct = 0
    total = 0
    results_per_image = []
    
    all_preds = []
    all_labels = []

    pbar = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Record per-image results
            batch_results = (predicted == labels).cpu().numpy().astype(int)
            results_per_image.extend(batch_results)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"[TEST] Test Set Accuracy: {accuracy:.2f}%")
    
    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Classifier")
    
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    parser.add_argument("--data_dir", type=str, help="Root directory for dataset", default=None)
    parser.add_argument("--model_name", type=str, help="Model name", default=None)
    parser.add_argument("--img_size", type=int, help="Image size", default=None)
    parser.add_argument("--compute_stats", type=bool, action="store_true", help="Compute dataset stats", default=None)
    parser.add_argument("--mean", type=float, nargs='+', help="Mean for normalization", default=None)
    parser.add_argument("--std", type=float, nargs='+', help="Std for normalization", default=None)
    parser.add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--num_workers", type=int, help="Number of data loader workers", default=None)
    parser.add_argument("--device", type=str, help="Device (cpu, cuda, mps, auto)", default=None)
    parser.add_argument("--checkpoint_path", type=str, help="Checkpoint filename", default=None)

    args = parser.parse_args()
    
    # Override config defaults
    if args.batch_size is not None: test_cfg.batch_size = args.batch_size
    if args.output_dir is not None: test_cfg.output_dir = args.output_dir
    if args.data_dir is not None: test_cfg.root_dir = args.data_dir
    if args.model_name is not None: test_cfg.model_name = args.model_name
    if args.img_size is not None: test_cfg.img_size = args.img_size
    if args.mean is not None: test_cfg.mean = args.mean
    if args.std is not None: test_cfg.std = args.std
    if args.seed is not None: test_cfg.seed = args.seed
    if args.num_workers is not None: test_cfg.num_workers = args.num_workers
    if args.device is not None: test_cfg.device = args.device
    if args.checkpoint_path is not None: test_cfg.checkpoint_path = args.checkpoint_path
        
    test(test_cfg)
