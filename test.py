import os
import shutil
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

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


def plot_precision_recall_curve(y_true, y_scores, classes, output_dir):
    """
    Plots the Precision-Recall curve for multiclass data.

    Args:
        y_true (list): True labels.
        y_scores (list or np.array): Predicted probabilities for each class.
        classes (list): List of class names.
        output_dir (str): Directory to save the precision-recall curve image.
    """
    # Binarize the output for multiclass
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    y_scores = np.array(y_scores)

    plt.figure(figsize=(10, 8))
    
    # Plot Precision-Recall curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {classes[i]}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Multiclass)')
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(plot_path)
    print(f"[INFO] Precision-Recall curve saved to '{plot_path}'")

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
    all_probs = []

    # Directory for wrong classifications
    wrong_classified_dir = os.path.join(config.output_dir, "wrong_classified")
    if config.save_wrong_images:
        os.makedirs(wrong_classified_dir, exist_ok=True)
        print(f"[INFO] Saving wrong classified images to '{wrong_classified_dir}'")
    
    global_idx = 0 # To track index in dataset

    pbar = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            batch_size = labels.size(0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Record per-image results
            batch_results = (predicted == labels).cpu().numpy().astype(int)
            results_per_image.extend(batch_results)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Save wrong classified images
            if config.save_wrong_images:
                # Move to CPU for processing
                preds_cpu = predicted.cpu().numpy()
                labels_cpu = labels.cpu().numpy()
                
                for i in range(batch_size):
                    if preds_cpu[i] != labels_cpu[i]:
                        # Get original image path
                        # Assuming test_loader is a Subset, we access the underlying dataset
                        dataset_idx = test_loader.dataset.indices[global_idx + i]
                        img_path, _ = test_loader.dataset.dataset.data[dataset_idx]
                        img_filename = os.path.basename(img_path)
                        
                        true_class = classes[labels_cpu[i]]
                        pred_class = classes[preds_cpu[i]]
                        
                        new_filename = f"True_{true_class}_Pred_{pred_class}_{img_filename}"
                        new_path = os.path.join(wrong_classified_dir, new_filename)
                        
                        try:
                            shutil.copy(img_path, new_path)
                        except Exception as e:
                            print(f"[WARN] Could not copy image {img_path}: {e}")

            global_idx += batch_size

    accuracy = 100 * correct / total
    f1_score_value = f1_score(all_labels, all_preds, average='weighted')

    print(f"[TEST] Test Set Accuracy: {accuracy:.2f}%")
    print(f"[TEST] Test Set F1 Score: {f1_score_value:.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    # Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, classes, config.output_dir)

    # Precision-Recall Curve
    plot_precision_recall_curve(all_labels, all_probs, classes, config.output_dir)

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
    parser.add_argument("--save_wrong_images", action="store_true", help="Save wrong classified images", default=None)

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
    if args.save_wrong_images is not None: test_cfg.save_wrong_images = args.save_wrong_images
        
    test(test_cfg)
