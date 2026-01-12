import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from config import train_cfg, TrainConfig
from utils import set_seed, get_device, get_model, save_model
from dataset import create_dataloaders


def plot_training_metrics(
    train_losses: list, 
    val_losses: list, 
    train_accs: list, 
    val_accs: list, 
    val_steps: list, 
    output_dir: str
):
    """
    Plots training and validation metrics.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        train_accs (list): List of training accuracies.
        val_accs (list): List of validation accuracies.
        val_steps (list): List of validation steps.
        output_dir (str): Directory to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Train vs Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_steps, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    
    # Train vs Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    if val_accs:
        plt.plot(val_steps, val_accs, label='Validation Acc', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    print(f"[INFO] Training metrics saved to '{plot_path}'")
    
    # Check if running in a notebook
    try:
        if 'IPKernelApp' in get_ipython().config:
            plt.show()
    except NameError:
        pass
    except AttributeError:
        pass


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int
):
    """
    Validate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run the validation on.
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.

    Returns:
        tuple: A tuple containing the average validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]" if num_epochs > 0 else f"Epoch {epoch} [Val]", leave=False)
    
    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_pbar.set_postfix({'loss': loss.item()})
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = 100 * val_correct / val_total
    
    return avg_val_loss, avg_val_acc


def train(config: TrainConfig = None):
    """
    Trains the model on the training set and validates on the validation set.

    Args:
        config (TrainConfig): The configuration object.
    """
    if config is None:
        config = train_cfg
    set_seed(config.seed)
    device = get_device(config.device)

    print(f"[TRAIN] Starting training of {config.model_name}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Data loading
    train_loader, val_loader, _ = create_dataloaders(config)
    
    # Model loading
    try:
        model = get_model(config.model_name, config.num_classes, pretrained=getattr(config, "pretrained", True)).to(device)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    
    # Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Scheduler Step Decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    
    best_val_acc = 0.0
    last_val_acc = 0.0
    last_epoch = 0
    last_best_path = ""
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    val_epochs_list = []
    
    try:
        for epoch in range(1, config.num_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs} [Train]", leave=False)
            
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                l1_reg = sum(p.abs().sum() for p in model.parameters())
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + config.l1_lambda * l1_reg + config.l2_lambda * l2_reg

                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            scheduler.step()
                
            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_train_acc = 100 * train_correct / train_total
            
            train_loss_history.append(avg_train_loss)
            train_acc_history.append(avg_train_acc)

            if epoch % config.val_epochs == 0:
                avg_val_loss, avg_val_acc = validate(model, val_loader, criterion, device, epoch, config.num_epochs)
                
                val_loss_history.append(avg_val_loss)
                val_acc_history.append(avg_val_acc)
                val_epochs_list.append(epoch)

                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
            
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc

                    if last_best_path and os.path.exists(last_best_path):
                        os.remove(last_best_path)

                    checkpoint_path = os.path.join(config.output_dir, config.checkpoint_path.replace(".pth", f"_{avg_val_acc:.2f}_{epoch}.pth"))
                    save_model(model, checkpoint_path, optimizer, avg_val_acc, epoch)
                    print(f"--> Best Model Saved (Acc: {best_val_acc:.2f}%) at {checkpoint_path}")

                    last_best_path = checkpoint_path

                last_val_acc = avg_val_acc
                last_epoch = epoch
            else:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
        
    except KeyboardInterrupt:
            print("\n[TRAIN] Training interrupted by user (Ctrl+C).")
            
    plot_training_metrics(train_loss_history, val_loss_history, train_acc_history, val_acc_history, val_epochs_list, config.output_dir)

    last_model_path = os.path.join(config.output_dir, config.checkpoint_path)
    save_model(model, last_model_path, optimizer, last_val_acc, last_epoch)
    print(f"--> Last Model Saved at {last_model_path}")
    print("[TRAIN] Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classifier")
    
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--lr", type=float, help="Learning rate", default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory", default=None)
    parser.add_argument("--data_dir", type=str, help="Root directory for dataset", default=None)
    parser.add_argument("--model_name", type=str, help="Model name", default=None)
    parser.add_argument("--val_epochs", type=int, help="Validation frequency (epochs)", default=None)
    parser.add_argument("--step_size", type=int, help="Scheduler step size", default=None)
    parser.add_argument("--gamma", type=float, help="Scheduler gamma", default=None)
    parser.add_argument("--l1_lambda", type=float, help="L1 regularization lambda", default=None)
    parser.add_argument("--l2_lambda", type=float, help="L2 regularization lambda", default=None)
    parser.add_argument("--val_split", type=float, help="Validation split ratio", default=None)
    parser.add_argument("--test_split", type=float, help="Test split ratio", default=None)
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
    if args.epochs is not None: train_cfg.num_epochs = args.epochs
    if args.batch_size is not None: train_cfg.batch_size = args.batch_size
    if args.lr is not None: train_cfg.learning_rate = args.lr
    if args.output_dir is not None: train_cfg.output_dir = args.output_dir
    if args.data_dir is not None: train_cfg.root_dir = args.data_dir
    if args.model_name is not None: train_cfg.model_name = args.model_name
    if args.val_epochs is not None: train_cfg.val_epochs = args.val_epochs
    if args.step_size is not None: train_cfg.step_size = args.step_size
    if args.gamma is not None: train_cfg.gamma = args.gamma
    if args.l1_lambda is not None: train_cfg.l1_lambda = args.l1_lambda
    if args.l2_lambda is not None: train_cfg.l2_lambda = args.l2_lambda
    if args.val_split is not None: train_cfg.val_split = args.val_split
    if args.test_split is not None: train_cfg.test_split = args.test_split
    if args.img_size is not None: train_cfg.img_size = args.img_size
    if args.compute_stats is not None: train_cfg.compute_stats = args.compute_stats
    if args.mean is not None: train_cfg.mean = args.mean
    if args.std is not None: train_cfg.std = args.std
    if args.seed is not None: train_cfg.seed = args.seed
    if args.num_workers is not None: train_cfg.num_workers = args.num_workers
    if args.device is not None: train_cfg.device = args.device
    if args.checkpoint_path is not None: train_cfg.checkpoint_path = args.checkpoint_path
        
    train(train_cfg)