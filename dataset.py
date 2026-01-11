import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm.auto import tqdm


class ClassificationDataset(Dataset):
    """
    Classification dataset class.

    Args:
        root_dir (str): Root directory of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.exists(root_dir):
            if os.path.exists('images'):
                 self.root_dir = 'images'
            elif os.path.exists(os.path.join('..', 'images')):
                 self.root_dir = os.path.join('..', 'images')
            else:
                 raise FileNotFoundError(f"Directory {root_dir} not found.")
            
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
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception:
            raise Exception(f"Image not found: {img_path}")
            
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_targets(self):
        return [label for _, label in self.data]


def get_dataset_stats(dataset: Dataset, batch_size: int, num_workers: int, img_size: int):
    """
    Calculate the mean and standard deviation of the dataset.

    Args:
        dataset (Dataset): Dataset object.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        img_size (int): Image size.

    Returns:
        Tuple[List[float], List[float]]: Tuple containing the mean and standard deviation of the dataset.
    """
    print("[INFO] Calculating mean and standard deviation of the dataset...")
    
    temp_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    stat_dataset = type(dataset)(dataset.root_dir, transform=temp_transform)
    loader = DataLoader(stat_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    for images, _ in tqdm(loader, desc="Stats", leave=False):
        batch_samples = images.size(0) # batch size
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    print(f"[INFO] Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.tolist(), std.tolist()


def create_dataloaders(config):
    """
    Create dataloaders for training, validation and testing.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple containing the train, validation and test dataloaders.
    """

    # Detect config properties
    temp_ds = ClassificationDataset(root_dir=config.root_dir, transform=None)
    config.num_classes = len(temp_ds.classes)

    # Stats
    if config.compute_stats:
        mean, std = get_dataset_stats(temp_ds, config.batch_size, config.num_workers, config.img_size)
        config.mean = mean
        config.std = std
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std) 
    ])

    full_dataset = ClassificationDataset(root_dir=config.root_dir, transform=transform)
    
    targets = full_dataset.get_targets()
    dataset_indices = list(range(len(full_dataset)))
    
    # Split Test vs (Train+Val)
    train_val_idx, test_idx = train_test_split(
        dataset_indices, test_size=config.test_split, random_state=config.seed, stratify=targets
    )
    
    # Split Train vs Val
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

    print(f"[INFO] Dataset -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    return train_loader, val_loader, test_loader