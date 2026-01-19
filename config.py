from typing import List


class TrainConfig:
    def __init__(
        self,
        root_dir: str,
        model_name: str,
        pretrained: bool,
        batch_size: int,
        num_epochs: int,
        val_epochs: int,
        learning_rate: float,
        step_size: int,
        gamma: float,
        l1_lambda: float,
        l2_lambda: float, 
        val_split: float,
        test_split: float,
        img_size: int,
        compute_stats: bool,
        mean: List[float],
        std: List[float],
        seed: int,
        num_workers: int,
        device: str,
        checkpoint_path: str,
        output_dir: str = "out"
    ):
        # Paths and Model
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_epochs = val_epochs
        self.learning_rate = learning_rate
        
        # Scheduler (Step Decay)
        self.step_size = step_size
        self.gamma = gamma
        
        # Regularization
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Split Dataset
        self.val_split = val_split
        self.test_split = test_split
        
        # Input Images
        self.img_size = img_size
        
        # Normalization
        self.compute_stats = compute_stats
        self.mean = mean
        self.std = std
        
        # System
        self.seed = seed
        self.num_workers = num_workers
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Runtime (Automatic detection)
        self.num_classes = 0
        self.input_channels = 3

    def __repr__(self):
        return (f"TrainConfig(root_dir='{self.root_dir}', model_name='{self.model_name}', "
                f"batch_size={self.batch_size}, num_epochs={self.num_epochs}, "
                f"lr={self.learning_rate}, device='{self.device}')")


class TestConfig:
    def __init__(
        self,
        root_dir: str,
        model_name: str,
        batch_size: int,
        val_split: float,
        test_split: float,
        img_size: int,
        mean: List[float],
        std: List[float],
        seed: int,
        num_workers: int,
        device: str,
        checkpoint_path: str,
        output_dir: str = "out_test",
        save_wrong_images: bool = False
    ):
        # Paths and Model
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.save_wrong_images = save_wrong_images
        self.model_name = model_name
        
        # Inference Parameters
        self.batch_size = batch_size
        
        # Data Loading
        self.val_split = val_split
        self.test_split = test_split
        
        # Input Images
        self.img_size = img_size
        
        # Normalization (must match training)
        self.mean = mean
        self.std = std
        self.compute_stats = False # Always False for inference
        
        # System
        self.seed = seed
        self.num_workers = num_workers
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Runtime
        self.num_classes = 0
        self.input_channels = 3

    def __repr__(self):
        return (f"TestConfig(root_dir='{self.root_dir}', model_name='{self.model_name}', "
                f"batch_size={self.batch_size}, device='{self.device}', "
                f"checkpoint='{self.checkpoint_path}')")


# Training Configuration
train_cfg = TrainConfig(
    # Paths and Model
    root_dir='images',      
    model_name=None,
    pretrained=True,
    output_dir="out",

    # Hyperparameters
    batch_size=32,
    num_epochs=5,
    val_epochs=1,
    learning_rate=0.001,

    # Scheduler (Step Decay)
    step_size=10,
    gamma=0.1,

    # Regularization
    l1_lambda=0.0001,
    l2_lambda=0.0001,

    # Split Dataset
    val_split=0.15,
    test_split=0.15,

    # Input Images
    img_size=256,

    # Normalization
    compute_stats=True,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],

    # System
    seed=42,
    num_workers=0,
    device="auto",
    checkpoint_path="best.pth",
)

# Test Configuration
test_cfg = TestConfig(
    # Paths and Model
    root_dir='images',
    model_name=None, # must match checkpoint model name
    output_dir="out_test",
    save_wrong_images=True,

    # Inference Params
    batch_size=32,
    
    # Data Split
    val_split=0.15, # must be the same as training for correct split of test set
    test_split=0.15, # must be the same as training for correct split of test set

    # Input Images
    img_size=256, # must match training

    # Normalization
    mean=[0.5, 0.5, 0.5], # must match training
    std=[0.5, 0.5, 0.5], # must match training

    # System
    seed=42, # must be the same as training for correct split of test set
    num_workers=0,
    device="auto",
    checkpoint_path="best.pth",
)