from typing import List

class Config:
    def __init__(
        self,
        root_dir: str,
        model_name: str,
        batch_size: int,
        num_epochs: int,
        val_epochs: int,
        learning_rate: float,
        step_size: int,
        gamma: float,
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
        
        # Hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.val_epochs = val_epochs
        self.learning_rate = learning_rate
        
        # Scheduler (Step Decay)
        self.step_size = step_size
        self.gamma = gamma
        
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
        return (f"Config(root_dir='{self.root_dir}', model_name='{self.model_name}', "
                f"batch_size={self.batch_size}, num_epochs={self.num_epochs}, "
                f"lr={self.learning_rate}, device='{self.device}')")

# Default Configuration Object
cfg = Config(
    # Paths and Model
    root_dir='images',      
    model_name='GC1',
    output_dir="out"

    # Hyperparameters
    batch_size=32,
    num_epochs=5,
    val_epochs=1,
    learning_rate=0.001,

    # Scheduler (Step Decay)
    step_size=10,
    gamma=0.1,

    # Split Dataset
    val_split=0.15,
    test_split=0.15,

    # Input Images
    img_size=256,

    # Normalization
    compute_stats=False,
    mean=[0.6582812666893005, 0.6344856023788452, 0.6075275540351868],
    std=[0.6582812666893005, 0.6344856023788452, 0.6075275540351868],

    # System
    seed=42,
    num_workers=0,
    device="auto",
    checkpoint_path="garbage_custom_1_best.pth",
)
