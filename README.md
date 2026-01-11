# Garbage_Classifier

## Dataset Structure
The dataset should be organized in the following folder structure:

```
root_dir/
    ├── class_name_1/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    ├── class_name_2/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    └── ...
```

The `root_dir` name can be configured in `config.py`.
Each subdirectory represents a class, and contains the images for that class.

## Configuration Parameters

The project uses two configuration classes in `config.py`: `TrainConfig` and `TestConfig`.

### Common Parameters (Train & Test)
| Parameter | Description | Accepted Values | Default |
|-----------|-------------|-----------------|---------|
| `root_dir` | Path to the root directory containing the dataset folders. | `str` (path) | `'images'` |
| `model_name` | Identifier for the model architecture (e.g., 'GC1'). | `str` | `None` |
| `output_dir` | Directory where outputs (checkpoints, plots) will be saved. | `str` (path) | `'out'` / `'out_test'` |
| `device` | Compute device to use. `'auto'` selects CUDA/MPS/CPU automatically. | `'cpu'`, `'cuda'`, `'mps'`, `'auto'` | `'auto'` |
| `seed` | Random seed for reproducibility. **Test Requirement**: Must match the training seed if relying on random splits to ensure the test set remains isolated. | `int` | `42` |
| `num_workers` | Number of subprocesses for data loading. | `int` | `0` |
| `img_size` | Size to which images are resized (square `size x size`). **Test Requirement**: Must match the training size. | `int` (pixels) | `256` |
| `mean` | Mean values for normalization [R, G, B]. **Test Requirement**: Must match the training values. | `list[float]` | `[0.5, 0.5, 0.5]` |
| `std` | Standard deviation for normalization [R, G, B]. **Test Requirement**: Must match the training values. | `list[float]` | `[0.5, 0.5, 0.5]` |
| `batch_size` | Number of samples per batch. Can vary between train and test. | `int` | `32` |
| `checkpoint_path` | Filename of the checkpoint to save (Training) or load (Testing). | `str` (filename) | `'best.pth'` |

### Training Specific Parameters (`TrainConfig`)
| Parameter | Description | Accepted Values | Default |
|-----------|-------------|-----------------|---------|
| `num_epochs` | Total number of training epochs. | `int` | `5` |
| `learning_rate` | Initial learning rate for the optimizer. | `float` | `0.001` |
| `val_epochs` | Frequency (in epochs) to perform validation. | `int` | `1` |
| `step_size` | Period of learning rate decay (StepLR scheduler). | `int` | `10` |
| `gamma` | Multiplicative factor of learning rate decay. | `float` | `0.1` |
| `val_split` | Fraction of data to use for validation. | `float` (0.0-1.0) | `0.15` |
| `test_split` | Fraction of data to reserve for testing. | `float` (0.0-1.0) | `0.15` |
| `compute_stats` | If `True`, computes dataset mean and std before training. Defaults to manual values if `False`. | `bool` | `True` |

### Testing Specific Parameters (`TestConfig`)
| Parameter | Description | Accepted Values | Default |
|-----------|-------------|-----------------|---------|
| `test_split` | Fraction of data used for testing. **Test Requirement**: Must match `TrainConfig.test_split` if loading data from the same source folders to ensure the same split is generated. | `float` (0.0-1.0) | `0.15` |