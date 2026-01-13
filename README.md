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
| `model_name` | Identifier for the model architecture. See the full list in the [Model Name List](#model-name-list) section. | `str` | `None` |
| `pretrained` | Whether to load pretrained weights (only used by some backbones). **Note**: applies to `Xception` and `Resnet18`; ignored by custom CNNs (`GC1`, `GC2`). | `bool` | `True` |
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
| `l1_lambda` | L1 regularization strength. Set to `0.0` to disable. | `float` | `0.0` |
| `l2_lambda` | L2 regularization strength. Set to `0.0` to disable. | `float` | `0.0` |
| `val_split` | Fraction of data to use for validation. | `float` (0.0-1.0) | `0.15` |
| `test_split` | Fraction of data to reserve for testing. | `float` (0.0-1.0) | `0.15` |
| `compute_stats` | If `True`, computes dataset mean and std before training. Defaults to manual values if `False`. | `bool` | `True` |

### Testing Specific Parameters (`TestConfig`)
| Parameter | Description | Accepted Values | Default |
|-----------|-------------|-----------------|---------|
| `test_split` | Fraction of data used for testing. **Test Requirement**: Must match `TrainConfig.test_split` if loading data from the same source folders to ensure the same split is generated. | `float` (0.0-1.0) | `0.15` |

## Models

### Model Name List

| `model_name` | Required input | Supports `pretrained` | Notes |
|---|---:|:---:|---|
| `Xception` | `3 x 299 x 299` | ✅ | Uses a `timm` backbone; input size is enforced in this repo |
| `Resnet18` | `3 x 224 x 224` | ✅ | Torchvision backbone with the final FC replaced for `num_classes` |
| `GC1` | `3 x 256 x 256` | ❌ | Custom CNN; test 1 |
| `GC2` | `3 x 256 x 256` | ❌ | Custom CNN with dropout and a deeper head |

### Xception (`model_name='Xception'`)

The Xception weights commonly referenced for this architecture are a PyTorch port of the Keras implementation (credited to tstandley, adapted by cadene):

- Required input: **`3 x 299 x 299`** (so set `img_size=299`)

* ##### `pretrained=True` (fine-tuning)

    When you fine-tune Xception with pretrained weights, keep these normalization values:

    ```python
    mean=[0.5, 0.5, 0.5]
    std=[0.5, 0.5, 0.5]
    ```

* ##### `pretrained=False` (training from scratch)

    If you train Xception from scratch, you may change normalization (`mean/std`) for your dataset, but you still must keep the required input size (`299 x 299`).

### Resnet18 (`model_name='Resnet18'`)

The ResNet-18 wrapper is based on the torchvision backbone with the classifier replaced to match `num_classes`.

- Required input: **`3 x 224 x 224`** (set `img_size=224`)
- `pretrained=True`: uses ImageNet weights; prefer ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- `pretrained=False`: you can use custom normalization, but the input size must stay `224 x 224`.

### GarbageCustom_1 / GC1 (`model_name='GC1'`)

GC1 is a fixed-shape CNN and enforces the input tensor shape at runtime:

- Required input: **`3 x 256 x 256`** (so set `img_size=256`)
- RGB only

### GarbageCustom_2 / GC2 (`model_name='GC2'`)

GC2 is a deeper fixed-shape CNN with dropout regularization:

- Required input: **`3 x 256 x 256`** (so set `img_size=256`)
- RGB only; no pretrained weights

## Garbage Dataset Normalization (Training From Scratch)

If you want to train a model **from scratch** on the **Garbage** dataset, set the normalization stats explicitly to:

```python
mean=[0.6582812666893005, 0.6344856023788452, 0.6075275540351868]
std=[0.6582812666893005, 0.6344856023788452, 0.6075275540351868]
```

In this case, use `compute_stats=False` so the provided `mean/std` are not overridden.