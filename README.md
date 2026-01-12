# FACTNet
Frequency-Driven Adaptive Calibration Transformer for Progress Aware Real-world Image Dehazing Network



## Project Structure

```
.
├── main.py                      # Main entry point for training and testing
├── train.py                     # Training script with loss functions and optimization
├── eval.py                      # Evaluation script for testing
├── valid.py                     # Validation script during training
├── utils.py                     # Utility functions (metrics, timers, etc.)
├── data/
│   ├── data_load.py            # Data loading and dataset classes
│   ├── data_augment.py          # Data augmentation transforms
│   └── nhhaze/                  # Dataset directory
│       ├── train/
│       │   ├── hazy/            # Hazy training images
│       │   └── gt/              # Ground truth training images
│       └── test/
│           ├── hazy/            # Hazy test images
│           └── gt/              # Ground truth test images
├── models/
│   ├── former.py                # Main DehazeFormer model architecture
│   ├── transformer_block.py    # Transformer block implementation
│   ├── layer.py                 # Patch embedding and unembedding layers
│   ├── attention.py             # Attention mechanism
│   ├── feed_forward.py          # Feed-forward network
│   ├── freq_processing.py       # Frequency domain processing
│   └── DegenerativePerceptionMoudel.py  # DPM fusion module
└── results/                     # Output directory for models and results
    └── Dehazeformer/
        ├── Best.pkl             # Best model checkpoint
        ├── model.pkl            # Latest model weights
        └── test/                # Test results
```

## Environment Configuration

The project uses the following main dependencies:

- **Python** 3.8+
- **PyTorch** 1.8+
- **CUDA** (for GPU acceleration)
- **torchvision**
- **numpy**
- **PIL/Pillow**
- **scikit-image**
- **matplotlib**
- **albumentations**

### Installation

1. Create a conda environment (recommended):
```bash
conda create -n dehazeformer python=3.8
conda activate dehazeformer
```

2. Install PyTorch (adjust CUDA version as needed):
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. Install other dependencies:
```bash
pip install numpy pillow scikit-image matplotlib albumentations tensorboard
```

## Dataset Preparation

### NH-Haze Dataset Structure

Prepare your dataset in the following structure:

```
data/nhhaze/
├── train/
│   ├── hazy/          # Hazy training images
│   └── gt/            # Ground truth training images (same filenames as hazy)
└── test/
    ├── hazy/          # Hazy test images
    └── gt/            # Ground truth test images (same filenames as hazy)
```

**Important Notes:**
- Image pairs (hazy and gt) must have the same filename
- Supported formats: `.png`, `.jpg`, `.jpeg`
- Images will be automatically loaded and paired based on matching filenames

### Data Augmentation

During training, the following augmentations are applied:
- Random crop (256×256)
- Random rotation and flip (8 modes)
- Random horizontal flip (50% probability)
- Random vertical flip (50% probability)

## Training and Testing

###培训

To train the model from scratch:

```bash
python main.py --mode train --data_dir ./data/nhhaze --batch_size 4 --learning_rate 1e-4 --num_epoch 30000
```

**Key Training Parameters:**
- `--mode`: Set to `train` for training mode
- `--data_dir`: Path to dataset directory (default: `./data/nhhaze`)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for optimizer (default: 1e-4)
- `--num_epoch`: Number of training epochs (default: 30000)
- `--print_freq`: Print frequency (default: 1)
- `--num_worker`: Number of data loading workers (default: 8)
- `--save_freq`: Model saving frequency in epochs (default: 100)
- `--valid_freq`: Validation frequency in epochs (default: 1)
- `--resume`: Path to checkpoint file for resuming training (optional)

**Training Features:**
- Uses AdamW optimizer with cosine annealing learning rate scheduler
- Warmup phase: 3% of total epochs (minimum 5, maximum 200 epochs)
- Loss function: Multi-scale L1 loss + Frequency domain L1 loss (weight: 0.1)
- Gradient clipping: max_norm=1.0
- Automatic checkpoint saving (latest, periodic, and best model)
- TensorBoard logging for training metrics

**Resume Training:**
```bash
python main.py --mode train 
```

### Testing

To test the trained model:

```bash
python main.py --mode test 
```

**Key Testing Parameters:**
- `--mode`: Set to `test` for testing mode
- `--data_dir`: Path to dataset directory
- `--test_model`: Path to model checkpoint file (default: `Best.pkl`)
- `--save_image`: Whether to save output images (default: True)

**Test Output:**
- Dehazed images saved to `results/Dehazeformer/test/`
- Metrics file (`metrics.txt`) containing PSNR, SSIM, and inference time for each image
- Average metrics printed to console

### Configuration Examples

**Training with custom settings:**
```bash
python main.py \
    --mode train \
    --data_dir ./data/nhhaze \
    --batch_size 8 \
    --learning_rate 2e-4 \
    --num_epoch 50000 \
    --save_freq 200 \
    --valid_freq 5 \
    --num_worker 4
```

**Testing with specific model:**
```bash
python main.py \
    --mode test \
    --data_dir ./data/nhhaze \
    --test_model results/Dehazeformer/checkpoint_1000.pth \
    --save_image True
```

## Pre-trained Models and Results

The trained model checkpoints are saved in `results/Dehazeformer/`:
- `Best.pkl`: Best model based on validation score (PSNR + 10×SSIM)
- `model.pkl`: Latest model weights
- `latest_checkpoint.pth`: Latest checkpoint with optimizer and scheduler states


**Note:** The complete models and weights will be publicly released after the paper is accepted.

**Pre-trained Model Download:**
- File: FACT
- Link: https://pan.baidu.com/s/1lfVckkSEck1Jqutit9LBaA
- Extraction Code: uybi

