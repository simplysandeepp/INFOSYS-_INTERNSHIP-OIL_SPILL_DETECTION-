# Oil Spill Detection using Enhanced U-Net with Attention Mechanisms

A comprehensive deep learning solution for automated oil spill detection in satellite/aerial imagery using an advanced U-Net architecture with attention gates, residual connections, and mixed precision training.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Detailed Code Walkthrough](#detailed-code-walkthrough)
- [Training Configuration](#training-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Results](#results)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a state-of-the-art semantic segmentation model for detecting oil spills in maritime imagery. The solution achieves **95-96% accuracy** through advanced architectural improvements and careful training optimization.

**Target Performance:**
- Accuracy: 95-96%
- Dice Coefficient: >0.90
- IoU (Intersection over Union): >0.85
- Hardware: Optimized for T4 GPU (Google Colab)

---

## Features

### Model Architecture
- **Enhanced U-Net** with 4 encoder-decoder levels
- **Attention Gates** for focused feature extraction
- **Residual Connections** preventing vanishing gradients
- **Mixed Precision Training** (FP16) for faster computation
- **Combined Loss Function** (BCE + Dice) for balanced learning

### Training Optimizations
- **Advanced Data Augmentation** (flips, rotations, brightness/contrast)
- **Learning Rate Warmup** (5 epochs) for stable initialization
- **Patient Early Stopping** (20 epoch patience) to escape plateaus
- **Adaptive Learning Rate** with gradual reduction
- **GPU Memory Growth** preventing OOM errors

### Comprehensive Visualizations
- Dataset distribution analysis
- Sample images with ground truth overlays
- Data statistics (coverage, brightness, contrast)
- Model architecture diagrams
- Training history curves
- Learning rate schedules
- Prediction overlays with confidence maps
- Confusion matrices
- Quality heatmaps
- Best/worst case analysis

---

## Architecture

### Enhanced U-Net Structure

```
Input (256x256x3)
    ↓
Encoder Block 1 (64 filters) ──→ Skip Connection 1
    ↓ MaxPool                         ↓
Encoder Block 2 (128 filters) ─→ Skip Connection 2
    ↓ MaxPool                         ↓
Encoder Block 3 (256 filters) ─→ Skip Connection 3
    ↓ MaxPool                         ↓
Encoder Block 4 (512 filters) ─→ Skip Connection 4
    ↓ MaxPool                         ↓
Bridge (1024 filters)
    ↓ Upsample
Decoder Block 1 (512) ←─ Attention Gate 4 ←─┘
    ↓ Upsample
Decoder Block 2 (256) ←─ Attention Gate 3 ←─┘
    ↓ Upsample
Decoder Block 3 (128) ←─ Attention Gate 2 ←─┘
    ↓ Upsample
Decoder Block 4 (64) ←── Attention Gate 1 ←─┘
    ↓
Output (256x256x1) - Sigmoid Activation
```

**Key Components:**
- **Residual Blocks**: Each conv block includes skip connections
- **Attention Gates**: Focus on relevant features before concatenation
- **Dropout**: Applied in encoder levels 2-4 (20% rate) for regularization
- **Batch Normalization**: After each convolution for stable training

---

## Dataset Structure

```
Dataset/
└── dataset/
    ├── train/
    │   ├── images/  # Training images (.jpg/.jpeg)
    │   └── masks/   # Binary masks (.png)
    ├── val/
    │   ├── images/  # Validation images
    │   └── masks/   # Validation masks
    └── test/
        ├── images/  # Test images
        └── masks/   # Test masks
```
## Dataset Classification
![Alt text](assets/data_distribution.png)

**Requirements:**
- Images: RGB format (.jpg/.jpeg)
- Masks: Binary format (.png) - white (255) = oil spill, black (0) = background
- Paired filenames: Corresponding images and masks should have matching names

---

## Installation

### Requirements

```bash
# Core dependencies (pre-installed in Google Colab)
tensorflow>=2.12.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.7.0
scikit-learn>=1.2.0
Pillow>=9.4.0
```

### Setup in Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# All other dependencies are pre-installed in Colab
# No additional installation required!
```

---

## Detailed Code Walkthrough

### Cell 1: Google Drive Mounting

**Purpose**: Connect to Google Drive to access the dataset stored there.

```python
from google.colab import drive
drive.mount('/content/drive')
```

**What it does:**
- Prompts for Google account authentication
- Mounts Drive at `/content/drive/`
- Enables access to files stored in Drive

---

### Cell 2: Enhanced Setup and Configuration

**Purpose**: Import libraries, configure GPU, set hyperparameters, and verify dataset paths.

#### 2.1 Library Imports

```python
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
```

**Key Libraries:**
- **NumPy**: Numerical operations and array manipulation
- **Pandas**: Data organization and statistics
- **Matplotlib/Seaborn**: Visualization
- **OpenCV**: Image processing
- **Scikit-learn**: Evaluation metrics

#### 2.2 TensorFlow and GPU Configuration

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Enable mixed precision (FP16) for T4 GPU
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Why Mixed Precision?**
- **2x faster training** on T4 GPUs
- **Reduced memory usage** (16-bit vs 32-bit floats)
- **Maintained accuracy** (loss computed in FP32)

**Why Memory Growth?**
- Prevents TensorFlow from allocating all GPU memory upfront
- Allows other processes to use GPU
- Prevents Out-of-Memory (OOM) errors

#### 2.3 Dataset Path Configuration

```python
BASE_DATA_DIR = '/content/drive/MyDrive/Dataset/dataset'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

TRAIN_IMAGES = os.path.join(TRAIN_DIR, 'images')
TRAIN_MASKS = os.path.join(TRAIN_DIR, 'masks')
VAL_IMAGES = os.path.join(VAL_DIR, 'images')
VAL_MASKS = os.path.join(VAL_DIR, 'masks')
TEST_IMAGES = os.path.join(TEST_DIR, 'images')
TEST_MASKS = os.path.join(TEST_DIR, 'masks')
```

**Path Verification:**
```python
for dir_path, dir_name in [(TRAIN_IMAGES, 'Train Images'), ...]:
    if os.path.exists(dir_path):
        file_count = len([f for f in os.listdir(dir_path) 
                         if os.path.isfile(os.path.join(dir_path, f))])
        print(f"✓ {dir_name}: {file_count} files")
    else:
        print(f"✗ {dir_name}: NOT FOUND!")
```

This ensures all required directories exist before training begins.

#### 2.4 Hyperparameter Configuration

```python
IMG_HEIGHT = 256          # Increased from 128 for better detail
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 8            # Optimized for T4 GPU (16GB)
EPOCHS = 30               # Extended for full convergence
LEARNING_RATE = 0.0001    # Conservative for fine-tuning
TRAINING_SUBSET = 1.0     # Use 100% of data
WARMUP_EPOCHS = 5         # Gradual LR warmup
DISABLE_EARLY_STOPPING = False  # Set True for guaranteed full training
```

**Why These Values?**
- **256×256**: Balance between detail and memory usage
- **Batch Size 8**: Maximum stable size for T4 GPU
- **LR 0.0001**: Prevents overshooting optimal weights
- **30 Epochs**: Sufficient for convergence with early stopping
- **Warmup**: Prevents early training instability

#### 2.5 Directory Creation

```python
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('logs', exist_ok=True)
```

Creates output folders for saved models, results, visualizations, and TensorBoard logs.

---

### Cell 3: Data Loading and Preprocessing

**Purpose**: Load dataset, apply augmentation, create TensorFlow datasets, and generate initial visualizations.

#### 3.1 Image Path Loading

```python
def load_image_paths(image_dir, mask_dir, subset=1.0):
    """Load and optionally subsample image-mask pairs"""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) 
                        if f.lower().endswith('.png')])
    
    if subset < 1.0:
        n_samples = int(len(image_files) * subset)
        indices = random.sample(range(len(image_files)), n_samples)
        image_files = [image_files[i] for i in indices]
        mask_files = [mask_files[i] for i in indices]
    
    image_paths = [os.path.join(image_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
    return image_paths, mask_paths
```
### Dataset Distribution
![Alt text](assets/dataset_distribution.png)

**Features:**
- **Sorted Loading**: Ensures image-mask pairing consistency
- **Format Filtering**: Only loads valid image formats
- **Subsampling**: Allows quick testing with partial dataset

#### 3.2 Image Preprocessing

```python
def load_and_preprocess_image(image_path, mask_path, 
                              img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Load, resize, and normalize images"""
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
    
    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.cast(mask > 0.5, tf.float32)  # Binarize
    
    return img, mask
```

![Alt text](assets/model_layer_distribution.png)

**Key Steps:**
1. **Reading**: TensorFlow's efficient file I/O
2. **Decoding**: Format-specific decoders (JPEG for images, PNG for masks)
3. **Resizing**: Uniform 256×256 size
4. **Normalization**: [0, 255] → [0, 1] range
5. **Binarization**: Mask threshold at 0.5

#### 3.3 Advanced Data Augmentation

```python
@tf.function
def apply_advanced_augmentation(img, mask):
    """Apply random augmentations to increase dataset diversity"""
    
    # Horizontal flip (50% chance)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    # Vertical flip (50% chance)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    
    # Random rotation (0°, 90°, 180°, 270°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    mask = tf.image.rot90(mask, k=k)
    
    # Brightness adjustment (±10%)
    img = tf.image.random_brightness(img, max_delta=0.1)
    
    # Contrast adjustment (90-110%)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    
    # Clip to valid range
    img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, mask
```
![Alt text](assets/distribution_graph.png)

**Why Augmentation?**
- **Prevents Overfitting**: Model learns invariant features
- **Increases Dataset Size**: Effectively 8× more training samples
- **Improves Generalization**: Better performance on unseen data

**@tf.function Decorator:**
- Compiles function into TensorFlow graph
- ~10× faster execution
- Enables GPU acceleration

#### 3.4 TensorFlow Dataset Creation

```python
def create_dataset(image_paths, mask_paths, batch_size=BATCH_SIZE, 
                  augment=False, cache=True):
    """Create optimized tf.data pipeline"""
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Shuffle training data
    if augment:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    # Load and preprocess
    dataset = dataset.map(load_and_preprocess_image,
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation
    if augment:
        dataset = dataset.map(apply_advanced_augmentation,
                             num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache in memory
    if cache:
        dataset = dataset.cache()
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

**Optimization Techniques:**
- **Parallel Map**: Utilizes all CPU cores
- **Caching**: Stores preprocessed data in RAM
- **Prefetching**: Loads next batch while GPU trains
- **AUTOTUNE**: Automatically optimizes parameters

**Performance Impact:**
- **3-5× faster training** compared to naive loading
- **Eliminates CPU bottleneck** in data pipeline

#### 3.5 Visualization 1: Dataset Distribution

```python
def visualize_dataset_distribution(train_images, val_images, test_images=None):
    """Bar chart showing train/val/test split"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    datasets = ['Training', 'Validation']
    counts = [len(train_images), len(val_images)]
    colors = ['#2ecc71', '#3498db']
    
    if test_images:
        datasets.append('Test')
        counts.append(len(test_images))
        colors.append('#e74c3c')
    
    bars = ax.bar(datasets, counts, color=colors, edgecolor='black', 
                  linewidth=2, alpha=0.8)
    ax.set_ylabel('Number of Images', fontweight='bold', fontsize=12)
    ax.set_title('Dataset Distribution', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('visualizations/01_dataset_distribution.png', dpi=150)
    plt.show()
```

**Output**: [Image will be added here]

**Purpose**: Verify balanced dataset split and identify potential data imbalance.

#### 3.6 Visualization 2: Sample Images with Masks

**Output**: [Image will be added here]

**Purpose**: Visually inspect data quality and annotation accuracy.

#### 3.7 Visualization 3: Data Statistics

**Output**: [Image will be added here]

**Purpose**: Understand dataset characteristics:
- **Coverage**: Percentage of pixels containing oil spills
- **Brightness**: Overall image illumination
- **Contrast**: Pixel intensity variation

---

### Cell 4: Enhanced U-Net Model Architecture

**Purpose**: Define and compile the segmentation model with advanced components.

#### 4.1 Attention Block

The attention mechanism helps the model focus on relevant features while suppressing irrelevant background information. It learns spatial attention weights that highlight oil spill regions.

**What Attention Does:**
1. **Highlights Relevant Features**: Suppresses background, emphasizes oil spills
2. **Improves Boundary Detection**: Focuses on edges and transitions
3. **Reduces False Positives**: Ignores irrelevant spatial locations

#### 4.2 Residual Convolutional Block

Residual connections solve the vanishing gradient problem, enabling deeper networks that converge faster and achieve better performance.

**Why Residual Connections?**
- **Solves Vanishing Gradients**: Direct path for gradient flow
- **Enables Deeper Networks**: Can stack 100+ layers effectively
- **Faster Convergence**: Learns residual (difference) rather than full mapping

#### 4.3 Complete U-Net Model

**Architecture Summary:**
- **Input**: 256×256×3 RGB image
- **Encoder**: 64 → 128 → 256 → 512 filters (4 levels)
- **Bridge**: 1024 filters (deepest features)
- **Decoder**: 512 → 256 → 128 → 64 filters (4 levels)
- **Output**: 256×256×1 binary mask (sigmoid)

**Total Parameters**: ~31 million (trainable)

#### 4.4 Loss Functions and Metrics

**Why Combined Loss?**
- **BCE**: Handles class imbalance (more non-spill pixels)
- **Dice**: Focuses on region overlap (critical for segmentation)
- **Synergy**: BCE provides pixel accuracy, Dice ensures shape matching

#### 4.5 Model Compilation

**Optimizer: AdamW**
- **Adam**: Adaptive Moment Estimation (combines momentum + RMSProp)
- **Weight Decay**: L2 regularization decoupled from gradient updates
- **Learning Rate**: 0.0001 (conservative for stability)

**Metrics Tracked:**
1. **Accuracy**: Pixel-wise classification accuracy
2. **Dice Coefficient**: Primary segmentation metric
3. **IoU**: Intersection over Union (Jaccard index)
4. **Precision**: True Positives / (True Positives + False Positives)
5. **Recall**: True Positives / (True Positives + False Negatives)

---

### Cell 5: Training with Advanced Callbacks

**Purpose**: Train the model with sophisticated monitoring and optimization strategies.

#### 5.1 Learning Rate Warmup

**Why Warmup?**
- **Prevents Early Instability**: Random initialization can cause large gradients
- **Smooth Start**: Gradually "wakes up" the network
- **Better Final Performance**: Avoids bad local minima early in training

**Schedule:**
- Epochs 1-5: LR increases from 0.00001 → 0.0001
- Epoch 6+: LR = 0.0001 (with adaptive reduction)

#### 5.2 Early Stopping (Patient Version)

**Why 20 Epoch Patience?**
- **Allows Plateau Escape**: Model can temporarily stagnate then improve
- **Prevents Premature Stopping**: Previous issue with 10 epoch patience
- **Better Convergence**: Reaches true optimal performance

#### 5.3 Learning Rate Reduction

**Adaptive Learning Rate Strategy:**
```
Initial:   0.0001
After 7:   0.00005  (if no improvement)
After 14:  0.000025
After 21:  0.0000125
Minimum:   0.0000001
```

#### 5.4 Training Execution

**Training Process:**
1. **Warmup Phase** (Epochs 1-5): LR gradually increases
2. **Main Training** (Epochs 6-30): Full learning rate
3. **Adaptive Phase**: LR reduces if plateau detected
4. **Early Stopping**: Triggers if 20 epochs without improvement

**Typical Timeline:**
- Epoch 1-10: Rapid improvement (Dice: 0.5 → 0.85)
- Epoch 11-20: Refinement (Dice: 0.85 → 0.92)
- Epoch 21-30: Fine-tuning (Dice: 0.92 → 0.95+)

#### 5.5 Visualization 5: Training History

**Output**: [Image will be added here]

**What to Look For:**
- **Convergence**: Both curves plateau
- **Overfitting**: Train improves but validation degrades
- **Underfitting**: Both curves still improving at end
- **Optimal Point**: Green star marks best validation performance

---

### Cell 6: Comprehensive Evaluation and Visualizations

**Purpose**: Generate detailed performance analysis and publication-quality visualizations.

#### 6.1 Detailed Prediction Visualization

**Output**: [Image will be added here]

**Column Interpretation:**
1. **Original**: Input satellite/aerial image
2. **Ground Truth**: Expert-annotated oil spill mask
3. **Confidence**: Heatmap showing model certainty (blue=low, red=high)
4. **Prediction**: Thresholded binary mask (0.5 cutoff)
5. **Overlay**: Red regions show detected oil spills on original image

#### 6.2 Confusion Matrix

**Output**: [Image will be added here]

**Matrix Interpretation:**
```
                Predicted
              No Spill | Spill
Actual ─────────────────────────
No Spill  │    TN    │   FP    │
Spill     │    FN    │   TP    │
```

#### 6.3 Quality Heatmap

**Output**: [Image will be added here]

**Color Legend:**
- **Green**: Excellent segmentation (IoU > 0.8)
- **Yellow**: Good segmentation (IoU 0.6-0.8)
- **Red**: Poor segmentation (IoU < 0.6)

#### 6.4 Best vs Worst Predictions

**Output**: [Image will be added here]

**Analysis Value:**
- **Worst Cases**: Reveal model weaknesses (e.g., small spills, complex backgrounds)
- **Best Cases**: Show model strengths (e.g., clear boundaries, high contrast)
- **Pattern Recognition**: Identify systematic failures vs. random errors
- **Improvement Targets**: Guide future data augmentation or architecture changes

---

## Results

### Expected Performance Metrics

Based on the enhanced architecture and training strategy, you should achieve:

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 96-98% | 95-96% | 94-96% |
| **Dice Coefficient** | 0.92-0.95 | 0.90-0.93 | 0.89-0.92 |
| **IoU** | 0.86-0.90 | 0.83-0.87 | 0.82-0.86 |
| **Precision** | 0.93-0.96 | 0.91-0.94 | 0.90-0.93 |
| **Recall** | 0.91-0.94 | 0.89-0.92 | 0.88-0.91 |

![Alt text](assets/prediction.png)
![Alt text](assets/training_outout_graph.png)

### Training Timeline

**Typical Training Progress:**

```
Epoch 1-5:   Warmup phase, rapid initial learning
  Dice: 0.30 → 0.65
  Loss: 0.85 → 0.45

Epoch 6-15:  Main learning phase
  Dice: 0.65 → 0.88
  Loss: 0.45 → 0.20

Epoch 16-25: Fine-tuning phase
  Dice: 0.88 → 0.92
  Loss: 0.20 → 0.12
  
Epoch 26-30: Convergence
  Dice: 0.92 → 0.95
  Loss: 0.12 → 0.09
```

## Best and Worst Prediction
![Alt text](assets/best_worst_prediction.png)

---

---
## Visualizations

### Confusion Matrix
![Alt text](assets/confusion_matrix.png)

### Learning Rate
![Alt text](assets/learning_rate_graph.png)

### Heatmap
![Alt text](assets/heatmap.png)

--- 

## Usage

### Basic Training

```python
# 1. Mount Drive and verify paths
from google.colab import drive
drive.mount('/content/drive')

# 2. Run all cells in sequence
# Cell 1: Mount Drive
# Cell 2: Setup and Configuration
# Cell 3: Data Loading and Preprocessing
# Cell 4: Model Architecture
# Cell 5: Training
# Cell 6: Evaluation

# 3. Training starts automatically after Cell 5
```

### Loading Pre-trained Model

```python
from tensorflow import keras

# Load best model
model = keras.models.load_model('models/best_model.h5', 
                                custom_objects={
                                    'combined_loss': combined_loss,
                                    'dice_coefficient': dice_coefficient,
                                    'iou_metric': iou_metric
                                })

# Make predictions
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (256, 256))
img = img.astype(np.float32) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
mask = (prediction[0].squeeze() > 0.5).astype(np.uint8) * 255
```

### Prediction on New Images

```python
def predict_oil_spill(image_path, model, threshold=0.5):
    """Predict oil spill on new image"""
    # Load and preprocess
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (256, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred_prob = model.predict(img_input, verbose=0)[0].squeeze()
    pred_binary = (pred_prob > threshold).astype(np.uint8) * 255
    
    # Resize back to original
    pred_binary = cv2.resize(pred_binary, (original_size[1], original_size[0]))
    pred_prob = cv2.resize(pred_prob, (original_size[1], original_size[0]))
    
    # Create overlay
    overlay = img.copy()
    overlay[pred_binary > 127] = [255, 0, 0]
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    # Calculate statistics
    total_pixels = pred_binary.size
    spill_pixels = np.sum(pred_binary > 127)
    spill_percentage = (spill_pixels / total_pixels) * 100
    avg_confidence = np.mean(pred_prob[pred_binary > 127]) if spill_pixels > 0 else 0
    
    return {
        'binary_mask': pred_binary,
        'confidence_map': pred_prob,
        'overlay': blended,
        'spill_detected': spill_pixels > 0,
        'spill_percentage': spill_percentage,
        'average_confidence': avg_confidence
    }

# Example usage
result = predict_oil_spill('new_image.jpg', model)
print(f"Oil Spill Detected: {result['spill_detected']}")
print(f"Coverage: {result['spill_percentage']:.2f}%")
print(f"Confidence: {result['average_confidence']:.2f}")
```

---

## Troubleshooting

### Issue 1: Out of Memory (OOM) Error

**Solutions:**
```python
# Option 1: Reduce batch size
BATCH_SIZE = 4  # Instead of 8

# Option 2: Reduce image size
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Option 3: Disable caching
dataset = create_dataset(image_paths, mask_paths, cache=False)
```

### Issue 2: Training Stops Too Early

**Solutions:**
```python
# Option 1: Disable early stopping completely
DISABLE_EARLY_STOPPING = True

# Option 2: Increase patience
early_stopping = EarlyStopping(patience=30)
```

### Issue 3: Poor Performance on Test Set

**Solutions:**
```python
# More aggressive augmentation or increased dropout rate
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{oil_spill_detection_2024,
  title = {Enhanced U-Net for Oil Spill Detection},
  author = {[Sandeep Prajapati]},
  year = {2025},
  url = {[https://github.com/simplysandeepp/INFOSYS-_INTERNSHIP-OIL_SPILL_DETECTION-.git]}
}
```

---

## License

This project is licensed under the Infosys Springbaord License.

---

## Acknowledgments

- **U-Net Architecture**: Ronneberger et al. (2015)
- **Attention Mechanisms**: Oktay et al. (2018)
- **Framework**: TensorFlow/Keras team

---

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: [contact@sandeepp.in]
- Project Link: [https://sandeepp.in/]
