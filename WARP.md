# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

`cadrille` is a multi-modal CAD reconstruction system that generates CadQuery code from point clouds, images, or text descriptions. It extends Qwen2-VL-2B-Instruct with a custom point cloud encoder using Fourier embeddings.

## Common Commands

### Training
```bash
# Train with point clouds and images (default)
python train.py --mode pc_img --use-text

# Train with only point clouds
python train.py --mode pc

# Train with only images
python train.py --mode img

# Train without text data
python train.py --mode pc_img
```

Training hyperparameters:
- Default batch size: 15 (without text), 8 (with text)
- Max steps: 120,000
- Learning rate: 2e-4 with cosine scheduler
- Saves checkpoints every 10,000 steps to `./work_dirs`

### Inference
```bash
# Run inference on DeepCAD test set with point clouds
python test.py --split deepcad_test_mesh --mode pc

# Run on Fusion360 test set
python test.py --split fusion360_test_mesh --mode pc

# Run with images
python test.py --split deepcad_test_mesh --mode img

# Run with text descriptions
python test.py --mode text

# Run on CPU (slower, but doesn't require GPU)
python test.py --split deepcad_test_mesh --mode pc --device cpu

# Specify custom checkpoint and output directory
python test.py --checkpoint-path maksimko123/cadrille-rl --py-path ./work_dirs/custom_output
```

Generated CadQuery code is saved to `./work_dirs/tmp_py` by default.

### Evaluation
```bash
# Evaluate predictions (calculates IoU, CD, invalidity ratio)
python evaluate.py

# Evaluate with custom paths
python evaluate.py --gt-mesh-path ./data/fusion360_test_mesh --pred-py-path ./work_dirs/tmp_py
```

**Important**: Before running evaluation, ensure the output directory (`--py-path` in test.py) is empty. The evaluation script also expects empty mesh/brep directories.

### Data Preparation
```bash
# Convert CAD-Recode CadQuery programs to meshes
python data/cadrecode2mesh.py
```

## Architecture

### Model Architecture
- **Base**: Qwen2-VL-2B-Instruct (vision-language model)
- **Custom Extension**: `FourierPointEncoder` for point cloud processing
  - Uses Fourier embeddings (8 frequencies) to encode 3D point coordinates
  - Projects to model's hidden size (matching Qwen2-VL's embedding dimension)
  - Point embeddings are injected at the beginning of the input sequence

### Multi-Modal Input Handling
The model supports three input modalities:
1. **Point clouds**: Encoded via `FourierPointEncoder`, prepended to text tokens
2. **Images**: Processed by Qwen2-VL's visual encoder (4 views concatenated)
3. **Text**: Direct text descriptions (Text2CAD dataset)

Input modality is tracked via `is_pc` and `is_img` boolean tensors throughout forward pass.

### Key Components

#### cadrille.py
- `Cadrille`: Main model class extending `Qwen2VLForConditionalGeneration`
- `FourierPointEncoder`: Encodes point clouds using Fourier features (3D → 51D → hidden_size)
- `collate()`: Custom collator that handles multi-modal batches with proper tokenization

The collator:
- Prepends pad tokens for point cloud embeddings
- Processes videos/images for Qwen2-VL
- Creates labels by masking everything except assistant responses (tokens between `151644,77091` and `151645`)

#### dataset.py
- `CadRecodeDataset`: Loads meshes and converts to point clouds or multi-view images
  - Point clouds: FPS sampled to 256 points, normalized
  - Images: 4 rendered views (fronts: [1,1,1], [-1,-1,-1], [-1,1,-1], [1,-1,1]) at 128×128
- `Text2CADDataset`: Text prompt → CadQuery code pairs

#### evaluate.py
- Uses non-daemon multiprocessing pool to safely execute generated CadQuery code
- Each code execution runs in isolated process with 3-second timeout (prevents CadQuery memory leaks)
- Metrics: IoU (3D intersection over union), Chamfer Distance, invalidity ratio

### Data Flow
1. **Training**: Mesh → [Point Cloud | Images] + Text → Model → CadQuery code tokens
2. **Inference**: Input modality → Tokenized + embedded → Generate up to 768 tokens → Save as .py files
3. **Evaluation**: .py files → Execute CadQuery → Render mesh → Compare with ground truth

## Pre-trained Models
- **SFT**: `maksimko123/cadrille` (supervised fine-tuning)
- **RL**: `maksimko123/cadrille-rl` (reinforcement learning fine-tuned)

Both models are available on HuggingFace and can be loaded directly as checkpoint paths.

## Datasets
Datasets must be downloaded from HuggingFace (requires git-lfs):
- **DeepCAD** (test): Meshes normalized to unit cube
- **Fusion360** (test): From Autodesk's Fusion360 Gallery
- **Text2CAD** (train/val/test): Text descriptions with CadQuery codes
- **CAD-Recode** (train/val): Large-scale synthetic CAD dataset

Expected directory structure is documented in `data/README.md`.

## Development Notes

### CPU Inference
- Use `--device cpu` flag to run inference without GPU
- CPU mode uses `float32` instead of `bfloat16` for better compatibility
- CPU mode uses `eager` attention instead of `flash_attention_2` (which requires CUDA)
- Batch size is automatically reduced for CPU: 1 for pc/img mode, 4 for text mode
- Performance is significantly slower than GPU, but useful for testing or GPU-constrained environments

### Point Cloud Processing
- Training uses 256 points via Farthest Point Sampling (FPS)
- Normalization: divide by `normalize_std_pc=100` for train/val, scale to [-1,1] for test
- Optional augmentation: Gaussian noise with scale 0.01

### Image Processing
- 4-view rendering using Open3D (headless mode required)
- Fixed camera distance: -0.9
- Uniform yellow color: [255, 255, 136]
- Views are concatenated into 2×2 grid

### Code Generation
- Target: CadQuery code that defines a variable `r` containing the CAD model
- Max generation length: 768 tokens
- Generated code is executed during evaluation to render 3D mesh

### Multi-Modal Training Strategy
- `mode='pc_img'`: Randomly samples either point cloud or image per example (50/50)
- `--use-text`: Adds Text2CAD dataset to training, changes batch size to 8

### Gradient Accumulation
- Without text: 2 steps (effective batch size: 30)
- With text: 4 steps (effective batch size: 32)

## Technical Requirements
- PyTorch 2.5.1 with CUDA 12.4
- Flash Attention 2 (for efficient transformer inference)
- Open3D (headless rendering build for image generation)
- PyTorch3D (for FPS sampling)
- CadQuery with OCP backend (for CAD code execution)
- See `Dockerfile` for complete dependency list
