# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Common Commands

### Setup & Dependencies
```bash
# Install dependencies (as per requirements.txt)
pip install torch torchvision "numpy<2" scikit-learn pyyaml pillow -q

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Dataset Analysis
```bash
# Analyze dataset statistics and class imbalance
python scripts/analyze_dataset.py

# Validate dataset integrity
python scripts/audit_dataset.py
```

### Component Testing
```bash
# Test data loading and augmentation pipeline
python scripts/test_data_pipeline.py

# Test model architecture and forward pass
python scripts/test_model.py

# Quick 1-epoch training test (for debugging)
python scripts/quick_train_test.py
```

### Training
```bash
# Train model with GPU (recommended)
python scripts/train.py --config configs/training_config.yaml --device cuda

# Train model with CPU (fallback)
python scripts/train.py --config configs/training_config.yaml --device cpu

# Resume training from checkpoint (if interrupted)
python scripts/train.py --config configs/training_config.yaml --device cuda --resume checkpoints/checkpoint_epoch_010.pth
```

### Evaluation
```bash
# Evaluate model on test set
python scripts/evaluate.py --model checkpoints/best_model.pth --device cuda

# Evaluate with custom checkpoint
python scripts/evaluate.py --model checkpoints/checkpoint_epoch_025.pth --device cuda
```

### Inference
```bash
# Single image prediction
python scripts/predict.py --model checkpoints/best_model.pth --image test_image.jpg --top-k 3

# Batch prediction on directory
python scripts/predict.py --model checkpoints/best_model.pth --batch path/to/images/ --output predictions.json

# Batch prediction with CSV output
python scripts/predict.py --model checkpoints/best_model.pth --batch path/to/images/ --output predictions.csv
```

### Configuration
```bash
# Edit training hyperparameters
nano configs/training_config.yaml

# Common adjustments:
# - batch_size: reduce for GPU OOM (e.g., 32)
# - learning_rate: adjust for convergence (e.g., 1e-5)
# - loss type: switch to "focal" for severe imbalance
# - freeze_backbone_layers: set to 3 for transfer learning
```

## 🏗️ Code Architecture & Structure

### High-Level Organization
The project follows a modular PyTorch structure with clear separation of concerns:

```
crop_disease_detection_model/
│
├── data/                 # Data handling: Dataset, transforms, preprocessing
│   ├── dataset.py       # Custom PyTorch Dataset with class weighting
│   ├── loader.py        # DataLoader creation utilities
│   ├── transforms.py    # Augmentation pipelines (flip, rotation, etc.)
│   └── preprocessing.py # Image normalization and resizing
│
├── models/               # Model architectures
│   ├── backbone.py      # Pretrained backbones (ResNet, EfficientNet, ViT)
│   ├── heads.py         # Classification head with dropout
│   ├── disease_detector.py # Main model class combining backbone + head
│   └── __init__.py
│
├── training/             # Training pipeline
│   ├── trainer.py       # Training loop with metrics tracking
│   ├── loss_functions.py # Cross-Entropy, Focal, Weighted losses
│   ├── metrics.py       # Accuracy, F1, precision, recall calculations
│   ├── callbacks.py     # Early stopping, model checkpointing
│   └── __init__.py
│
├── inference/            # Deployment & inference
│   ├── predictor.py     # Model loading and prediction interface
│   └── __init__.py
│
├── evaluation/           # Model evaluation
│   ├── evaluator.py     # Test set evaluation and metrics generation
│   └── __init__.py
│
├── scripts/              # Executable workflow scripts
│   ├── train.py         # Main training orchestrator
│   ├── evaluate.py      # Test set evaluation
│   ├── predict.py       # Single/batch inference
│   ├── analyze_dataset.py # Dataset statistics and recommendations
│   ├── audit_dataset.py # Dataset validation and integrity checks
│   ├── test_data_pipeline.py # Data pipeline unit tests
│   ├── test_model.py    # Model architecture unit tests
│   └── quick_train_test.py # Quick 1-epoch training verification
│
├── configs/              # Configuration files
│   ├── training_config.yaml # Training hyperparameters (primary config)
│   └── inference_config.yaml # Inference settings (batch size, thresholds)
│
├── checkpoints/          # Model weights and metadata
│   ├── best_model.pth    # Best validation checkpoint
│   ├── class_mapping.json # Class index to name mapping
│   └── model_config.json # Model architecture configuration
│
└── logs/                 # Training and evaluation outputs
    ├── training_history.csv # Per-epoch metrics (train/val loss/accuracy)
    └── confusion_matrices/ # Saved confusion matrices from evaluation
```

### Key Design Patterns
1. **Configuration-Driven**: All training/inference parameters controlled via YAML configs
2. **Modular Components**: Each module (data, model, training, etc.) has clear responsibilities
3. **Factory Patterns**: Model and dataset creation through factory functions
4. **Callback System**: Training uses callbacks for early stopping, checkpointing
5. **Separation of Concerns**: Training logic decoupled from model architecture

### Data Flow
1. `data.dataset.DiseaseDataset` → loads and augments images
2. `data.loader.get_data_loaders` → creates train/val/test DataLoaders
3. `models.disease_detector.DiseaseDetectionModel` → defines model architecture
4. `training.trainer.ModelTrainer` → handles training loop with callbacks
5. `inference.predictor.Predictor` → loads model and makes predictions
6. `evaluation.evaluator.ModelEvaluator` → computes metrics and generates reports

### Extension Points
- **New Backbones**: Add to `models.backbone` and register in factory
- **New Losses**: Implement in `training.loss_functions` and reference in config
- **New Metrics**: Add to `training.metrics` and update trainer/evaluator
- **Augmentation Pipelines**: Modify `data.transforms` for custom transforms

### Development Workflow
1. **Component Testing**: Always run test scripts before modifying core components
   - `python scripts/test_data_pipeline.py` for data changes
   - `python scripts/test_model.py` for architecture changes
2. **Configuration Changes**: Modify `configs/training_config.yaml` for experiments
3. **Training Monitoring**: Check `logs/training_history.csv` for progress
4. **Debugging**: Use `scripts/quick_train_test.py` for fast iteration
5. **Validation**: Run full evaluation after training with `scripts/evaluate.py`

### Important Notes
- GPU training strongly recommended (4-6 hours vs 24-48 hours on CPU)
- Always verify dataset paths in scripts if moving the project
- Class mapping is saved with checkpoints - required for inference
- Logs directory grows over time - periodically clear old checkpoint files
- The project uses relative paths from repository root - maintain structure when copying