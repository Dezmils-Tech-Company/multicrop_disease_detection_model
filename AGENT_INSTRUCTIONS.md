# Crop Disease Detection CV Model - Development Instructions

## PROJECT OVERVIEW
**Objective:** Build a computer vision model to scan crop images and simultaneously:
1. Identify the crop type (Apple, Corn, Tomato, etc.)
2. Identify the disease (Black Rot, Rust, Blight, etc.)

**Dataset Available:** 152,127 preprocessed images across 53 classes
- 121,207 training images
- 15,431 validation images  
- 15,489 test images

---

## PART 1: DATASET STRUCTURE & UNDERSTANDING

### Current Dataset Layout
```
dataset/
├── train/          # 121,207 images organized by class
├── val/            # 15,431 images organized by class
├── test/           # 15,489 images organized by class
├── dataset_metadata.json
```

### Class Organization
- 53 total classes in format: `{CROP}___{DISEASE_OR_HEALTH}`
- Examples: `Tomato___Leaf_Spot`, `Apple___Healthy`, `Corn___Common_Rust`
- Classes span multiple crop types: Apple, Cashew, Cassava, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, Tomato

### Data Pipeline Already Available
Refer to notebooks in `notebooks/` directory:
1. `01_exploratory_data_analysis.ipynb` - Dataset statistics and distribution
2. `02_data_validation_cleaning.ipynb` - Data quality checks
3. `03_visualization_analysis.ipynb` - Class distribution visualization
4. `04_normalization_standardization.ipynb` - Image normalization details
5. `05_class_imbalance_handling.ipynb` - Imbalance strategies (some classes have 13k images, others have 800)
6. `06_data_pipeline.ipynb` - Complete data loading pipeline

---

## PART 2: ARCHITECTURE & MODEL SELECTION

### Why Pretrained Transfer Learning?
Since you have pretrained data, use transfer learning to:
- Reduce training time (weeks → hours)
- Improve accuracy with limited compute
- Leverage learned features from ImageNet or agricultural datasets

### Recommended Pretrained CV Model Choices

#### Option A: ResNet-Based (Recommended for balanced speed/accuracy)
- **ResNet50** or **ResNet152** (ImageNet pretrained)
  - Good for general feature extraction
  - Transfer to 53-class classification
  - ~25MB model size for ResNet50

#### Option B: Vision Transformers (Best accuracy, higher compute)
- **ViT-Base** (Vision Transformer)
  - State-of-the-art accuracy
  - Better for domain-specific crops
  - Requires more VRAM during training

#### Option C: EfficientNet (Mobile/Edge deployment)
- **EfficientNet-B4** or **B5**
  - Smaller models, faster inference
  - Good accuracy-to-size ratio
  - Best for API deployment

#### Option D: DenseNet (Alternative)
- **DenseNet121** or **DenseNet161**
  - Efficient feature reuse
  - Lower memory footprint than ResNet

### Multi-Task Learning Approach (ADVANCED)
Since the task is to identify BOTH crop AND disease:
- **Shared backbone:** Pretrained model (ResNet50/EfficientNet)
- **Crop classifier head:** Output 11 crop types
- **Disease classifier head:** Output 5-7 disease types per crop
- **Joint training:** Single loss combining both tasks

**Simpler Alternative:** Single 53-class classifier (crop + disease as one label)
- Easier to implement
- Still achieves the goal
- Start here, advance to multi-task if needed

---

## PART 3: PROJECT FILE ORGANIZATION

### Directory Structure to Create
```
crop_disease_detection_model/
│
├── data/                          # Data handling modules
│   ├── __init__.py
│   ├── dataset.py                # Custom PyTorch Dataset class
│   ├── transforms.py             # Data augmentation pipelines
│   ├── loader.py                 # DataLoader utilities
│   └── preprocessing.py          # Image normalization, resizing
│
├── models/                        # Model definitions
│   ├── __init__.py
│   ├── backbone.py               # Load pretrained models (ResNet, EfficientNet, etc.)
│   ├── heads.py                  # Classification heads
│   ├── disease_detector.py       # Main model class combining backbone + heads
│   └── model_factory.py          # Create model instances with different configs
│
├── training/                      # Training pipeline
│   ├── __init__.py
│   ├── trainer.py                # Main training loop
│   ├── loss_functions.py         # Custom losses (focal loss, weighted CE)
│   ├── metrics.py                # Accuracy, F1, confusion matrix, per-class metrics
│   ├── callbacks.py              # Early stopping, model checkpointing, LR scheduling
│   └── config.py                 # Training hyperparameters
│
├── inference/                     # Deployment & inference
│   ├── __init__.py
│   ├── predictor.py              # Load model & make predictions
│   ├── batch_processor.py        # Process multiple images
│   └── postprocessor.py          # Confidence thresholding, crop/disease extraction
│
├── evaluation/                    # Model evaluation
│   ├── __init__.py
│   ├── evaluator.py              # Test set evaluation
│   ├── visualizer.py             # Confusion matrices, ROC curves
│   └── report_generator.py       # Generate evaluation reports
│
├── api/                           # FastAPI/Flask endpoints (already exists)
│   ├── main.py                   # API entry point
│   ├── endpoints/
│   │   ├── predict.py            # Prediction endpoint
│   │   ├── batch_predict.py      # Batch prediction
│   │   └── health.py             # Health check endpoint
│   └── schemas.py                # Request/response validation
│
├── scripts/                       # Standalone runnable scripts
│   ├── train.py                  # Main training script (orchestrator)
│   ├── evaluate.py               # Evaluate on test set
│   ├── predict.py                # Single image prediction CLI
│   ├── export_model.py           # Export to ONNX/TorchScript
│   ├── data_stats.py             # Print dataset statistics
│   └── audit_dataset.py          # Check for corrupted images
│
├── notebooks/                     # Jupyter notebooks (already exists)
│   ├── (existing EDA notebooks)
│   ├── 07_model_training.ipynb   # Interactive model training & experiments
│   ├── 08_inference_testing.ipynb # Test predictions on samples
│   └── 09_results_analysis.ipynb # Analyze results & failure cases
│
├── configs/                       # Configuration files (YAML/JSON)
│   ├── model_configs.yaml        # Model architecture choices
│   ├── training_configs.yaml     # Hyperparameters (LR, batch size, epochs)
│   ├── data_configs.yaml         # Data paths, transforms
│   └── inference_configs.yaml    # Deployment settings
│
├── checkpoints/                   # Model weights storage
│   ├── resnet50_best.pth         # Best model checkpoint
│   ├── resnet50_latest.pth       # Latest checkpoint
│   └── (other model variants)
│
├── logs/                          # Training logs
│   ├── training_metrics.csv      # Per-epoch metrics
│   ├── confusion_matrices/       # Saved confusion matrices
│   └── tensorboard/              # TensorBoard event files
│
├── tests/                         # Unit tests (already exists)
│   ├── test_data_loading.py
│   ├── test_model.py
│   ├── test_inference.py
│   └── test_api.py
│
├── requirements.txt              # Already has torch, torchvision, etc.
├── .env                          # Secrets, API keys
├── build_metadata.py             # Already exists
└── README.md                      # Updated with model training guide
```

---

## PART 4: TRAINING PIPELINE (HIGH-LEVEL WORKFLOW)

### Phase 1: Data Preparation
1. **Load metadata** from `dataset/dataset_metadata.json`
2. **Create PyTorch Dataset class** that:
   - Loads images from disk
   - Applies class label encoding (0-52 for 53 classes)
   - Handles missing/corrupted images gracefully
3. **Create DataLoaders** for train/val/test with:
   - Batch size: 32-128 (depends on GPU memory)
   - Shuffle: True for train, False for val/test
4. **Define augmentations** for training:
   - Random horizontal/vertical flips
   - Random rotation (±15 degrees)
   - Random brightness/contrast adjustments
   - Random crop/zoom
   - Normalize with ImageNet stats (mean, std)

### Phase 2: Model Setup
1. **Select pretrained backbone:**
   - Load ResNet50 pretrained on ImageNet
   - Remove final classification layer
2. **Freeze early layers:** (optional, for faster training)
   - Freeze first 3 residual blocks
   - Only train final block + classification head
3. **Add custom head:**
   - Fully connected layer(s) leading to 53 output neurons
   - Softmax activation for multi-class classification
4. **Move to GPU** if available

### Phase 3: Training Setup
1. **Define loss function:**
   - Cross-Entropy Loss (standard)
   - OR Focal Loss if class imbalance is severe
2. **Define optimizer:**
   - Adam with learning rate 1e-4 to 1e-5 for fine-tuning
   - OR SGD with momentum if Adam is unstable
3. **Define learning rate scheduler:**
   - Reduce LR by 0.1 if validation loss plateaus
   - Warmup learning rate for first 5 epochs
4. **Define metrics to track:**
   - Top-1 accuracy
   - Top-5 accuracy (optional)
   - Per-class F1 score
   - Macro/weighted F1

### Phase 4: Training Loop
1. **For each epoch:**
   - **Train phase:**
     - Forward pass on batch
     - Compute loss
     - Backward pass
     - Optimizer step
     - Track metrics
   - **Validation phase:**
     - Forward pass on val set
     - Compute loss & accuracy
     - Save checkpoint if validation accuracy improves
     - Early stopping if no improvement for 10 epochs
2. **Log metrics:**
   - Save to CSV for analysis
   - Optional: TensorBoard logging
3. **Expected training time:** 20-50 epochs (4-12 hours on modern GPU)

### Phase 5: Evaluation
1. **Load best checkpoint** from training
2. **Run on test set:**
   - Compute accuracy, precision, recall, F1
3. **Generate confusion matrix** for 53 classes
4. **Per-class analysis:**
   - Identify which crop-disease pairs have low accuracy
   - Analyze failure patterns
5. **Save evaluation report** as JSON/CSV

### Phase 6: Inference Pipeline
1. **Load trained model** + metadata
2. **For incoming image:**
   - Preprocess (resize to 224x224, normalize)
   - Forward pass
   - Get softmax probabilities for all 53 classes
   - Extract top-3 predictions
3. **Post-process:**
   - Extract crop type from class name
   - Extract disease from class name
   - Return: `{crop: "Tomato", disease: "Leaf_Spot", confidence: 0.94}`

---

## PART 5: TRAINING CONFIGURATION (HYPERPARAMETERS)

### Model Configuration
```yaml
model_type: "resnet50"  # or "efficientnet_b4", "vit_base", "densenet121"
pretrained: true
freeze_backbone_layers: 3  # 0 = no freezing, 3 = freeze 3 blocks
input_size: 224  # or 256, 384 depending on model
```

### Training Configuration
```yaml
batch_size: 64
learning_rate: 1e-4
optimizer: "adam"  # or "sgd"
num_epochs: 50
early_stopping_patience: 10
weight_decay: 1e-4
gradient_clip_norm: 1.0
warmup_epochs: 5
lr_scheduler: "cosine"  # or "reduce_on_plateau"
```

### Data Configuration
```yaml
train_split_path: "dataset/train"
val_split_path: "dataset/val"
test_split_path: "dataset/test"
image_extensions: [".jpg", ".jpeg", ".png"]
handle_missing: "skip"  # or "error"
```

### Augmentation Configuration
```yaml
horizontal_flip: true
vertical_flip: true
rotation: 15  # degrees
brightness: 0.2
contrast: 0.2
crop_scale: [0.8, 1.0]  # random crop 80-100% of image
```

---

## PART 6: TASK BREAKDOWN FOR AI AGENT

### When tasked with model development, the agent should:

1. **Data Validation**
   - Verify dataset structure matches expectations
   - Check for corrupted images
   - Validate class distribution
   - Generate dataset report

2. **Model Selection & Configuration**
   - Select appropriate pretrained backbone based on compute constraints
   - Configure input size and preprocessing
   - Define model head architecture

3. **Data Pipeline Implementation**
   - Create custom PyTorch Dataset class
   - Implement augmentation pipeline
   - Create DataLoaders with proper settings
   - Test data loading for speed/correctness

4. **Model Architecture Implementation**
   - Load pretrained model
   - Attach classification head
   - Implement model forward pass
   - Validate model with random input

5. **Training Implementation**
   - Implement training loop with loss calculation
   - Implement validation loop
   - Add checkpointing logic
   - Add early stopping mechanism
   - Log metrics per epoch

6. **Hyperparameter Tuning**
   - Experiment with learning rates (1e-5, 1e-4, 1e-3)
   - Test batch sizes (32, 64, 128)
   - Adjust optimizer choice
   - Monitor convergence

7. **Evaluation**
   - Run on test set
   - Generate confusion matrix
   - Calculate per-class metrics
   - Identify failure cases

8. **Inference Pipeline**
   - Implement model loading
   - Implement preprocessing pipeline matching training
   - Add confidence scoring
   - Extract crop and disease from predictions

9. **API Integration**
   - Create FastAPI endpoint for single image
   - Create batch prediction endpoint
   - Add input validation
   - Return structured response

10. **Testing & Validation**
    - Unit tests for data loading
    - Unit tests for model forward pass
    - Integration tests for inference
    - API endpoint tests

---

## PART 7: EXPECTED OUTCOMES & METRICS

### Performance Targets
- **Top-1 Accuracy:** 92-96% (achievable with ResNet50 + fine-tuning)
- **Per-class F1:** >0.85 for most classes
- **Inference Speed:** <100ms per image on CPU

### Deliverables
1. Trained model checkpoint (.pth file)
2. Model metadata (class names, input size, normalization stats)
3. Test set evaluation report (confusion matrix, per-class metrics)
4. Working API endpoint accepting crop images
5. Inference script for batch predictions
6. Documentation for model retraining

---

## PART 8: KEY CONSIDERATIONS

### Class Imbalance
- Tomato___Leaf_Spot: 13,309 images
- Strawberry___Leaf_Scorch: 1,774 images
- **Solution:** Weighted Cross-Entropy Loss or Focal Loss

### Data Quality
- Some images in `dataset/quarantine/` may be problematic
- Run audit before training
- Consider image quality filtering

### Transfer Learning Strategy
- Start with ImageNet pretrained weights
- Use moderate learning rate (1e-4) to preserve pretrained features
- Option to fine-tune only last layer first, then unfreeze

### Deployment Considerations
- Export model to ONNX for faster inference
- Consider quantization for mobile deployment
- Implement confidence threshold to reject low-confidence predictions

---

## QUICK START CHECKLIST

- [ ] Run `build_metadata.py` to verify dataset
- [ ] Set up project directory structure as outlined
- [ ] Create config files for model/training/data
- [ ] Implement PyTorch Dataset class
- [ ] Load and test pretrained model
- [ ] Create training loop with logging
- [ ] Train on small subset first (1 epoch) to verify pipeline
- [ ] Full training run with monitoring
- [ ] Evaluate on test set
- [ ] Implement inference pipeline
- [ ] Create API endpoints
- [ ] Test end-to-end workflow
- [ ] Documentation and model export
