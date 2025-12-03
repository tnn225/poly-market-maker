# TransformerClassifier Hyperparameter Tuning Guide

## Overview
This guide provides comprehensive tuning strategies for the FT Transformer classifier to maximize `total_pnl` while maintaining good precision and recall.

## Key Metrics
- **Primary Goal**: Maximize `total_pnl` (profit and loss)
- **Secondary Metrics**: Precision, Recall, F1, ROC-AUC, PR-AUC
- **Trade-off**: Higher recall = more trades = more opportunities but potentially lower precision

## Hyperparameter Presets

### 1. "Balanced" (Recommended Starting Point)
**Best for**: General use, maximizing total_pnl with balanced precision/recall

```python
TransformerClassifier(
    n_layers=5, d_token=96, n_heads=8,
    attention_dropout=0.2, ff_dropout=0.2,
    batch_size=1024, epochs=150,
    learning_rate=2e-4, weight_decay=1e-4,
    use_feature_scaling=True,
    use_focal_loss=True, focal_alpha=0.3, focal_gamma=2.5,
    label_smoothing=0.05,
    warmup_epochs=10, use_cosine_annealing=True
)
```

**Characteristics**:
- Moderate model capacity (5 layers, 96 token dim)
- Strong regularization (0.2 dropout, 1e-4 weight decay)
- Focal loss for hard examples
- Balanced class weighting (1.5x positive class boost)

### 2. "High Capacity"
**Best for**: Large datasets, complex patterns, when you have GPU and time

```python
TransformerClassifier(
    n_layers=6, d_token=128, n_heads=8,
    attention_dropout=0.25, ff_dropout=0.25,
    batch_size=512, epochs=200,
    learning_rate=1.5e-4, weight_decay=1.5e-4,
    use_feature_scaling=True,
    use_focal_loss=True, focal_alpha=0.3, focal_gamma=2.5,
    label_smoothing=0.05,
    warmup_epochs=15, use_cosine_annealing=True
)
```

**Characteristics**:
- High model capacity (6 layers, 128 token dim)
- Very strong regularization to prevent overfitting
- Longer training (200 epochs)
- Requires more GPU memory

### 3. "Lightweight"
**Best for**: Quick iterations, limited compute, fast experimentation

```python
TransformerClassifier(
    n_layers=3, d_token=48, n_heads=6,
    attention_dropout=0.1, ff_dropout=0.1,
    batch_size=1024, epochs=80,
    learning_rate=4e-4, weight_decay=5e-6,
    use_feature_scaling=True,
    use_focal_loss=False,
    label_smoothing=0.0,
    warmup_epochs=3, use_cosine_annealing=True
)
```

**Characteristics**:
- Small model (3 layers, 48 token dim)
- Lighter regularization
- Faster training
- Good for baseline and quick tests

### 4. "High Recall" (Less Conservative)
**Best for**: When you want to capture more opportunities, maximize trades

```python
TransformerClassifier(
    n_layers=5, d_token=96, n_heads=8,
    attention_dropout=0.15, ff_dropout=0.15,
    batch_size=1024, epochs=150,
    learning_rate=2e-4, weight_decay=1e-4,
    use_feature_scaling=True,
    use_focal_loss=True, focal_alpha=0.4, focal_gamma=2.0,
    label_smoothing=0.0,
    warmup_epochs=10, use_cosine_annealing=True
)
```

**Characteristics**:
- Moderate capacity
- Higher focal_alpha (0.4) for more positive class focus
- Less regularization (0.15 dropout)
- Optimized for maximum recall

## Hyperparameter Tuning Strategy

### Architecture Hyperparameters

#### `n_layers` (Number of Transformer Layers)
- **Range**: 3-8
- **Default**: 5
- **Effect**: More layers = more capacity but slower training and higher risk of overfitting
- **Tuning**: Start with 4-5, increase if underfitting, decrease if overfitting

#### `d_token` (Token Dimension)
- **Range**: 32-192
- **Default**: 96
- **Effect**: Larger = more representational capacity per feature
- **Tuning**: Must be divisible by `n_heads`. Common values: 48, 64, 96, 128
- **Memory**: Doubling d_token roughly quadruples memory usage

#### `n_heads` (Attention Heads)
- **Range**: 4-16
- **Default**: 8
- **Effect**: More heads = more diverse attention patterns
- **Constraint**: `d_token % n_heads == 0`
- **Tuning**: Usually 8 is optimal, increase for very large models

### Regularization Hyperparameters

#### `attention_dropout` & `ff_dropout`
- **Range**: 0.05-0.3
- **Default**: 0.2
- **Effect**: Higher = stronger regularization, prevents overfitting
- **Tuning**:
  - If overfitting: Increase to 0.25-0.3
  - If underfitting: Decrease to 0.1-0.15
  - Start with 0.15-0.2

#### `weight_decay` (L2 Regularization)
- **Range**: 1e-6 to 1e-3
- **Default**: 1e-4
- **Effect**: Higher = stronger weight regularization
- **Tuning**:
  - If overfitting: Increase to 1.5e-4 or 2e-4
  - If underfitting: Decrease to 5e-5 or 1e-5

#### `label_smoothing`
- **Range**: 0.0-0.1
- **Default**: 0.05
- **Effect**: Prevents overconfidence, improves calibration
- **Tuning**: Start with 0.0, add 0.05 if model is overconfident

### Training Hyperparameters

#### `batch_size`
- **Range**: 256-2048
- **Default**: 1024
- **Effect**: Larger = more stable gradients, faster training (on GPU)
- **Tuning**:
  - GPU: Use 512-2048 (larger is better if memory allows)
  - CPU: Use 256-512
  - Rule: Use largest batch size that fits in memory

#### `learning_rate`
- **Range**: 1e-5 to 1e-3
- **Default**: 2e-4
- **Effect**: Higher = faster learning but less stable
- **Tuning**:
  - Start with 2e-4 or 3e-4
  - If loss oscillates: Decrease to 1e-4
  - If training too slow: Increase to 4e-4 (with caution)
  - Scale with batch_size: `lr ≈ 2e-4 * (batch_size / 1024)`

#### `epochs`
- **Range**: 50-300
- **Default**: 150
- **Effect**: More epochs = more training, but early stopping should prevent overfitting
- **Tuning**: Set high (150-200), rely on early stopping

### Loss Function Hyperparameters

#### `use_focal_loss`
- **Options**: True/False
- **Default**: True (recommended)
- **Effect**: Focal loss focuses on hard examples, improves recall
- **When to use**: Always recommended for imbalanced datasets

#### `focal_alpha`
- **Range**: 0.1-0.5
- **Default**: 0.3
- **Effect**: Higher = more focus on positive class
- **Tuning**:
  - For higher recall: Increase to 0.4
  - For balanced: Keep at 0.25-0.3
  - For higher precision: Decrease to 0.2

#### `focal_gamma`
- **Range**: 1.0-3.0
- **Default**: 2.5
- **Effect**: Higher = more focus on hard examples
- **Tuning**:
  - If model struggles with hard examples: Increase to 3.0
  - If model is too aggressive: Decrease to 2.0

### Learning Rate Schedule

#### `use_cosine_annealing`
- **Options**: True/False
- **Default**: True (recommended)
- **Effect**: Smooth learning rate decay, better convergence
- **Alternative**: ReduceLROnPlateau (more adaptive but less smooth)

#### `warmup_epochs`
- **Range**: 3-20
- **Default**: 10
- **Effect**: Gradual learning rate increase at start
- **Tuning**: 10-15% of total epochs, minimum 5

### Class Weighting

#### Positive Class Boost (in code: `positive_class_boost`)
- **Range**: 1.0-2.5
- **Default**: 1.5
- **Effect**: Higher = more positive predictions = higher recall
- **Tuning**:
  - Conservative: 1.2-1.3
  - Balanced: 1.5
  - Aggressive: 1.8-2.0
  - Very aggressive: 2.0-2.5

## Tuning Workflow

### Step 1: Baseline
Start with "Balanced" preset and train for full epochs.

### Step 2: Diagnose
Check validation metrics:
- **Overfitting**: Train metrics >> Val metrics → Increase regularization
- **Underfitting**: Both metrics low → Increase capacity or learning rate
- **Low Recall**: Increase positive class boost or focal_alpha
- **Low Precision**: Decrease positive class boost or increase regularization

### Step 3: Iterate
Make one change at a time:
1. Adjust regularization (dropout, weight_decay)
2. Adjust capacity (n_layers, d_token)
3. Adjust loss function (focal_alpha, focal_gamma)
4. Adjust class weighting (positive_class_boost)

### Step 4: Fine-tune
Once close to optimal:
- Fine-tune learning rate (±20%)
- Adjust warmup epochs
- Try label smoothing (0.0 → 0.05)

## Common Issues and Solutions

### Issue: Model not learning (loss not decreasing)
**Solutions**:
- Check learning rate (too low?)
- Check batch size (too large?)
- Check feature scaling (enabled?)
- Check data quality

### Issue: Overfitting (train >> val)
**Solutions**:
- Increase dropout (0.2 → 0.25-0.3)
- Increase weight_decay (1e-4 → 1.5e-4)
- Decrease model capacity (n_layers: 5 → 4, d_token: 96 → 64)
- Add label smoothing (0.0 → 0.05)

### Issue: Underfitting (both metrics low)
**Solutions**:
- Increase model capacity (n_layers: 4 → 5-6, d_token: 64 → 96-128)
- Decrease regularization (dropout: 0.2 → 0.15)
- Increase learning rate (2e-4 → 3e-4)
- Train longer (epochs: 100 → 150-200)

### Issue: Low Recall (missing opportunities)
**Solutions**:
- Increase positive_class_boost (1.5 → 1.8-2.0)
- Increase focal_alpha (0.3 → 0.4)
- Decrease regularization slightly
- Use "High Recall" preset

### Issue: Low Precision (too many false positives)
**Solutions**:
- Decrease positive_class_boost (1.5 → 1.2-1.3)
- Increase regularization (dropout: 0.15 → 0.2)
- Decrease focal_alpha (0.3 → 0.25)
- Increase weight_decay

### Issue: Low total_pnl despite good metrics
**Solutions**:
- Check spread threshold in evaluation
- Adjust positive_class_boost to find optimal trade-off
- Try different loss functions (focal vs weighted BCE)
- Check if model is calibrated (use Brier score)

## Grid Search Recommendations

If you want to do systematic hyperparameter search, focus on:

1. **High Impact** (search these first):
   - `d_token`: [64, 96, 128]
   - `n_layers`: [4, 5, 6]
   - `learning_rate`: [1.5e-4, 2e-4, 3e-4]
   - `positive_class_boost`: [1.3, 1.5, 1.8]

2. **Medium Impact**:
   - `attention_dropout`, `ff_dropout`: [0.15, 0.2, 0.25]
   - `weight_decay`: [5e-5, 1e-4, 1.5e-4]
   - `focal_alpha`: [0.25, 0.3, 0.4]

3. **Low Impact** (fine-tuning):
   - `n_heads`: [6, 8]
   - `label_smoothing`: [0.0, 0.05]
   - `warmup_epochs`: [5, 10, 15]

## Best Practices

1. **Always use feature scaling** (`use_feature_scaling=True`)
2. **Use focal loss** for imbalanced datasets (`use_focal_loss=True`)
3. **Use cosine annealing** for better convergence
4. **Monitor validation metrics** during training
5. **Use early stopping** (automatic, patience = 15% of epochs)
6. **Start with "Balanced" preset** and adjust from there
7. **Make one change at a time** to understand impact
8. **Track total_pnl** as primary metric, not just accuracy/AUC

## Expected Performance

With "Balanced" preset on typical market data:
- **ROC-AUC**: 0.75-0.85
- **PR-AUC**: 0.70-0.80
- **Precision**: 55-65%
- **Recall**: 45-55%
- **F1**: 0.50-0.60
- **Total PnL**: Positive (varies by dataset and spread)

## Notes

- Early stopping monitors recall-weighted F1 (beta=1.5) to balance precision/recall
- Model automatically uses GPU if available (CUDA or MPS)
- Batch size should be adjusted based on available memory
- Training time scales roughly with: `n_layers * d_token^2 * batch_size`

