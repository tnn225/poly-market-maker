import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from rtdl import FTTransformer
from rtdl.modules import FeatureTokenizer, Transformer
from poly_market_maker.dataset import Dataset


class TransformerClassifier:
    """
    FT Transformer (Feature Tokenizer Transformer) for tabular data classification.
    
    Uses the rtdl library's FTTransformer implementation, which is a proven
    architecture for tabular data that treats features as tokens.
    """
    def __init__(
        self,
        n_layers=5,  # Optimized: 5 layers for balanced capacity
        d_token=96,  # Optimized: 96 token dim for good capacity
        n_heads=8,   # Standard: 8 heads (d_token must be divisible by n_heads)
        attention_dropout=0.2,  # Optimized: stronger regularization
        ff_dropout=0.2,  # Optimized: stronger regularization
        activation='gelu',  # Standard: GELU for transformers
        batch_size=1024,  # Optimized: large batch for stability
        epochs=150,  # Optimized: more epochs with early stopping
        learning_rate=2e-4,  # Optimized: stable learning rate
        weight_decay=1e-4,  # Optimized: stronger L2 regularization
        device=None,
        use_feature_scaling=True,  # Essential: always use for tabular data
        use_focal_loss=True,  # Optimized: enable for imbalanced datasets
        focal_alpha=0.3,  # Optimized: focus on positive class
        focal_gamma=2.5,  # Optimized: focus on hard examples
        label_smoothing=0.05,  # Optimized: prevent overconfidence
        warmup_epochs=10,  # Optimized: gradual warmup
        use_cosine_annealing=True  # Recommended: better convergence
    ):
        self.model = None
        self.feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.n_layers = n_layers
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ff_dropout = ff_dropout
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_feature_scaling = use_feature_scaling
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        self.use_cosine_annealing = use_cosine_annealing
        self.scaler = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def fit(self, X, y):
        # Convert to numpy arrays if needed
        if hasattr(X, "values"):
            self.feature_cols = X.columns.tolist() if hasattr(X, "columns") else None
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        
        # Ensure y is 1D
        if len(y.shape) > 1:
            y = y.flatten()
        
        # Split data (no shuffle to preserve temporal order)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)
        
        # Feature scaling (RobustScaler is more robust to outliers than StandardScaler)
        if self.use_feature_scaling:
            self.scaler = RobustScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            print("Applied RobustScaler for feature normalization")
        
        num_features = X.shape[1]
        
        # Compute class weights for imbalanced datasets
        # Increase positive class weight to make model less conservative
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Boost positive class weight to encourage more positive predictions (less conservative)
        # This will increase recall at the cost of some precision
        # Adjust this multiplier (1.3-2.0) based on desired aggressiveness:
        # - 1.3: Slightly less conservative
        # - 1.5: Balanced (default)
        # - 1.8: More aggressive, higher recall
        # - 2.0: Very aggressive, maximum recall
        positive_class_boost = 1.5  # Can be tuned: 1.3-2.0
        if 1 in class_weight_dict:
            class_weight_dict[1] = class_weight_dict[1] * positive_class_boost
        
        print(f"Class weights (boosted positive): {class_weight_dict}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        sample_weights_tensor = torch.FloatTensor(sample_weights).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, sample_weights_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Create FT Transformer model
        # Build feature tokenizer and transformer components
        import inspect
        
        try:
            # Try building components with common parameter names
            feature_tokenizer = FeatureTokenizer(
                n_num_features=num_features,
                cat_cardinalities=[],
                d_token=self.d_token
            )
        except TypeError as e:
            # Try alternative parameter names
            try:
                feature_tokenizer = FeatureTokenizer(
                    d_numerical=num_features,
                    categories=[],
                    d_token=self.d_token
                )
            except TypeError:
                sig = inspect.signature(FeatureTokenizer.__init__)
                raise ValueError(
                    f"FeatureTokenizer API mismatch. Available parameters: {list(sig.parameters.keys())}. "
                    f"Error: {e}"
                )
        
        # Build transformer with all required parameters
        # ffn_d_hidden is typically 4 * d_token for transformer architectures
        ffn_d_hidden = 4 * self.d_token if self.d_token else 192
        
        # Map activation string to rtdl-compatible format
        # rtdl expects 'ReLU', 'GELU' (capitalized) or PyTorch module
        activation_map = {
            'gelu': 'GELU',
            'relu': 'ReLU',
            'reglu': 'ReGLU',
            'geglu': 'GEGLU'
        }
        ffn_activation = activation_map.get(self.activation.lower(), 'ReLU')
        
        transformer = Transformer(
            d_token=self.d_token,
            n_blocks=self.n_layers,
            attention_n_heads=self.n_heads,
            attention_dropout=self.attention_dropout,
            attention_initialization='kaiming',  # Default initialization
            attention_normalization='LayerNorm',  # Default normalization
            ffn_d_hidden=ffn_d_hidden,  # Feed-forward hidden dimension (4 * d_token)
            ffn_dropout=self.ff_dropout,
            ffn_activation=ffn_activation,  # Mapped to rtdl-compatible format
            ffn_normalization='LayerNorm',  # Default normalization
            residual_dropout=0.0,
            prenormalization=True,  # Standard for transformers
            first_prenormalization=False,  # Standard setting
            last_layer_query_idx=None,  # Use all tokens
            n_tokens=None,  # Will be inferred from input
            kv_compression_ratio=None,  # No compression
            kv_compression_sharing=None,  # No compression
            head_activation='ReLU',  # Default activation for head
            head_normalization='LayerNorm',  # Default normalization for head
            d_out=1
        )
        
        self.model = FTTransformer(
            feature_tokenizer=feature_tokenizer,
            transformer=transformer
        ).to(self.device)
        
        # Loss function - Focal Loss or Weighted BCE
        if self.use_focal_loss:
            # Focal Loss for handling hard examples
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
                    super().__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    self.reduction = reduction
                
                def forward(self, inputs, targets):
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        inputs, targets, reduction='none'
                    )
                    pt = torch.exp(-bce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                    return focal_loss
            
            criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='none')
            print(f"Using Focal Loss (alpha={self.focal_alpha}, gamma={self.focal_gamma})")
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            print("Using Weighted BCE Loss")
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler - Cosine Annealing with Warmup or ReduceLROnPlateau
        if self.use_cosine_annealing:
            # Cosine annealing with warmup for better convergence
            def get_lr(epoch):
                if epoch < self.warmup_epochs:
                    # Linear warmup
                    return self.learning_rate * (epoch + 1) / self.warmup_epochs
                else:
                    # Cosine annealing
                    progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
                    return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
            
            self.get_lr = get_lr  # Store as instance method
            scheduler = None  # Will be handled manually
            print(f"Using Cosine Annealing with {self.warmup_epochs} warmup epochs")
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=True
            )
            print("Using ReduceLROnPlateau scheduler")
        
        # Training loop
        # Use recall-weighted F1 to make model less conservative
        # This balances precision and recall but gives more weight to recall
        best_val_metric = -1.0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        # Early stopping patience: adjust based on epochs
        # Rule of thumb: patience = 10-20% of total epochs
        patience = max(15, int(self.epochs * 0.15))  # 15% of epochs, minimum 15
        
        print("\n" + "="*60)
        print("FT Transformer Training (rtdl implementation)")
        print("="*60)
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60 + "\n")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_X, batch_y, batch_weights in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (rtdl FTTransformer expects (x_num, x_cat) where x_cat can be None)
                logits = self.model(batch_X, None).squeeze(-1)
                
                # Apply label smoothing if enabled
                if self.label_smoothing > 0:
                    batch_y_smooth = batch_y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                else:
                    batch_y_smooth = batch_y
                
                loss = criterion(logits, batch_y_smooth)
                loss = (loss * batch_weights).mean()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                
                train_loss += loss.item()
                
                # Store predictions for metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    train_preds.extend(probs.cpu().numpy())
                    train_targets.extend(batch_y.cpu().numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    logits = self.model(batch_X, None).squeeze(-1)
                    loss = criterion(logits, batch_y).mean()
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits)
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            
            train_pred_binary = (train_preds > 0.5).astype(int)
            val_pred_binary = (val_preds > 0.5).astype(int)
            
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            train_f1 = f1_score(train_targets, train_pred_binary)
            val_f1 = f1_score(val_targets, val_pred_binary)
            
            train_precision = precision_score(train_targets, train_pred_binary, zero_division=0)
            val_precision = precision_score(val_targets, val_pred_binary, zero_division=0)
            
            train_recall = recall_score(train_targets, train_pred_binary, zero_division=0)
            val_recall = recall_score(val_targets, val_pred_binary, zero_division=0)
            
            # PR-AUC
            train_pr_auc = auc(*precision_recall_curve(train_targets, train_preds)[:2][::-1])
            val_pr_auc = auc(*precision_recall_curve(val_targets, val_preds)[:2][::-1])
            
            # Update learning rate
            beta = 1.5
            if val_precision + val_recall > 0:
                recall_weighted_f1 = (1 + beta**2) * (val_precision * val_recall) / (beta**2 * val_precision + val_recall)
            else:
                recall_weighted_f1 = 0.0
            
            if self.use_cosine_annealing:
                # Manual learning rate scheduling with cosine annealing
                new_lr = self.get_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                scheduler.step(recall_weighted_f1)
            
            # Print progress
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, AUC: {train_auc:.4f}, "
                      f"PR-AUC: {train_pr_auc:.4f}, F1: {train_f1:.4f}, "
                      f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, AUC: {val_auc:.4f}, "
                      f"PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}, "
                      f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                      f"Recall-F1: {recall_weighted_f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping and model checkpointing
            # Use recall-weighted F1 (already calculated above) to make model less conservative
            # Also track PR-AUC as secondary metric (important for imbalanced data)
            min_delta = 5e-4  # Minimum improvement threshold
            
            # Check if this is a new best (using recall-weighted F1 as primary metric)
            is_best = recall_weighted_f1 > best_val_metric + min_delta
            
            if is_best:
                best_val_metric = recall_weighted_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                best_epoch = epoch + 1
                
                # Save best model
                os.makedirs('data/models', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'feature_cols': self.feature_cols,
                    'num_features': num_features,
                    'scaler': self.scaler,  # Save scaler for inference
                    'config': {
                        'n_layers': self.n_layers,
                        'd_token': self.d_token,
                        'n_heads': self.n_heads,
                        'attention_dropout': self.attention_dropout,
                        'ff_dropout': self.ff_dropout,
                        'activation': self.activation,
                        'use_feature_scaling': self.use_feature_scaling,
                    },
                    'best_metrics': {
                        'recall_weighted_f1': recall_weighted_f1,
                        'pr_auc': val_pr_auc,
                        'auc': val_auc,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1,
                        'epoch': best_epoch
                    }
                }, 'data/models/transformer_classifier_best.pt')
                print(f"  ✓ New best model! (Recall-F1: {recall_weighted_f1:.4f}, PR-AUC: {val_pr_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation recall-weighted F1: {best_val_metric:.4f} at epoch {best_epoch}")
                    break
        
        # Load best model and print summary
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("\n" + "="*60)
            print("Training Complete!")
            print("="*60)
            print(f"Best model found at epoch {best_epoch}")
            print(f"Best validation recall-weighted F1: {best_val_metric:.4f}")
            print(f"Total epochs trained: {epoch + 1}")
            print("="*60)
        else:
            print("\nWarning: No model checkpoint was saved. Training may not have improved.")
        
        return self.model

    def predict_proba(self, X, batch_size=64):
        """
        Generate predictions for the test dataset.
        
        Returns a numpy array of predictions in sklearn format (n_samples, 2).
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if hasattr(X, "values"):
            X = X.values
        
        # Apply feature scaling if used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                logits = self.model(batch_X, None).squeeze(-1)
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Convert to sklearn format: [1-p, p] for each sample
        prob_positive = predictions.flatten()
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    def get_probability(self, price, target, seconds_left, bid, ask):
        """Generate a single probability prediction for the provided snapshot."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        if target <= 0 or price <= 0:
            raise ValueError("price and target must be positive numbers.")

        row = {
            'delta': float(price - target),
            'percent': float(price - target) / target,
            'log_return': float(np.log(price / target)),
            'time': float(seconds_left / 900.0),
            'seconds_left': float(seconds_left),
            'bid': float(bid),
            'ask': float(ask),
        }
   
        if not hasattr(self, 'feature_cols') or self.feature_cols is None:
            raise ValueError("feature_cols not set. Make sure to call fit() with a DataFrame or set feature_cols manually.")
        
        X = np.array([[row[k] for k in self.feature_cols]], dtype='float32')
        
        # Apply feature scaling if used during training
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor, None).squeeze(-1)
            prediction = torch.sigmoid(logits).cpu().numpy()[0]
        
        return float(np.clip(prediction, 0.0, 1.0))

    def save(self, filepath='data/models/transformer_classifier.pt'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model. Default: 'data/models/transformer_classifier.pt'
        
        Raises:
            ValueError: If model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Get number of features (needed for model reconstruction)
        if hasattr(self, 'scaler') and self.scaler is not None:
            num_features = self.scaler.n_features_in_
        elif hasattr(self, 'feature_cols') and self.feature_cols is not None:
            num_features = len(self.feature_cols)
        else:
            raise ValueError("Cannot determine number of features. Model may not be properly trained.")
        
        # Prepare data to save
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'feature_cols': self.feature_cols,
            'num_features': num_features,
            'scaler': self.scaler,  # Save scaler for inference
            'config': {
                'n_layers': self.n_layers,
                'd_token': self.d_token,
                'n_heads': self.n_heads,
                'attention_dropout': self.attention_dropout,
                'ff_dropout': self.ff_dropout,
                'activation': self.activation,
                'use_feature_scaling': self.use_feature_scaling,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'use_focal_loss': self.use_focal_loss,
                'focal_alpha': self.focal_alpha,
                'focal_gamma': self.focal_gamma,
                'label_smoothing': self.label_smoothing,
                'warmup_epochs': self.warmup_epochs,
                'use_cosine_annealing': self.use_cosine_annealing,
            }
        }
        
        torch.save(save_data, filepath)
        print(f"Model saved to: {filepath}")
        print(f"  - Features: {len(self.feature_cols) if self.feature_cols else 'N/A'}")
        print(f"  - Architecture: {self.n_layers} layers, d_token={self.d_token}, n_heads={self.n_heads}")
        print(f"  - Feature scaling: {self.use_feature_scaling}")

    @classmethod
    def load(cls, filepath='data/models/transformer_classifier.pt', device=None):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file. Default: 'data/models/transformer_classifier.pt'
            device: Device to load model on. If None, auto-detects (cuda/mps/cpu)
        
        Returns:
            TransformerClassifier: Loaded model instance ready for inference
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted or incompatible
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(filepath, map_location='cpu')  # Load to CPU first
            
            # Extract configuration
            config = checkpoint.get('config', {})
            
            # Set device
            if device is None:
                device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device_obj = torch.device(device)
            
            # Create model instance with saved configuration
            model = cls(
                n_layers=config.get('n_layers', 5),
                d_token=config.get('d_token', 96),
                n_heads=config.get('n_heads', 8),
                attention_dropout=config.get('attention_dropout', 0.2),
                ff_dropout=config.get('ff_dropout', 0.2),
                activation=config.get('activation', 'gelu'),
                batch_size=config.get('batch_size', 1024),
                epochs=config.get('epochs', 150),
                learning_rate=config.get('learning_rate', 2e-4),
                weight_decay=config.get('weight_decay', 1e-4),
                device=device_obj,
                use_feature_scaling=config.get('use_feature_scaling', True),
                use_focal_loss=config.get('use_focal_loss', True),
                focal_alpha=config.get('focal_alpha', 0.3),
                focal_gamma=config.get('focal_gamma', 2.5),
                label_smoothing=config.get('label_smoothing', 0.05),
                warmup_epochs=config.get('warmup_epochs', 10),
                use_cosine_annealing=config.get('use_cosine_annealing', True)
            )
            
            # Restore feature columns and scaler
            model.feature_cols = checkpoint.get('feature_cols')
            model.scaler = checkpoint.get('scaler')
            
            # Reconstruct model architecture
            num_features = checkpoint.get('num_features')
            if num_features is None:
                raise ValueError("Cannot determine number of features from checkpoint")
            
            # Build feature tokenizer
            try:
                feature_tokenizer = FeatureTokenizer(
                    n_num_features=num_features,
                    cat_cardinalities=[],
                    d_token=model.d_token
                )
            except TypeError:
                feature_tokenizer = FeatureTokenizer(
                    d_numerical=num_features,
                    categories=[],
                    d_token=model.d_token
                )
            
            # Build transformer
            ffn_d_hidden = 4 * model.d_token
            activation_map = {
                'gelu': 'GELU',
                'relu': 'ReLU',
                'reglu': 'ReGLU',
                'geglu': 'GEGLU'
            }
            ffn_activation = activation_map.get(model.activation.lower(), 'ReLU')
            
            transformer = Transformer(
                d_token=model.d_token,
                n_blocks=model.n_layers,
                attention_n_heads=model.n_heads,
                attention_dropout=model.attention_dropout,
                attention_initialization='kaiming',
                attention_normalization='LayerNorm',
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=model.ff_dropout,
                ffn_activation=ffn_activation,
                ffn_normalization='LayerNorm',
                residual_dropout=0.0,
                prenormalization=True,
                first_prenormalization=False,
                last_layer_query_idx=None,
                n_tokens=None,
                kv_compression_ratio=None,
                kv_compression_sharing=None,
                head_activation='ReLU',
                head_normalization='LayerNorm',
                d_out=1
            )
            
            # Create and load model
            model.model = FTTransformer(
                feature_tokenizer=feature_tokenizer,
                transformer=transformer
            ).to(device_obj)
            
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.model.eval()  # Set to evaluation mode
            
            # Print load information
            print(f"Model loaded from: {filepath}")
            print(f"  - Device: {device_obj}")
            print(f"  - Features: {len(model.feature_cols) if model.feature_cols else 'N/A'}")
            print(f"  - Architecture: {model.n_layers} layers, d_token={model.d_token}, n_heads={model.n_heads}")
            print(f"  - Feature scaling: {model.use_feature_scaling}")
            
            # Print best metrics if available
            best_metrics = checkpoint.get('best_metrics')
            if best_metrics:
                print(f"  - Best validation metrics:")
                print(f"    * Recall-weighted F1: {best_metrics.get('recall_weighted_f1', 'N/A'):.4f}")
                print(f"    * PR-AUC: {best_metrics.get('pr_auc', 'N/A'):.4f}")
                print(f"    * AUC: {best_metrics.get('auc', 'N/A'):.4f}")
                print(f"    * Precision: {best_metrics.get('precision', 'N/A'):.4f}")
                print(f"    * Recall: {best_metrics.get('recall', 'N/A'):.4f}")
                print(f"    * Best epoch: {best_metrics.get('epoch', 'N/A')}")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Error loading model from {filepath}: {str(e)}")


def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask', 'z_score', 'prob_est']

    # ============================================================================
    # HYPERPARAMETER TUNING PRESETS
    # ============================================================================
    # Choose a preset or customize:
    #
    # PRESET 1: "Balanced" (Recommended starting point)
    # - Good balance between capacity and regularization
    # - Optimized for PnL maximization with reasonable precision
    # - Best for: General use, maximizing total_pnl
    #
    # PRESET 2: "High Capacity" (More complex patterns)
    # - Larger model with more parameters
    # - Requires more data and longer training
    # - Best for: Large datasets, complex patterns
    #
    # PRESET 3: "Lightweight" (Faster training)
    # - Smaller model, faster training
    # - Good for: Quick iterations, limited compute
    #
    # PRESET 4: "High Recall" (More trades, less conservative)
    # - Optimized for maximum recall
    # - May have lower precision but more opportunities
    # - Best for: When you want to capture more opportunities
    # ============================================================================
    if True:
    # PRESET: "Balanced" (Recommended)
        model = TransformerClassifier(
        # Architecture
        n_layers=5,              # Increased from 4: more depth for complex patterns
        d_token=96,              # Increased from 64: more capacity per token
        n_heads=8,               # Keep at 8 (d_token must be divisible by n_heads)
        attention_dropout=0.2,   # Increased from 0.15: stronger regularization
        ff_dropout=0.2,          # Increased from 0.15: stronger regularization
        activation='gelu',       # GELU is standard for transformers
        
        # Training
        batch_size=1024,         # Large batch for stability
        epochs=150,              # Increased from 100: allow more training
        learning_rate=2e-4,      # Reduced from 3e-4: more stable with larger model
        weight_decay=1e-4,       # Increased from 1e-5: stronger L2 regularization
        
        # Features
        use_feature_scaling=True, # Essential for tabular data
        
        # Loss function
        use_focal_loss=True,     # Enable Focal Loss for hard examples
        focal_alpha=0.3,         # Increased from 0.25: more focus on positive class
        focal_gamma=2.5,         # Increased from 2.0: more focus on hard examples
        
        # Regularization
        label_smoothing=0.05,     # Small amount: prevents overconfidence
        
        # Learning rate schedule
        warmup_epochs=10,        # Increased from 5: more gradual warmup
        use_cosine_annealing=True # Better convergence than ReduceLROnPlateau
    )
    
    # Alternative presets (uncomment to use):
    
    # PRESET 2: "High Capacity"
    # model = TransformerClassifier(
    #     n_layers=6, d_token=128, n_heads=8,
    #     attention_dropout=0.25, ff_dropout=0.25,
    #     batch_size=512, epochs=200, learning_rate=1.5e-4, weight_decay=1.5e-4,
    #     use_feature_scaling=True, use_focal_loss=True,
    #     focal_alpha=0.3, focal_gamma=2.5, label_smoothing=0.05,
    #     warmup_epochs=15, use_cosine_annealing=True
    # )
    
    # PRESET 3: "Lightweight"
    # model = TransformerClassifier(
    #     n_layers=3, d_token=48, n_heads=6,
    #     attention_dropout=0.1, ff_dropout=0.1,
    #     batch_size=1024, epochs=80, learning_rate=4e-4, weight_decay=5e-6,
    #     use_feature_scaling=True, use_focal_loss=False,
    #     label_smoothing=0.0, warmup_epochs=3, use_cosine_annealing=True
    # )
    
    # PRESET 4: "High Recall" (Less conservative)
    # model = TransformerClassifier(
    #     n_layers=5, d_token=96, n_heads=8,
    #     attention_dropout=0.15, ff_dropout=0.15,
    #     batch_size=1024, epochs=150, learning_rate=2e-4, weight_decay=1e-4,
    #     use_feature_scaling=True, use_focal_loss=True,
    #     focal_alpha=0.4, focal_gamma=2.0, label_smoothing=0.0,
    #     warmup_epochs=10, use_cosine_annealing=True
    # )
    
    
        model.fit(train_df[feature_cols], train_df['label'])
    
    # Save model
        model.save('./data/models/transformer_classifier.pt')
        print("\n" + "="*60)
    
    # Example: Load the saved model
    print("Loading saved model for inference...")
    loaded_model = TransformerClassifier.load('./data/models/transformer_classifier.pt')
    
    # Evaluate with loaded model
    prob = loaded_model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print("\nEvaluation Results (using loaded model):")
    print(ret)
    
    # Example predictions
    print(f"\nExample prediction: {loaded_model.get_probability(87684.42, 87498.59, 60, 0.53, 0.55)}")
    print(f"Example prediction: {loaded_model.get_probability(87398.59, 87584.42, 60, 0.45, 0.47)}")


if __name__ == "__main__":
    main()
