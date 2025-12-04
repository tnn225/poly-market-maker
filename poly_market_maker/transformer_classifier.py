import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import optuna
from rtdl import FTTransformer
from rtdl.modules import FeatureTokenizer, Transformer
from poly_market_maker.dataset import Dataset



FEATURE_COLS = ['log_return', 'time', 'bid']

class TransformerClassifier:
    """
    FT Transformer (Feature Tokenizer Transformer) optimized to maximize PnL = is_up - bid.
    
    Uses PnL-weighted loss function to directly optimize for profit and loss.
    """
    def __init__(
        self,
        n_layers=4,
        d_token=48,
        n_heads=8,
        attention_dropout=0.2,
        ff_dropout=0.2,
        activation='gelu',
        batch_size=1024,
        epochs=100,
        learning_rate=2e-4,
        weight_decay=1e-4,
        device=None,
        use_feature_scaling=True,
        label_smoothing=0.05,
        warmup_epochs=10,
        use_cosine_annealing=True,
        spread=0.0,  # Spread for trade decision
        target_trade_rate=0.25,  # Target trade rate (25% = 0.25)
        trade_rate_reg_strength=0.5  # Strength of trade rate regularization
    ):
        self.model = None
        self.feature_cols = None
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
        self.label_smoothing = label_smoothing
        self.warmup_epochs = warmup_epochs
        self.use_cosine_annealing = use_cosine_annealing
        self.spread = spread
        self.target_trade_rate = target_trade_rate
        self.trade_rate_reg_strength = trade_rate_reg_strength
        self.scaler = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

    def fit(self, X, y):
        """
        Fit the model to maximize PnL = is_up - bid.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Features including 'bid' column
        y : pandas.Series or numpy.ndarray
            Labels (is_up: 1 if price went up, 0 otherwise)
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame with 'bid' column")
        
        if 'bid' not in X.columns:
            raise ValueError("X must contain 'bid' column for PnL optimization")
        
        self.feature_cols = X.columns.tolist()
        
        # Extract bid and is_up
        bid_values = X['bid'].values
        if hasattr(y, "values"):
            is_up = y.values
        else:
            is_up = y
        
        # Ensure y is 1D
        if len(is_up.shape) > 1:
            is_up = is_up.flatten()
        
        # Calculate PnL weights: (is_up - bid)
        # Keep original PnL values - positive for profitable, negative for unprofitable
        pnl_weights = is_up.astype(float) - bid_values.astype(float)
        
        # Prepare features (excluding bid from features if needed, or keep it)
        X_features = X.values
        
        # Split data (no shuffle to preserve temporal order)
        split_idx = int(len(X_features) * 0.9)
        X_train, X_val = X_features[:split_idx], X_features[split_idx:]
        y_train, y_val = is_up[:split_idx], is_up[split_idx:]
        bid_train, bid_val = bid_values[:split_idx], bid_values[split_idx:]
        pnl_weights_train, pnl_weights_val = pnl_weights[:split_idx], pnl_weights[split_idx:]
        
        # Feature scaling
        if self.use_feature_scaling:
            self.scaler = RobustScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            print("Applied RobustScaler for feature normalization")
        
        num_features = X_features.shape[1]
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        bid_train_tensor = torch.FloatTensor(bid_train).to(self.device)
        pnl_weights_train_tensor = torch.FloatTensor(pnl_weights_train).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        bid_val_tensor = torch.FloatTensor(bid_val).to(self.device)
        pnl_weights_val_tensor = torch.FloatTensor(pnl_weights_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, bid_train_tensor, pnl_weights_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor, bid_val_tensor, pnl_weights_val_tensor)
        
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
        import inspect
        
        try:
            feature_tokenizer = FeatureTokenizer(
                n_num_features=num_features,
                cat_cardinalities=[],
                d_token=self.d_token
            )
        except TypeError:
            try:
                feature_tokenizer = FeatureTokenizer(
                    d_numerical=num_features,
                    categories=[],
                    d_token=self.d_token
                )
            except TypeError:
                sig = inspect.signature(FeatureTokenizer.__init__)
                raise ValueError(
                    f"FeatureTokenizer API mismatch. Available parameters: {list(sig.parameters.keys())}"
                )
        
        ffn_d_hidden = 4 * self.d_token
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
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=self.ff_dropout,
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
        
        self.model = FTTransformer(
            feature_tokenizer=feature_tokenizer,
            transformer=transformer
        ).to(self.device)
        
        # Custom PnL-optimized loss that encourages selectivity
        class PnLOptimizedLoss(nn.Module):
            def __init__(self, target_trade_rate, reg_strength):
                super().__init__()
                self.target_trade_rate = target_trade_rate
                self.reg_strength = reg_strength
            
            def forward(self, logits, targets, bids, pnl_weights):
                """
                Loss that directly optimizes for PnL by:
                1. Encouraging high prob when (is_up - bid) > 0
                2. Discouraging high prob when (is_up - bid) < 0
                3. Adding regularization to control trade rate
                """
                probs = torch.sigmoid(logits)
                
                # Expected PnL for each sample: prob * (is_up - bid)
                # We want to maximize this, so minimize negative
                expected_pnl = probs * pnl_weights
                
                # Base BCE loss for calibration
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets, reduction='none'
                )
                
                # Combine: minimize negative expected PnL + BCE for calibration
                # Add penalty for predicting high prob when PnL is negative
                negative_pnl_mask = pnl_weights < 0
                penalty = torch.where(
                    negative_pnl_mask,
                    probs * torch.abs(pnl_weights),  # Penalize high prob for negative PnL
                    torch.zeros_like(probs)
                )
                
                # Regularization: penalize deviation from target trade rate
                # This controls selectivity to achieve desired trade rate
                avg_prob = probs.mean()
                trade_rate_reg = self.reg_strength * (avg_prob - self.target_trade_rate) ** 2
                
                # Total loss: negative expected PnL + BCE + penalty + regularization
                loss = -expected_pnl.mean() + 0.1 * bce_loss.mean() + penalty.mean() + trade_rate_reg
                
                return loss
        
        criterion = PnLOptimizedLoss(self.target_trade_rate, self.trade_rate_reg_strength)
        print(f"Using PnL-optimized loss with target trade rate: {self.target_trade_rate*100:.1f}%")
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        if self.use_cosine_annealing:
            def get_lr(epoch):
                if epoch < self.warmup_epochs:
                    return self.learning_rate * (epoch + 1) / self.warmup_epochs
                else:
                    progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
                    return self.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
            
            self.get_lr = get_lr
            scheduler = None
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
        
        # Training loop - optimize for PnL
        best_val_pnl = float('-inf')
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        patience = max(15, int(self.epochs * 0.15))
        
        print("\n" + "="*60)
        print("FT Transformer Training - Maximizing PnL = is_up - bid")
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
            
            for batch_X, batch_y, batch_bid, batch_pnl_weights in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(batch_X, None).squeeze(-1)
                
                # Apply label smoothing if enabled
                if self.label_smoothing > 0:
                    batch_y_smooth = batch_y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                else:
                    batch_y_smooth = batch_y
                
                # PnL-optimized loss
                loss = criterion(logits, batch_y_smooth, batch_bid, batch_pnl_weights)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
            val_bids = []
            val_is_up = []
            
            with torch.no_grad():
                for batch_X, batch_y, batch_bid, batch_pnl_weights in val_loader:
                    logits = self.model(batch_X, None).squeeze(-1)
                    # Use a simple BCE for validation loss tracking
                    val_bce = nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
                    val_loss += val_bce.item()
                    
                    probs = torch.sigmoid(logits)
                    val_preds.extend(probs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
                    val_bids.extend(batch_bid.cpu().numpy())
                    val_is_up.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_bids = np.array(val_bids)
            val_is_up = np.array(val_is_up)
            
            # Calculate PnL: sum of (is_up - bid) for trades where prob - spread > bid
            val_actions = (val_preds - self.spread > val_bids)
            val_pnl = ((val_is_up[val_actions] - val_bids[val_actions]).sum() if val_actions.sum() > 0 else 0.0)
            
            # Calculate trade rate for monitoring
            val_trade_rate = val_actions.sum() / len(val_actions) if len(val_actions) > 0 else 0.0
            
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
            
            train_pr_auc = auc(*precision_recall_curve(train_targets, train_preds)[:2][::-1])
            val_pr_auc = auc(*precision_recall_curve(val_targets, val_preds)[:2][::-1])
            
            # Update learning rate
            if self.use_cosine_annealing:
                new_lr = self.get_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            else:
                scheduler.step(val_pnl)
            
            # Print progress
            if (epoch + 1) % 1 == 0:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Epoch {epoch+1}/{self.epochs} - {current_time}")
                print(f"  Train - Loss: {train_loss/len(train_loader):.4f}, AUC: {train_auc:.4f}, "
                      f"PR-AUC: {train_pr_auc:.4f}, F1: {train_f1:.4f}, "
                      f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                print(f"  Val   - Loss: {val_loss/len(val_loader):.4f}, AUC: {val_auc:.4f}, "
                      f"PR-AUC: {val_pr_auc:.4f}, F1: {val_f1:.4f}, "
                      f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
                      f"PnL: {val_pnl:.4f}, TradeRate: {val_trade_rate:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping based on PnL
            min_delta = 0.1
            is_best = val_pnl > best_val_pnl + min_delta
            
            if is_best:
                best_val_pnl = val_pnl
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                best_epoch = epoch + 1
                
                # Save best model
                os.makedirs('data/models', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'feature_cols': self.feature_cols,
                    'num_features': num_features,
                    'scaler': self.scaler,
                    'config': {
                        'n_layers': self.n_layers,
                        'd_token': self.d_token,
                        'n_heads': self.n_heads,
                        'attention_dropout': self.attention_dropout,
                        'ff_dropout': self.ff_dropout,
                        'activation': self.activation,
                        'use_feature_scaling': self.use_feature_scaling,
                        'spread': self.spread,
                    },
                    'best_metrics': {
                        'pnl': val_pnl,
                        'pr_auc': val_pr_auc,
                        'auc': val_auc,
                        'precision': val_precision,
                        'recall': val_recall,
                        'f1': val_f1,
                        'epoch': best_epoch
                    }
                }, 'data/models/transformer_classifier_best.pt')
                print(f"  ✓ New best model! (PnL: {val_pnl:.4f}, PR-AUC: {val_pr_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation PnL: {best_val_pnl:.4f} at epoch {best_epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("\n" + "="*60)
            print("Training Complete!")
            print("="*60)
            print(f"Best model found at epoch {best_epoch}")
            print(f"Best validation PnL: {best_val_pnl:.4f}")
            print(f"Total epochs trained: {epoch + 1}")
            print("="*60)
        else:
            print("\nWarning: No model checkpoint was saved. Training may not have improved.")
        
        return self.model

    def predict_proba(self, X, batch_size=64):
        """Generate predictions for the test dataset."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        if hasattr(X, "values"):
            X = X.values
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                logits = self.model(batch_X, None).squeeze(-1)
                probs = torch.sigmoid(logits)
                predictions.extend(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        prob_positive = predictions.flatten()
        prob_negative = 1 - prob_positive
        return np.column_stack([prob_negative, prob_positive])

    def get_probability(self, price, target, seconds_left, bid, ask, z_score, prob_est):
        """Generate a single probability prediction."""
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
            'z_score': float(z_score),
            'prob_est': float(prob_est),
        }
   
        if not hasattr(self, 'feature_cols') or self.feature_cols is None:
            raise ValueError("feature_cols not set. Make sure to call fit() with a DataFrame or set feature_cols manually.")
        
        X = np.array([[row[k] for k in self.feature_cols]], dtype='float32')
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor, None).squeeze(-1)
            prediction = torch.sigmoid(logits).cpu().numpy()[0]
        
        return float(np.clip(prediction, 0.0, 1.0))

    def save(self, filepath='data/models/transformer_classifier.pt'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if hasattr(self, 'scaler') and self.scaler is not None:
            num_features = self.scaler.n_features_in_
        elif hasattr(self, 'feature_cols') and self.feature_cols is not None:
            num_features = len(self.feature_cols)
        else:
            raise ValueError("Cannot determine number of features. Model may not be properly trained.")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'feature_cols': self.feature_cols,
            'num_features': num_features,
            'scaler': self.scaler,
            'config': {
                'n_layers': self.n_layers,
                'd_token': self.d_token,
                'n_heads': self.n_heads,
                'attention_dropout': self.attention_dropout,
                'ff_dropout': self.ff_dropout,
                'activation': self.activation,
                'use_feature_scaling': self.use_feature_scaling,
                'spread': self.spread,
            }
        }
        
        torch.save(save_data, filepath)
        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath='data/models/transformer_classifier.pt', device=None):
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            config = checkpoint.get('config', {})
            
            if device is None:
                device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device_obj = torch.device(device)
            
            model = cls(
                n_layers=config.get('n_layers', 4),
                d_token=config.get('d_token', 64),
                n_heads=config.get('n_heads', 4),
                attention_dropout=config.get('attention_dropout', 0.2),
                ff_dropout=config.get('ff_dropout', 0.2),
                activation=config.get('activation', 'gelu'),
                device=device_obj,
                use_feature_scaling=config.get('use_feature_scaling', True),
                spread=config.get('spread', 0.0)
            )
            
            model.feature_cols = checkpoint.get('feature_cols')
            model.scaler = checkpoint.get('scaler')
            
            num_features = checkpoint.get('num_features')
            if num_features is None:
                raise ValueError("Cannot determine number of features from checkpoint")
            
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
            
            model.model = FTTransformer(
                feature_tokenizer=feature_tokenizer,
                transformer=transformer
            ).to(device_obj)
            
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.model.eval()
            
            print(f"Model loaded from: {filepath}")
            print(f"  - Device: {device_obj}")
            print(f"  - Features: {len(model.feature_cols) if model.feature_cols else 'N/A'}")
            
            best_metrics = checkpoint.get('best_metrics')
            if best_metrics:
                print(f"  - Best validation PnL: {best_metrics.get('pnl', 'N/A'):.4f}")
            
            return model
            
        except Exception as e:
            raise ValueError(f"Error loading model from {filepath}: {str(e)}")


def train_model():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df


    model = TransformerClassifier(
        n_layers=3,
        d_token=64,
        n_heads=4,
        attention_dropout=0.2,
        ff_dropout=0.2,
        activation='gelu',
        batch_size=1024,
        epochs=10,
        learning_rate=2e-4,
        weight_decay=1e-4,
        use_feature_scaling=True,
        label_smoothing=0.05,
        warmup_epochs=10,
        use_cosine_annealing=True,
        spread=0.0,
        target_trade_rate=0.25,  # Target 25% trade rate
        trade_rate_reg_strength=0.5  # Regularization strength
    )
    
    model.fit(train_df[FEATURE_COLS], train_df['label'])
    model.save('./data/models/transformer_classifier.pt')
    return model

def evaluate_model():
    dataset = Dataset()
    test_df = dataset.test_df

    model = TransformerClassifier.load('./data/models/transformer_classifier.pt')
    
    prob = model.predict_proba(test_df[FEATURE_COLS])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.0)
    print("\nEvaluation Results (using loaded model):")
    print(ret)

def tune_hyperparameters(n_trials=100, epochs_per_trial=5):
    """
    Use Optuna to tune hyperparameters for TransformerClassifier to maximize PnL.
    
    Parameters:
    -----------
    n_trials : int
        Number of Optuna trials to run
    epochs_per_trial : int
        Number of epochs to train per trial (reduced for faster search)
    """
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df
    
    # Split train into train and validation for tuning
    split_idx = int(len(train_df) * 0.9)
    train_tune_df = train_df.iloc[:split_idx].copy()
    val_tune_df = train_df.iloc[split_idx:].copy()
    
    def objective(trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 3, 8)
        d_token = trial.suggest_categorical("d_token", [32, 48, 64, 96, 128])
        # n_heads must divide d_token evenly
        n_heads_options = [h for h in [4, 8, 16] if d_token % h == 0]
        if not n_heads_options:
            n_heads_options = [4]  # fallback
        n_heads = trial.suggest_categorical("n_heads", n_heads_options)
        
        attention_dropout = trial.suggest_float("attention_dropout", 0.1, 0.4)
        ff_dropout = trial.suggest_float("ff_dropout", 0.1, 0.4)
        activation = trial.suggest_categorical("activation", ["gelu", "relu"])
        batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 20)
        use_cosine_annealing = trial.suggest_categorical("use_cosine_annealing", [True, False])
        spread = trial.suggest_float("spread", 0.0, 0.1)
        target_trade_rate = trial.suggest_float("target_trade_rate", 0.15, 0.35)  # 15% to 35%
        trade_rate_reg_strength = trial.suggest_float("trade_rate_reg_strength", 0.1, 1.0)
        
        print(f"\nTrial {trial.number}: n_layers={n_layers}, d_token={d_token}, n_heads={n_heads}, "
              f"lr={learning_rate:.2e}, batch_size={batch_size}")
        
        try:
            # Create model with suggested hyperparameters
            model = TransformerClassifier(
                n_layers=n_layers,
                d_token=d_token,
                n_heads=n_heads,
                attention_dropout=attention_dropout,
                ff_dropout=ff_dropout,
                activation=activation,
                batch_size=batch_size,
                epochs=epochs_per_trial,  # Use fewer epochs for faster search
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                use_feature_scaling=True,
                label_smoothing=label_smoothing,
                warmup_epochs=warmup_epochs,
                use_cosine_annealing=use_cosine_annealing,
                spread=spread
            )
            
            # Train on tuning set
            model.fit(train_tune_df[FEATURE_COLS], train_tune_df['label'])
            
            # Evaluate on validation set
            val_prob = model.predict_proba(val_tune_df[FEATURE_COLS])
            val_tune_df_eval = val_tune_df.copy()
            val_tune_df_eval['probability'] = val_prob[:, 1]
            
            # Calculate PnL
            ret = dataset.evaluate_model_metrics(
                val_tune_df_eval, 
                probability_column='probability', 
                spread=spread
            )
            
            val_pnl = ret.get('total_pnl', 0.0)
            if np.isnan(val_pnl):
                val_pnl = 0.0
            
            print(f"Trial {trial.number} PnL: {val_pnl:.4f}")
            
            # Clean up GPU memory if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return float(val_pnl)
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float('-inf')  # Return very bad score for failed trials
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="transformer_classifier_pnl_optimization"
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna hyperparameter tuning")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}")
    print(f"Epochs per trial: {epochs_per_trial}")
    print(f"Training samples: {len(train_tune_df)}")
    print(f"Validation samples: {len(val_tune_df)}")
    print(f"{'='*60}\n")
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Print results
    print(f"\n{'='*60}")
    print("Optuna Tuning Complete!")
    print(f"{'='*60}")
    print(f"Best PnL: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    best_model = TransformerClassifier(
        n_layers=study.best_params['n_layers'],
        d_token=study.best_params['d_token'],
        n_heads=study.best_params['n_heads'],
        attention_dropout=study.best_params['attention_dropout'],
        ff_dropout=study.best_params['ff_dropout'],
        activation=study.best_params['activation'],
        batch_size=study.best_params['batch_size'],
        epochs=150,  # Use full epochs for final model
        learning_rate=study.best_params['learning_rate'],
        weight_decay=study.best_params['weight_decay'],
        use_feature_scaling=True,
        label_smoothing=study.best_params['label_smoothing'],
        warmup_epochs=study.best_params['warmup_epochs'],
        use_cosine_annealing=study.best_params['use_cosine_annealing'],
        spread=study.best_params['spread'],
        target_trade_rate=study.best_params.get('target_trade_rate', 0.25),
        trade_rate_reg_strength=study.best_params.get('trade_rate_reg_strength', 0.5)
    )
    
    best_model.fit(train_df[FEATURE_COLS], train_df['label'])
    best_model.save('./data/models/transformer_classifier_optuna.pt')
    
    # Evaluate on test set
    print("\nEvaluating best model on test set...")
    test_prob = best_model.predict_proba(test_df[FEATURE_COLS])
    test_df_eval = test_df.copy()
    test_df_eval['probability'] = test_prob[:, 1]
    ret = dataset.evaluate_model_metrics(
        test_df_eval, 
        probability_column='probability', 
        spread=study.best_params['spread']
    )
    print("\nTest Set Results (best Optuna model):")
    print(ret)
    
    return best_model, study.best_params



def main():
    tune_hyperparameters(n_trials=10, epochs_per_trial=10)
    # evaluate_model()

if __name__ == "__main__":
    main()
