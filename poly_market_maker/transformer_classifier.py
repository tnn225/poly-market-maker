import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
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
        n_layers=4,
        d_token=48,
        n_heads=8,
        attention_dropout=0.1,
        ff_dropout=0.1,
        activation='gelu',
        batch_size=128,
        epochs=100,
        learning_rate=3e-4,
        weight_decay=1e-5,
        device=None
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
        
        num_features = X.shape[1]
        
        # Compute class weights for imbalanced datasets
        # Increase positive class weight to make model less conservative
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Boost positive class weight to encourage more positive predictions (less conservative)
        # This will increase recall at the cost of some precision
        if 1 in class_weight_dict:
            class_weight_dict[1] = class_weight_dict[1] * 1.5  # Increase positive class weight by 50%
        
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
        
        # Loss function (weighted BCE)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Training loop
        # Use recall-weighted F1 to make model less conservative
        # This balances precision and recall but gives more weight to recall
        best_val_metric = -1.0
        best_model_state = None
        patience_counter = 0
        patience = 15
        
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
                loss = criterion(logits, batch_y)
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
            
            # Update learning rate based on recall-weighted F1
            beta = 1.5
            if val_precision + val_recall > 0:
                recall_weighted_f1 = (1 + beta**2) * (val_precision * val_recall) / (beta**2 * val_precision + val_recall)
            else:
                recall_weighted_f1 = 0.0
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
            if recall_weighted_f1 > best_val_metric + 5e-4:  # min_delta
                best_val_metric = recall_weighted_f1
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save best model
                os.makedirs('data/models', exist_ok=True)
                torch.save({
                    'model_state_dict': best_model_state,
                    'feature_cols': self.feature_cols,
                    'num_features': num_features,
                    'config': {
                        'n_layers': self.n_layers,
                        'd_token': self.d_token,
                        'n_heads': self.n_heads,
                        'attention_dropout': self.attention_dropout,
                        'ff_dropout': self.ff_dropout,
                        'activation': self.activation,
                    }
                }, 'data/models/transformer_classifier_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation recall-weighted F1: {best_val_metric:.4f} at epoch {epoch+1 - patience_counter}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with validation recall-weighted F1: {best_val_metric:.4f}")
        
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
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor, None).squeeze(-1)
            prediction = torch.sigmoid(logits).cpu().numpy()[0]
        
        return float(np.clip(prediction, 0.0, 1.0))


def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

    # Create FT Transformer with optimized hyperparameters
    model = TransformerClassifier(
        n_layers=4,              # Number of transformer layers
        d_token=48,               # Token dimension
        n_heads=8,                # Number of attention heads
        attention_dropout=0.1,    # Attention dropout
        ff_dropout=0.1,            # Feed-forward dropout
        activation='gelu',        # Activation function
        batch_size=128,
        epochs=100,
        learning_rate=3e-4,       # Learning rate for AdamW
        weight_decay=1e-5         # Weight decay for regularization
    )
    
    model.fit(train_df[feature_cols], train_df['label'])
    
    # Evaluate
    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print("\nEvaluation Results:")
    print(ret)
    
    # Example predictions
    print(f"\nExample prediction: {model.get_probability(87684.42, 87498.59, 60, 0.53, 0.55)}")
    print(f"Example prediction: {model.get_probability(87398.59, 87584.42, 60, 0.45, 0.47)}")


if __name__ == "__main__":
    main()
