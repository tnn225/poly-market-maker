import os
import pandas as pd
import numpy as np
from poly_market_maker.dataset import Dataset

import h2o
from h2o.automl import H2OAutoML


MAX_RUNTIME_SECS = 120
MAX_MODELS = 10
# Initialize H2O (safe to call multiple times - will reuse existing cluster if running)
# We'll initialize it lazily in the class or main function instead

class H2OClassifier:
    def __init__(self, max_models=MAX_MODELS, max_runtime_secs=MAX_RUNTIME_SECS, seed=42):
        """
        Initialize H2O AutoML Classifier
        
        Parameters:
        -----------
        max_models : int
            Maximum number of models to train
        max_runtime_secs : int
            Maximum runtime in seconds
        seed : int
            Random seed for reproducibility
        """
        self.model = None
        self.automl = None
        self.feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed

    def fit(self, X, y):
        """
        Train H2O AutoML model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Training features
        y : pandas.Series or pandas.DataFrame
            Training labels
        """
        # Store feature columns if X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_cols = X.columns.tolist()
        
        # Convert pandas DataFrame/Series to H2O Frame
        if isinstance(X, pd.DataFrame):
            train_h2o = h2o.H2OFrame(X)
        else:
            # If numpy array, convert to DataFrame first
            train_h2o = h2o.H2OFrame(pd.DataFrame(X, columns=self.feature_cols))
        
        # Convert y to H2O Frame
        if isinstance(y, pd.Series):
            y_df = pd.DataFrame({y.name if y.name else 'label': y})
        elif isinstance(y, pd.DataFrame):
            y_df = y
        else:
            # If numpy array, convert to DataFrame
            y_df = pd.DataFrame({'label': y})
        
        y_h2o = h2o.H2OFrame(y_df)
        
        # Combine features and target
        train_h2o['label'] = y_h2o
        
        # Convert label to categorical (factor) for binary classification
        train_h2o['label'] = train_h2o['label'].asfactor()
        
        # Identify feature columns (exclude label)
        feature_cols_h2o = [col for col in train_h2o.columns if col != 'label']
        
        # Initialize and train AutoML
        self.automl = H2OAutoML(
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            sort_metric="AUC",
            balance_classes=False,
            stopping_metric="AUC",
            stopping_tolerance=0.001,
            stopping_rounds=3,
        )
        
        self.automl.train(
            x=feature_cols_h2o,
            y='label',
            training_frame=train_h2o
        )
        
        # Get the best model
        self.model = self.automl.leader

        if self.model is None:
            # No models were trained – likely due to insufficient or invalid data
            n_rows = train_h2o.nrows
            n_cols = train_h2o.ncols
            raise RuntimeError(
                f"H2O AutoML did not train any models (leader is None). "
                f"Training frame shape: {n_rows} rows x {n_cols} cols. "
                f"Check that there are enough non-NA rows and both classes are present in 'label'."
            )

        print(f"H2O AutoML training complete. Best model: {self.model.model_id}")
        print(f"Best model AUC: {self.model.auc()}")

    def predict_proba(self, X):
        """
        Generate probability predictions
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features
        
        Returns:
        --------
        numpy.ndarray of shape (n_samples, 2)
            Probability predictions where:
            - Column 0: probability of class 0 (negative)
            - Column 1: probability of class 1 (positive)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Convert pandas DataFrame/Series to H2O Frame
        if isinstance(X, pd.DataFrame):
            test_h2o = h2o.H2OFrame(X)
        else:
            # If numpy array, convert to DataFrame first
            test_h2o = h2o.H2OFrame(pd.DataFrame(X, columns=self.feature_cols))
        
        # Get predictions
        predictions = self.model.predict(test_h2o)
        
        # Extract probabilities
        # H2O predictions have columns: predict, p0, p1
        # Try multi-threaded conversion if polars/pyarrow are available, otherwise use single-threaded
        try:
            prob_negative = predictions['p0'].as_data_frame(use_multi_thread=True).values.flatten()
            prob_positive = predictions['p1'].as_data_frame(use_multi_thread=True).values.flatten()
        except (TypeError, AttributeError):
            # Fall back to single-threaded if multi-threaded is not supported
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="h2o")
                prob_negative = predictions['p0'].as_data_frame().values.flatten()
                prob_positive = predictions['p1'].as_data_frame().values.flatten()
        
        # Return in sklearn format: (n_samples, 2)
        return np.column_stack([prob_negative, prob_positive])

    def save(self, path):
        """Save the model to a file"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        model_path = h2o.save_model(self.model, path=path, force=True)
        print(f"Model saved to: {model_path}")
        return model_path

    def load(self, path):
        """Load a model from a file"""
        self.model = h2o.load_model(path)
        print(f"Model loaded from: {path}")
        return self.model

def main():
    # Explicitly initialize H2O and fail fast with a clear error if it cannot start
    try:
        h2o.init()
    except Exception as e:
        raise RuntimeError(
            "Failed to start or connect to an H2O cluster. "
            "Ensure Java is installed and available on PATH, then retry."
        ) from e

    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
    
    model = H2OClassifier(max_models=MAX_MODELS, max_runtime_secs=MAX_RUNTIME_SECS)

    filename = 'data/models/h2o_classifier'
    if os.path.exists(filename):
        try:
            # Try to load the model (H2O saves models in a directory)
            model.load(filename)
            print(f"Model loaded from: {filename}")
        except Exception as e:
            print(f"Could not load model from {filename}: {e}")
            print("Training new model...")
            model.fit(train_df[feature_cols], train_df['label'])
            model.save(filename)
            print(f"Model saved to: {filename}")
    else:
        print("Training new model...")
        model.fit(train_df[feature_cols], train_df['label'])
        model.save(filename)
        print(f"Model saved to: {filename}")
    
    # Make predictions
    prob = model.predict_proba(test_df[feature_cols])
    test_df['probability'] = prob[:, 1]
    ret = dataset.evaluate_model_metrics(test_df, probability_column='probability', spread=0.05)
    print(ret)

if __name__ == "__main__":
    main()
