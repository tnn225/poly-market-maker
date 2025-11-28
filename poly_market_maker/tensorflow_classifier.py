import os

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from poly_market_maker.dataset import Dataset

class TensorflowClassifier:
    def __init__(self):
        self.model = None
        self.feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.batch_size = 64
        self.epochs = 20
        self.learning_rate = 1e-3
    def fit(self, X, y):
        # Convert to numpy arrays if needed (e.g., from pandas DataFrame)
        if hasattr(X, "values"):
            self.feature_cols = X.columns.tolist() if hasattr(X, "columns") else None
            X = X.values
        if hasattr(y, "values"):
            y = y.values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
            .shuffle(len(X_train)) \
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
            .batch(self.batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        # Get input shape from the data
        input_shape = X.shape[1] if len(X.shape) > 1 else X.shape[0]
        normalizer = tf.keras.layers.Normalization(axis=-1, input_shape=(input_shape,))
        normalizer.adapt(X)

        self.model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(
                256,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=20,
            verbose=1
        )
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
        
        # TensorFlow predict() returns probabilities directly (shape: n_samples, 1)
        # Convert to sklearn format (n_samples, 2) with [1-p, p]
        predictions = self.model.predict(X, batch_size=batch_size, verbose=0)
        # Reshape if needed
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        # Convert to sklearn format: [1-p, p] for each sample
        prob_positive = predictions.flatten()
        prob_negative = 1 - prob_positive
        self.y_pred = np.column_stack([prob_negative, prob_positive])
        return self.y_pred

    def get_probability(self, price, target, seconds_left, bid, ask):
        """Generate a single probability prediction for the provided snapshot."""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

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
        # TensorFlow predict() returns probabilities directly (shape: 1, 1) for sigmoid output
        prediction = self.model.predict(X, verbose=0).flatten()[0]
        return float(np.clip(prediction, 0.0, 1.0))

def main():
    dataset = Dataset()
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']

    model = TensorflowClassifier()
    model.fit(train_df[feature_cols], train_df['label'])
    # Note: To save the model, use: model.model.save('path/to/model')
    # model.model.save('data/models/tensorflow_classifier')

    prob = model.predict_proba(test_df[feature_cols])
    print(prob)

if __name__ == "__main__":
    main()