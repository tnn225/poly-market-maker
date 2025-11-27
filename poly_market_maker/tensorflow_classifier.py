import os

import tensorflow as tf
import numpy as np
from poly_market_maker.dataset import Dataset

class TensorflowClassifier:
    def __init__(self):
        self.model = None

    def fit(self, X, y, batch_size=64):
        self.dataset = Dataset()
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X, y)) \
            .shuffle(len(X)) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        normalizer = tf.keras.layers.Normalization(axis=-1)
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

        Returns a numpy array of predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")        
        
        if hasattr(X, "values"):
            X = X.values
        self.y_pred = self.model.predict_proba(X)
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
   
        X = np.array([[row[k] for k in self.feature_cols]], dtype='float32')
        prediction = self.model.predict(X).flatten()[0]
        return float(np.clip(prediction, 0.0, 1.0))

def main():
    model = TensorflowClassifier(filename='./data/models/tensorflow.keras')

    prob = model.get_prediction(87576.878879, 87628.002972, 686, 0.4, 0.41)
    print(f'Probability {prob:.4f}')

if __name__ == "__main__":
    main()