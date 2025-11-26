import tensorflow as tf
import numpy as np
from poly_market_maker.dataset import Dataset

class Model:
    def __init__(self):
        self.dataset = Dataset()
        self.model = None
        self.feature_cols = ['delta', 'percent', 'log_return', 'time', 'seconds_left', 'bid', 'ask']
        self.label_col = 'label'  # binary classification target (is_up)

    def train(self, batch_size=64):
        X = self.dataset.train_df[self.feature_cols].astype('float32').values
        y = self.dataset.train_df[self.label_col].astype('float32').values

        val_X = self.dataset.test_df[self.feature_cols].astype('float32').values
        val_y = self.dataset.test_df[self.label_col].astype('float32').values

        self.train_dataset = tf.data.Dataset.from_tensor_slices((X, y)) \
            .shuffle(len(X)) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_X, val_y)) \
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

    def predict(self, batch_size=64):
        """
        Generate predictions for the test dataset.

        Returns a numpy array of predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if not hasattr(self.dataset, 'test_df'):
            raise ValueError("Test dataset not available. Call dataset.train_test_split() first.")
        
        X = self.dataset.test_df[self.feature_cols].astype('float32').values
        self.y_pred = self.model.predict(X, batch_size=batch_size).flatten()
        return self.y_pred

    def get_prediction(self, price, target, seconds_left, bid, ask):
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
   
        features = np.array(
            [[row[k] for k in self.feature_cols]],
            dtype='float32'
        )

        prediction = self.model.predict(features, batch_size=1).flatten()[0]
        return float(np.clip(prediction, 0.0, 1.0))

    def evaluate(self, spread: float = 0.05):
        """
        Evaluate the classifier on the test dataset.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        probs = self.predict()

        if not hasattr(self.dataset, 'test_df'):
            raise ValueError("Test dataset not available. Call dataset.train_test_split() first.")
        
        test_df = self.dataset.test_df.copy()
        test_df['probability'] = np.clip(probs, 0.0, 1.0)
        
        metrics = self.dataset.evaluate_model_metrics(
            test_df, 
            probability_column='probability', 
            spread=spread
        )
        
        summary, buy_df = self.dataset.evaluate_strategy(
            test_df, 
            spread=spread, 
            probability_column='probability'
        )
        
        # Combine metrics
        evaluation_results = {
            **metrics,
            **summary
        }
        
        print("\n" + "="*80)
        print("Model Evaluation Results")
        print("="*80)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Total P&L: {metrics['total_pnl']:.2f}")
        print(f"\nStrategy Summary:")
        print(f"  Total rows: {summary['total_rows']}")
        print(f"  Eligible rows: {summary['eligible_rows']}")
        print(f"  Buy trades: {summary['buy_trades']}")
        print(f"  Total revenue: {summary['total_revenue']:.2f}")
        print(f"  Total cost: {summary['total_cost']:.2f}")
        print(f"  Total P&L: {summary['total_pnl']:.2f}")
        print(f"  Avg P&L per trade: {summary['avg_pnl_per_trade']:.4f}")
        print("="*80 + "\n")
        
        return evaluation_results

def main():
    model = Model()
    model.train()
    model.evaluate()

    prob = model.get_prediction(87576.878879, 87628.002972, 686, 0.4, 0.41)
    print(f'Probability {prob:.4f}')

if __name__ == "__main__":
    main()