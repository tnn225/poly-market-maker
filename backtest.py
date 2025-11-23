import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from poly_market_maker.utils import setup_logging
from poly_market_maker.prediction_engine import PricePrediction
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Ensure ./data exists
os.makedirs("./data", exist_ok=True)

class Backtest:
    """
    Backtest class for analyzing market data and predicting price movements.
    
    Processes historical price data to:
    - Calculate target prices and price direction (is_up)
    - Generate predictions using PricePrediction engine
    - Analyze P&L optimization strategies
    """
    
    def __init__(self, dates):
        """
        Initialize Backtest with historical data from specified dates.
        
        Args:
            dates: List of date strings in format "YYYY-MM-DD"
        """
        # Load data from CSV files for each date
        dataframes = []
        for date in dates:
            df = self.read_rows(date)
            if df is not None and not df.empty:
                dataframes.append(df)
        
        # Combine all dataframes
        if dataframes:
            self.df = pd.concat(dataframes, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=["timestamp", "price", "bid", "ask"])
        
        # Process data: add target, is_up, delta, and predictions
        self._add_target_and_is_up()
        self._add_prediction()

    def read_rows(self, date):
        """
        Read price data from CSV file for a specific date.
        
        Args:
            date: Date string in format "YYYY-MM-DD"
            
        Returns:
            DataFrame with columns: timestamp, price, bid, ask, or None if file not found
        """
        path = f"./data/price_{date}.csv"
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return None
        
        try:
            df = pd.read_csv(path)
            return df
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    def _add_target_and_is_up(self):
        """
        Calculate target price and price direction (is_up) for each 15-minute interval.
        
        For each interval:
        - target: price at the start of the interval (timestamp % 900 == 0)
        - is_up: True if price at interval end (interval + 900) >= target, False otherwise
        
        Also calculates delta = log(price / target) as a log-return metric.
        """
        # Initialize columns
        self.df['target'] = None
        self.df['is_up'] = None
        
        # Calculate 15-minute interval start time (rounded down to nearest 900 seconds)
        self.df['interval'] = self.df['timestamp'] // 900 * 900
        
        # Create price lookup dictionary for O(1) access
        price_dict = dict(zip(self.df['timestamp'], self.df['price']))
        
        # Calculate target and is_up for each unique interval
        interval_values = {}
        for interval in self.df['interval'].unique():
            target = price_dict.get(interval)
            
            if target is not None:
                # Get price at the end of the interval (interval + 900 seconds)
                next_interval_price = price_dict.get(interval + 900)
                
                if next_interval_price is not None:
                    # is_up = True if price goes up or stays same
                    interval_values[interval] = {
                        'target': target,
                        'is_up': target <= next_interval_price
                    }
        
        # Apply the same target and is_up to all rows in [interval, interval+900)
        for interval, values in interval_values.items():
            mask = (self.df['interval'] == interval)
            self.df.loc[mask, 'target'] = values['target']
            self.df.loc[mask, 'is_up'] = values['is_up']

        # Filter out rows where target or is_up couldn't be calculated
        self.df = self.df[(self.df['is_up'].notna()) & (self.df['target'].notna())].copy()

        # Calculate delta as log-return: log(price / target)
        # This represents the logarithmic difference between current price and target
        # Use pandas operations to avoid numpy/pandas compatibility issues
        self.df['price'] = self.df['price'].astype(float)
        self.df['target'] = self.df['target'].astype(float)
        self.df["delta"] = np.log(self.df["price"] / self.df["target"])

        # Clean up temporary interval column
        self.df = self.df.drop(columns=['interval'])
        

    def _add_prediction(self):
        """
        Calculate prediction probability (up) using PricePrediction engine.
        
        For each row:
        - seconds_left: Time remaining in the current 15-minute interval
        - up: Probability that price will reach or exceed target by interval end
        """
        # Calculate interval start time
        self.df['interval'] = self.df['timestamp'] // 900 * 900
        
        # Calculate seconds remaining in current interval
        # Formula: 900 - (timestamp - interval_start)
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])
        
        # Initialize prediction engine with historical price data
        prediction_engine = PricePrediction()
        
        # Populate engine with all prices from dataframe for volatility estimation
        prediction_engine.prices.clear()
        for price in self.df['price'].values:
            prediction_engine.prices.append(price)
        
        # Recalculate volatility (sigma) using historical prices
        prediction_engine.sigma = prediction_engine.estimate_sigma()
        
        # Calculate prediction probability for each row
        # up = probability that price >= target given current price, target, and time remaining
        self.df['up'] = self.df.apply(
            lambda row: prediction_engine.get_probability(
                row['price'], 
                row['target'], 
                row['seconds_left']
            ),
            axis=1
        )
        
        # Clean up temporary interval column
        self.df = self.df.drop(columns=['interval'])

def train_model(df):
    """
    Train a Random Forest classifier to predict is_up from delta and seconds_left.
    
    Args:
        df: DataFrame with columns 'delta', 'seconds_left', and 'is_up'
        
    Returns:
        Trained RandomForestClassifier model, or None if insufficient data
    """
    # Prepare features (X) and target (y)
    X = df[['delta', 'seconds_left']].copy()
    y = df['is_up'].copy()
    
    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        logger.warning("No valid data for model training")
        return None
    
    # Convert target to integer (0 or 1) for classification
    # is_up is boolean, convert to 0/1 for sklearn
    y = y.astype(bool).astype(int)
    
    # Split data: 80% training, 20% testing
    # stratify ensures balanced class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest classifier
    # n_estimators: number of trees
    # max_depth: limit tree depth to prevent overfitting
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print("Model Performance")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"{'='*60}\n")
    
    # Display feature importance to understand which feature is more predictive
    feature_importance = pd.DataFrame({
        'feature': ['delta', 'seconds_left'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Feature Importance:")
    print(feature_importance.to_string(index=False))
    print()
    
    return model

def train_and_evaluate_models(df, top_n=5):
    """
    Train multiple models with different configurations and select top performers.
    
    Args:
        df: DataFrame with columns 'delta', 'seconds_left', and 'is_up'
        top_n: Number of top models to return (default: 5)
        
    Returns:
        List of tuples: (model, model_name, metrics_dict) for top N models
    """
    # Prepare features and target
    X = df[['delta', 'seconds_left']].copy()
    y = df['is_up'].copy()
    
    # Remove rows with missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        logger.warning("No valid data for model training")
        return []
    
    # Convert target to integer
    y = y.astype(bool).astype(int)
    
    # Check if we have both classes for stratification
    unique_classes = y.unique()
    use_stratify = len(unique_classes) > 1
    
    # Split data
    # Use stratify only if we have both classes, otherwise it will fail
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        logger.warning(f"Only one class found in target ({unique_classes}), not using stratification")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Define model configurations to test
    model_configs = [
        # Random Forest variations
        {
            'name': 'RandomForest_100_10',
            'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        },
        {
            'name': 'RandomForest_200_15',
            'model': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        },
        {
            'name': 'RandomForest_50_5',
            'model': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        },
        # Gradient Boosting variations
        {
            'name': 'GradientBoosting_100_3',
            'model': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        },
        {
            'name': 'GradientBoosting_200_5',
            'model': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        },
        # Logistic Regression
        {
            'name': 'LogisticRegression',
            'model': LogisticRegression(random_state=42, max_iter=1000)
        },
        # Decision Tree
        {
            'name': 'DecisionTree_10',
            'model': DecisionTreeClassifier(max_depth=10, random_state=42)
        },
        {
            'name': 'DecisionTree_15',
            'model': DecisionTreeClassifier(max_depth=15, random_state=42)
        },
    ]
    
    # Train and evaluate all models
    results = []
    print(f"\n{'='*80}")
    print(f"Training and Evaluating {len(model_configs)} Models")
    print(f"{'='*80}\n")
    
    for config in model_configs:
        try:
            model = config['model']
            model_name = config['name']
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            results.append((model, model_name, metrics))
            
            print(f"{model_name:30s} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {config['name']}: {e}")
            continue
    
    # Sort by F1 score (or accuracy) and select top N
    results.sort(key=lambda x: x[2]['f1_score'], reverse=True)
    top_models = results[:top_n]
    
    print(f"\n{'='*80}")
    print(f"Top {top_n} Models (ranked by F1 Score)")
    print(f"{'='*80}")
    for i, (model, name, metrics) in enumerate(top_models, 1):
        print(f"{i}. {name:30s} | F1: {metrics['f1_score']:.4f} | Accuracy: {metrics['accuracy']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"{'='*80}\n")
    
    return top_models

def analyze_pnl_strategy(df):
    """
    Analyze optimal trading strategy to maximize P&L.
    
    P&L Formula: revenue - cost
    - revenue: 1 for each true positive (is_up = True), 0 otherwise
    - cost: bid price paid for each trade
    
    Analyzes multiple strategies:
    1. P&L by bid price ranges
    2. P&L by prediction probability (up) ranges
    3. Optimal spread threshold for buy condition
    4. Best delta and seconds_left combinations
    5. Expected value analysis
    
    Args:
        df: DataFrame with columns: is_up, bid, up, delta, seconds_left
    """
    print(f"\n{'='*80}")
    print("P&L Optimization Analysis")
    print(f"{'='*80}")
    
    # Calculate P&L for each row
    # P&L = revenue (1 if true positive) - cost (bid price)
    df['pnl'] = df['is_up'].astype(int) - df['bid']
    
    # Strategy 1: Analyze P&L by bid price ranges
    # Identifies which bid price ranges are most profitable
    print("\n1. P&L by Bid Ranges:")
    bid_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bid_labels = [f"{bid_bins[i]:.1f}-{bid_bins[i+1]:.1f}" for i in range(len(bid_bins)-1)]
    df['bid_range'] = pd.cut(df['bid'], bins=bid_bins, labels=bid_labels, include_lowest=True)
    
    bid_analysis = df.groupby('bid_range').agg({
        'pnl': ['sum', 'mean', 'count'],
        'is_up': 'mean',  # True positive rate
        'bid': 'mean'
    }).round(4)
    bid_analysis.columns = ['Total P&L', 'Avg P&L per Trade', 'Count', 'True Positive Rate', 'Avg Bid']
    print(bid_analysis)
    
    # Strategy 2: Analyze P&L by prediction probability (up) ranges
    # Shows how prediction confidence correlates with profitability
    print("\n2. P&L by Up (Prediction) Ranges:")
    up_bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    up_labels = [f"{up_bins[i]:.1f}-{up_bins[i+1]:.1f}" for i in range(len(up_bins)-1)]
    df['up_range'] = pd.cut(df['up'], bins=up_bins, labels=up_labels, include_lowest=True)
    
    up_analysis = df.groupby('up_range').agg({
        'pnl': ['sum', 'mean', 'count'],
        'is_up': 'mean',
        'bid': 'mean'
    }).round(4)
    up_analysis.columns = ['Total P&L', 'Avg P&L per Trade', 'Count', 'True Positive Rate', 'Avg Bid']
    print(up_analysis)
    
    # Strategy 3: Find optimal spread threshold for buy condition
    # Tests different spread values in condition: up - spread > bid
    # Goal: Find spread that maximizes total P&L
    print("\n3. Optimal Spread Threshold Analysis:")
    spread_values = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2]
    spread_results = []
    
    for spread in spread_values:
        # Buy condition: prediction probability minus spread must exceed bid
        action_mask = df['up'] - spread > df['bid']
        action_df = df[action_mask]
        
        if len(action_df) > 0:
            total_pnl = action_df['pnl'].sum()
            avg_pnl = action_df['pnl'].mean()
            count = len(action_df)
            true_positive_rate = action_df['is_up'].mean()
            avg_bid = action_df['bid'].mean()
            
            spread_results.append({
                'Spread': spread,
                'Total P&L': total_pnl,
                'Avg P&L per Trade': avg_pnl,
                'Count': count,
                'True Positive Rate': true_positive_rate,
                'Avg Bid': avg_bid
            })
    
    spread_df = pd.DataFrame(spread_results)
    if not spread_df.empty:
        print(spread_df.to_string(index=False))
        best_spread = spread_df.loc[spread_df['Total P&L'].idxmax()]
        print(f"\nBest Spread: {best_spread['Spread']:.3f} with Total P&L: {best_spread['Total P&L']:.2f}")
    
    # Strategy 4: Analyze P&L by delta and seconds_left combinations
    # Identifies which combinations of price deviation and time remaining are most profitable
    print("\n4. P&L by Delta and Seconds_left (Top 10 combinations):")
    # Use quantile-based binning to create equal-sized groups
    df['delta_bin'] = pd.qcut(df['delta'], q=10, duplicates='drop', labels=False)
    df['seconds_bin'] = pd.qcut(df['seconds_left'], q=5, duplicates='drop', labels=False)
    
    combo_analysis = df.groupby(['delta_bin', 'seconds_bin']).agg({
        'pnl': ['sum', 'mean', 'count'],
        'is_up': 'mean',
        'bid': 'mean',
        'delta': 'mean',
        'seconds_left': 'mean'
    }).round(4)
    combo_analysis.columns = ['Total P&L', 'Avg P&L', 'Count', 'TP Rate', 'Avg Bid', 'Avg Delta', 'Avg Seconds']
    combo_analysis = combo_analysis.sort_values('Total P&L', ascending=False).head(10)
    print(combo_analysis)
    
    # Strategy 5: Expected value analysis
    # E[P&L] = P(True Positive) * 1 - bid
    # Compares expected vs actual P&L by prediction probability ranges
    print("\n5. Expected Value Analysis (E[P&L] = P(TP) * 1 - bid):")
    df['expected_pnl'] = df['is_up'] - df['bid']
    ev_analysis = df.groupby('up_range').agg({
        'expected_pnl': 'mean',
        'pnl': 'mean',
        'is_up': 'mean',
        'bid': 'mean',
        'up': 'mean'
    }).round(4)
    ev_analysis.columns = ['Expected P&L', 'Actual Avg P&L', 'True Positive Rate', 'Avg Bid', 'Avg Up']
    ev_analysis = ev_analysis.sort_values('Expected P&L', ascending=False)
    print(ev_analysis)
    
    print(f"\n{'='*80}\n")

def main():
    """
    Main function to run backtest analysis.
    
    Steps:
    1. Load last 7 days of historical data
    2. Analyze P&L optimization strategies
    3. Train ML model to predict price direction
    4. Add model predictions to dataframe
    """
    # Calculate date range: last 7 days from today
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    dates.reverse()  # Sort oldest to newest for chronological processing
    
    # Initialize backtest with historical data
    backtest = Backtest(dates)
    
    # Analyze P&L optimization to find best trading strategies
    analyze_pnl_strategy(backtest.df)
    
    # Train multiple models and select top performers
    top_models = train_and_evaluate_models(backtest.df, top_n=5)
    
    # Use the best model (first in the list) for predictions
    if top_models:
        best_model, best_model_name, best_metrics = top_models[0]
        print(f"\nUsing best model: {best_model_name}")
        print(f"Best model metrics - F1: {best_metrics['f1_score']:.4f}, "
              f"Accuracy: {best_metrics['accuracy']:.4f}")
        
        # Add predictions from best model to dataframe
        mask = (backtest.df[['delta', 'seconds_left']].notna().all(axis=1) & 
                backtest.df['is_up'].notna())
        backtest.df.loc[mask, 'predicted_is_up'] = best_model.predict(
            backtest.df.loc[mask, ['delta', 'seconds_left']]
        )
        print(f"Predictions from best model added to dataframe")
        
        # Optionally: Add predictions from all top models
        for i, (model, name, metrics) in enumerate(top_models[:3], 1):  # Top 3 models
            col_name = f'predicted_is_up_model_{i}'
            backtest.df.loc[mask, col_name] = model.predict(
                backtest.df.loc[mask, ['delta', 'seconds_left']]
            )
        print(f"Predictions from top 3 models added to dataframe")
    else:
        print("No models were successfully trained")

if __name__ == "__main__":
    main()