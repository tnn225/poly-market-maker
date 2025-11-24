import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from poly_market_maker.utils import setup_logging
from poly_market_maker.prediction_engine import PredictionEngine
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Ensure ./data exists
os.makedirs("./data", exist_ok=True)

class Backtest:
    # Store percentile thresholds for delta binning (shared across instances)
    _delta_percentiles = None
    
    def __init__(self, dates):
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

        self.df['timestamp'] = self.df['timestamp'].astype(int)
        self.df['price'] = self.df['price'].astype(float)
        self.df['bid'] = self.df['bid'].astype(float)
        self.df['ask'] = self.df['ask'].astype(float)

        self._add_target_and_is_up()

    def read_rows(self, date):
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

        self.df["delta"] = self.df["price"] - self.df["target"]
        
        self.df['seconds_left'] = 900 - (self.df['timestamp'] - self.df['interval'])

        self.df = self.df.drop(columns=['interval'])
        
        # Bin delta into 100 percentile bins using fixed percentile thresholds
        # This allows handling new delta values outside the original range
        delta_values = self.df['delta'].dropna()
        if len(delta_values) > 0:
            delta_percentiles = np.percentile(delta_values, np.linspace(0, 100, 101))
            # Remove duplicates while preserving order
            delta_percentiles = pd.Series(delta_percentiles).drop_duplicates().values
            # Ensure min and max are extended for handling out-of-range values
            if len(delta_percentiles) > 1:
                delta_percentiles[0] = delta_values.min() - 1e-10
                delta_percentiles[-1] = delta_values.max() + 1e-10
                
                # Store percentiles sorted for faster binary search lookup
                self._delta_percentiles = np.sort(delta_percentiles)
                
                try:
                    self.df['delta_bin'] = pd.cut(
                        self.df['delta'], 
                        bins=delta_percentiles,
                        labels=False,
                        include_lowest=True
                    )
                except (ValueError, IndexError):
                    # Fallback if cut fails
                    unique_deltas = delta_values.nunique()
                    n_bins = min(100, unique_deltas)
                    if n_bins > 1:
                        self.df['delta_bin'] = pd.qcut(
                            self.df['delta'], 
                            q=n_bins, 
                            labels=False, 
                            duplicates='drop'
                        )
                    else:
                        self.df['delta_bin'] = 0
            else:
                self.df['delta_bin'] = 0
        else:
            self.df['delta_bin'] = 0
        
        # Bin seconds_left into 10 bins
        self.df['seconds_left_bin'] = pd.cut(
            self.df['seconds_left'],
            bins=10,
            labels=False,
            include_lowest=True
        )
        
        # Fill NaN values
        self.df['delta_bin'] = self.df['delta_bin'].fillna(0).astype(int)
        self.df['seconds_left_bin'] = self.df['seconds_left_bin'].fillna(0).astype(int)
        
        # Create table grouped by seconds_left_bin and delta_bin
        self._create_table()

    def _create_table(self):
        """Create table grouped by seconds_left_bin and delta_bin"""
        self.table = self.df.groupby(['seconds_left_bin', 'delta_bin']).agg({
            'is_up': ['count', 'mean']
        }).round(4)
        
        self.table.columns = ['count', 'mean']
        self.table = self.table.reset_index()

    def get_delta_bin(self, delta: float) -> int:
        """Get delta bin number using binary search for O(log n) lookup"""
        if self._delta_percentiles is None or len(self._delta_percentiles) < 2:
            return 0
        
        percentiles = self._delta_percentiles
        
        # Handle values outside the range
        if delta < percentiles[0]:
            return 0
        if delta >= percentiles[-1]:
            return len(percentiles) - 2
        
        # Use binary search for O(log n) lookup
        idx = np.searchsorted(percentiles, delta, side='right') - 1
        # Ensure idx is within valid range
        idx = max(0, min(idx, len(percentiles) - 2))
        return idx



def main():
    spread = 0.1   # 10% spread
    
    today = datetime.now()
    dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    dates.reverse()
    
    # Initialize PredictionEngine to get probability function
    prediction_engine = PredictionEngine()
    
    # Initialize Backtest with data
    backtest = Backtest(dates)
    
    # Calculate probability for each row
    backtest.df['probability'] = backtest.df.apply(
        lambda row: prediction_engine.get_probability(
            row['price'],
            row['target'],
            row['seconds_left']
        ),
        axis=1
    )
    
    # Action = buy if get_probability > bid + spread
    backtest.df['action'] = backtest.df['probability'] > (backtest.df['bid'] + spread)
    
    # Filter to only buy actions
    df_buy = backtest.df[backtest.df['action'] == True].copy()
    
    # Calculate P&L
    # Revenue = is_up (1 if True, 0 if False)
    # Cost = bid
    # P&L = Revenue - Cost
    df_buy['revenue'] = df_buy['is_up'].astype(float)
    df_buy['cost'] = df_buy['bid']
    df_buy['pnl'] = df_buy['revenue'] - df_buy['cost']
    
    # Display summary statistics
    print(f"\n{'='*80}")
    print("Backtest Results")
    print(f"{'='*80}")
    print(f"Total rows: {len(backtest.df)}")
    print(f"Buy actions: {len(df_buy)}")
    print(f"Total Revenue: {df_buy['revenue'].sum():.2f}")
    print(f"Total Cost: {df_buy['cost'].sum():.2f}")
    print(f"Total P&L: {df_buy['pnl'].sum():.2f}")
    print(f"Average P&L per trade: {df_buy['pnl'].mean():.4f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()