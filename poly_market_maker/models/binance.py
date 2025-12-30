import requests
import time
import os
import numpy as np
from datetime import datetime, timezone, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


BASE_URL = "https://api.binance.com/api/v3/klines"

filename = "./data/binance_data.csv"

class Binance:
    def __init__(self, symbol="BTCUSDT", interval="15m"):
        self.symbol = symbol
        self.interval = interval
        self.df = self.fetch(start_time=datetime.now(timezone.utc) - timedelta(days=365), end_time=datetime.now(timezone.utc))
        self.df = self._add_features(self.df)

    def fetch(self, start_time=None, end_time=None, limit=1000, sleep=0.2):
        """Fetch klines data. Returns DataFrame with ~35K rows for 1 year at 15m intervals."""
        # Load from file if it exists
        if os.path.exists(filename):
            print(f"Loading data from {filename}")
            df = pd.read_csv(filename)
            # Ensure timestamp columns are int
            df["open_time"] = df["open_time"].astype(int)
            df["close_time"] = df["close_time"].astype(int)
            return df
        
        print(f"Fetching data from {start_time} to {end_time}")
        
        start_ms = self._to_ms(start_time or datetime.now(timezone.utc) - timedelta(days=365))
        end_ms = self._to_ms(end_time or datetime.now(timezone.utc))
        
        all_rows = []
        current_start = start_ms
        
        while current_start < end_ms:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": limit,
            }
            
            r = requests.get(BASE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
            
            all_rows.extend(data)
            current_start = data[-1][0] + 1
            time.sleep(sleep)
        
        columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                   "quote_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"]
        
        df = pd.DataFrame(all_rows, columns=columns)
        df["open_time"] = (df["open_time"] / 1000).astype(int)  # Convert ms to seconds
        df["close_time"] = (df["close_time"] / 1000).astype(int)  # Convert ms to seconds
        
        for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]:
            df[col] = df[col].astype(float)
        # Ensure data directory exists
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Saved data to {filename}")
        return df

    def _add_features(self, df):
        df["taker_sell_base_vol"] = df["volume"] - df["taker_buy_base_vol"]
        df["taker_sell_quote_vol"] = df["quote_volume"] - df["taker_buy_quote_vol"]

        df["buy_ratio"] = df["taker_buy_base_vol"] / df["volume"]

        df["num_trades"] = df["num_trades"].astype(int)
        
        # Calculate technical features
        range_ = df['high'] - df['low']
        df["taker_sell_base_vol"] = df["volume"] - df["taker_buy_base_vol"]
        df["taker_sell_quote_vol"] = df["quote_volume"] - df["taker_buy_quote_vol"]

        df["buy_ratio"] = df["taker_buy_base_vol"] / df["volume"]

        df["num_trades"] = df["num_trades"].astype(int)
        
        # Calculate technical features
        range_ = df['high'] - df['low']
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        df['body_pct'] = body / range_.replace(0, np.nan)
        df['upper_wick_pct'] = upper_wick / range_.replace(0, np.nan)
        df['lower_wick_pct'] = lower_wick / range_.replace(0, np.nan)
        df['close_pos'] = (df['close'] - df['low']) / range_.replace(0, np.nan)
        
        # Returns (backward looking - past returns, not future)
        df['ret_1'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['ret_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        df['ret_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        df['ret_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # EMA differences with multiple periods (PC1 shows ema_diff is important)
        for ema_period in [10, 20, 50]:
            ema = df['close'].ewm(span=ema_period, adjust=False).mean()
            df[f'ema_diff_{ema_period}'] = (df['close'] - ema) / df['close']
        
        # Momentum: rate of change of returns
        df['ret_momentum'] = df['ret_1'] - df['ret_1'].shift(1)
        
        # Volatility: rolling std of returns
        df['ret_volatility'] = df['ret_1'].rolling(window=10).std()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['atr_pct'] = atr / df['close']
        
        # Volume features
        vol_window = 20
        vol_mean = df['volume'].rolling(window=vol_window).mean()
        vol_std = df['volume'].rolling(window=vol_window).std()
        df['vol_ratio'] = df['volume'] / vol_mean.replace(0, np.nan)
        df['vol_z'] = (df['volume'] - vol_mean) / vol_std.replace(0, np.nan)
        
        # Hour cyclical encoding
        df['datetime'] = pd.to_datetime(df['open_time'], unit='s', utc=True)
        df['hour'] = df['datetime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df = df.drop(columns=['datetime', 'hour'])
        
        # Feature interactions (based on PCA: important features appear together)
        df['body_vol_interaction'] = df['body_pct'] * df['vol_ratio']
        df['ret_vol_interaction'] = df['ret_1'] * df['vol_z']
        df['wick_balance'] = df['upper_wick_pct'] - df['lower_wick_pct']  # Net wick direction
        
        df['is_up'] = (df['close'] >= df['open']).astype(int)
        df['label'] = (df['close'].shift(-1) >= df['open'].shift(-1)).astype(int)
        return df
    
    def _to_ms(self, ts):
        """Convert to milliseconds timestamp."""
        if isinstance(ts, int):
            return ts
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "")).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        if isinstance(ts, datetime):
            return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)
        raise ValueError("Unsupported timestamp type")

    def train_test_split(self, test_ratio: float = 0.2):
        df_sorted = self.df.sort_values('open_time').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        split_idx = max(1, min(split_idx, len(df_sorted) - 1))
        self.train_df = df_sorted.iloc[:split_idx].copy()
        self.test_df = df_sorted.iloc[split_idx:].copy()
        return self.train_df, self.test_df


def evaluate(test_df, label_column: str = 'label', probability_column: str = 'probability', threshold: float = 0.5):
    y_true = test_df[label_column].astype(float)
    y_pred_proba = test_df[probability_column].astype(float)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    test_df['y_true'] = y_true
    test_df['y_pred_proba'] = y_pred_proba
    test_df['y_pred'] = y_pred
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"Model Performance Metrics")
    print(f"{'='*60}")
    print(f"ROC-AUC:      {auc:.4f}")
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"\nConfusion Matrix (threshold={threshold}):")
    print(f"  True Neg: {cm[0,0]:6d}  |  False Pos: {cm[0,1]:6d}")
    print(f"  False Neg: {cm[1,0]:6d}  |  True Pos:  {cm[1,1]:6d}")
    print(f"\nLabel Distribution:")
    class_0_count = int((y_true == 0).sum())
    class_1_count = int((y_true == 1).sum())
    total = len(y_true)
    print(f"  Class 0: {class_0_count:6d} ({class_0_count/total*100:.2f}%)")
    print(f"  Class 1: {class_1_count:6d} ({class_1_count/total*100:.2f}%)")
    print(f"{'='*60}\n")

    return test_df

def main():
    binance = Binance(symbol="BTCUSDT", interval="15m")
    
    train_df, test_df = binance.train_test_split()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    # Feature selection based on PCA analysis (prioritizing high-variance components)
    feature_cols = [
        # PC1 (20.63% variance): Returns and EMA - most important
        'ret_1',          # Past return over 1 period
        'ret_3',          # Past return over 3 periods
        'ret_5',          # Past return over 5 periods
        'ret_10',         # Past return over 10 periods
        'ema_diff_10',    # EMA(10) difference
        'ema_diff_20',    # EMA(20) difference (original)
        'ema_diff_50',    # EMA(50) difference
        'ret_momentum',   # Rate of change of returns
        'ret_volatility', # Volatility of returns
        
        # PC2 (16.63% variance): Volume metrics
        'vol_z',          # Volume z-score
        'vol_ratio',      # Volume relative to mean
        
        # PC3-PC4 (27.40% combined): Candlestick patterns
        'body_pct',       # Body size as percentage of range
        'upper_wick_pct', # Upper wick percentage
        'lower_wick_pct', # Lower wick percentage
        'close_pos',      # Close position in range
        'wick_balance',   # Net wick direction (upper - lower)
        
        # PC5-PC8 (25.93% combined): ATR and time features
        'atr_pct',        # ATR as percentage
        'hour_sin',       # Hour sine encoding
        'hour_cos',       # Hour cosine encoding
        
        # Feature interactions (combining important features)
        'body_vol_interaction',  # body_pct * vol_ratio
        'ret_vol_interaction',   # ret_1 * vol_z
    ]

    model.fit(train_df[feature_cols], train_df['label'])
    test_df['probability'] = model.predict_proba(test_df[feature_cols])[:, 1]
    
    # Plot stacked histogram of probabilities by label
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by label
    prob_label_0 = test_df[test_df['label'] == 0]['probability'].dropna()
    prob_label_1 = test_df[test_df['label'] == 1]['probability'].dropna()
    
    # Create stacked histogram
    plt.hist([prob_label_0, prob_label_1], bins=50, stacked=False, 
             color=['red', 'green'], alpha=0.7, edgecolor='black',
             label=['Label=0 (Red)', 'Label=1 (Green)'])
    
    plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Threshold 0.5')
    plt.xlabel('Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Probabilities by Label (Test Set)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()