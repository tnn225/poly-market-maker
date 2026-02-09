import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class VolumeSpikeMartingaleBacktest:
    """
    Backtest strategy:
    - Detect volume spike (> X times average)
    - Enter opposite direction of spike candle
    - Martingale: if lose, double size up to N times
    """

    def __init__(
        self,
        data_path="data/btcusdt_data_15m.csv",
        vol_ma_period=20,
        vol_spike_multiplier=3.0,
        vol_prev_multiplier=2.0,
        min_delta=200.0,
        base_shares=10,
        max_martingale_levels=5,
        cost_per_share=0.5,
        win_per_share=1.0,
    ):
        """
        Args:
            data_path: Path to CSV data file
            vol_ma_period: Period to calculate Volume MA (default 20)
            vol_spike_multiplier: Volume spike threshold (X times Volume MA, default 3x)
            vol_prev_multiplier: Volume vs previous kline multiplier (default 2x)
            min_delta: Minimum price delta (absolute) to qualify as spike (default $200)
            base_shares: Base position size (10 shares = $5 cost)
            max_martingale_levels: Max martingale attempts
            cost_per_share: Cost per share when betting ($0.5 default, so 10 shares = $5)
            win_per_share: Win amount per share ($1.0 default, so 10 shares = $10 win)
        """
        self.data_path = data_path
        self.vol_ma_period = vol_ma_period
        self.vol_spike_multiplier = vol_spike_multiplier
        self.vol_prev_multiplier = vol_prev_multiplier
        self.min_delta = min_delta
        self.base_shares = base_shares
        self.max_martingale_levels = max_martingale_levels
        self.cost_per_share = cost_per_share
        self.win_per_share = win_per_share

        self.df = None
        self.trades = []

    def load_and_prepare_data(self):
        """Load data and calculate indicators."""
        # 1. Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records from {self.data_path}")

        # 2. Calculate up/down and delta
        self.df['delta'] = self.df['close_price'] - self.df['open_price']
        self.df['direction'] = np.where(self.df['delta'] > 0, 'up', 'down')
        self.df['delta_pct'] = (self.df['delta'] / self.df['open_price']) * 100

        # 3. Calculate Volume MA 20
        self.df['vol_ma'] = self.df['volume_btc'].rolling(
            window=self.vol_ma_period, min_periods=self.vol_ma_period
        ).mean()

        # 4. Detect volume spike conditions
        self.df['vol_ratio'] = self.df['volume_btc'] / self.df['vol_ma']
        self.df['abs_delta'] = self.df['delta'].abs()
        # Volume vs previous kline (must be >= 2x)
        self.df['vol_prev'] = self.df['volume_btc'].shift(1)
        self.df['vol_prev_ratio'] = self.df['volume_btc'] / self.df['vol_prev']

        self.df['is_spike'] = (
            (self.df['vol_ratio'] > self.vol_spike_multiplier) &
            (self.df['abs_delta'] > self.min_delta) &
            (self.df['vol_prev_ratio'] >= self.vol_prev_multiplier)
        )

        print(f"Data prepared. Columns: {list(self.df.columns)}")
        print(f"Volume MA period: {self.vol_ma_period}, Spike multiplier: {self.vol_spike_multiplier}x, Min delta: ${self.min_delta}")
        print(f"Volume vs prev kline: >= {self.vol_prev_multiplier}x")
        print(f"Volume spikes detected: {self.df['is_spike'].sum()}")

        return self.df

    def run_backtest(self):
        """Run the backtest with martingale strategy."""
        self.trades = []
        df = self.df.copy()

        i = self.vol_ma_period  # Start after enough data for vol_ma

        while i < len(df) - 1:
            row = df.iloc[i]

            # Check for volume spike
            if not row['is_spike']:
                i += 1
                continue

            # Volume spike detected - start martingale sequence
            spike_direction = row['direction']
            trade_direction = 'down' if spike_direction == 'up' else 'up'  # Opposite

            martingale_trades = self._execute_martingale_sequence(
                df, i, trade_direction
            )

            self.trades.extend(martingale_trades)

            # Move index past the martingale sequence
            if martingale_trades:
                last_trade = martingale_trades[-1]
                i = last_trade['exit_index'] + 1
            else:
                i += 1

        return self.trades

    def _execute_martingale_sequence(self, df, spike_index, trade_direction):
        """Execute martingale sequence starting from spike."""
        sequence_trades = []
        current_shares = self.base_shares
        entry_index = spike_index + 1  # Enter at next candle

        for level in range(self.max_martingale_levels):
            if entry_index >= len(df):
                break

            entry_row = df.iloc[entry_index]

            # Bet for single candle: open -> close of same candle
            entry_price = entry_row['open_price']
            exit_price = entry_row['close_price']

            # Determine if bet is correct (binary up/down)
            if trade_direction == 'up':
                is_win = exit_price > entry_price
            else:  # down
                is_win = exit_price < entry_price

            # Calculate PnL (binary bet: win = +$10 - $5 cost = +$5 net, lose = -$5 cost)
            bet_cost = current_shares * self.cost_per_share
            if is_win:
                pnl = current_shares * self.win_per_share - bet_cost  # Win $10 - $5 cost = +$5 net
            else:
                pnl = -bet_cost  # Lose $5

            trade = {
                'spike_index': spike_index,
                'spike_time': df.iloc[spike_index]['timestamp'],
                'entry_index': entry_index,
                'entry_time': entry_row['timestamp'],
                'exit_index': entry_index,
                'exit_time': entry_row['timestamp'],
                'direction': trade_direction,
                'martingale_level': level + 1,
                'shares': current_shares,
                'bet_cost': bet_cost,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'is_win': is_win,
                'vol_ratio': df.iloc[spike_index]['vol_ratio'],
            }
            sequence_trades.append(trade)

            if is_win:
                # Win - stop martingale sequence
                break
            else:
                # Lose - double position and continue
                current_shares *= 2
                entry_index += 1

        return sequence_trades

    def get_statistics(self):
        """Calculate backtest statistics."""
        if not self.trades:
            return {"error": "No trades executed"}

        trades_df = pd.DataFrame(self.trades)

        total_trades = len(trades_df)
        winning_trades = trades_df['is_win'].sum()
        losing_trades = total_trades - winning_trades

        total_pnl = trades_df['pnl'].sum()
        total_bet_cost = trades_df['bet_cost'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()

        # Martingale sequence stats
        sequences = trades_df.groupby('spike_index').agg({
            'pnl': 'sum',
            'martingale_level': 'max',
            'is_win': 'last'  # Did sequence end in win?
        }).reset_index()

        winning_sequences = sequences['is_win'].sum()
        total_sequences = len(sequences)

        # Stats by martingale level
        level_stats = trades_df.groupby('martingale_level').agg({
            'pnl': ['count', 'sum', 'mean'],
            'is_win': 'sum'
        })

        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades * 100 if total_trades > 0 else 0,
            'total_bet_cost': total_bet_cost,
            'total_pnl': total_pnl,
            'roi': (total_pnl / total_bet_cost * 100) if total_bet_cost > 0 else 0,
            'avg_pnl': avg_pnl,
            'max_win': max_win,
            'max_loss': max_loss,
            'total_sequences': total_sequences,
            'winning_sequences': winning_sequences,
            'sequence_win_rate': winning_sequences / total_sequences * 100 if total_sequences > 0 else 0,
            'level_stats': level_stats,
        }

        return stats

    def print_report(self):
        """Print detailed backtest report."""
        stats = self.get_statistics()

        if "error" in stats:
            print(stats["error"])
            return

        print("\n" + "=" * 70)
        print("BACKTEST REPORT - Volume Spike Martingale Strategy")
        print("=" * 70)

        print(f"\n--- Parameters ---")
        print(f"Volume MA Period:         {self.vol_ma_period}")
        print(f"Volume Spike Multiplier:  {self.vol_spike_multiplier}x (vs Volume MA)")
        print(f"Volume Prev Multiplier:   {self.vol_prev_multiplier}x (vs prev kline)")
        print(f"Min Delta:                ${self.min_delta}")
        print(f"Base Shares:              {self.base_shares}")
        print(f"Max Martingale Levels:    {self.max_martingale_levels}")
        print(f"Cost per Share:           ${self.cost_per_share}")
        print(f"Win per Share:            ${self.win_per_share}")
        print(f"(10 shares: cost ${self.base_shares * self.cost_per_share}, win ${self.base_shares * self.win_per_share})")

        print(f"\n--- Trade Statistics ---")
        print(f"Total Trades:             {stats['total_trades']}")
        print(f"Winning Trades:           {stats['winning_trades']}")
        print(f"Losing Trades:            {stats['losing_trades']}")
        print(f"Win Rate:                 {stats['win_rate']:.2f}%")

        print(f"\n--- PnL Statistics (Binary Bet) ---")
        print(f"Total Bet Cost:           ${stats['total_bet_cost']:,.2f}")
        print(f"Total PnL:                ${stats['total_pnl']:,.2f}")
        print(f"ROI:                      {stats['roi']:.2f}%")
        print(f"Average PnL per Trade:    ${stats['avg_pnl']:,.2f}")
        print(f"Max Win:                  ${stats['max_win']:,.2f}")
        print(f"Max Loss:                 ${stats['max_loss']:,.2f}")

        print(f"\n--- Martingale Sequence Statistics ---")
        print(f"Total Sequences:          {stats['total_sequences']}")
        print(f"Winning Sequences:        {stats['winning_sequences']}")
        print(f"Sequence Win Rate:        {stats['sequence_win_rate']:.2f}%")

        print(f"\n--- Stats by Martingale Level ---")
        print(stats['level_stats'])

        print("\n" + "=" * 70)

    def export_trades(self, output_path="data/backtest_btc_up_or_down.csv"):
        """Export trades to CSV."""
        if not self.trades:
            print("No trades to export")
            return

        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(output_path, index=False)
        print(f"Trades exported to {output_path}")

    def show_sample_trades(self, n=10):
        """Show sample trades."""
        if not self.trades:
            print("No trades")
            return

        trades_df = pd.DataFrame(self.trades)
        print(f"\n--- Sample Trades (first {n}) ---")
        print(trades_df.head(n).to_string(index=False))

    def visualize_trades(self, output_path="data/btc_trades_chart.png", show=True):
        """Visualize entry points on price chart."""
        if self.df is None or not self.trades:
            print("No data or trades to visualize")
            return

        trades_df = pd.DataFrame(self.trades)
        df = self.df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                            gridspec_kw={'height_ratios': [3, 1, 1]})

        # Plot 1: Price with entry points
        ax1.plot(df['timestamp'], df['close_price'], color='#2196F3', linewidth=0.8,
                 label='BTC Price', alpha=0.8)

        # Mark volume spikes
        spikes = df[df['is_spike'] == True]
        ax1.scatter(pd.to_datetime(spikes['timestamp']), spikes['close_price'],
                    color='yellow', marker='*', s=100, label='Volume Spike', zorder=3, alpha=0.7)

        # Mark ALL entry points (including martingale retries) - show individual trade win/lose
        # Up trades
        up_trades = trades_df[trades_df['direction'] == 'up']
        up_win = up_trades[up_trades['is_win'] == True]
        up_lose = up_trades[up_trades['is_win'] == False]

        if len(up_win) > 0:
            ax1.scatter(pd.to_datetime(up_win['entry_time']), up_win['entry_price'],
                        color='#00E676', marker='^', s=120, label=f'Up Win ({len(up_win)})',
                        zorder=5, edgecolors='black', linewidths=0.5)
        if len(up_lose) > 0:
            ax1.scatter(pd.to_datetime(up_lose['entry_time']), up_lose['entry_price'],
                        color='#FF5252', marker='^', s=120, label=f'Up Lose ({len(up_lose)})',
                        zorder=5, edgecolors='black', linewidths=0.5)

        # Down trades
        down_trades = trades_df[trades_df['direction'] == 'down']
        down_win = down_trades[down_trades['is_win'] == True]
        down_lose = down_trades[down_trades['is_win'] == False]

        if len(down_win) > 0:
            ax1.scatter(pd.to_datetime(down_win['entry_time']), down_win['entry_price'],
                        color='#00E676', marker='v', s=120, label=f'Down Win ({len(down_win)})',
                        zorder=5, edgecolors='black', linewidths=0.5)
        if len(down_lose) > 0:
            ax1.scatter(pd.to_datetime(down_lose['entry_time']), down_lose['entry_price'],
                        color='#FF5252', marker='v', s=120, label=f'Down Lose ({len(down_lose)})',
                        zorder=5, edgecolors='black', linewidths=0.5)

        ax1.set_title('BTC/USDT 15m - Volume Spike Martingale Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USDT)', fontsize=11)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # Plot 2: Volume with spike threshold
        ax2.bar(df['timestamp'], df['volume_btc'], color='#64B5F6', alpha=0.6, width=0.01)
        ax2.plot(df['timestamp'], df['vol_ma'], color='orange', linewidth=1.5, label=f'Volume MA {self.vol_ma_period}')
        ax2.plot(df['timestamp'], df['vol_ma'] * self.vol_spike_multiplier,
                 color='red', linestyle='--', linewidth=1, label=f'{self.vol_spike_multiplier}x Volume MA')
        ax2.set_ylabel('Volume (BTC)', fontsize=11)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        # Plot 3: Cumulative PnL
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        ax3.plot(pd.to_datetime(trades_df['entry_time']), trades_df['cum_pnl'],
                 color='#4CAF50', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.fill_between(pd.to_datetime(trades_df['entry_time']), trades_df['cum_pnl'], 0,
                         where=(trades_df['cum_pnl'] >= 0), color='green', alpha=0.3)
        ax3.fill_between(pd.to_datetime(trades_df['entry_time']), trades_df['cum_pnl'], 0,
                         where=(trades_df['cum_pnl'] < 0), color='red', alpha=0.3)
        ax3.set_ylabel('Cumulative PnL ($)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

        plt.tight_layout()
        if show:
            print("Showing interactive chart... (close window to continue)")
            plt.show()
        else:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Chart saved to {output_path}")


def run_parameter_sweep():
    """Run backtest with different parameters for tuning."""
    results = []

    # Parameter combinations to test
    vol_spike_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6, 7]
    vol_ma_periods = [10, 20, 30, 40, 50, 100]
    max_levels = [3, 4, 5, 6, 7]

    for spike_mult in vol_spike_multipliers:
        for ma_period in vol_ma_periods:
            for max_lvl in max_levels:
                bt = VolumeSpikeMartingaleBacktest(
                    vol_ma_period=ma_period,
                    vol_spike_multiplier=spike_mult,
                    max_martingale_levels=max_lvl,
                )
                bt.load_and_prepare_data()
                bt.run_backtest()
                stats = bt.get_statistics()

                if "error" not in stats:
                    results.append({
                        'vol_spike_mult': spike_mult,
                        'vol_ma_period': ma_period,
                        'max_martingale': max_lvl,
                        'total_trades': stats['total_trades'],
                        'total_bet_cost': stats['total_bet_cost'],
                        'total_pnl': stats['total_pnl'],
                        'roi': stats['roi'],
                        'win_rate': stats['win_rate'],
                        'seq_win_rate': stats['sequence_win_rate'],
                    })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_pnl', ascending=False)

    print("\n" + "=" * 80)
    print("PARAMETER SWEEP RESULTS (sorted by Total PnL)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv("data/parameter_sweep_results.csv", index=False)
    print("\nResults saved to data/parameter_sweep_results.csv")

    return results_df

def run_backtest():
    # Default backtest
    print("Running backtest...")
    bt = VolumeSpikeMartingaleBacktest(
        data_path="./data/btcusdt_data_15m.csv",
        vol_ma_period=100,
        vol_spike_multiplier=2,
        vol_prev_multiplier=2,
        min_delta=0,
        base_shares=100,
        max_martingale_levels=5,
    )

    df = bt.load_and_prepare_data()
    print(f"\nData sample:")
    print(df[['timestamp', 'delta', 'abs_delta', 'direction',
              'volume_btc', 'vol_ma', 'vol_ratio', 'vol_prev_ratio', 'is_spike']].head(30).tail(15))

    bt.run_backtest()
    bt.print_report()
    bt.show_sample_trades(15)
    bt.export_trades()
    bt.visualize_trades()



if __name__ == "__main__":
    # run_parameter_sweep()

    run_backtest()
    
