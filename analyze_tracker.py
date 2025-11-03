#!/usr/bin/env python3
"""
GPS Tracker Battery Analysis Script

Analyzes GPS tracker data to investigate battery drain by tracking:
- Battery level (batl) over time
- GSM signal level (gsmlev) fluctuations
- Correlation between signal strength variations and battery consumption

The script detects charging cycles (when battery reaches 100%) and analyzes
each discharge period separately to identify patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re
import sys


def parse_params(params_str):
    """Parse the params string into a dictionary."""
    params = {}
    # Split by comma and parse key=value pairs
    pairs = params_str.split(', ')
    for pair in pairs:
        if '=' in pair:
            key, value = pair.split('=', 1)
            params[key.strip()] = value.strip()
    return params


def load_and_process_data(csv_file):
    """Load CSV and extract batl and gsmlev from params column."""
    df = pd.read_csv(csv_file)

    # Parse params column
    df['params_dict'] = df['params'].apply(parse_params)
    df['batl'] = df['params_dict'].apply(lambda x: int(x.get('batl', 0)))
    df['gsmlev'] = df['params_dict'].apply(lambda x: int(x.get('gsmlev', 0)))
    df['batv'] = df['params_dict'].apply(lambda x: float(x.get('batv', 0)))
    df['bats'] = df['params_dict'].apply(lambda x: int(x.get('bats', 0)))

    # Convert datetime
    df['dt_server'] = pd.to_datetime(df['dt_server'])

    return df


def detect_discharge_cycles(df):
    """
    Detect periods where battery is at 100% and starts going down.
    Returns a list of (start_idx, end_idx) tuples for each discharge cycle.
    """
    cycles = []
    in_cycle = False
    cycle_start_idx = None

    for idx in range(len(df)):
        batl = df.iloc[idx]['batl']

        # Start of a discharge cycle: battery at 100%
        if batl == 100 and not in_cycle:
            cycle_start_idx = idx
            in_cycle = True

        # End of cycle: battery drops to very low or gets recharged back to high level
        # We'll end the cycle when battery reaches 100% again or at the end of data
        elif batl == 100 and in_cycle and idx > cycle_start_idx + 10:  # At least 10 records after start
            # This is the start of a new cycle, so end the previous one
            cycles.append((cycle_start_idx, idx - 1))
            cycle_start_idx = idx

    # Add the last cycle if we're still in one
    if in_cycle and cycle_start_idx is not None:
        cycles.append((cycle_start_idx, len(df) - 1))

    return cycles


def calculate_cycle_statistics(df, start_idx, end_idx, cycle_num, change_time):
    """Calculate statistics for a discharge cycle."""
    period_df = df.iloc[start_idx:end_idx+1].copy()

    if len(period_df) == 0:
        return None

    start_time = period_df.iloc[0]['dt_server']
    end_time = period_df.iloc[-1]['dt_server']

    # Determine if this cycle is before or after the GSM change
    if start_time < change_time:
        mode = "Before GSM Lock (Auto)"
    else:
        mode = "After GSM Lock (4G Only)"

    stats = {
        'cycle_num': cycle_num,
        'mode': mode,
        'start_time': start_time,
        'end_time': end_time,
        'records': len(period_df),
        'duration_hours': (end_time - start_time).total_seconds() / 3600,
        'batl_start': period_df.iloc[0]['batl'],
        'batl_end': period_df.iloc[-1]['batl'],
        'batl_change': period_df.iloc[-1]['batl'] - period_df.iloc[0]['batl'],
        'gsmlev_mean': period_df['gsmlev'].mean(),
        'gsmlev_std': period_df['gsmlev'].std(),
        'gsmlev_min': period_df['gsmlev'].min(),
        'gsmlev_max': period_df['gsmlev'].max(),
        'gsmlev_range': period_df['gsmlev'].max() - period_df['gsmlev'].min(),
        'batv_start': period_df.iloc[0]['batv'],
        'batv_end': period_df.iloc[-1]['batv'],
        'batv_min': period_df['batv'].min(),
        'batv_max': period_df['batv'].max(),
    }

    # Calculate battery drain rate (% per hour)
    if stats['duration_hours'] > 0:
        stats['batl_drain_rate_per_hour'] = stats['batl_change'] / stats['duration_hours']
    else:
        stats['batl_drain_rate_per_hour'] = 0

    return stats


def plot_cycle_comparison(df, cycles, change_time, all_stats):
    """Create visualizations comparing discharge cycles."""
    # Overall timeline plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    # Plot 1: Battery level over time with cycle boundaries
    axes[0].plot(df['dt_server'], df['batl'], marker='o', markersize=1, linewidth=1)
    axes[0].axvline(x=change_time, color='red', linestyle='--', linewidth=2, label='GSM Band Lock Change')

    # Mark cycle boundaries
    for i, (start_idx, end_idx) in enumerate(cycles):
        start_time = df.iloc[start_idx]['dt_server']
        axes[0].axvline(x=start_time, color='green', linestyle=':', alpha=0.5, linewidth=1)

    axes[0].set_ylabel('Battery Level (%)')
    axes[0].set_title('Battery Level Over Time - Discharge Cycles Marked')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot 2: GSM signal level over time
    axes[1].plot(df['dt_server'], df['gsmlev'], marker='o', markersize=1, linewidth=1, color='orange')
    axes[1].axvline(x=change_time, color='red', linestyle='--', linewidth=2, label='GSM Band Lock Change')
    axes[1].set_ylabel('GSM Signal Level')
    axes[1].set_title('GSM Signal Level Over Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Battery voltage over time
    axes[2].plot(df['dt_server'], df['batv'], marker='o', markersize=1, linewidth=1, color='green')
    axes[2].axvline(x=change_time, color='red', linestyle='--', linewidth=2, label='GSM Band Lock Change')
    axes[2].set_ylabel('Battery Voltage (V)')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Battery Voltage Over Time')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    output_file = 'tracker_analysis_cycles.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nCycle timeline plot saved to: {output_file}")

    # Comparison bar charts
    if len(all_stats) > 0:
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

        cycle_labels = [f"Cycle {s['cycle_num']}\n{s['mode'][:10]}" for s in all_stats]
        drain_rates = [s['batl_drain_rate_per_hour'] for s in all_stats]
        gsm_stds = [s['gsmlev_std'] for s in all_stats]
        gsm_means = [s['gsmlev_mean'] for s in all_stats]
        durations = [s['duration_hours'] for s in all_stats]

        colors = ['blue' if s['mode'].startswith('Before') else 'green' for s in all_stats]

        # Battery drain rate comparison
        axes2[0, 0].bar(range(len(all_stats)), drain_rates, color=colors)
        axes2[0, 0].set_ylabel('Battery Drain (% per hour)')
        axes2[0, 0].set_title('Battery Drain Rate per Cycle')
        axes2[0, 0].set_xticks(range(len(all_stats)))
        axes2[0, 0].set_xticklabels(cycle_labels, rotation=45, ha='right', fontsize=8)
        axes2[0, 0].grid(True, alpha=0.3, axis='y')
        axes2[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # GSM signal stability (std dev)
        axes2[0, 1].bar(range(len(all_stats)), gsm_stds, color=colors)
        axes2[0, 1].set_ylabel('GSM Signal Std Dev')
        axes2[0, 1].set_title('GSM Signal Stability per Cycle (lower = better)')
        axes2[0, 1].set_xticks(range(len(all_stats)))
        axes2[0, 1].set_xticklabels(cycle_labels, rotation=45, ha='right', fontsize=8)
        axes2[0, 1].grid(True, alpha=0.3, axis='y')

        # GSM signal mean
        axes2[1, 0].bar(range(len(all_stats)), gsm_means, color=colors)
        axes2[1, 0].set_ylabel('GSM Signal Mean')
        axes2[1, 0].set_title('Average GSM Signal Strength per Cycle')
        axes2[1, 0].set_xticks(range(len(all_stats)))
        axes2[1, 0].set_xticklabels(cycle_labels, rotation=45, ha='right', fontsize=8)
        axes2[1, 0].grid(True, alpha=0.3, axis='y')

        # Cycle duration
        axes2[1, 1].bar(range(len(all_stats)), durations, color=colors)
        axes2[1, 1].set_ylabel('Duration (hours)')
        axes2[1, 1].set_title('Cycle Duration')
        axes2[1, 1].set_xticks(range(len(all_stats)))
        axes2[1, 1].set_xticklabels(cycle_labels, rotation=45, ha='right', fontsize=8)
        axes2[1, 1].grid(True, alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', label='Before GSM Lock'),
                          Patch(facecolor='green', label='After GSM Lock')]
        fig2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

        plt.tight_layout()
        output_file2 = 'tracker_cycle_comparison.png'
        plt.savefig(output_file2, dpi=150, bbox_inches='tight')
        print(f"Cycle comparison plot saved to: {output_file2}")


def main():
    # Configuration
    csv_file = 'fifotrack1Q3 2025-11-01 00_00_00-2025-11-04 00_00_00.csv'
    change_time = pd.to_datetime('2025-11-03 09:42:00')

    print("=" * 80)
    print("GPS Tracker Battery Drain Analysis - Discharge Cycle Report")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {csv_file}")
    df = load_and_process_data(csv_file)
    print(f"Total records: {len(df)}")
    print(f"Time range: {df['dt_server'].min()} to {df['dt_server'].max()}")

    # Detect discharge cycles
    print("\n" + "=" * 80)
    print("DETECTING DISCHARGE CYCLES")
    print("=" * 80)
    cycles = detect_discharge_cycles(df)
    print(f"\nFound {len(cycles)} discharge cycles")

    # Calculate statistics for each cycle
    all_stats = []
    for i, (start_idx, end_idx) in enumerate(cycles):
        stats = calculate_cycle_statistics(df, start_idx, end_idx, i + 1, change_time)
        if stats:
            all_stats.append(stats)

    # Print detailed report for each cycle
    print("\n" + "=" * 80)
    print("DISCHARGE CYCLE REPORTS")
    print("=" * 80)

    for stats in all_stats:
        print(f"\n{'='*80}")
        print(f"CYCLE {stats['cycle_num']}: {stats['mode']}")
        print(f"{'='*80}")
        print(f"  Time Period:")
        print(f"    Start:    {stats['start_time']}")
        print(f"    End:      {stats['end_time']}")
        print(f"    Duration: {stats['duration_hours']:.2f} hours ({stats['duration_hours']/24:.2f} days)")
        print(f"    Records:  {stats['records']}")

        print(f"\n  Battery Performance:")
        print(f"    Start Level:  {stats['batl_start']}%")
        print(f"    End Level:    {stats['batl_end']}%")
        print(f"    Total Change: {stats['batl_change']:+d}%")
        print(f"    Drain Rate:   {stats['batl_drain_rate_per_hour']:.2f}% per hour")
        if stats['duration_hours'] > 0 and stats['batl_change'] < 0:
            hours_to_empty = abs(stats['batl_start'] / stats['batl_drain_rate_per_hour'])
            print(f"    Est. Time to 0%: {hours_to_empty:.1f} hours ({hours_to_empty/24:.1f} days)")

        print(f"\n  Battery Voltage:")
        print(f"    Start:  {stats['batv_start']:.2f}V")
        print(f"    End:    {stats['batv_end']:.2f}V")
        print(f"    Min:    {stats['batv_min']:.2f}V")
        print(f"    Max:    {stats['batv_max']:.2f}V")

        print(f"\n  GSM Signal Performance:")
        print(f"    Mean:     {stats['gsmlev_mean']:.2f}")
        print(f"    Std Dev:  {stats['gsmlev_std']:.2f} (lower = more stable)")
        print(f"    Range:    {stats['gsmlev_min']} - {stats['gsmlev_max']} (span: {stats['gsmlev_range']})")

    # Summary comparison between before and after GSM lock
    before_cycles = [s for s in all_stats if s['mode'].startswith('Before')]
    after_cycles = [s for s in all_stats if s['mode'].startswith('After')]

    if before_cycles and after_cycles:
        print("\n" + "=" * 80)
        print("SUMMARY: BEFORE vs AFTER GSM BAND LOCK")
        print("=" * 80)

        # Calculate averages
        before_avg_drain = sum(s['batl_drain_rate_per_hour'] for s in before_cycles) / len(before_cycles)
        after_avg_drain = sum(s['batl_drain_rate_per_hour'] for s in after_cycles) / len(after_cycles)

        before_avg_gsm_std = sum(s['gsmlev_std'] for s in before_cycles) / len(before_cycles)
        after_avg_gsm_std = sum(s['gsmlev_std'] for s in after_cycles) / len(after_cycles)

        before_avg_gsm_mean = sum(s['gsmlev_mean'] for s in before_cycles) / len(before_cycles)
        after_avg_gsm_mean = sum(s['gsmlev_mean'] for s in after_cycles) / len(after_cycles)

        print(f"\nBefore GSM Lock (Auto Mode) - {len(before_cycles)} cycle(s):")
        print(f"  Average Battery Drain: {before_avg_drain:.2f}% per hour")
        print(f"  Average GSM Stability: {before_avg_gsm_std:.2f} (std dev)")
        print(f"  Average GSM Strength:  {before_avg_gsm_mean:.2f}")

        print(f"\nAfter GSM Lock (4G Only) - {len(after_cycles)} cycle(s):")
        print(f"  Average Battery Drain: {after_avg_drain:.2f}% per hour")
        print(f"  Average GSM Stability: {after_avg_gsm_std:.2f} (std dev)")
        print(f"  Average GSM Strength:  {after_avg_gsm_mean:.2f}")

        print(f"\nImpact of GSM Band Lock:")
        drain_improvement = ((before_avg_drain - after_avg_drain) / abs(before_avg_drain)) * 100
        stability_improvement = ((before_avg_gsm_std - after_avg_gsm_std) / before_avg_gsm_std) * 100

        print(f"  Battery Drain: {drain_improvement:+.1f}% ({'better' if drain_improvement > 0 else 'worse'})")
        print(f"  GSM Stability: {stability_improvement:+.1f}% ({'better' if stability_improvement > 0 else 'worse'})")
        print(f"  GSM Strength:  {after_avg_gsm_mean - before_avg_gsm_mean:+.2f}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    plot_cycle_comparison(df, cycles, change_time, all_stats)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
