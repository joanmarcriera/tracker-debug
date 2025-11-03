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


def detect_discharge_cycles(df, min_duration_hours=1.0, min_battery_drop=5):
    """
    Detect periods where battery is discharging (going down).
    A discharge cycle starts when battery is at 100% or starts decreasing,
    and ends when:
    - Battery starts charging again (bats changes or battery level increases)
    - Battery reaches very low level (near 0%)
    - Battery reaches 100% again (recharged)
    - End of data

    Only includes meaningful discharge cycles (not brief charging interruptions).

    Args:
        df: DataFrame with battery data
        min_duration_hours: Minimum duration in hours to consider a valid cycle
        min_battery_drop: Minimum battery level drop to consider a valid cycle

    Returns a list of (start_idx, end_idx) tuples for each discharge cycle.
    """
    cycles = []
    in_cycle = False
    cycle_start_idx = None
    prev_batl = None
    prev_bats = None

    for idx in range(len(df)):
        batl = df.iloc[idx]['batl']
        bats = df.iloc[idx]['bats']  # Battery charging status (1=charging, 0=discharging)

        # Start of a discharge cycle: battery at 100% or high level and not charging
        if not in_cycle:
            if batl == 100 or (prev_batl is not None and batl < prev_batl and bats == 0):
                cycle_start_idx = idx
                in_cycle = True

        # Detect end of discharge cycle
        elif in_cycle:
            # End conditions:
            # 1. Battery starts charging (bats changes from 0 to 1)
            # 2. Battery level increases significantly (more than 2%, indicating charging started)
            # 3. Battery reaches 100% again
            # 4. Battery is very low (<=5%)

            battery_increased = prev_batl is not None and batl > prev_batl + 2
            started_charging = prev_bats == 0 and bats == 1
            reached_full = batl == 100 and idx > cycle_start_idx + 10
            battery_very_low = batl <= 5

            if started_charging or battery_increased or reached_full or battery_very_low:
                # End the current cycle
                # Use idx-1 if we detected charging/increase, otherwise use idx
                end_idx = idx - 1 if (started_charging or battery_increased) else idx
                cycles.append((cycle_start_idx, end_idx))

                # Start a new cycle if battery is at 100% after this
                if reached_full:
                    cycle_start_idx = idx
                    in_cycle = True
                else:
                    in_cycle = False
                    cycle_start_idx = None

        prev_batl = batl
        prev_bats = bats

    # Add the last cycle if we're still in one
    if in_cycle and cycle_start_idx is not None:
        cycles.append((cycle_start_idx, len(df) - 1))

    # Filter out short cycles that are just charging interruptions
    valid_cycles = []
    for start_idx, end_idx in cycles:
        period_df = df.iloc[start_idx:end_idx+1]

        # Calculate duration
        duration_hours = (period_df.iloc[-1]['dt_server'] - period_df.iloc[0]['dt_server']).total_seconds() / 3600

        # Calculate battery drop
        battery_start = period_df.iloc[0]['batl']
        battery_end = period_df.iloc[-1]['batl']
        battery_drop = battery_start - battery_end

        # Only include cycles that meet minimum criteria
        if duration_hours >= min_duration_hours or battery_drop >= min_battery_drop:
            valid_cycles.append((start_idx, end_idx))

    return valid_cycles


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

    # Determine end reason
    end_reason = "Unknown"
    if end_idx < len(df) - 1:
        # Not at the end of data
        if period_df.iloc[-1]['batl'] <= 5:
            end_reason = "Battery depleted (≤5%)"
        elif df.iloc[end_idx + 1]['bats'] == 1 and period_df.iloc[-1]['bats'] == 0:
            end_reason = "Charging started"
        elif df.iloc[end_idx + 1]['batl'] > period_df.iloc[-1]['batl'] + 2:
            end_reason = "Battery level increased"
        elif df.iloc[end_idx + 1]['batl'] == 100:
            end_reason = "Recharged to 100%"
    else:
        end_reason = "End of data"

    stats = {
        'cycle_num': cycle_num,
        'mode': mode,
        'start_time': start_time,
        'end_time': end_time,
        'end_reason': end_reason,
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
        'bats_start': period_df.iloc[0]['bats'],
        'bats_end': period_df.iloc[-1]['bats'],
    }

    # Calculate battery drain rate (% per hour)
    if stats['duration_hours'] > 0:
        stats['batl_drain_rate_per_hour'] = stats['batl_change'] / stats['duration_hours']
    else:
        stats['batl_drain_rate_per_hour'] = 0

    # Calculate normalized metrics (projected to 24 hours for comparison)
    if stats['duration_hours'] > 0:
        stats['batl_change_per_24h'] = (stats['batl_change'] / stats['duration_hours']) * 24
        stats['estimated_hours_to_empty'] = abs(stats['batl_start'] / stats['batl_drain_rate_per_hour']) if stats['batl_drain_rate_per_hour'] != 0 else float('inf')
    else:
        stats['batl_change_per_24h'] = 0
        stats['estimated_hours_to_empty'] = float('inf')

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
    import glob
    import os

    # Find all CSV files in current directory
    csv_files = sorted(glob.glob('*.csv'))

    if len(csv_files) == 0:
        print("No CSV files found in current directory!")
        return

    # If multiple CSV files, let user choose
    if len(csv_files) > 1:
        print("Available CSV files:")
        for i, f in enumerate(csv_files, 1):
            # Get file size
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"  {i}. {f} ({size_mb:.2f} MB)")

        # Check if file specified as command line argument
        if len(sys.argv) > 1:
            try:
                choice = int(sys.argv[1])
                if 1 <= choice <= len(csv_files):
                    csv_file = csv_files[choice - 1]
                else:
                    print(f"Invalid choice. Please select 1-{len(csv_files)}")
                    return
            except ValueError:
                # Treat as filename
                csv_file = sys.argv[1]
                if not os.path.exists(csv_file):
                    print(f"File not found: {csv_file}")
                    return
        else:
            # Interactive selection
            while True:
                try:
                    choice = int(input(f"\nSelect file (1-{len(csv_files)}): "))
                    if 1 <= choice <= len(csv_files):
                        csv_file = csv_files[choice - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(csv_files)}")
                except ValueError:
                    print("Please enter a valid number")
                except (KeyboardInterrupt, EOFError):
                    print("\nCancelled")
                    return
    else:
        csv_file = csv_files[0]

    # GSM band lock change time - adjust this for your specific case
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
    print("Filtering criteria:")
    print("  - Minimum duration: 1.0 hours")
    print("  - OR minimum battery drop: 5%")
    cycles = detect_discharge_cycles(df, min_duration_hours=1.0, min_battery_drop=5)
    print(f"\nFound {len(cycles)} meaningful discharge cycles (excluding short charging interruptions)")

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
        print(f"    Start:       {stats['start_time']}")
        print(f"    End:         {stats['end_time']}")
        print(f"    End Reason:  {stats['end_reason']}")
        print(f"    Duration:    {stats['duration_hours']:.2f} hours ({stats['duration_hours']/24:.2f} days)")
        print(f"    Records:     {stats['records']}")

        print(f"\n  Battery Performance:")
        print(f"    Start Level:       {stats['batl_start']}%")
        print(f"    End Level:         {stats['batl_end']}%")
        print(f"    Total Change:      {stats['batl_change']:+d}%")
        print(f"    Drain Rate:        {stats['batl_drain_rate_per_hour']:.2f}% per hour")
        print(f"    Projected 24h:     {stats['batl_change_per_24h']:.1f}% (normalized for comparison)")
        if stats['estimated_hours_to_empty'] != float('inf'):
            print(f"    Est. Time to 0%:   {stats['estimated_hours_to_empty']:.1f} hours ({stats['estimated_hours_to_empty']/24:.1f} days)")
        else:
            print(f"    Est. Time to 0%:   N/A (battery not draining)")

        print(f"\n  Battery Voltage:")
        print(f"    Start:  {stats['batv_start']:.2f}V")
        print(f"    End:    {stats['batv_end']:.2f}V")
        print(f"    Min:    {stats['batv_min']:.2f}V")
        print(f"    Max:    {stats['batv_max']:.2f}V")

        print(f"\n  Charging Status:")
        print(f"    Start: {'Charging' if stats['bats_start'] == 1 else 'Discharging'}")
        print(f"    End:   {'Charging' if stats['bats_end'] == 1 else 'Discharging'}")

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
