# GPS Tracker Battery Analysis Tool

A Python tool to analyze GPS tracker battery drain patterns and help debug battery performance issues. Supports detecting discharge cycles, tracking GSM signal stability, and comparing performance before/after configuration changes.

## Features

- 📊 **Discharge Cycle Detection**: Automatically identifies meaningful battery discharge cycles
- 🔋 **Battery Drain Analysis**: Calculates drain rates, projected battery life, and normalized metrics
- 📡 **GSM Signal Tracking**: Monitors signal strength and stability (fluctuations)
- 📈 **Visual Reports**: Generates timeline plots and comparison charts
- ⚙️ **Configuration Comparison**: Compare performance before/after changes (e.g., GSM band lock)
- 📁 **Deterministic Output**: Output filenames based on input CSV for easy tracking

## Use Case

This tool was created to debug rapid battery drain in GPS tracker devices. It helps identify:
- How long the battery lasts under normal usage
- Whether GSM signal fluctuations correlate with battery drain
- Impact of configuration changes (like locking GSM to 4G only)
- Patterns in battery performance across different time periods

## Installation

### Prerequisites

- Python 3.8+
- pyenv (recommended for environment management)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd fifotrack-debug
```

2. Create and activate a Python environment:
```bash
pyenv virtualenv 3.11 fifotrack
pyenv activate fifotrack
```

3. Install dependencies:
```bash
pip install pandas matplotlib
```

## Data Format

The tool expects CSV files with the following structure:

```csv
dt_server,dt_tracker,lat,lng,altitude,angle,speed,params
2025-11-01 18:59:14,2025-11-01 18:59:14,52.01327,0.248775,0,0,0,"alm_code=0, batl=24, bats=1, batv=3.53, cellid=81C4478, gpslev=0, gsmlev=25, lac=3E0, lowbat=0, mcc=234, mnc=10"
```

**Required fields in `params` column:**
- `batl`: Battery level (0-100%)
- `bats`: Battery charging status (0=discharging, 1=charging)
- `batv`: Battery voltage
- `gsmlev`: GSM signal level

## Usage

### Interactive Mode

Run without arguments to select a file interactively:

```bash
python3 analyze_tracker.py
```

### Command Line Mode

Analyze a specific file:

```bash
python3 analyze_tracker.py --file "fifotrack1Q3 2025-11-01 00_00_00-2025-11-04 00_00_00.csv"
```

Analyze without GSM lock comparison:

```bash
python3 analyze_tracker.py --file "fifotrack2Q3 2025-10-01 00_00_00-2025-11-04 00_00_00.csv" --no-gsm-lock
```

Specify custom GSM lock timestamp:

```bash
python3 analyze_tracker.py --file "data.csv" --gsm-lock "2025-11-03 10:00:00"
```

### Batch Processing

Process all CSV files at once:

```bash
./analyze_all.sh
```

### Command Line Options

```
Options:
  -h, --help            Show help message and exit
  --file FILE, -f FILE  CSV file to analyze (filename or number from list)
  --gsm-lock TIMESTAMP  Timestamp when GSM band was locked to 4G
                        (default: 2025-11-03 09:42:00)
  --no-gsm-lock         Disable GSM lock analysis (no before/after comparison)
```

## Output

The tool generates the following outputs:

### Console Report

Detailed text report including:
- Discharge cycle detection summary
- Per-cycle statistics (duration, battery drain rate, GSM signal metrics)
- End reason for each cycle (battery depleted, charging started, etc.)
- Before/after GSM lock comparison (if applicable)
- Overall summary statistics

### Visualization Files

Two PNG files per input CSV:

1. **`<input>_analysis_cycles.png`**: Timeline showing battery level, GSM signal, and voltage over time with cycle regions highlighted

2. **`<input>_cycle_comparison.png`**: Bar charts comparing:
   - Battery drain rate per cycle
   - GSM signal stability (lower = more stable)
   - Average GSM signal strength
   - Cycle duration

Example filenames:
```
fifotrack1Q3_2025-11-01_00_00_00-2025-11-04_00_00_00_analysis_cycles.png
fifotrack1Q3_2025-11-01_00_00_00-2025-11-04_00_00_00_cycle_comparison.png
```

## How It Works

### Discharge Cycle Detection

A discharge cycle is defined as:

- **Start**: Battery at 100%
- **End**: One of the following occurs:
  - Battery increases by 10%+ from minimum (device placed on charger)
  - Battery depletes to ≤5% (complete discharge)
  - Battery reaches 100% again (fully recharged)
  - End of data

**Filtering**: Only cycles meeting minimum criteria are included:
- Duration ≥ 1 hour OR battery drop ≥ 5%

This prevents short charging interruptions from creating false cycles.

### Metrics Calculated

Per cycle:
- Duration (hours/days)
- Battery drain rate (% per hour)
- Projected 24h battery change (normalized comparison)
- Estimated time to 0%
- GSM signal mean, standard deviation, min/max
- Battery voltage range
- End reason

### GSM Lock Comparison

When a GSM lock timestamp is provided, the tool:
1. Classifies each cycle as "Before" or "After" the change
2. Calculates average metrics for each period
3. Shows percentage improvement/degradation
4. Highlights impact on battery drain and signal stability

## Example Output

```
================================================================================
DISCHARGE CYCLE REPORTS
================================================================================

================================================================================
CYCLE 1: Before GSM Lock (Auto)
================================================================================
  Time Period:
    Start:       2025-11-01 22:15:08
    End:         2025-11-02 11:27:41
    End Reason:  Battery depleted (≤5%)
    Duration:    13.21 hours (0.55 days)
    Records:     1186

  Battery Performance:
    Start Level:       100%
    End Level:         4%
    Total Change:      -96%
    Drain Rate:        -7.27% per hour
    Projected 24h:     -174.4% (normalized for comparison)
    Est. Time to 0%:   13.8 hours (0.6 days)

  GSM Signal Performance:
    Mean:     13.47
    Std Dev:  7.38 (lower = more stable)
    Range:    1 - 25 (span: 24)

================================================================================
SUMMARY: BEFORE vs AFTER GSM BAND LOCK
================================================================================

Before GSM Lock (Auto Mode) - 2 cycle(s):
  Average Battery Drain: -6.04% per hour
  Average GSM Stability: 7.37 (std dev)

After GSM Lock (4G Only) - 1 cycle(s):
  Average Battery Drain: -7.69% per hour
  Average GSM Stability: 7.98 (std dev)

Impact of GSM Band Lock:
  Battery Drain: +27.2% (better)
  GSM Stability: -8.3% (worse)
```

## Interpreting Results

### Battery Drain Rate
- **-7% per hour**: Battery would last ~14 hours from 100% to 0%
- **Lower absolute value = better** (less drain)
- Compare cycles to identify problematic periods

### GSM Signal Stability (Std Dev)
- **Lower = more stable** (less fluctuation)
- High fluctuation may correlate with higher power consumption
- Values typically range from 5-10

### GSM Signal Strength (Mean)
- **Higher = stronger signal**
- Stronger signals generally require less power to maintain
- Values typically range from 10-25

## Troubleshooting

### No cycles detected
- Check if CSV has data with battery at 100%
- Verify `params` column contains `batl` values
- Try lowering filtering thresholds in the code

### Wrong cycle boundaries
- Adjust `recharge_threshold` parameter (default: 10%)
- Check if data has many small battery fluctuations

### Missing visualizations
- Ensure matplotlib is installed: `pip install matplotlib`
- Check for write permissions in output directory

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

Created to debug battery drain issues in Fifotrack GPS devices, but applicable to any similar GPS tracker data format.
