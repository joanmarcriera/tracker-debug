#!/bin/bash
# Analyze all GPS tracker CSV files
# This script runs the analysis for all tracker files and generates consistent output filenames

cd "$(dirname "$0")"

echo "=========================================="
echo "Analyzing all GPS tracker files..."
echo "=========================================="
echo

# Activate pyenv environment
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
pyenv activate fifotrack

# Analyze fifotrack1Q3 with October data (with GSM lock)
echo "1/3: Processing fifotrack1Q3 (October data)..."
python3 analyze_tracker.py --file "fifotrack1Q3 2025-10-01 00_00_00-2025-11-04 00_00_00.csv" --gsm-lock "2025-11-03 09:42:00"
echo

# Analyze fifotrack1Q3 with November data (with GSM lock)
echo "2/3: Processing fifotrack1Q3 (November data)..."
python3 analyze_tracker.py --file "fifotrack1Q3 2025-11-01 00_00_00-2025-11-04 00_00_00.csv" --gsm-lock "2025-11-03 09:42:00"
echo

# Analyze fifotrack2Q3 (no GSM lock)
echo "3/3: Processing fifotrack2Q3 (no GSM lock)..."
python3 analyze_tracker.py --file "fifotrack2Q3 2025-10-01 00_00_00-2025-11-04 00_00_00.csv" --no-gsm-lock
echo

echo "=========================================="
echo "All analyses complete!"
echo "=========================================="
echo
echo "Generated files:"
ls -1 fifotrack*_analysis_cycles.png fifotrack*_cycle_comparison.png 2>/dev/null | sort
