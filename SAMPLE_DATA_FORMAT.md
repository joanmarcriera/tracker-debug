# Sample Data Format

This document describes the expected CSV format for GPS tracker data.

## CSV Structure

The input CSV file should have the following columns:

```csv
dt_server,dt_tracker,lat,lng,altitude,angle,speed,params
```

### Column Descriptions

- `dt_server`: Server timestamp (YYYY-MM-DD HH:MM:SS)
- `dt_tracker`: Tracker timestamp (YYYY-MM-DD HH:MM:SS)
- `lat`: Latitude
- `lng`: Longitude
- `altitude`: Altitude in meters
- `angle`: Direction angle
- `speed`: Speed
- `params`: Comma-separated key=value pairs (see below)

## Params Field

The `params` field is a string containing comma-separated key=value pairs. 

### Required Keys

The following keys are **required** in the params field:

- `batl`: Battery level (0-100 integer)
- `bats`: Battery charging status (0=discharging, 1=charging)
- `batv`: Battery voltage (decimal, e.g., 3.97)
- `gsmlev`: GSM signal level (integer, typically 0-30)

### Example Params String

```
"alm_code=0, batl=85, bats=0, batv=3.97, cellid=81C4478, gpslev=0, gsmlev=16, lac=3E0, lowbat=0, mcc=234, mnc=10"
```

## Complete Example Row

```csv
2025-11-01 18:59:14,2025-11-01 18:59:14,52.01327,0.248775,0,0,0,"alm_code=0, batl=24, bats=1, batv=3.53, cellid=81C4478, gpslev=0, gsmlev=25, lac=3E0, lowbat=0, mcc=234, mnc=10"
```

## Data Requirements

For meaningful analysis, the data should:

1. Span at least one complete discharge cycle (battery going from 100% to lower levels)
2. Have records at regular intervals (e.g., every 30 seconds)
3. Include at least several hours of data
4. Contain battery level data ranging from 100% down to lower levels

## Exporting Data from Fifotrack

If you're using a Fifotrack GPS device:

1. Log into your Fifotrack web platform
2. Go to Reports → Historical Track
3. Select your device and date range
4. Export as CSV
5. Use the exported CSV directly with this tool

## Testing

To verify your CSV format is correct, try running:

```bash
python3 analyze_tracker.py --file your_data.csv
```

If the format is incorrect, you'll see an error message indicating which field is missing or malformed.
