# Contributing to GPS Tracker Battery Analysis Tool

Thank you for considering contributing to this project!

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the Issues section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Sample data if relevant (anonymized)

### Submitting Changes

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test thoroughly with sample data
5. Commit with clear messages (`git commit -m "Add feature: description"`)
6. Push to your fork (`git push origin feature/your-feature-name`)
7. Create a Pull Request

### Code Style

- Follow PEP 8 guidelines for Python code
- Add docstrings to all functions
- Use type hints where appropriate
- Comment complex logic
- Keep functions focused and single-purpose

### Testing

Before submitting:

- Test with multiple CSV files
- Verify both with and without GSM lock scenarios
- Check that output files are generated correctly
- Ensure plots render properly

### Documentation

- Update README.md if adding features
- Add docstrings to new functions
- Update SAMPLE_DATA_FORMAT.md if changing data requirements

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/fifotrack-debug.git
cd fifotrack-debug

# Create virtual environment
pyenv virtualenv 3.11 fifotrack-dev
pyenv activate fifotrack-dev

# Install dependencies
pip install -r requirements.txt

# Make your changes and test
python3 analyze_tracker.py --file sample_data.csv
```

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.

Thank you for helping improve this tool!
