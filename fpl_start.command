#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Show current directory for debugging
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required packages
echo "Installing/updating required packages..."
pip install -q streamlit pandas numpy scikit-learn pulp requests

# Check if fpl.py exists
if [ ! -f "fpl.py" ]; then
    echo "Error: fpl.py not found in current directory!"
    echo "Please make sure this script is in the same folder as fpl.py"
    read -p "Press any key to exit..."
    exit 1
fi

# Run the Streamlit app
echo "Starting FPL Assistant..."
echo "Opening browser at http://localhost:8501"
streamlit run fpl.py --server.headless false --server.port 8501

# Keep terminal open after app closes
echo ""
echo "FPL Assistant has stopped."
read -p "Press any key to close this window..."