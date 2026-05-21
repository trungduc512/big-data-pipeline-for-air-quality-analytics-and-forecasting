#!/bin/bash
# Script to run the Air Quality Monitoring Dashboard

echo "🌬️  Starting Air Quality Monitoring Dashboard..."
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the Flask app
echo "Starting Flask server..."
echo "Dashboard will be available at: http://localhost:5000"
echo ""
uv run app.py

