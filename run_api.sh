#!/bin/bash

# Script to launch the FastAPI application

echo "NYC Taxi Trip Duration Prediction API"
echo "======================================"
echo ""

# Activate venv
source venv/bin/activate

# Check if FastAPI is installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting API server..."
echo "API will be available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch the API
cd api
python main.py
