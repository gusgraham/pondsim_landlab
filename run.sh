#!/bin/bash

# Pondsim Startup Script
# Automatically configures the environment and launches the requested interface.

# Get the absolute path to the project root
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$ROOT_DIR/src"

# Use virtual environment if it exists
if [ -d "$ROOT_DIR/.venv" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
else
    # Fallback to system python
    PYTHON="python3"
fi

# Function to show usage
show_help() {
    echo "Pondsim — Overland Flow Simulation"
    echo ""
    echo "Usage: ./run.sh [gui|cli|test] [args...]"
    echo ""
    echo "Commands:"
    echo "  gui          Start the Pondsim Qt Graphical Interface (default)"
    echo "  cli          Run the Headless CLI for automated pipelines"
    echo "  test         Run the unit test suite"
    echo ""
    echo "Examples:"
    echo "  ./run.sh gui"
    echo "  ./run.sh cli --dem path/to/dem.tif --sources path/to/src.geojson --out-dir ./results"
    echo "  ./run.sh test"
}

# Handle help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Default to gui if no command provided or if provided string looks like an option
if [[ -z "$1" ]] || [[ "$1" == --* ]]; then
    COMMAND="gui"
else
    COMMAND="$1"
    shift
fi

case "$COMMAND" in
    gui)
        echo "Starting Pondsim GUI..."
        "$PYTHON" -m pondsim.app "$@"
        ;;
    cli)
        echo "Starting Pondsim CLI..."
        "$PYTHON" -m pondsim.cli "$@"
        ;;
    test)
        echo "Running tests..."
        if "$PYTHON" -c "import pytest" &>/dev/null; then
            "$PYTHON" -m pytest "$@"
        else
            echo "Error: pytest not found in environment."
            exit 1
        fi
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        show_help
        exit 1
        ;;
esac
