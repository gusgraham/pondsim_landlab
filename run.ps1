$Root = $PSScriptRoot
$env:PYTHONPATH = "$Root\src"

# Path to the virtual environment python
$Python = if (Test-Path "$Root\.venv\Scripts\python.exe") { 
    "$Root\.venv\Scripts\python.exe" 
} else { 
    "python" 
}

# Get arguments
$Command = $args[0]
$RemainingArgs = if ($args.Count -gt 1) { $args[1..($args.Count-1)] } else { @() }

if (-not $Command -or $Command -eq "-h" -or $Command -eq "--help") {
    Write-Host "Pondsim — Overland Flow Simulation"
    Write-Host ""
    Write-Host "Usage: .\run.ps1 [gui|cli|test] [args...]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  gui          Start the Pondsim Qt Graphical Interface (default)"
    Write-Host "  cli          Run the Headless CLI for automated pipelines"
    Write-Host "  test         Run the unit test suite"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1 gui"
    Write-Host "  .\run.ps1 cli --dem path/to/dem.tif"
    Write-Host "  .\run.ps1 test"
    exit
}

switch ($Command) {
    "gui" {
        Write-Host "Starting Pondsim GUI..."
        & $Python -m pondsim.app @RemainingArgs
    }
    "cli" {
        Write-Host "Starting Pondsim CLI..."
        & $Python -m pondsim.cli @RemainingArgs
    }
    "test" {
        Write-Host "Running tests..."
        & $Python -m pytest @RemainingArgs
    }
    default {
        # Fallback to gui if no recognized command is given
        Write-Host "Starting Pondsim GUI..."
        & $Python -m pondsim.app @args
    }
}
