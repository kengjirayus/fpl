# FPL Weekly Assistant 🏟️

AI-powered Fantasy Premier League analysis tool that helps you optimize your team with data-driven insights.

## ✨ Features

- **Live FPL Data**: Pulls real-time data from official FPL API
- **AI Predictions**: Uses machine learning to predict player points for next gameweek
- **Team Optimization**: Suggests optimal Starting XI and bench order
- **Transfer Suggestions**: Recommends transfers based on your strategy (Free, Hits, or Wildcard)
- **Thai Language Support**: User interface with Thai translations

## 📋 Requirements

- Python 3.8+
- Internet connection (for FPL API access)

### Python Dependencies
```
streamlit
pandas
numpy
scikit-learn
pulp
requests
```

---

## 🍎 macOS Setup

### Quick Start (Recommended)

1. **Download/Clone** this repository to your desired folder
2. **Copy** the `fpl_start.command` file to the same folder as `fpl.py`
3. **Double-click** `fpl_start.command` to launch the app
4. The app will automatically:
   - Create/activate virtual environment
   - Install required packages
   - Find available port and start the app
   - Open your browser to the app URL

### Manual Setup

If you prefer manual setup:

```bash
# Navigate to the project folder
cd /path/to/your/fpl/folder

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy scikit-learn pulp requests

# Run the app
streamlit run fpl.py
```

### Troubleshooting macOS

- **Permission denied**: Run `chmod +x fpl_start.command` in Terminal
- **Port already in use**: The script will automatically find next available port
- **Virtual environment issues**: Delete `.venv` folder and run the script again

---

## 🪟 Windows Setup

### Method 1: Batch File (Recommended)

1. **Create** `fpl_start.bat` file in the same folder as `fpl.py`
2. **Copy** this content to the batch file:

```batch
@echo off
cd /d "%~dp0"

echo Current directory: %CD%
echo Files in directory:
dir /b

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install required packages
echo Installing/updating required packages...
pip install -q streamlit pandas numpy scikit-learn pulp requests

REM Check if fpl.py exists
if not exist "fpl.py" (
    echo Error: fpl.py not found in current directory!
    echo Please make sure this script is in the same folder as fpl.py
    pause
    exit /b 1
)

REM Find available port
set PORT=8501
:check_port
netstat -an | find ":%PORT% " > nul
if %errorlevel% == 0 (
    set /a PORT+=1
    echo Port %PORT% is in use, trying next port...
    goto check_port
)

REM Run the Streamlit app
echo Starting FPL Assistant on port %PORT%...
echo Opening browser at http://localhost:%PORT%
streamlit run fpl.py --server.headless false --server.port %PORT%

echo.
echo FPL Assistant has stopped.
pause
```

3. **Double-click** `fpl_start.bat` to launch the app

### Method 2: PowerShell Script

1. **Create** `fpl_start.ps1` file:

```powershell
# Change to script directory
Set-Location $PSScriptRoot

# Create virtual environment if it doesn't exist
if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".venv\Scripts\Activate.ps1"

# Install packages
Write-Host "Installing/updating packages..."
pip install -q streamlit pandas numpy scikit-learn pulp requests

# Find available port
$port = 8501
while ((Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue)) {
    $port++
    Write-Host "Port $($port-1) is in use, trying port $port..."
}

# Run app
Write-Host "Starting FPL Assistant on port $port..."
streamlit run fpl.py --server.port $port

Read-Host "Press Enter to exit"
```

2. **Right-click** → "Run with PowerShell"

### Troubleshooting Windows

- **Python not found**: Install Python from [python.org](https://python.org) and add to PATH
- **Execution policy error**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell
- **Port issues**: The scripts automatically find available ports
- **Virtual environment errors**: Delete `.venv` folder and run the script again

---

## 🚀 How to Use

1. **Launch the app** using your platform's method above
2. **Enter your FPL Team ID** in the sidebar
   - Find your Team ID in the FPL website URL: `https://fantasy.premierleague.com/entry/YOUR_ID_HERE/`
3. **Choose transfer strategy**:
   - **Free Transfer**: Only use free transfers
   - **Allow Hit (AI Suggest)**: AI will suggest when taking hits is worth it
   - **Wildcard/Free Hit**: Optimize entire 15-man squad
4. **View recommendations**:
   - Top projected players for next gameweek
   - Optimal Starting XI and bench order
   - Captain and Vice-captain suggestions
   - Transfer recommendations

## 📁 Project Structure

```
FPL/
├── fpl.py                 # Main Streamlit application
├── fpl_start.command      # macOS launcher script
├── fpl_start.bat          # Windows launcher script (create this)
├── fpl_start.ps1          # Windows PowerShell script (create this)
├── .venv/                 # Virtual environment (auto-created)
└── README.md              # This file
```

## 🔧 Advanced Configuration

### Custom Port
To use a specific port, edit the launch script and change:
- macOS/Linux: `--server.port 8502`
- Windows: `set PORT=8502`

### Headless Mode
To run without auto-opening browser:
- Change `--server.headless false` to `--server.headless true`

### Debug Mode
Add `--logger.level debug` to the streamlit command for detailed logging.

## 🆘 Support

### Common Issues

1. **"Module not found" errors**: Virtual environment not activated properly
2. **"Port already in use"**: Close existing Streamlit apps or use different port
3. **API errors**: Check internet connection and FPL website availability
4. **Optimization errors**: Usually due to invalid team configurations

### Getting Help

- Check the terminal/command prompt output for error messages
- Ensure all files are in the same directory
- Verify your FPL Team ID is correct
- Make sure you have internet connection

---

## 📄 License

This project is for educational and personal use only. Fantasy Premier League data belongs to the Premier League.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

---

**Happy FPL managing! 🏆**
