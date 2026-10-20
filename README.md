# CrackSense AI — Flask Application

## Project Structure
```
crack_flask/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── models/                 # Place your trained .pth file here
│   └── crack_classifier_final.pth   ← Copy your trained model here
├── static/
│   └── uploads/            # Auto-created
└── templates/
    ├── index.html           # Home / landing page
    ├── task1.html           # Task 1: Image Classification
    └── task2.html           # Task 2: Failure Progression Analysis
```

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model
Copy `crack_classifier_final.pth` (or `best_crack_model.pth`) into the `models/` folder:
```bash
cp /path/to/crack_classifier_final.pth models/
```

If you want to use a different filename, update `MODEL_PATH` in `app.py`.

### 3. Set your Gemini API Key
Get a free key at: https://aistudio.google.com/app/apikey

**Option A — Environment variable (recommended):**
```bash
# Windows
set GEMINI_API_KEY=your_key_here

# Linux/Mac
export GEMINI_API_KEY=your_key_here
```

**Option B — Edit app.py directly:**
```python
GEMINI_API_KEY = 'your_actual_key_here'
```

### 4. Run the app
```bash
python app.py
```

Open: http://localhost:5000

## Features

### Task 1 — Crack Classification
- Upload any structural image
- EfficientNet-B3 classifies into: BULGING / COMBINED / SHEAR / TENSION CRACK
- Confidence scores with animated bar charts
- Radar + bar chart dashboard
- Severity level and recommended action

### Task 2 — Failure Progression Analysis
- Upload INITIAL + END STAGE images
- Gemini Vision AI analyzes damage progression
- ML model classifies the end-stage crack type
- Full dashboard: failure rate %, structural integrity gauge, KPI cards
- Observations, recommended actions, time-to-critical estimate

## Notes
- If Gemini API key is not set, Task 2 will show demo/fallback data
- If model file is not found, the app runs with random weights (for UI testing)
- Supports JPG, PNG, BMP image formats
