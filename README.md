# Video Feature Extraction Tool

A Python tool for analyzing videos and extracting visual and temporal features using OpenCV.

## Features

- **Shot Cut Detection** - Identifies scene transitions
- **Motion Analysis** - Quantifies movement using optical flow  
- **Text Detection** - Extracts text using OCR (Tesseract)
- **Object/Person Detection** - Detects people and objects (YOLO/Haar Cascade)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/raihanulislam00/Video_Feature_Extraction_Tool.git
cd Video_Feature_Extraction_Tool
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: (Optional) Install Tesseract for OCR
```bash
brew install tesseract  # macOS
# or
sudo apt-get install tesseract-ocr  # Linux
```

## How to Run

### Quick Start
```bash
# Activate virtual environment
source venv/bin/activate

# Run with sample video
python example_usage.py sample_test_video.mp4

# Run with your own video
python example_usage.py path/to/your/video.mp4
```

### Use as Python Module
```python
from video_feature_extractor import VideoFeatureExtractor

# Analyze video
extractor = VideoFeatureExtractor("video.mp4", sample_rate=30)
features = extractor.extract_all_features()

# Print summary
extractor.print_summary()

# Save results
extractor.save_features("output.json")
```

### Run Jupyter Notebook
```bash
jupyter notebook video_analysis_demo.ipynb
```

## Output

Results are automatically saved as JSON files. Example output:

```json
{
  "video_info": { "fps": 30, "duration_seconds": 5.0 },
  "shot_cuts": { "total_cuts": 0 },
  "motion_analysis": { "average_motion": 0.37, "motion_category": "Static/Very Low" },
  "text_detection": { "text_present_ratio": 1.0, "keywords": ["frame"] },
  "object_detection": { "total_people_detected": 0, "dominance": "unknown" }
}
```

## Requirements

- Python 3.7+
- opencv-python
- numpy
- pytesseract (optional)
- matplotlib (optional)

## Troubleshooting

If you encounter issues:
1. Make sure virtual environment is activated
2. Check that all dependencies are installed: `pip list`
3. For OCR issues, ensure Tesseract is installed
4. For YOLO detection, download model files separately (optional)
