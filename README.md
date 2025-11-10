# Video Feature Extraction Tool

A Python tool for analyzing videos and extracting visual and temporal features using OpenCV.

## Features

- **Shot Cut Detection** - Identifies scene transitions
- **Motion Analysis** - Quantifies movement using optical flow  
- **Text Detection** - Extracts text using OCR (Tesseract)
- **Object/Person Detection** - Detects people and objects (YOLO/Haar Cascade)

## Installation

```bash
# Install dependencies
pip3 install -r requirements.txt

# Optional: Install Tesseract for OCR
brew install tesseract  # macOS
```

## Usage

### Command Line
```bash
python3 video_feature_extractor.py your_video.mp4
```

### Python Script
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

### Jupyter Notebook
```bash
jupyter notebook video_analysis_demo.ipynb
```

## Output

Results are saved as JSON:

```json
{
  "video_info": { "fps": 30, "duration_seconds": 5.0 },
  "shot_cuts": { "total_cuts": 12 },
  "motion_analysis": { "average_motion": 5.23, "motion_category": "Moderate" },
  "text_detection": { "text_present_ratio": 0.35, "keywords": ["hello"] },
  "object_detection": { "total_people_detected": 45, "dominance": "people" }
}
```

## Requirements

- Python 3.7+
- opencv-python
- numpy
- pytesseract (optional)
- matplotlib (optional)

## Examples

See `example_usage.py` for detailed usage examples.
