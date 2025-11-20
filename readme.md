An AI-powered intrusion detection system that detects people entering restricted zones in video footage using the YOLOv8n model.


## Features
- Detect humans using YOLO
- Define restricted areas in video
- Track movement inside zones
- Trigger warnings on intrusion
- Clean project architecture (DRY, KISS)


## Folder Structure

├── detecttor
    ├── __init__py
    ├── intrusion_detector.py
    ├── utils.py
├── screenshots
    ├── larm.png

├── zones
    ├── __init__.py
    ├── zone_manager.py

├── main.py
├── restricted_zones.json
├── readme.md
├── test.mp4



## Installation
pip install -r requirements.txt


## Usage
python main.py 

## Technologies
- Python
- OpenCV
- YOLOv8n




