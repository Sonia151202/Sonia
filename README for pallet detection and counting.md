# 📦 Pallet Detection with YOLOv8 (Ultralytics)

This project demonstrates object detection for **pallets** using the YOLOv8 (You Only Look Once) architecture on a custom dataset. It is designed for deployment in industrial or logistics settings where pallet counting and tracking are essential.

## 📁 Dataset

Custom dataset hosted on [Kaggle Datasets](https://www.kaggle.com/datasets/bommasonia/yolov8).  
Format: YOLOv8-compliant directory structure:
pallet detection.v1i.yolov8/
├── train/
├── valid/
├── test/
└── data.yaml

## 🔧 Requirements

Install the required libraries:
pip install ultralytics opencv-python matplotlib
This notebook was built and tested on the Kaggle Python environment.

🚀 Features
Training with both YOLOv8m and YOLOv8s models.

Custom training loop for pallet detection.

Evaluation using mAP (mean Average Precision) metrics.

Prediction visualization for quick model insight.

📌 Workflow Summary
1. Clone this repo and setup
git clone https://github.com/your-username/pallet-detection-yolov8.git
cd pallet-detection-yolov8
2. Dataset Import (KaggleHub)
   
import kagglehub
kagglehub.login()
kagglehub.dataset_download('bommasonia/yolov8')
4. Training YOLOv8m
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='yolov8m_run'
)
4. Evaluation

metrics = model.val(data='data.yaml', split='test')
5. Visualize Predictions
Visualizes first 10 predicted test images using matplotlib.

6. Training YOLOv8s (lightweight)

model = YOLO('yolov8s.pt')
model.train(
    data='pallet.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='pallet_yolov8s'
)
7. Evaluation on Best Model

model = YOLO('runs/detect/pallet_yolov8s/weights/best.pt')
results = model.val(data='data.yaml', split='test')
📊 Sample Output
makefile

Precision:    0.9234
Recall:       0.8876
mAP@50:       0.9512
mAP@50-95:    0.7854
📸 Example Predictions
<div align="center"> <img src="images/sample1.jpg" width="300"> <img src="images/sample2.jpg" width="300"> </div>
💡 Future Work
Export model to ONNX/TFLite for deployment.

Add integration with live camera feeds or RTSP stream.

Convert project into a Streamlit or Gradio app for demos.

🙋‍♀️ Author
Developed by Bomma Sonia
Data Science | AI | Deep Learning | Computer Vision

⭐️ Show your support
Give a ⭐️ if this project helped you!
