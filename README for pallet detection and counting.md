# Pallet Detection and Counting using YOLOv8

This project performs pallet detection and counting using YOLOv8 object detection models (`YOLOv8m` and `YOLOv8s`). It uses annotated datasets prepared in YOLO format and trains models using Ultralytics' YOLOv8 implementation.

## 📦 Dataset

The dataset is stored on Kaggle and structured in YOLOv8 format with the following directory structure:

```
pallet detection.v1i.yolov8/
├── train/
│   └── images/, labels/
├── valid/
│   └── images/, labels/
├── test/
│   └── images/, labels/
└── data.yaml
```

## 🚀 Getting Started

### Install Requirements

```bash
pip install ultralytics
pip install split-folders
```

### Load Dataset from Kaggle

```python
import kagglehub
kagglehub.login()
kagglehub.dataset_download('bommasonia/yolov8')
```

## 🔧 Training the Model

### YOLOv8m Training

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.train(
    data='data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    project='pallet_detection_project',
    name='yolov8m_run',
    exist_ok=True
)
```

### YOLOv8s Training

```python
model = YOLO('yolov8s.pt')
model.train(
    data='pallet.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    name='pallet_yolov8s'
)
```

## 📊 Evaluation

```python
metrics = model.val(data='data.yaml', split='test')
print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
```

## 🖼️ Visualizing Predictions

Predictions on test images are saved and visualized using OpenCV and Matplotlib.

## 📁 Directory Structure

```
.
├── runs/detect/predict_test/
├── pallet.yaml
└── data.yaml
```

## 🧠 Model Used

- **YOLOv8m.pt** – Medium model variant
- **YOLOv8s.pt** – Small model variant

## 📌 Results

Performance metrics such as Precision, Recall, mAP@50, and mAP@50-95 are printed after evaluation. Annotated predictions are saved in the `runs/detect/` directory.

## 🧑‍💻 Author

**Bomma Sonia** - Data Science & AI Professional

## 📝 License

This project is licensed under the MIT License.
