# Football Player Role Classification

A CNN-based image classification system that identifies four roles in football match images: **ball**, **goalkeeper**, **player**, and **referee**.

This project was developed as part of the Pattern Recognition course at the University of Europe for Applied Sciences (MSc Software Engineering).

---

## Project Overview

The system uses object crops extracted from real match footage and classifies them into four categories using three different CNN architectures:

- **Custom CNN** — a lightweight convolutional network built from scratch
- **MobileNetV2** — a pre-trained model fine-tuned with transfer learning
- **ResNet50** — a deeper pre-trained model fine-tuned with transfer learning

Grad-CAM visualizations are included to explain which image regions influenced each model's predictions. An inference demo shows the best model (MobileNetV2) applied to a real match image with bounding boxes drawn for every annotated object.

---

## Results

| Model | Accuracy | F1-Score (macro) |
|---|---|---|
| Custom CNN | 67.87% | 0.72 |
| **MobileNetV2** | **76.39%** | **0.78** |
| ResNet50 | 71.35% | 0.74 |

---

## Dataset

- **Source:** [Football Players Detection — Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- **License:** CC BY 4.0
- **Classes:** ball, goalkeeper, player, referee
- **Raw images:** 663 match images (train + valid + test)
- **Extracted crops:** 15,636 individual object crops
- **Balanced dataset:** 3,878 images (up to 1,500 per class)

---

## Project Structure

```
football-classifier/
├── main.py                 # Main pipeline — runs all steps in order
├── prepare_dataset.py      # Extracts crops from YOLO annotations
├── balance_dataset.py      # Balances class distribution
├── train.py                # Trains all three CNN models
├── gradcam.py              # Generates Grad-CAM visualizations
├── inference_demo.py       # Runs MobileNetV2 on a real match image
├── requirements.txt
├── data/                   # Raw YOLO dataset (downloaded from Roboflow)
│   ├── train/
│   ├── valid/
│   └── test/
├── dataset/                # Extracted crops by class
├── dataset_balanced/       # Balanced dataset used for training
└── outputs/                # Saved models and generated figures
    ├── best_custom_cnn.keras
    ├── best_mobilenet_ft.keras
    ├── best_resnet_ft.keras
    ├── class_distribution.pdf
    ├── sample_images.pdf
    ├── curves_Custom_CNN.pdf
    ├── curves_MobileNetV2.pdf
    ├── curves_ResNet50.pdf
    ├── cm_Custom_CNN.pdf
    ├── cm_MobileNetV2.pdf
    ├── cm_ResNet50.pdf
    ├── model_comparison.pdf
    ├── gradcam_Custom_CNN.pdf
    ├── gradcam_MobileNetV2.pdf
    ├── gradcam_ResNet50.pdf
    └── inference_demo.pdf
```

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Step 0 — Download the dataset (required before anything else)

The dataset is not included in this repository due to its size. Download it from Roboflow before running the pipeline:

```bash
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_API_KEY')
project = rf.workspace('roboflow-jvuqo').project('football-players-detection-3zvbc')
project.version(1).download('yolov8', location='./data')
"
```

Replace `YOUR_API_KEY` with your Roboflow API key (available at https://app.roboflow.com/settings/api).

### Option 1 — Full pipeline (recommended)

After downloading the dataset, run all remaining steps with a single command:

```bash
python main.py
```

This will execute in order: crop extraction → balancing → training → Grad-CAM → inference demo.

### Option 2 — Step by step

```bash
# Step 1: Extract crops from YOLO annotations
python prepare_dataset.py

# Step 2: Balance the dataset
python balance_dataset.py

# Step 3: Train the models
python train.py

# Step 4: Generate Grad-CAM visualizations
python gradcam.py

# Step 5: Run inference on a real match image
python inference_demo.py
```

---

## Output Figures

| Figure | Description |
|---|---|
| `class_distribution.pdf` | Bar chart showing the balanced class distribution |
| `sample_images.pdf` | Grid of sample images from each class |
| `curves_*.pdf` | Training and validation accuracy/loss curves per model |
| `cm_*.pdf` | Confusion matrix per model |
| `model_comparison.pdf` | Bar chart comparing validation accuracy across models |
| `gradcam_*.pdf` | Grad-CAM heatmaps showing model attention regions |
| `inference_demo.pdf` | MobileNetV2 applied to a real match image with bounding boxes |

---

## Methodology

1. **Dataset preparation** — Raw YOLO detection dataset converted to classification format by extracting bounding box crops
2. **Balancing** — Random undersampling to limit each class to 1,500 images
3. **Augmentation** — Rotation, flipping, zoom, brightness variation, and shifting
4. **Training** — Custom CNN trained from scratch; MobileNetV2 and ResNet50 trained with two-phase transfer learning (feature extraction + fine-tuning)
5. **Evaluation** — Accuracy, precision, recall, F1-score, confusion matrix
6. **Explainability** — Grad-CAM applied to the last convolutional layer of each model
7. **Inference demo** — MobileNetV2 applied to a full match image with ground truth bounding boxes and model predictions overlaid

---

## Author

Marcelo Augusto
MSc Software Engineering — University of Europe for Applied Sciences
Pattern Recognition — Summer 2026