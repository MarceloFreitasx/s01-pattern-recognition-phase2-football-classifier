---
title: Football Role Classifier
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: cc-by-4.0
---

# Football Role Classifier

Interactive demo for the football player role classification project.

## Modes

1. **Match Analysis** — upload a match frame (or try an example). **YOLOv8** detects
   objects and **MobileNetV2** classifies each crop.
2. **Crop Explorer + Grad-CAM** — upload a single object crop, get class
   probabilities and a Grad-CAM explanation heatmap.

Classes: **ball**, **goalkeeper**, **player**, **referee**.

## Run locally

From the project root:

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Required model files in `demo/model/`:

- `best_mobilenet_ft.keras` — classifier (~31 MB)
- `yolo_detector.pt` — detector (~6 MB)

## Train the YOLO detector (optional)

If `yolo_detector.pt` is missing, train on the Roboflow dataset:

```bash
yolo detect train model=yolov8n.pt data=demo/yolo_data.yaml epochs=20 imgsz=640 \
  project=demo/model name=detector device=cpu
cp demo/model/detector/weights/best.pt demo/model/yolo_detector.pt
```

Update `path:` in `demo/yolo_data.yaml` to point to your local `data/` folder.

## Deploy on Hugging Face Spaces

1. Create an account at [huggingface.co](https://huggingface.co).
2. Click **New Space** and choose **Gradio** as the SDK.
3. Upload the contents of this `demo/` folder (or connect a Git repo).
4. Include both model files. Use **Git LFS** for large weights:

   ```bash
   git lfs install
   git lfs track "*.keras" "*.pt"
   ```

5. Once the Space builds, share the public URL with your professor.

## Notes

- Upload works best on **broadcast-style match images** similar to the training data.
- Crop Explorer expects a **single object crop** (224×224 after resize).
- Example match frames are in `examples/matches/`; example crops in `examples/`.
