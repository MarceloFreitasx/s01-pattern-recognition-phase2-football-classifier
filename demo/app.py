"""
Football Player Role Classification — Gradio demo.

Two modes:
  1. Match Analysis — upload a match frame (YOLO + MobileNetV2) or try an example.
  2. Crop Explorer — classify a single crop and show Grad-CAM.

Run locally:
    pip install -r requirements.txt
    python app.py
"""

import os

# Avoid TF/PyTorch GPU conflicts when both models are loaded in the same process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import sys
from pathlib import Path

import gradio as gr

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from detector import load_yolo

load_yolo()

from constants import CLASSES
from gradcam_utils import explain_crop
from match_analysis import annotate_match, annotate_uploaded_match, list_match_examples
from model_utils import load_mobilenet

match_model = load_mobilenet()
crop_model = match_model

CROP_EXAMPLES = [
    str(DEMO_DIR / "examples" / name)
    for name in ["ball.jpg", "goalkeeper.jpg", "player.jpg", "referee.jpg"]
]

MATCH_EXAMPLE_ROWS = [
    [str(DEMO_DIR / "examples" / "matches" / f"{name}.jpg"), name]
    for name in list_match_examples()
]

MATCH_DESCRIPTION = """
Upload a **match frame** or pick an **example** below.

- **Examples** use dataset annotations (ground truth) — same logic as
  `outputs/inference_demo.png`: box colours from the true class, label shows
  `true class → MobileNetV2 prediction`.
- **Your upload** uses **YOLOv8** to find objects (box colours) and **MobileNetV2**
  to classify each crop (shown in the label).
"""

CROP_DESCRIPTION = """
Upload a **tight crop** of a single object (ball, goalkeeper, player, or referee).
The model returns class probabilities and a **Grad-CAM** heatmap highlighting
which image regions influenced the prediction.
"""


def run_match_analysis(image, example_name: str | None):
    if example_name:
        return annotate_match(match_model, example_name)
    if image is None:
        raise gr.Error("Upload a match image or click one of the examples below.")
    try:
        return annotate_uploaded_match(match_model, image)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def run_crop_explorer(image):
    if image is None:
        raise gr.Error("Please upload a crop image first.")
    return explain_crop(crop_model, image)


with gr.Blocks(title="Football Role Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Football Role Classifier")
    gr.Markdown(
        "Interactive demo for classifying football match objects into "
        "**ball**, **goalkeeper**, **player**, and **referee**."
    )

    with gr.Tabs():
        with gr.Tab("Match Analysis"):
            gr.Markdown(MATCH_DESCRIPTION)
            with gr.Row():
                with gr.Column(scale=1):
                    match_input = gr.Image(type="pil", label="Upload match frame")
                    example_name = gr.Textbox(visible=False)
                    analyze_btn = gr.Button("Analyze Match", variant="primary")
                    gr.Examples(
                        examples=MATCH_EXAMPLE_ROWS,
                        inputs=[match_input, example_name],
                        label="Example match frames (ground-truth mode)",
                    )
                with gr.Column(scale=2):
                    match_output = gr.Image(type="pil", label="Annotated frame")
                    match_summary = gr.Markdown(label="Summary")

            match_input.upload(lambda: None, outputs=example_name)

            analyze_btn.click(
                fn=run_match_analysis,
                inputs=[match_input, example_name],
                outputs=[match_output, match_summary],
            )

        with gr.Tab("Crop Explorer + Grad-CAM"):
            gr.Markdown(CROP_DESCRIPTION)
            with gr.Row():
                with gr.Column():
                    crop_input = gr.Image(type="pil", label="Upload crop")
                    classify_btn = gr.Button("Classify", variant="primary")
                    gr.Examples(
                        examples=[[p] for p in CROP_EXAMPLES],
                        inputs=crop_input,
                        label="Example crops",
                    )
                with gr.Column():
                    crop_summary = gr.Markdown(label="Result")
                    crop_probs = gr.Label(num_top_classes=len(CLASSES), label="Probabilities")
                    gradcam_output = gr.Image(type="numpy", label="Grad-CAM overlay")

            classify_btn.click(
                fn=run_crop_explorer,
                inputs=crop_input,
                outputs=[crop_probs, crop_summary, gradcam_output],
            )
            crop_input.change(
                fn=run_crop_explorer,
                inputs=crop_input,
                outputs=[crop_probs, crop_summary, gradcam_output],
            )

if __name__ == "__main__":
    demo.launch()
