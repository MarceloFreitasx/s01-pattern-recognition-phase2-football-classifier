# Presentation Video Script — 10 minutes
**Project:** An Explainable Transfer-Learning Framework for On-Pitch Role Classification in Broadcast Football Images
**Author:** Marcelo Augusto

> Language note: slides + narration below are in English to match the paper.
> Ask if you want a Portuguese version of the narration.
> Pace: ~140–150 words/min. Total target ≈ 10:00. Times are cumulative budgets.

---

## Slide 1 — Title (0:00–0:30)
**On slide:**
- Title of the paper
- Your name, University of Europe for Applied Sciences
- Pattern Recognition Course Project
- (optional) one football broadcast image in the background

**Narration:**
"Hi, I'm Marcelo. In this video I'll present my Pattern Recognition project: an explainable, transfer-learning framework for classifying on-pitch roles — ball, goalkeeper, player, and referee — directly from broadcast football images. I'll cover the problem, the dataset, my methodology, the main results, and what I learned."

---

## Slide 2 — Problem & Motivation (0:30–1:30)
**On slide:**
- "Every match = hours of video, mostly unused"
- Roles to identify: ball / goalkeeper / player / referee
- Today: manual labelling or expensive proprietary tools
- Goal: low-cost, automatic role recognition for smaller clubs

**Narration:**
"Every televised match produces hours of footage, but turning that into useful data still depends on manual work or expensive commercial systems that smaller clubs can't afford. A first step for any analytics — tactics, scouting, live stats — is knowing what each object on the pitch is. So my goal was to build an affordable model that recognises the four basic roles from ordinary broadcast crops, and — just as importantly — to check whether it can be *trusted*, not just whether it scores well."

---

## Slide 3 — Research Questions (1:30–2:15)
**On slide (RQ1–RQ5, short form):**
- RQ1: How accurately can CNNs classify the four roles?
- RQ2: Custom CNN vs transfer learning — which is better?
- RQ3: Effect of preprocessing, augmentation, balancing?
- RQ4: Do Grad-CAM & SHAP show meaningful attention?
- RQ5: Main errors & what's needed for real deployment?

**Narration:**
"I framed the project around five research questions: how accurately CNNs can classify the roles; whether a custom network or transfer learning works better; how much preprocessing and class balancing matter; whether explainability methods confirm the model looks at the right regions; and finally, what the main errors are and what real deployment would require. Everything in the results maps back to these five questions."

---

## Slide 4 — Dataset (2:15–3:15)
**On slide:**
- Roboflow "Football Players Detection", CC BY 4.0
- 663 match images → 15,636 object crops
- 4 classes, YOLO bounding boxes
- **Insert Table 1 (dataset composition)** + a couple of sample crops (Fig. samples)
- Highlight: player = 13,239 vs ball = 405 → strong imbalance

**Narration:**
"I used a public, openly-licensed Roboflow dataset of 663 annotated match images. I cropped every bounding box, giving 15,636 object crops across the four classes. The key challenge is visible in this table: it's extremely imbalanced — over thirteen thousand player crops versus only four hundred balls. I kept the original train/validation/test split at the image level so crops from the same frame never leak across splits, and I balanced *only* the training set, leaving validation and test realistically imbalanced."

---

## Slide 5 — Methodology / Workflow (3:15–4:15)
**On slide:**
- **Insert workflow figure (Figures/workflow.pdf)**
- Steps: crops → leakage-free split → train-only balancing + augmentation → 3 models → training → evaluation → Grad-CAM + SHAP + robustness

**Narration:**
"Here's the full pipeline. After extracting and cleaning crops, I preserve the match-level split, balance only the training data with undersampling plus class weights, and apply label-preserving augmentation — rotation, shifts, zoom, brightness, and horizontal flip. I deliberately avoided strong colour changes, because kit colour is exactly what separates goalkeepers and referees from players. Then I train three models and evaluate them not only with accuracy, but with per-class and macro F1, explainability, and robustness tests."

---

## Slide 6 — Models (4:15–5:15)
**On slide:**
- **Custom CNN** (from scratch): 4 conv blocks → GAP → Dense → softmax; **0.46M params**
- **Insert architecture figure (Figures/architecture.pdf)**
- **MobileNetV2** (~3.05M) and **ResNet50** (~24.8M): two-phase transfer learning (freeze head → fine-tune last layers)

**Narration:**
"I compare three models under identical conditions. The baseline is a compact custom CNN with four convolutional blocks and only 0.46 million parameters. Against it I fine-tune two ImageNet-pretrained backbones: the lightweight MobileNetV2 and the much larger ResNet50. Both use a two-phase strategy — first I train only the new head with the backbone frozen, then I unfreeze the last layers and fine-tune at a low learning rate. This lets me ask whether extra capacity actually helps on a small, imbalanced dataset."

---

## Slide 7 — Experimental Setup (5:15–5:45)
**On slide:**
- **Insert Table 2 (hyperparameters)** — Adam, weighted cross-entropy, batch 32, early stopping, ReduceLROnPlateau
- Hardware: NVIDIA RTX 3080; TensorFlow/Keras; reproducible scripts + Kaggle notebook

**Narration:**
"All models share the same setup: Adam optimizer, weighted categorical cross-entropy to penalise minority-class errors, batch size 32, early stopping, and a learning-rate scheduler. Everything ran on a single RTX 3080 in TensorFlow and Keras, with the code organised into reproducible scripts and a Kaggle notebook."

---

## Slide 8 — Results: Performance (5:45–7:15)  ← MOST IMPORTANT
**On slide:**
- **Insert Table 3 (results)** and **Table 4 (per-class F1)**
- Custom CNN: 77.9% acc, **macro F1 0.67** (best balanced)
- MobileNetV2: 69.1% acc, macro F1 0.60
- ResNet50: **84.0% acc** but **macro F1 0.23**
- **Insert ResNet50 test confusion matrix** (all mass in "player")

**Narration:**
"This is the core result, and it's counter-intuitive. By raw accuracy, ResNet50 *wins* with 84 percent. But look at its macro F1 — only 0.23. The confusion matrix explains why: ResNet50 predicts 'player' for *every single crop*. Since players are 84 percent of the test set, it gets high accuracy while completely failing on ball, goalkeeper, and referee, scoring zero F1 on all three. The little custom CNN, by contrast, gives the most balanced performance — 78 percent accuracy and a 0.67 macro F1 — and it does that with fifty times fewer parameters. So the headline lesson is: under class imbalance, accuracy alone is misleading."

---

## Slide 9 — Explainability: Grad-CAM & SHAP (7:15–8:00)
**On slide:**
- **Insert Grad-CAM (Custom CNN) + SHAP (Custom CNN) side by side**
- Custom CNN / MobileNetV2: attention on the object (body, ball)
- ResNet50: diffuse / background → confirms it ignores content

**Narration:**
"To check trust, I used two explainability methods. For the custom CNN and MobileNetV2, both Grad-CAM and SHAP put the evidence on the object itself — the player's body or the ball. For ResNet50, neither method shows consistent object focus. Two independent methods agreeing gives me confidence this isn't an artifact: the high-accuracy model simply isn't using the image content. Explainability turned a 'best' model into a clearly untrustworthy one."

---

## Slide 10 — Robustness (8:00–8:30)
**On slide:**
- **Insert robustness figure (robustness_accuracy.pdf)**
- 5 corruptions × 5 severities
- Custom CNN & MobileNetV2 respond; ResNet50 = flat line (degenerate)

**Narration:**
"I also stress-tested the models with blur, noise, brightness, occlusion, and JPEG compression. The custom CNN and MobileNetV2 degrade as you'd expect. ResNet50 produces a perfectly flat line at every severity — but that's not robustness, it's the same collapse: a model that always says 'player' is trivially invariant to any corruption. This independently confirms the explainability finding."

---

## Slide 11 — Limitations & Future Work (8:30–9:00)
**On slide:**
- Small, single-source dataset; few ball/goalkeeper crops
- One split, single run (no statistical testing)
- Classifies crops only (needs an external detector)
- Future: more balanced data (esp. goalkeepers), external validation, YOLO integration, focal loss, uncertainty

**Narration:**
"The main limitations: the dataset is small and from a single source, the minority classes have very few test crops, and I report a single run on one split. The system also classifies pre-made crops rather than detecting objects in full frames. The most valuable next steps are a larger, better-balanced dataset — especially more goalkeepers — external validation on other broadcasts, and integrating the classifier with a YOLO detector for an end-to-end system."

---

## Slide 12 — Conclusion (9:00–9:40)
**On slide:**
- Best *balanced* model: custom CNN (0.67 macro F1, 0.46M params)
- Highest accuracy ≠ best model (ResNet50 collapse)
- Grad-CAM + SHAP + robustness expose the shortcut
- Smallest model = most reliable & deployable

**Narration:**
"To conclude: the compact custom CNN was the most balanced and the most deployable model, while the highest-accuracy model was actually the worst once we looked beyond accuracy. The big takeaway is methodological — macro F1, per-class metrics, explainability, and robustness together are what reveal whether a model is genuinely working. For this task, the smallest and cheapest model was also the most trustworthy."

---

## Slide 13 — Thank You / Links (9:40–10:00)
**On slide:**
- "Thank you"
- GitHub repo link
- Kaggle notebook link
- Contact / questions

**Narration:**
"Thanks for watching. The full code is on GitHub, the experiments are reproducible in the linked Kaggle notebook, and all details are in the paper. I'm happy to answer any questions."

---

## Production checklist
- [ ] Export each referenced figure/table as a clean image for the slides.
- [ ] Create `Figures/workflow.pdf` and `Figures/architecture.pdf` first (slides 5 & 6 need them).
- [ ] Keep ~13 slides; rehearse once to confirm ≈10:00.
- [ ] Record screen + voice (e.g., PowerPoint "Record", OBS, or Loom).
- [ ] Show the Kaggle notebook briefly during slide 7 or 8 to prove "results match code".
- [ ] Speak slowly on slide 8 — it's the key contribution.
