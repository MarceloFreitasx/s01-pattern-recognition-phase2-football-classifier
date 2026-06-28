# Presentation Video Script — 10 minutes
**Project:** An Explainable Transfer-Learning Framework for On-Pitch Role Classification in Broadcast Football Images
**Author:** Marcelo Augusto

> Language note: slides + narration below are in English to match the paper.
> Ask if you want a Portuguese version of the narration.
> Pace: ~140–150 words/min. Total target ≈ 10:00. Times are cumulative budgets.
> Slide deck: an animated web version of these slides lives in `slides/index.html`
> (open it in a browser, or run `python3 -m http.server` inside `slides/`). It includes
> live charts with the real numbers below — handy for screen-recording the talk.
>
> **Required format (per instructor): max 10:00 video = 2:00 live demo + 8:00 slides.**
> - **Part A (0:00–2:00):** screen-record the public app and narrate the demo (script below).
> - **Part B (2:00–10:00):** present the slides. The 13 slides below were originally budgeted
>   for 10:00, so to fit 8:00 keep narration tight and trim the lower-priority slides
>   (Slide 7 setup → ~20s; Slides 3 & 11 → ~30s each). Slide 8 stays the centerpiece.
>
> ⚠️ The HF Space sleeps when idle — open it ~1 min before recording so it is awake.
>
> **Speaker notes cues (press `S` in the browser):**
> - **Negrito** (`.say`) = texto para falar em voz alta
> - **Cinza itálico** (`.hint`) = orientação de palco — não ler (ex.: “apontar gráfico”, “contadores animam”)
> - **`→ AVANÇAR`** (yellow) — press `→` or Space *after* reading the text above it; reveals the next bullet/card/fragment on the current slide.
> - **`→ SLIDE`** (blue) — press `→` once more to go to the **next slide** (when there are no hidden items left).
> - **`DEMO`** (purple) — switch to screen-recording the Hugging Face app (Part A only).
> Read each **bold** block aloud, then hit the cue — slides stay in sync with your speech.

---

## Part A — Live Demo (0:00–2:00)
**Public link:** https://huggingface.co/spaces/marcelofreitasue/football-role-classifier

**Two ways to record:**

| Modo | Como |
|------|------|
| **Contínuo (recomendado)** | Slide Title (apresentação) → Slide Live Demo → grava tela HF → volta aos slides (Problem em diante) |
| **Separado (formato professor)** | Grave só a demo (0–2 min), depois grave os slides; no Title use a variante “You just saw…” e **pule** o slide Live Demo |

**On screen (demo):** the live Hugging Face Space (not slides).

**What to show (record your screen):**
1. Open the public link; show it loads in a plain browser — "no install, just a URL."
2. Upload a clear **player** crop → show predicted role + confidence.
3. Upload a **ball** crop → correct, high confidence.
4. Upload a **goalkeeper** or **referee** crop → use it to mention these are the harder classes.
5. (Optional) Upload a tiny/blurry crop → show lower confidence, motivating the robustness analysis.

**Narration (após intro no slide Title):**
"Let me show you the site. I'll open the public Hugging Face link — no install, just a browser tab. That's the whole point: low cost, easy access. I upload a player crop — the app returns the predicted role and a confidence score straight away. Now a ball — usually correct, often with high confidence; once it's cropped, the ball is actually one of the easier classes. A goalkeeper or a referee — trickier; in a small crop they can look a lot like outfield players, and you'll see that in the numbers later. The app runs the compact Custom CNN — the model that turned out most balanced in my tests. Okay — back to the slides. Let me explain how I built and evaluated it."

---

---

# Part B — Slides (2:00–10:00 · 8 minutes)

## Slide 1 — Title (início · ~0:25 se contínuo, ou 2:00–2:25 se demo já gravada)

**On slide:**
- Title of the paper
- Your name, University of Europe for Applied Sciences
- Pattern Recognition Course Project
- Live-demo link + football broadcast image in the background
- Callout: *First: live demo · Then: methodology & results*

**Narration (gravação contínua — começo do vídeo):**
"Hi, I'm Marcelo Augusto, from the University of Europe for Applied Sciences. This is my Pattern Recognition course project — an explainable transfer-learning framework that classifies four on-pitch roles in football broadcasts: ball, goalkeeper, player, and referee. The idea is affordable, trustworthy classification from ordinary broadcast crops — not just a high accuracy number, but something you can actually deploy and inspect. I'll start with a quick live demo of the public web app, then walk through the dataset, the three models I compared, and the results that surprised me."

**Narration (se demo já foi gravada antes — Parte A separada):**
"You just saw the project running live. I'm Marcelo Augusto, from the University of Europe for Applied Sciences. What follows is the work behind that demo — how I built a framework to classify four roles in football broadcasts, the dataset, the models, and what surprised me in the results." *(Pule o slide Live Demo.)*

---

## Slide 1b — Live Demo intro (~0:25–2:00 · só na gravação contínua)

**On slide:** HF Space link, bullets, callout *The next 2 minutes are a live walkthrough of the app.*

**Narration:** *(mesmo texto da Part A acima; depois → SLIDE para Problem)*

---

## Slide 2 — Problem & Motivation
**On slide:**
- "Every match = hours of video, mostly unused"
- Roles to identify: ball / goalkeeper / player / referee
- Today: manual labelling or expensive proprietary tools
- Goal: low-cost, automatic role recognition for smaller clubs

**Narration:**
"So that was the app running live. The question behind it: every match on TV generates hours of video — but most of it never becomes usable data. You still need manual labelling or expensive commercial tools, and that's out of reach for smaller clubs and independent researchers. Before you can do anything useful — tactics, scouting, live stats — you have to know what's in the frame. So I wanted a model that could do that from ordinary broadcast crops, without a huge budget — and I didn't want to stop at accuracy. I needed to know whether the model is actually trustworthy, not just whether it scores well."

---

## Slide 3 — Research Questions (1:30–2:15)
**On slide (RQ1–RQ5, short form):**
- RQ1: How accurately can CNNs classify the four roles?
- RQ2: Custom CNN vs transfer learning — which is better?
- RQ3: Effect of preprocessing, augmentation, balancing?
- RQ4: Do Grad-CAM & SHAP show meaningful attention?
- RQ5: Main errors & what's needed for real deployment?

**Narration:**
"I organised the project around five questions. RQ1: can CNNs actually classify these four roles from broadcast crops? RQ2: does a small custom network beat transfer learning — MobileNetV2 and ResNet50? RQ3: how much do preprocessing, augmentation, and balancing matter when the data is this skewed? RQ4: when I look inside the model with Grad-CAM and SHAP, is it paying attention to the object — or to something else? RQ5: where does it fail, and what would it take to deploy this for real? Each results section answers one of them."

---

## Slide 4 — Dataset (2:15–3:15)
**On slide:**
- Roboflow "Football Players Detection", CC BY 4.0
- 663 match images → 15,636 object crops
- 4 classes, YOLO bounding boxes
- **Insert Table 1 (dataset composition)** + a couple of sample crops (Fig. samples)
- Highlight: player = 13,239 vs ball = 405 → strong imbalance

**Narration:**
"The data comes from a public Roboflow dataset — Football Players Detection, CC BY 4.0. Six hundred and sixty-three match images; every YOLO bounding box became its own crop — 15,636 samples across four classes. I kept the original split at the match level so crops from the same frame never leak across train, validation and test. Balancing only touched training; validation and test stay imbalanced on purpose, because that's what deployment looks like. The table tells the story: over thirteen thousand player crops, four hundred balls — a thirty-three-times gap, and players alone are eighty-four percent of the test set."

---

## Slide 5 — Methodology / Workflow (3:15–4:15)
**On slide:**
- **Insert workflow figure (Figures/workflow.pdf)**
- Steps: crops → leakage-free split → train-only balancing + augmentation → 3 models → training → evaluation → Grad-CAM + SHAP + robustness

**Narration:**
"Here's the pipeline end to end. Extract and clean crops, preserve the match-level split, balance training only with undersampling and class weights, apply label-preserving augmentation — rotation, shifts, zoom, brightness, horizontal flip. I deliberately avoided strong colour changes, because kit colour is how you tell a goalkeeper or referee from an outfield player. Then train three models and evaluate them with macro F1, per-class metrics, Grad-CAM, SHAP, and a robustness stress test — not just accuracy."

---

## Slide 6 — Models (4:15–5:15)
**On slide:**
- **Custom CNN** (from scratch): 4 conv blocks → GAP → Dense → softmax; **0.46M params**
- **Insert architecture figure (Figures/architecture.pdf)**
- **MobileNetV2** (~3.05M) and **ResNet50** (~24.8M): two-phase transfer learning (freeze head → fine-tune last layers)

**Narration:**
"Three models, same data, same recipe. The baseline is a compact custom CNN — four conv blocks, 0.46 million parameters, trained from scratch. Against it, MobileNetV2 at 3.05M and ResNet50 at 24.8M, both ImageNet pre-trained. Same two-phase strategy: freeze the backbone, train the head, then unfreeze the last layers and fine-tune slowly — thirty layers for MobileNet, fifteen for ResNet. The honest question: on a dataset this small and this skewed, does a bigger network actually help — or does it just learn to cheat?"

---

## Slide 7 — Experimental Setup (5:15–5:45)
**On slide:**
- **Insert Table 2 (hyperparameters)** — Adam, weighted cross-entropy, batch 32, early stopping, ReduceLROnPlateau
- Hardware: NVIDIA RTX 3080; TensorFlow/Keras; reproducible scripts + Kaggle notebook

**Narration:**
"Training was identical across models: Adam, weighted cross-entropy so mistakes on rare classes hurt more, batch 32, early stopping, ReduceLROnPlateau — always keeping the best validation checkpoint, not the last epoch. One RTX 3080, TensorFlow/Keras, scripts plus a Kaggle notebook. Fair warning: one run per model, one split — I treat small gaps as suggestive, not statistically proven."

---

## Slide 8 — Results: Performance (5:45–7:15)  ← MOST IMPORTANT
**On slide:**
- **Insert Table 3 (results)** and **Table 4 (per-class F1)**
- Custom CNN: 77.9% acc, **macro F1 0.67** (best balanced)
- MobileNetV2: 69.1% acc, macro F1 0.60
- ResNet50: **84.0% acc** but **macro F1 0.23**
- Per-class F1 (Ball / GK / Player / Ref): Custom CNN 0.94 / 0.30 / 0.87 / 0.58 · MobileNetV2 0.82 / 0.18 / 0.80 / 0.59 · ResNet50 0.00 / 0.00 / 0.91 / 0.00
- Goalkeeper is the hardest *real* class for every model
- **Insert ResNet50 test confusion matrix** (all mass in "player")

**Narration:**
"This is the slide to slow down on — the result that surprised me. ResNet50 has the highest accuracy, eighty-four percent. If you stopped there, you'd call it the winner. But macro F1 is 0.23. The confusion matrix explains why: it predicts 'player' for every single crop. Since players are eighty-four percent of the test set, that shortcut buys high accuracy while scoring zero F1 on ball, goalkeeper, and referee. The custom CNN gives the most balanced performance — seventy-eight percent accuracy, 0.67 macro F1 — with fifty times fewer parameters. Among models that actually distinguish classes, goalkeeper is the hardest. Bottom line: under class imbalance, accuracy on its own will lie to you."

---

## Slide 9 — Explainability: Grad-CAM & SHAP (7:15–8:00)
**On slide:**
- **Insert Grad-CAM (Custom CNN) + SHAP (Custom CNN) side by side**
- Custom CNN / MobileNetV2: attention on the object (body, ball)
- ResNet50: diffuse / background → confirms it ignores content

**Narration:**
"Numbers aren't enough — I wanted to see where the models look. For the custom CNN and MobileNetV2, Grad-CAM and SHAP both focus on the object: the player's body, the ball. For ResNet50, the maps are diffuse, attention drifting to the background. Two very different techniques, same conclusion: the high-accuracy model isn't really using the image. Explainability turned the apparent winner into an obvious no-go."

---

## Slide 10 — Robustness (8:00–8:30)
**On slide:**
- **Insert robustness figure (robustness_accuracy.pdf)**
- 5 corruptions × 5 severities (blur, noise, low brightness, occlusion, JPEG)
- Custom CNN & MobileNetV2 respond (e.g. CNN 77.9% → ~9% under strong brightness loss)
- ResNet50 = flat line at 84.0% (degenerate, not robust)

**Narration:**
"I also hammered the test set with blur, noise, brightness loss, occlusion, and JPEG compression. The custom CNN and MobileNetV2 degrade as you'd expect — the CNN drops from about seventy-eight to nine percent under strong darkening. ResNet50 draws a flat line at eighty-four everywhere. That looks like robustness; it isn't. It's the same shortcut — always 'player' — so corruption doesn't change the output. Third piece of evidence, same story."

---

## Slide 11 — Limitations & Future Work (8:30–9:00)
**On slide:**
- Small, single-source dataset; few ball/goalkeeper crops
- One split, single run (no statistical testing)
- Classifies crops only (needs an external detector)
- Future: more balanced data (esp. goalkeepers), external validation, YOLO integration, focal loss, uncertainty

**Narration:**
"Being upfront about limits: one public dataset, not many ball or goalkeeper crops, one run per model on one split. The classifier works on crops you already have — you'd still need a detector like YOLO for full frames. What I'd do next: more data, especially goalkeepers; validate on other broadcasts; wire this into a detector for an end-to-end pipeline."

---

## Slide 12 — Conclusion (9:00–9:40)
**On slide:**
- Best *balanced* model: custom CNN (0.67 macro F1, 0.46M params)
- Highest accuracy ≠ best model (ResNet50 collapse)
- Grad-CAM + SHAP + robustness expose the shortcut
- Smallest model = most reliable & deployable

**Narration:**
"Wrapping up: the custom CNN was the most balanced and the one I deployed — 0.67 macro F1 at 0.46M parameters. The highest-accuracy model was actually the worst once you looked per class. Grad-CAM, SHAP, and robustness all pointed at the same shortcut. For this task, the smallest model was also the most trustworthy. The takeaway I'd take forward: macro F1, per-class metrics, explainability, and robustness together — not accuracy on its own."

---

## Slide 13 — Thank You / Links (9:40–10:00)
**On slide:**
- "Thank you"
- GitHub repo link
- Kaggle notebook link
- Contact / questions

**Narration:**
"Thanks for watching. The live app is on Hugging Face, the code and Kaggle notebook are linked, and the paper has the full write-up. Happy to take questions."

---

## Production checklist
- [ ] Animated web deck ready at `slides/index.html` (reveal.js + Chart.js). Press `S` for speaker notes, `F` for fullscreen, `→`/Space to advance.
- [ ] Export each referenced figure/table as a clean image for the slides.
- [ ] Create `Figures/workflow.pdf` and `Figures/architecture.pdf` first (slides 5 & 6 need them).
- [ ] Keep ~13 slides; rehearse once to confirm ≈10:00.
- [ ] Record screen + voice (e.g., PowerPoint "Record", OBS, or Loom).
- [ ] Show the Kaggle notebook briefly during slide 7 or 8 to prove "results match code".
- [ ] Speak slowly on slide 8 — it's the key contribution.
