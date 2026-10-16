# CrowdDensity Mini

> **7% — AI / Machine Learning / Computer Vision Learning Progress**

A lightweight computer vision MVP for **person detection and crowd counting** using YOLO11n trained on the full annotated CrowdHuman dataset.

<p align="center">
  <b>Image → Person Detection → Person Count → Visual Result</b>
</p>

---

# Overview

**CrowdDensity Mini** is a computer vision project built to explore how AI can assist in estimating the number of visible people in crowded public spaces.

The project focuses on:

- detecting people,
- counting detected persons,
- visualizing bounding boxes,
- saving inference results,
- understanding failure cases in dense crowds.

The current version is intentionally positioned as a:

> **person detection + crowd counting baseline**

It is **not yet a crowd density-map estimator**.

The system does not perform:

- face recognition,
- identity tracking,
- demographic inference,
- biometric identification.

---

# Problem Background

Public spaces such as:

- events,
- transport areas,
- sidewalks,
- stations,
- malls,
- public facilities,

can become crowded quickly.

Manual counting becomes increasingly difficult when:

- people overlap,
- subjects are partially occluded,
- people appear very small,
- crowd density becomes high,
- the scene contains many people at different distances.

Computer vision can assist operators by automatically detecting visible people and providing an approximate count.

However, counting people from bounding boxes is not the same as estimating crowd density.

This project therefore starts from a simpler and more defensible MVP:

> **Can a lightweight object detector identify and count people in crowded images reasonably well?**

---

# Current Project Status

```text
Status      : MVP / Experimental Baseline
Task        : Person Detection + Crowd Counting
Model       : YOLO11n
Framework   : Ultralytics YOLO / PyTorch
Input       : Crowd / pedestrian image
Output      : Bounding boxes + detected person count
```

Current pipeline:

```text
Input Image
     ↓
Resize / YOLO Preprocessing
     ↓
YOLO11n
     ↓
Person Detection
     ↓
Confidence Filtering
     ↓
Bounding Boxes
     ↓
Person Count
     ↓
Annotated Image + JSON Result
```

---

# Dataset

The project uses the **CrowdHuman** dataset.

CrowdHuman is designed for human detection in crowded scenes and contains images with:

- high person density,
- heavy occlusion,
- partially visible people,
- small human instances,
- difficult crowd conditions.

The dataset was converted from its original annotation format into YOLO detection format.

---

# Dataset Used

The final experiment used the complete annotated training and validation splits available in the project environment.

```text
Training images   : 15,000
Validation images : 4,370
```

No annotated train or validation images were intentionally subsampled in the final experiment.

The training pipeline verified:

```text
all_annotated_train_used = true
all_annotated_val_used   = true
```

---

# Validation Instances

The validation split contains:

```text
99,481 annotated person instances
```

This makes the validation set substantially more difficult than ordinary sparse-object detection datasets.

Some images contain very large numbers of people.

---

# Annotation Conversion

CrowdHuman annotations were originally stored in:

```text
annotation_train.odgt
annotation_val.odgt
```

The pipeline converts valid human bounding boxes into YOLO format:

```text
class_id x_center y_center width height
```

Only valid non-ignored person boxes are used.

The project currently uses one class:

```text
0: person
```

---

# Full-Data Experiment

The final experiment uses:

```text
Model       : YOLO11n
Epochs      : 15
Image Size  : 640
Batch Size  : 16
Workers     : 2
Device      : NVIDIA Tesla T4
Task        : Object Detection
Classes     : 1
```

Training uses pretrained YOLO11n weights.

The model is fine-tuned specifically for:

```text
person
```

---

# Training Configuration

Important final-run settings:

```text
epochs      = 15
imgsz       = 640
batch       = 16
workers     = 2
max_det     = 400
seed        = 42
pretrained  = True
```

The optimizer was selected automatically by Ultralytics.

Final training used:

```text
AdamW
```

with Automatic Mixed Precision enabled.

---

# Why YOLO11n?

YOLO11n was selected because this project prioritizes:

- lightweight inference,
- fast experimentation,
- relatively small model size,
- simple deployment,
- strong object-detection baseline,
- suitability for portfolio-scale MVP development.

The objective was not to use the largest model available.

Instead:

> **The goal is to establish a practical baseline that can later be improved.**

---

# Final Validation Results

The final full-data model achieved:

| Metric | Result |
|---|---:|
| Precision | **83.70%** |
| Recall | **69.68%** |
| mAP@50 | **80.92%** |
| mAP@50–95 | **48.90%** |

Exact values:

```text
Precision   : 0.8370084562
Recall      : 0.6967773970
mAP50       : 0.8092446719
mAP50-95    : 0.4889942301
```

---

# Metric Interpretation

## Precision — 83.70%

When the model predicts a person, many detections correspond to actual annotated people.

The relatively strong precision indicates that false-positive detections are reasonably controlled.

---

## Recall — 69.68%

The model still misses some annotated people.

This is expected in difficult CrowdHuman images containing:

- severe occlusion,
- tiny persons,
- overlapping bodies,
- dense background crowds.

Recall is therefore an important limitation of the current baseline.

---

## mAP50 — 80.92%

The model performs reasonably well when detections are evaluated at IoU 0.50.

---

## mAP50-95 — 48.90%

Performance decreases under stricter localization thresholds.

This indicates that bounding-box localization still has significant room for improvement.

---

# Training Behaviour

During the 15 training epochs:

- training losses consistently decreased,
- validation losses also decreased,
- precision improved,
- recall improved,
- mAP50 improved,
- mAP50-95 improved.

No major divergence between train and validation curves was observed during the final short training run.

The final epoch achieved the strongest validation performance in the recorded training history.

---

# Counting Experiment

Detection boxes can also be used to estimate the visible number of people.

A small counting evaluation was performed on:

```text
50 randomly selected validation images
```

The model prediction count was compared against the number of annotated persons.

Result:

```text
Mean Absolute Error = 5.34 persons
```

This means that, across this specific 50-image validation sample, the predicted count differed from the annotated count by approximately:

```text
5.34 people per image on average
```

---

# Important Counting Limitation

The counting experiment is only a small demonstration.

It is NOT equivalent to a full crowd-counting benchmark.

The result depends on:

- confidence threshold,
- IoU threshold,
- occlusion,
- person size,
- bounding-box quality,
- CrowdHuman annotation conventions.

Therefore:

```text
MAE = 5.34
```

must always be described as:

> **a 50-image counting demo result**

and not as the universal counting error of the system.

---

# Confusion Matrix

The repository contains:

```text
training_plots/confusion_matrix.png
training_plots/confusion_matrix_normalized.png
```

Important:

YOLO detection confusion matrices contain a special:

```text
background
```

row / column.

This is different from an ordinary image-classification confusion matrix.

The values therefore must not be interpreted as simple binary classification counts.

---

# Label Distribution

The repository also includes:

```text
training_plots/labels.jpg
```

The visualization shows the distribution of annotated human bounding boxes.

CrowdHuman contains many:

- narrow person boxes,
- tall human boxes,
- overlapping bounding boxes,
- small-scale people.

This reflects the difficulty of crowded pedestrian detection.

---

# Qualitative Test Inference

The official CrowdHuman test image folder was used for qualitative inference.

The project generated predictions on:

```text
20 official test images
```

These examples are stored under:

```text
assets/results/
```

Example outputs contain:

```text
person bounding boxes
confidence scores
```

---

# Why Test Metrics Are Not Reported

The project environment does not contain:

```text
annotation_test.odgt
```

Therefore, official CrowdHuman test images are used only for:

```text
qualitative inference
```

and not for quantitative test metrics.

The reported Precision, Recall, mAP50, and mAP50-95 values are from the annotated validation set.

---

# Streamlit MVP

CrowdDensity Mini includes a simple Streamlit interface.

Run:

```bash
streamlit run app.py
```

The app allows users to:

- upload a crowd image,
- adjust confidence threshold,
- adjust IoU threshold,
- run person detection,
- view annotated detections,
- see detected person count,
- inspect detection confidence,
- save inference results.

---

# Application Flow

```text
Upload Image
     ↓
Analyze Crowd
     ↓
YOLO11n Inference
     ↓
Filter Predictions
     ↓
Count Person Boxes
     ↓
Display Detection Result
     ↓
Save JPG
     ↓
Save JSON
```

---

# Inference Defaults

Default application settings:

```text
Confidence Threshold : 0.25
IoU Threshold        : 0.50
Image Size           : 640
Maximum Detections   : 400
```

These values can be changed through the Streamlit interface.

Changing the confidence threshold can significantly affect the detected person count.

---

# Saved Results

New inference outputs are stored in:

```text
assets/results/
```

Image output:

```text
crowd_result_YYYYMMDD_HHMMSS.jpg
```

Metadata output:

```text
crowd_result_YYYYMMDD_HHMMSS.json
```

Example JSON information:

```json
{
  "task": "person_detection_and_counting",
  "detected_person_count": 42,
  "confidence_threshold": 0.25,
  "iou_threshold": 0.5
}
```

---

# Important Interpretation

The displayed person count represents:

> **the number of person bounding boxes detected by the model**

It does NOT guarantee the true number of people in the scene.

Possible errors include:

```text
Undercount
```

when people are missed.

And:

```text
Overcount
```

when incorrect or duplicate detections occur.

---

# Failure Cases

The current model is especially challenged by:

## Severe Occlusion

People may be mostly hidden behind other people.

---

## Tiny Persons

People far away from the camera occupy only a small number of pixels.

---

## Extremely Dense Crowds

Large amounts of overlapping detections make localization difficult.

---

## Confidence Threshold

Increasing the threshold can remove valid low-confidence detections.

Lowering it may increase false positives.

---

## Domain Shift

Images from:

- CCTV cameras,
- drones,
- unusual camera angles,
- night scenes,
- low-resolution cameras,

may differ from CrowdHuman.

Performance should not be assumed to transfer perfectly.

---

# What This Project Is

CrowdDensity Mini currently is:

> **A lightweight person detection and crowd counting MVP trained on CrowdHuman.**

It demonstrates:

- dataset conversion,
- full-data training,
- object detection,
- evaluation,
- counting from detections,
- qualitative testing,
- Streamlit deployment,
- artifact tracking.

---

# What This Project Is NOT

The current model is NOT:

- a density-map estimator,
- an exact crowd counter,
- a face-recognition system,
- a tracking system,
- a production crowd-safety system,
- a crowd-behaviour prediction system.

Do not interpret the model output as an exact occupancy measurement.

---

# Privacy

This MVP detects:

```text
person
```

It does not attempt to identify who those people are.

The current project does not use:

- face recognition,
- names,
- biometric identity,
- demographic classification.

The intended concept is:

> **anonymous crowd monitoring**

rather than personal identification.

---

# Repository Structure

```text
CrowdDensity-Mini/
│
├── assets/
│   └── results/
│       ├── *_pred.jpg
│       ├── crowd_result_*.jpg
│       └── crowd_result_*.json
│
├── models/
│   └── crowddensity_yolo11n_full_best.pt
│
├── notebooks/
│   └── crowd*.ipynb
│
├── reports/
│   ├── counting_demo_50.csv
│   ├── final_summary.json
│   └── manifest.csv
│
├── src/
│
├── training_plots/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── labels.jpg
│   ├── results.csv
│   └── results.png
│
├── .gitignore
├── AGENTS.md
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

# Artifact Description

## `models/`

Contains the final trained model:

```text
crowddensity_yolo11n_full_best.pt
```

---

## `reports/`

Contains experiment reports.

### `final_summary.json`

Summary of:

- dataset usage,
- training configuration,
- detection metrics,
- counting demo,
- limitations.

### `manifest.csv`

Dataset manifest generated during annotation conversion.

### `counting_demo_50.csv`

Ground-truth versus predicted person counts for the 50-image counting demonstration.

---

## `training_plots/`

Contains Ultralytics training artifacts:

```text
results.png
results.csv
confusion_matrix.png
confusion_matrix_normalized.png
labels.jpg
```

---

## `assets/results/`

Contains qualitative inference examples and Streamlit inference outputs.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/MENSTRUE/CrowdDensity-Mini.git
```

Enter the project:

```bash
cd CrowdDensity-Mini
```

Create a virtual environment if desired:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
streamlit run app.py
```

Streamlit will open the local application in your browser.

Upload a crowd image and click:

```text
Analyze Crowd
```

---

# Requirements

Core runtime dependencies:

```text
streamlit
ultralytics
opencv-python-headless
pillow
numpy
```

---

# Reproducibility

The training experiment used:

```text
Seed = 42
```

and deterministic execution was requested through Ultralytics.

However, exact reproducibility can still be affected by:

- CUDA versions,
- PyTorch versions,
- Ultralytics versions,
- GPU architecture,
- dependency updates.

---

# Development Lessons

This project reinforced several important lessons.

## Detection Metrics Are Not Counting Metrics

A detector can have good mAP while still producing meaningful count errors.

---

## Crowded Scenes Are Difficult

Occlusion and tiny objects remain challenging even when the model performs reasonably well overall.

---

## Thresholds Matter

The final count is affected by inference configuration.

---

## Full Data Matters

The final project uses all available annotated train and validation images rather than relying only on a small prototype subset.

---

## Evaluation Must Match the Claim

This project reports:

```text
person detection metrics
```

and a separate:

```text
50-image counting demo
```

These two evaluations should not be confused.

---

# Current Limitations

The current MVP is limited by:

- object-detection-based counting,
- severe crowd occlusion,
- small person instances,
- a single dataset domain,
- no temporal tracking,
- no density-map estimation,
- no occupancy-zone logic,
- no real-time camera integration,
- limited counting evaluation.

---

# Future Development

Potential future development:

```text
Crowd Image / Video
        ↓
Person Detection
        ↓
Tracking
        ↓
Zone-Based Counting
        ↓
Temporal Occupancy
        ↓
Crowd Density Estimation
        ↓
Warning Threshold
        ↓
Human Operator Review
```

Potential improvements include:

- ByteTrack / BoT-SORT tracking,
- entrance / exit counting,
- ROI-based occupancy,
- density heatmaps,
- dedicated crowd-counting networks,
- CSRNet-style density estimation,
- perspective-aware density estimation,
- video stream support,
- domain-specific fine-tuning,
- low-light evaluation,
- CCTV deployment experiments.

---

# Product Direction

The long-term product idea is not simply:

```text
count people
```

but:

```text
assist public-space operators in understanding occupancy conditions
```

Potential future outputs could include:

```text
Current occupancy
Crowd trend
Zone occupancy
Threshold warning
Density heatmap
```

Human operators should remain responsible for operational decisions.

---

# Public Progress

```text
7% — CrowdDensity Mini
```

This percentage represents personal learning and development progress.

It does not represent model accuracy or project completion percentage.

---

# Final Result

```text
Project             : CrowdDensity Mini
Status              : MVP / Experimental Baseline
Task                : Person Detection + Crowd Counting
Dataset             : CrowdHuman
Model               : YOLO11n

Training Images     : 15,000
Validation Images   : 4,370
Validation Persons  : 99,481

Epochs              : 15
Image Size          : 640
Batch Size          : 16

Precision           : 83.70%
Recall              : 69.68%
mAP50               : 80.92%
mAP50-95            : 48.90%

Counting Demo       : 50 validation images
Count MAE           : 5.34 persons
```

---

# Disclaimer

CrowdDensity Mini is an experimental portfolio project.

The model should not be treated as an exact occupancy measurement system or used as the sole basis for safety-critical decisions.

Detection and counting performance may change significantly across environments.

---

# Author

**Wafa Bila Syaefurokhman**

Focus:

```text
Android Development
Artificial Intelligence
Machine Learning
Computer Vision
```

GitHub:

```text
https://github.com/MENSTRUE
```

Project Repository:

```text
https://github.com/MENSTRUE/CrowdDensity-Mini
```

---

# Closing Note

CrowdDensity Mini is not intended to demonstrate a perfect crowd-counting system.

The goal is to build a complete and reproducible MVP:

```text
Problem
↓
Dataset
↓
Training
↓
Evaluation
↓
Failure Analysis
↓
Inference Application
↓
Documentation
↓
Next Iteration
```

The most important result is not only the model metric.

It is understanding:

> **where the baseline works, where it fails, and what should be improved next.**