# AGENTS.md

# CrowdDensity Mini — AI Coding Agent Guidelines

This document defines the rules, constraints, verified project state, and development conventions that AI coding agents must follow when modifying this repository.

Every AI coding agent working on this project should read this file before making changes.

The primary goals are:

```text
correctness
reproducibility
honest evaluation
maintainability
clear claims
minimal unnecessary complexity
```

---

# 1. Project Identity

Project:

```text
CrowdDensity Mini
```

Public progress label:

```text
7%
```

Repository:

```text
https://github.com/MENSTRUE/CrowdDensity-Mini
```

Current status:

```text
MVP / Experimental Baseline
```

Current task:

```text
Person detection + crowd counting
```

Model:

```text
YOLO11n
```

Dataset:

```text
CrowdHuman
```

---

# 2. Core Project Rule

The most important rule is:

> Never overclaim what the model can do.

The current model detects visible persons using object detection.

The number shown by the application is:

```text
number of detected person bounding boxes
```

It is NOT guaranteed to be:

```text
the exact number of humans in the scene
```

---

# 3. Correct Project Description

Preferred:

```text
YOLO11n person detection and crowd counting baseline trained on CrowdHuman.
```

Acceptable:

```text
Crowd person detection MVP.
```

Acceptable:

```text
Detection-based crowd counting baseline.
```

Do NOT describe the current system as:

```text
exact crowd density estimator
density-map estimator
production occupancy system
crowd safety prediction system
human tracking platform
face recognition system
```

---

# 4. Current Model

Final model:

```text
models/crowddensity_yolo11n_full_best.pt
```

Architecture:

```text
YOLO11n
```

Framework:

```text
Ultralytics YOLO
PyTorch
```

Task:

```text
Object Detection
```

Number of classes:

```text
1
```

Class mapping:

```text
0: person
```

---

# 5. Verified Dataset State

Final experiment uses:

```text
Training images   : 15,000
Validation images : 4,370
```

All annotated train data was used:

```text
true
```

All annotated validation data was used:

```text
true
```

Agents must NOT describe the final model as being trained only on the earlier subset experiment.

The final model is the:

```text
FULL DATA v4
```

experiment.

---

# 6. Validation Ground Truth

The final validation set contains:

```text
99,481 annotated person instances
```

This should be described as:

```text
validation person instances
```

Do not describe this number as total dataset persons unless explicitly verified.

---

# 7. Official Test Set

The CrowdHuman test image folder is available.

However:

```text
annotation_test.odgt
```

is NOT available in the current project experiment.

Therefore, official test images are used only for:

```text
qualitative inference
```

Do NOT report quantitative test accuracy, mAP, precision, recall, or count error from the official test folder.

---

# 8. Final Training Configuration

Verified final configuration:

```text
Model      : YOLO11n
Epochs     : 15
Image Size : 640
Batch Size : 16
Workers    : 2
Seed       : 42
Max Det    : 400
```

Training device:

```text
NVIDIA Tesla T4
```

The Kaggle environment detected two T4 GPUs, but the final notebook intentionally used:

```text
device=0
```

for training stability.

Do not claim multi-GPU training.

---

# 9. Optimizer

Training used:

```text
optimizer="auto"
```

Ultralytics selected:

```text
AdamW
```

Final logged optimizer configuration included:

```text
lr = 0.002
momentum = 0.9
```

Agents should not change optimizer claims unless a new experiment is executed.

---

# 10. Final Detection Metrics

Verified final validation metrics:

```text
Precision   : 0.8370084562118518
Recall      : 0.6967773970140276
mAP50       : 0.8092446719459494
mAP50-95    : 0.48899423011949256
```

Preferred public rounding:

```text
Precision   : 83.70%
Recall      : 69.68%
mAP50       : 80.92%
mAP50-95    : 48.90%
```

These metrics come from:

```text
CrowdHuman validation split
```

Never label these as official test-set metrics.

---

# 11. Counting Demo

A counting demonstration was performed on:

```text
50 randomly selected validation images
```

using:

```text
confidence threshold = 0.25
IoU threshold        = 0.50
max detections       = 400
```

Result:

```text
Count MAE = 5.34 persons
```

This MUST be described as:

```text
50-image validation counting demo MAE
```

Do NOT write:

```text
CrowdDensity Mini has a universal MAE of 5.34.
```

Do NOT compare this MAE directly against dedicated crowd-density models unless a fair benchmark is actually performed.

---

# 12. Detection vs Counting

Agents must preserve the distinction between:

```text
object detection evaluation
```

and:

```text
counting evaluation
```

Detection metrics include:

```text
Precision
Recall
mAP50
mAP50-95
```

Counting metric currently includes only:

```text
MAE on a 50-image validation sample
```

Good detection mAP does not automatically imply exact counting.

---

# 13. Confusion Matrix Rule

Files:

```text
training_plots/confusion_matrix.png
training_plots/confusion_matrix_normalized.png
```

These are YOLO detection confusion matrices.

They include:

```text
person
background
```

The `background` row / column is part of object detection evaluation.

Do NOT interpret this matrix as an ordinary 2-class classification confusion matrix.

Do NOT claim that:

```text
background is a trained second class
```

The trained detector contains only:

```text
person
```

---

# 14. Training Plots

Expected training artifacts:

```text
training_plots/
    confusion_matrix.png
    confusion_matrix_normalized.png
    labels.jpg
    results.csv
    results.png
```

Additional Ultralytics curves may be added if available:

```text
PR_curve.png
P_curve.png
R_curve.png
F1_curve.png
```

Do not fabricate missing plots.

---

# 15. Application

Main application:

```text
app.py
```

Framework:

```text
Streamlit
```

Expected application flow:

```text
Upload Image
     ↓
YOLO11n Inference
     ↓
Person Bounding Boxes
     ↓
Person Count
     ↓
Confidence Summary
     ↓
Annotated Result
     ↓
JSON Result
```

---

# 16. Inference Defaults

Default inference settings:

```text
imgsz   = 640
conf    = 0.25
iou     = 0.50
max_det = 400
```

The application allows the user to modify:

```text
confidence threshold
IoU threshold
```

Any UI changes must preserve clear visibility of these settings.

---

# 17. Count Interpretation

The application count is calculated using:

```text
number of person detections
```

Conceptually:

```python
person_count = len(person_detections)
```

This is not a density-map integral.

This is not a tracking-based unique visitor count.

This is not an exact ground-truth occupancy value.

---

# 18. Confidence Interpretation

Detection confidence refers to:

```text
model confidence for an individual person detection
```

Do NOT describe confidence as:

```text
crowd density percentage
occupancy certainty
safety level
crowd risk score
```

unless a future validated system explicitly implements those concepts.

---

# 19. Saved Inference Outputs

New application inference results should be saved under:

```text
assets/results/
```

Recommended naming:

```text
crowd_result_YYYYMMDD_HHMMSS.jpg
crowd_result_YYYYMMDD_HHMMSS.json
```

Existing qualitative test predictions may use:

```text
*_pred.jpg
```

Both formats may remain.

---

# 20. JSON Result

Recommended JSON structure:

```json
{
  "timestamp": "...",
  "source_filename": "...",
  "model": "crowddensity_yolo11n_full_best.pt",
  "task": "person_detection_and_counting",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.5,
  "detected_person_count": 42,
  "average_detection_confidence": 0.61,
  "maximum_detection_confidence": 0.92,
  "detections": []
}
```

Future schema changes should remain backward understandable.

---

# 21. Failure Cases

Agents must not hide the current system's failure cases.

Important limitations include:

```text
severe occlusion
tiny persons
very dense crowds
low-resolution images
domain shift
threshold sensitivity
```

The detector may undercount when people are strongly overlapping.

The detector may also produce false detections depending on the image domain.

---

# 22. Privacy Rule

The project is designed around:

```text
anonymous person detection
```

Do NOT add:

```text
face recognition
identity recognition
name lookup
biometric identification
demographic inference
```

unless explicitly requested for a legitimate future research direction.

The default project direction should remain privacy-preserving.

---

# 23. Repository Structure

Expected repository structure:

```text
CrowdDensity-Mini/
│
├── assets/
│   └── results/
│
├── models/
│   └── crowddensity_yolo11n_full_best.pt
│
├── notebooks/
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

Do not reorganize the repository without a clear reason.

---

# 24. Reports

## `reports/final_summary.json`

Primary machine-readable experiment summary.

Agents should prefer this file when programmatically loading experiment metadata.

---

## `reports/manifest.csv`

Contains converted dataset metadata.

Do not delete without explicit reason.

---

## `reports/counting_demo_50.csv`

Contains:

```text
filename
ground_truth_count
predicted_count
absolute_error
```

This file is the evidence behind the reported:

```text
MAE = 5.34
```

---

# 25. README Rule

`README.md` is the main public project documentation.

Public statements must remain consistent with:

```text
reports/final_summary.json
training logs
model artifacts
reports/counting_demo_50.csv
```

Do not silently update README metrics without running a new experiment.

---

# 26. Metric Rounding

Preferred display:

```text
83.70%
69.68%
80.92%
48.90%
5.34 persons
```

Keep raw values only when detailed reproducibility is useful.

Do not round the same metric differently across different documentation files without reason.

---

# 27. Source of Truth Priority

When information conflicts, use this priority:

```text
1. Verified final experiment output
2. final_summary.json
3. reports generated by final run
4. final model metadata
5. README.md
6. AGENTS.md
7. older experiment notes
```

Never replace verified final results with earlier subset experiment metrics.

---

# 28. Previous Subset Experiment

An earlier prototype used a subset of CrowdHuman.

Those metrics are NOT the current final project results.

The final project must use:

```text
FULL DATA v4
```

results.

Agents should avoid reintroducing old subset metrics into README, app, or public posts.

---

# 29. Model Loading

Always load:

```text
models/crowddensity_yolo11n_full_best.pt
```

Preferred code:

```python
from ultralytics import YOLO

model = YOLO(
    "models/crowddensity_yolo11n_full_best.pt"
)
```

Do not silently replace the trained model with:

```text
yolo11n.pt
```

during normal application inference.

The generic pretrained model may only be used when intentionally starting a new experiment.

---

# 30. Application Error Handling

The Streamlit application should:

- verify model existence,
- validate uploaded images,
- handle corrupted files,
- show readable error messages,
- avoid raw crashes where practical.

Do not silently ignore errors.

---

# 31. Path Handling

Prefer:

```python
from pathlib import Path
```

Example:

```python
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crowddensity_yolo11n_full_best.pt"
```

Never hardcode developer-local paths such as:

```text
D:\porto\100 day\day 7\
```

inside production application code.

Kaggle-specific paths are acceptable inside training notebooks.

---

# 32. Dependency Policy

Keep runtime dependencies minimal.

Expected runtime requirements:

```text
streamlit
ultralytics
opencv-python-headless
pillow
numpy
```

Training-only dependencies may exist inside notebooks.

Do not introduce:

```text
TensorFlow
Detectron2
OpenMMLab
```

or another large framework unless it provides a clear project benefit.

---

# 33. Python Style

Prefer:

- small functions,
- descriptive names,
- clear comments,
- explicit paths,
- readable errors,
- minimal duplication.

Avoid:

- unnecessary abstraction,
- giant monolithic functions,
- deeply nested logic,
- unexplained magic constants,
- unused imports,
- silent exceptions.

---

# 34. Jupyter / Kaggle Logging

Long-running notebook stages must print progress.

Recommended format:

```text
[BOOT] ...
[1/10] ...
[2/10] ...
[8/10] FULL DATA TRAINING START
[9/10] ...
[10/10] ...
[DONE] ...
```

The notebook intentionally uses heartbeat output because Kaggle sessions have previously shown:

```text
Running
Output 0 B
empty logs
```

without actual execution progress.

Do not remove heartbeat logs without a strong reason.

---

# 35. Dataset Conversion

CrowdHuman source annotations use ODGT files.

The final pipeline uses:

```text
fbox
```

person boxes.

Ignored annotations should remain excluded.

YOLO coordinates must remain normalized:

```text
x_center
y_center
width
height
```

all relative to image dimensions.

---

# 36. Data Integrity

Final expected records:

```text
Train records : 15,000
Val records   : 4,370
```

Training conversion should fail loudly if these are unexpectedly incomplete.

Do not silently continue with missing data unless explicitly running a diagnostic experiment.

---

# 37. Do Not Commit Raw Dataset

Never commit the full CrowdHuman dataset into this repository.

The repository should contain:

```text
model
reports
plots
notebook
demo outputs
application code
documentation
```

but not the raw multi-gigabyte dataset.

---

# 38. Gitignore

The `.gitignore` should exclude:

```text
dataset/
datasets/
data/

*.zip
*.tar
*.tar.gz
*.7z

.venv/
venv/

__pycache__/
*.pyc

.env
.env.*

runs/
tmp/
temp/

.ipynb_checkpoints/
```

Do not ignore the final model unless GitHub size constraints require it.

Current final model is small enough for normal repository storage.

---

# 39. Secrets

Never commit:

```text
Kaggle credentials
API keys
GitHub tokens
Hugging Face tokens
passwords
access tokens
.env
```

Use environment variables where needed.

---

# 40. Git Commit Style

Prefer descriptive conventional commits.

Examples:

```text
feat: add Streamlit crowd inference app
```

```text
feat: add full-data CrowdHuman model
```

```text
docs: add complete CrowdDensity documentation
```

```text
fix: correct person counting logic
```

```text
refactor: simplify inference result handling
```

```text
chore: organize training artifacts
```

Avoid vague messages such as:

```text
update
code
test
stuff
fix
```

for future commits.

---

# 41. Public Branding

Public project label:

```text
7%
```

Do NOT publicly use:

```text
Day 7/100
7/100 Challenge
100 Days Challenge
Day 7
```

The percentage represents:

```text
personal learning / growth progress
```

not:

```text
model accuracy
project completion
literal project count
```

---

# 42. Public Post Structure

Preferred public communication follows:

```text
Hook
Problem
MVP
Model Choice
Evidence
Lesson
Limitation
CTA
```

Always include at least one honest limitation.

Do not publish metrics without context.

---

# 43. Public Claim Example

Good:

```text
YOLO11n achieved 83.70% precision,
69.68% recall, 80.92% mAP50,
and 48.90% mAP50-95 on the
CrowdHuman validation split.
```

Good:

```text
A 50-image counting demo produced
an MAE of 5.34 persons.
```

Bad:

```text
The system counts crowds with 83.70% accuracy.
```

Bad:

```text
The system accurately knows how many people are present.
```

Bad:

```text
CrowdDensity Mini is a production-ready crowd monitoring system.
```

---

# 44. Density Terminology

The repository name contains:

```text
CrowdDensity
```

but the current system does NOT estimate a density map.

Therefore always clarify:

```text
Current implementation:
person detection + detection-based counting
```

Potential future implementation:

```text
density-map estimation
```

Do not allow the project name alone to create a false technical claim.

---

# 45. Potential Future Models

Future experiments MAY explore:

```text
CSRNet
MCNN
Bayesian crowd counting
density-map regression
YOLO + tracking
ByteTrack
BoT-SORT
```

These are roadmap ideas only.

Do not describe them as currently implemented.

---

# 46. Tracking

Current version:

```text
no temporal tracking
```

If tracking is added later:

```text
person detection
↓
track ID
↓
temporal count
```

must be evaluated separately.

Do not assume detection count and unique-person count are equivalent.

---

# 47. Zone-Based Counting

Potential future use:

```text
camera frame
↓
person detector
↓
ROI polygon
↓
people inside zone
↓
occupancy
```

This is NOT currently part of the baseline unless implemented in code.

---

# 48. Density Map Roadmap

Potential future crowd-density architecture:

```text
Input Image
     ↓
Feature Extractor
     ↓
Density Regression
     ↓
Density Map
     ↓
Integral of Density Map
     ↓
Estimated Crowd Count
```

This would represent a fundamentally different task from the current YOLO detector.

Do not mix evaluation metrics between the two methods.

---

# 49. Human Review

Any future operational version should support:

```text
AI detection
↓
occupancy estimate
↓
warning / recommendation
↓
human operator review
```

AI output should assist operational decisions rather than replace human judgment in safety-critical contexts.

---

# 50. Privacy-Preserving Direction

Preferred long-term direction:

```text
anonymous occupancy understanding
```

rather than:

```text
individual identification
```

If new features are proposed, consider whether they are necessary for the actual problem.

Avoid unnecessary identity-related processing.

---

# 51. Performance Claims

Measured validation inference speed exists in the training log.

However, do not publish latency or FPS as production performance unless:

- hardware is specified,
- preprocessing is included appropriately,
- benchmarking methodology is documented.

A single Ultralytics validation timing line is not enough to claim real-world FPS.

---

# 52. Never Fabricate Results

Agents must NEVER invent:

```text
accuracy
precision
recall
mAP
MAE
FPS
latency
dataset size
test metrics
benchmark rankings
improvements
```

If something has not been measured, write:

```text
Not measured yet.
```

---

# 53. Mandatory Validation After App Changes

After modifying `app.py`, verify:

```text
1. Model file loads successfully.
2. Image upload works.
3. YOLO inference completes.
4. Only person detections are counted.
5. Bounding boxes render correctly.
6. Person count is displayed.
7. Threshold controls work.
8. Result image saves correctly.
9. JSON result saves correctly.
10. No density-map claim is introduced.
```

---

# 54. Mandatory Validation After Model Changes

If the model is replaced:

verify:

```text
class names
image size
model task
metrics
checkpoint path
app compatibility
README metrics
AGENTS metrics
```

Never replace the model without updating documentation.

---

# 55. README and AGENTS Consistency

After changing final experiment results, update:

```text
README.md
AGENTS.md
reports/final_summary.json
app sidebar metrics
```

where applicable.

All public values should remain consistent.

---

# 56. Current Verified Final State

```text
Project              : CrowdDensity Mini
Progress             : 7%
Status               : MVP / Experimental Baseline

Task                 : Person Detection + Crowd Counting
Dataset              : CrowdHuman
Model                : YOLO11n

Train Images         : 15,000
Validation Images    : 4,370
Validation Instances : 99,481

Epochs               : 15
Image Size           : 640
Batch                : 16

Precision            : 83.70%
Recall               : 69.68%
mAP50                : 80.92%
mAP50-95             : 48.90%

Counting Demo Images : 50
Counting Demo MAE    : 5.34 persons
```

---

# 57. Current Model Path

```text
models/crowddensity_yolo11n_full_best.pt
```

This is the current model expected by the application.

---

# 58. Current Output Directories

```text
assets/results/
reports/
training_plots/
```

Keep their responsibilities separate.

```text
assets/results/
```

for inference visuals and JSON.

```text
reports/
```

for experiment data and summaries.

```text
training_plots/
```

for model-training visualization.

---

# 59. Engineering Philosophy

This repository prioritizes:

```text
real data
verified metrics
working code
clear limitations
reproducibility
simple deployment
honest documentation
```

over:

```text
inflated claims
unverified benchmarks
buzzwords
unnecessary complexity
```

---

# 60. Agent Decision Rule

Before making a change, ask:

```text
Does this make the system more correct,
more useful,
more maintainable,
more reproducible,
or easier to evaluate?
```

If not, avoid unnecessary complexity.

---

# 61. Final Reminder

CrowdDensity Mini currently is:

> **A YOLO11n person detection and detection-based crowd counting MVP trained on the full annotated CrowdHuman train and validation splits used in this project.**

It is not currently:

> **A validated density-map estimator or production occupancy monitoring system.**

Every future contribution must preserve that distinction until a future implementation and evaluation genuinely support stronger claims.