# Edge-Vision Defect Inspection Pipeline



An end-to-end industrial optical inspection and metrology system implemented in Python, PyTorch, OpenCV, and ONNX Runtime, evaluated on the **MVTec AD** (`screw`) benchmark.



The pipeline performs unsupervised surface anomaly localization using multi-scale deep feature extraction (ResNet-18) combined with spatial neighborhood aggregation and nearest-neighbor retrieval. Detected anomalies are segmented and measured in physical units (mm²) via an optical post-processing module for automated **PASS/FAIL** quality control.



---



## Visual Results & Industrial Inspection



### 1. Optical Metrology & Quality Gating (Pass / Fail)

Automated defect detection comparing a nominal screw (PASS) against an anomalous unit with a damaged thread/head (FAIL):



![Industrial Inspection HUD](assets/metrology_inspection.png)



### 2. Multi-Scale Feature Anomaly Localization

Continuous anomaly heatmap generated via intermediate feature space deviation (layer2 + layer3) with local neighborhood pooling:



![Anomaly Heatmap Localization](assets/heatmap_localization.png)



---



## Key Performance Results

| Metric | Baseline (no pooling, 10 % memory bank) | Feature Aggregation (Optimized) | Target / Requirement |
| :--- | :---: | :---: | :---: |
| **AUROC (Overall)** | 81.49 % | **92.42 %** | > 90.0 % |
| **Precision** | 96.00 % | **98.55 %** | Low False Alarm Rate |
| **Defect Sensitivity (Recall)** | 40.34 % | **57.14 %** | High Defect Capture |
| **F1-Score** | 56.80 % | **72.34 %** | Balanced Performance |

Both columns use the same protocol (decision threshold = 95th percentile of 5-fold cross-validated nominal scores, never tuned on the test set) and the same 160 test images (41 nominal, 119 defective). The only differences are the 3x3 neighborhood pooling and the memory bank size (10 % vs 20 % of the patches). The baseline can be reproduced with `python evaluate.py --baseline`.

### Inference Latency Benchmark (Intel Core i9-13900HX, CPU)

Feature-extraction backbone only (ResNet-18 up to layer3), 224x224x3 frames. Median latency over 200 iterations, ranges observed over 4 runs (the laptop CPU frequency varies between runs, the relative gain stays stable):

| Runtime Engine | Latency / Frame | Throughput | Latency Reduction |
| :--- | :---: | :---: | :---: |
| **PyTorch Native (CPU)** | 9.3 – 15.3 ms | 65 – 108 FPS | Baseline |
| **ONNX Runtime (CPU)** | **3.5 – 5.1 ms** | **196 – 284 FPS** | **-62 to -70 %** |



---



## System Architecture



```text

Input Frame (224x224x3)

         │

         ▼

[Frozen ResNet-18 Backbone] ────► Multi-scale feature maps (layer2: 28x28, layer3: 14x14)

         │

         ▼

[Local Neighborhood Pooling] ───► 3x3 AvgPool2d spatial smoothing (reduces optical noise)

         │

         ▼

[Memory Bank Indexing] ─────────► Nearest-Neighbor distance query (k-NN, Euclidean)

         │

         ▼

[Raw Anomaly Heatmap] ──────────► Absolute metric thresholding (calibrated on held-out nominal parts)

         │

         ▼

[OpenCV Post-Processing] ───────► Connected Components Analysis (cv2.connectedComponentsWithStats)

         │

         ▼

[Metrology & Decision HUD] ─────► Defect bounding box, real area (mm²), PASS/FAIL classification

```



---



## Repository Structure



```text

defect-vision-screw/

├── .github/workflows/ci.yml  # CI: runs the unit tests on every push and pull request

├── assets/                   # Visual inspection results and screenshots

│   ├── heatmap_localization.png

│   └── metrology_inspection.png

├── data/

│   └── screw/                # MVTec AD dataset partition (train/good, test splits)

├── models/

│   ├── backbone.onnx         # Optimized static ONNX computational graph (backbone only)

│   └── threshold.json        # Decision threshold written by evaluate.py (not versioned)

├── src/

│   ├── data_loader.py        # PyTorch Dataset and DataLoader with ImageNet transforms

│   ├── feature_extractor.py  # Frozen ResNet-18 intermediate extractor with AvgPool2d

│   ├── anomaly_detector.py   # Multi-scale patch alignment and k-NN memory bank

│   └── inspector.py          # Metrology module, OpenCV bounding boxes, and Pass/Fail gating

├── tests/                    # Unit tests (pytest) on synthetic data, no dataset needed

│   ├── test_data_loader.py   # Label parsing and 5-fold split (disjoint, full coverage)

│   ├── test_detector.py      # Feature fusion shapes (128 + 256 -> 384 channels)

│   └── test_inspector.py     # PASS/FAIL decision and pixel -> mm² conversion

├── setup_data.py             # Automated dataset ingestion from public mirrors

├── explore_data.py           # Optical frame inspection visualizer

├── evaluate.py               # Benchmark script (AUROC, Precision, Recall, Confusion Matrix)

├── visualize_heatmap.py      # Spatial anomaly heatmap visualizer (Jet overlay)

├── run_inspection.py         # Full quality control execution with metric HUD

├── export_onnx.py            # Static ONNX graph export (Opset 18)

├── benchmark_latency.py      # Profiling script comparing PyTorch CPU vs ONNX Runtime

├── requirements.txt          # Pinned runtime dependencies

├── requirements-dev.txt      # Runtime dependencies + pytest

└── pytest.ini                # pytest configuration

```



---



## Getting Started



### 1. Environment Setup

Tested with Python 3.13. To also run the unit tests, install `requirements-dev.txt` instead and run `pytest`.



```powershell

git clone https://github.com/victor-saunier/defect-vision-screw.git

cd defect-vision-screw

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

```



### 2. Dataset Ingestion



```powershell

python setup_data.py

```



### 3. Pipeline Evaluation



```powershell

python evaluate.py

python evaluate.py --baseline   # optional: initial version, for comparison

```



### 4. Industrial Metrology & Inspection

Requires step 3 first: `evaluate.py` calibrates the decision threshold and saves it to `models/threshold.json`.



```powershell

python run_inspection.py

```



### 5. ONNX Export & Latency Benchmark



```powershell

python export_onnx.py

python benchmark_latency.py

```



---



## Engineering Highlights



* **Unsupervised Anomaly Modeling:** Solves manufacturing class imbalance by constructing a nominal geometric memory bank exclusively from 320 non-defective parts (`train/good`).

* **Noise Rejection via Spatial Pooling:** Mitigates high-frequency surface noise and specular reflection by applying local neighborhood pooling (`AvgPool2d`, kernel=3) on intermediate feature representations.

* **Calibrated Physical Metrology:** Converts segmented pixel clusters into physical surface measurements (mm²) using an optical calibration factor (50 µm / pixel), with the threshold calibrated (95th percentile) with 5-fold cross-validation on the 320 nominal training parts, never on the test set, and areas rescaled to the native 1024 px resolution.

* **Edge-Optimized Runtime:** Eliminates autograd overhead and leverages AVX/SIMD vectorization via ONNX Runtime, achieving a backbone throughput of roughly **200 – 280 FPS** on a laptop CPU (the k-NN search is not included in this measure).
