# LDFA-DETR: A Lightweight Dynamic Feature Aggregation Detector for Real-Time UAV Infrared Target Detection

This repository accompanies the manuscript **“LDFA-DETR: A Lightweight Dynamic Feature Aggregation Detector for Real-Time UAV Infrared Target Detection”**, submitted to *The Journal of Supercomputing*. It describes the LDFA-DETR implementation, trained checkpoints, and the training/evaluation workflows associated with the manuscript.

LDFA-DETR is built on **Ultralytics RT-DETR** and integrates three components:

- **LDFE-Net** — a lightweight feature-extraction backbone using dynamic multi-shape depthwise convolutions.
- **ASTE** — an adaptive sparse Transformer enhancement module for intra-scale feature refinement, incorporating existing mechanisms such as ASSA, DyT, Mona, and SEFN.
- **GPMF** — gated parallel multi-scale fusion for cross-scale interaction, including multi-dilation processing and channel-gated up-/downsampling.

The manuscript evaluates LDFA-DETR primarily on the **HIT-UAV infrared** dataset and additionally on the **VisDrone2019 visible-light** detection dataset. The models for these two datasets are trained and evaluated **separately**; the VisDrone2019 evaluation is **not** a zero-shot infrared-to-visible transfer experiment.

## 1. Environment and installation

### Reported experimental environment

| Item | Configuration used in the manuscript |
| --- | --- |
| Operating system | Windows 11 |
| CPU | Intel Core i5-14600KF |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| Python | 3.9.25 |
| PyTorch | 2.8.0 |
| CUDA environment | 12.9 (as reported; check the CUDA version of the installed PyTorch build separately) |

The reported measurements were obtained on the desktop GPU above. **A minimum GPU-memory requirement and onboard UAV performance have not been established.**

### Installation

```bash
# Replace the placeholder with the URL of your public GitHub repository.
git clone https://github.com/USERNAME/REPOSITORY.git LDFA-DETR
cd LDFA-DETR

# Create and activate a suitable Python environment, then:
pip install -r requirements.txt
```

**Important:** LDFA-DETR uses custom Ultralytics modules. If this repository contains a modified `ultralytics/` package, make sure Python imports that local version, not an unrelated installed release. If the repository provides packaging metadata, an editable installation may be used:

```bash
pip install -e .
python -c "import ultralytics; print(ultralytics.__file__)"
```

The `pip install -e .` step applies **only if** the repository contains the required packaging files. Record the exact Ultralytics commit/version and dependency versions used for the released experiments in `requirements.txt` or a separate environment file.

## 2. Datasets

**Dataset images and annotations are not redistributed in this repository.** Download them from their original sources and convert annotations to the format expected by the training scripts.

| Dataset | Modality | Classes | Training | Validation | Test |
| --- | --- | ---: | ---: | ---: | ---: |
| HIT-UAV | Infrared thermal | 5 | 2,029 | 290 | 579 |
| VisDrone2019-DET (hereafter VisDrone2019) | Visible light | 10 | 6,471 | 548 | 1,610 (test-dev) |

### HIT-UAV

- Dataset publication: https://doi.org/10.1038/s41597-023-02066-6
- Dataset archive: https://zenodo.org/records/7633134
- Repository configuration: `hit-uav/dataset.yaml`
- Five categories used in the manuscript: **Person, Car, Bicycle, OtherVehicle, DontCare**. Keep the **class IDs and ordering** in the released YAML consistent with the checkpoint and the paper's evaluation.
- Reproduce the paper's **2,029/290/579** train/validation/test split. If this split was generated locally, publish the split file lists or the exact procedure and seed used to produce it.

### VisDrone2019

- Official project: https://github.com/VisDrone/VisDrone-Dataset
- Dataset documentation: https://docs.ultralytics.com/datasets/detect/visdrone/
- Repository configuration: `visdrone2019/dataset.yaml`
- Ten detection categories; use the **VisDrone2019-DET image dataset**, not the video-tracking subsets.
- The **1,610-image test split is test-dev**. Record the label conversion and handling of ignored regions so that the evaluation protocol can be reproduced.

### Directory layout

A conventional Ultralytics detection layout is shown below. **Follow the `train`, `val`, and `test` paths in each dataset YAML if your repository uses a different layout.**

```text
hit-uav/
   dataset.yaml
   images/{train,val,test}/
   labels/{train,val,test}/
visdrone2019/
   dataset.yaml
   images/{train,val,test}/
   labels/{train,val,test}/
```

Edit the `path` field in each dataset YAML to match your local storage location. Ensure the category mappings, annotation conversion, image paths, and evaluation split match those used for the reported results.

## 3. Model configurations and checkpoints

Model configuration paths in this release are listed relative to `ultralytics/cfg/models/rt-detr/`.

| Model | Configuration | Role in the manuscript |
| --- | --- | --- |
| RT-DETR-R18 | `rtdetr-r18.yaml` | Main baseline |
| RT-DETR-R34 | `rtdetr-r34.yaml` | Additional comparison |
| **LDFA-DETR** | **`LDFA-DETR.yaml`** | Proposed complete model: LDFE-Net + ASTE + GPMF |

### Trained checkpoints

The following checkpoint filenames are used in this README. **Confirm the files exist and are downloadable before publication.**

| File | Training dataset | Paper's reported evaluation |
| --- | --- | --- |
| `weights/best_hit_uav.pt` | HIT-UAV | mAP50 **83.50%**, mAP50:95 **53.60%** |
| `weights/best_visdrone.pt` | VisDrone2019 | mAP50 **38.24%**, mAP50:95 **22.04%** |



## 4. Training and evaluation

### Paper training settings

The following settings are reported in Section 4.2 of the manuscript. Verify that the released scripts use the same settings; an unspecified script default should not be assumed to match the paper.

| Setting | Value |
| --- | --- |
| Input image size | 640 × 640 |
| Training epochs | 200 |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.0001 |
| Weight decay | 0.0005 |
| Learning-rate scheduling | `cos_lr=False`, `lrf=1.0` |
| Color augmentation | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` |
| Geometric augmentation | `translate=0.1`, `scale=0.5`, `fliplr=0.5` |
| Mosaic / MixUp | Disabled |
| Random seed | `0` |
| Deterministic setting | Enabled in the reported setup |
| Automatic mixed precision | Disabled |

### Training

The repository's intended entry point is:

```bash
python train.py
```

Configure `train.py` to select the required model YAML and dataset YAML (`hit-uav/dataset.yaml` or `visdrone2019/dataset.yaml`) and to match the settings above. Check the script defaults against the paper settings before running. If using an Ultralytics CLI command instead, first confirm that the custom modules are registered in the active local package.

### Validation

```bash
python val.py
```

Select the **matching checkpoint and dataset** in `val.py`: `weights/best_hit_uav.pt` with HIT-UAV, or `weights/best_visdrone.pt` with VisDrone2019. Run the paper's evaluation split rather than assuming that a script's default `val` split is equivalent to the test/test-dev split.

The main paper metrics are **Precision, Recall, mAP50, mAP50:95, parameter count, FLOPs, and FPS**. The HIT-UAV analysis additionally reports **per-class AP50** and COCO-style **small-, medium-, and large-object AP**. For the size-specific AP values, the manuscript uses IoU thresholds **0.50:0.05:0.95** and **a maximum of 100 detections per image**. Check the evaluation script or COCO-format export procedure for support before claiming these outputs are automatically generated by `val.py`.

## 5. Results reported in the manuscript

| Dataset | Model | Precision (%) | Recall (%) | mAP50 (%) | mAP50:95 (%) | Params (M) | FLOPs (G) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIT-UAV | RT-DETR-R18 | 86.93 | 73.81 | 79.13 | 51.50 | 19.87 | 57.0 |
| HIT-UAV | **LDFA-DETR** | **86.36** | **77.57** | **83.50** | **53.60** | **15.58** | **46.2** |
| VisDrone2019 | RT-DETR-R18 | 54.73 | 39.02 | 37.36 | 21.60 | 19.87 | 57.0 |
| VisDrone2019 | **LDFA-DETR** | **55.44** | **39.69** | **38.24** | **22.04** | **15.60** | **46.2** |

The manuscript reports **53.56 FPS** for LDFA-DETR and **68.35 FPS** for RT-DETR-R18 on HIT-UAV using its desktop test environment. Different runtime, library versions, device settings, or evaluation protocols may change the observed values. **Peak memory usage, device power consumption, and actual onboard/flight performance were not evaluated in the manuscript.**

These are **reported paper results**, not a claim that this README or the repository has been independently validated to reproduce every number.

## 6. Acknowledgments and licensing

This work builds on the [Ultralytics](https://github.com/ultralytics/ultralytics) implementation of RT-DETR and other appropriately cited components. We thank their authors and the open-source community.

Ultralytics distributes its code under **AGPL-3.0 or an Enterprise license**, depending on the use case. Check the applicable licensing and attribution obligations before distributing modified framework code or model weights. Include the appropriate license and notices in the actual release. See [Ultralytics licensing information](https://www.ultralytics.com/license).

## 7. Citation

If this repository supports your research, please cite the corresponding manuscript when an appropriate public citation becomes available:

> Liu H, Lu Y, Xu H. *LDFA-DETR: A Lightweight Dynamic Feature Aggregation Detector for Real-Time UAV Infrared Target Detection.* Manuscript submitted to *The Journal of Supercomputing*.

**A journal volume, pages, DOI, or formal BibTeX record will be added only after publication details are available.**\
