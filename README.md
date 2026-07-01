<h1 align="center">
  DAPWH
</h1>

<h3 align="center">
  Descriptor: Parasitoid Wasps and Associated Hymenoptera Dataset (DAPWH)
</h3>

<p align="center">
  <strong>João Manoel Herrera Pinheiro</strong><sup>1</sup> &middot;
  <strong>Gabriela do Nascimento Herrera</strong><sup>2</sup> &middot;
  <strong>Luciana Bueno dos Reis Fernandes</strong><sup>2</sup> &middot;
  <strong>Alvaro Doria dos Santos</strong><sup>3</sup> &middot;
  <strong>Ricardo V. Godoy</strong><sup>1</sup><br>
  <strong>Eduardo A. B. Almeida</strong><sup>4</sup> &middot;
  <strong>Helena Carolina Onody</strong><sup>5</sup> &middot;
  <strong>Marcelo Andrade da Costa Vieira</strong><sup>1</sup> &middot;
  <strong>Angelica Maria Penteado-Dias</strong><sup>2</sup> &middot;
  <strong>Marcelo Becker</strong><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup> São Carlos School of Engineering, University of São Paulo (USP), São Carlos, Brazil<br>
  <sup>2</sup> Department of Ecology and Evolutionary Biology, Federal University of São Carlos (UFSCar), São Carlos, Brazil<br>
  <sup>3</sup> Federal University of Tocantins (UFT), Porto Nacional, Brazil<br>
  <sup>4</sup> Department of Biology (FFCLRP), University of São Paulo (USP), Ribeirão Preto, Brazil<br>
  <sup>5</sup> Deputado Jesualdo Cavalcanti Campus, State University of Piauí (UESPI), Corrente, Brazil
</p>

<p align="center">
  <em>IEEE Data Descriptions, 2026</em>
</p>

<p align="center">
  <a href="https://doi.org/10.1109/IEEEDATA.2026.3683381">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="Paper">
  </a>&nbsp;
  <a href="https://arxiv.org/abs/2602.20028">
    <img src="https://img.shields.io/badge/arXiv-2606.31941-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>&nbsp;
  <a href="https://zenodo.org/records/18501018">
    <img src="https://img.shields.io/badge/Dataset-Available-2ea44f?style=flat-square&logo=databricks&logoColor=white" alt="Dataset">
  </a>
</p>

![Ichneumonidae Detection Example](00134_yolov12.png)
*An Ichneumonidae wasp detected.*

![Braconidae Detection Example](604_yolov12.png)
*A Braconidae wasp detected.*

This repository contains the data processing pipelines and training workflows developed for the DAPWH dataset. It includes scripts for preprocessing, dataset split, model training, and evaluation of classification and detection models.
## Dataset Availability
Comprising the families Ichneumonidae and Braconidae, these parasitoid wasps are ecologically critical for the regulation of insect populations, yet they remain one of the most taxonomically challenging groups due to their cryptic morphology and vast number of undescribed species. To address the scarcity of robust digital resources for these key groups, we present a curated image dataset designed to advance automated identification systems. The dataset contains 3,556 high-resolution images, primarily focused on Neotropical Ichneumonidae and Braconidae, while also including supplementary families such as Andrenidae, Apidae, Bethylidae, Chrysididae, Colletidae, Halictidae, Megachilidae, Pompilidae, and Vespidae to improve model robustness. Crucially, a subset of 1,739 images is annotated in COCO format, featuring multi-class bounding boxes for the full insect body, wing venation, and scale bars. This resource provides a foundation for developing computer vision models capable of identify this families.

The DAPWH dataset is publicly available on Zenodo:

- [https://zenodo.org/records/18501018](https://zenodo.org/records/18501018)
- DOI: [https://doi.org/10.5281/zenodo.18501017](https://doi.org/10.5281/zenodo.18501017)

The paper citation can be found in:

[Dataset of Parasitoid Wasps and Associated Hymenoptera (DAPWH)](https://doi.org/10.1109/IEEEDATA.2026.3683381)

```latex
@ARTICLE{11483097,
  author={Pinheiro, Joao Manoel Herrera and Do Nascimento Herrera, Gabriela and Fernandes, Luciana Bueno Dos Reis and Santos, Alvaro Doria Dos and Godoy, Ricardo V. and Almeida, Eduardo A. B. and Onody, Helena Carolina and Vieira, Marcelo Andrade Da Costa and Penteado-Dias, Angelica Maria and Becker, Marcelo},
  journal={IEEE Data Descriptions}, 
  title={Descriptor: Parasitoid Wasps and Associated Hymenoptera Dataset (DAPWH)}, 
  year={2026},
  volume={3},
  number={},
  pages={265-274},
  keywords={Antennas;Electronic mail;Pixel;Local area networks;Location awareness;Presence network agents;Protocols;Communication systems;Message systems;Mobile communication;Arthropod;biodiversity;convolutional neural network;taxonomy},
  doi={10.1109/IEEEDATA.2026.3683381}}
```

## Scripts

This repository includes the following Python scripts to support dataset preparation and model training workflows:

* **`cls_split_data.py`**
  Splits the classification dataset into training, validation, and test subsets.

* **`cls_train_models.py`**
  Trains state-of-the-art image classification models.

* **`scripts/split_data.py`**
  Splits the COCO-format detection subset into training, validation, and test sets.

* **`scripts/coco2yolo.py`**
  Converts COCO-style annotations into YOLO-format annotations (bounding boxes or segmentation masks).

## Classification

This section describes how to replicate the classification benchmarks for the **DAWPDset**. We evaluated several state-of-the-art architectures, including **EfficientNetV2-S**, **YOLOv12s-cls**, **ConvNeXt-Tiny**, **ResNet-50**, and **VGG16**.

### Prerequisites

1. **Download the Dataset**: Ensure you have downloaded the **DAPWH** folder.
2. **Environment**: A Python environment with `torch`, `torchvision`, and `ultralytics` installed is recommended. If required, download the pretrained weight file (`yolov12s-cls.pt`) from the official YOLO website before training.
3. **Hardware**: Benchmarks were performed on a workstation equipped with an **NVIDIA RTX 4090**.

### 1. Data Preparation (`cls_split_data.py`)

Before training, you must split the raw images (organized by **Dorsal_Ventral**, **Frontal**, and **Lateral** views) into training, validation, and test sets.

* **Action**: Open `cls_split_data.py` and update the `dataset_folder` variable to point to your local path, exemple: `/Documents/DAPWH`.
* **Function**: This script ensures that each family (e.g., **Ichneumonidae**, **Braconidae**) is balanced across the splits to maintain taxonomic integrity.

###  Model Training (`cls_train_models.py`)

This script handles the training for all architectures

* **Usage**: Update the `MODEL_LIST` within the script to select which architectures to train:
```python
MODEL_LIST = [
    ('efficientnet_v2_s', 'torch', 'DEFAULT'),
    ('convnext_tiny',     'torch', 'DEFAULT'),
    ('vit_b_16',          'torch', 'DEFAULT'),
    ('vgg16',             'torch', 'DEFAULT'),
    ('resnet50',          'torch', 'DEFAULT'),
    ('yolov12s-cls.pt',    'yolo',  None)
]
```

### Directory Structure Requirement

To ensure the scripts run correctly, maintain the following structure:

```text
DAPWH/
├── Dorsal_Ventral/
│   ├── Ichneumonidae/
│   └── Braconidae/
├── Frontal/
│   └── ...
└── Lateral/
    └── ...
```

## Detection

This section outlines the workflow for object detection within the **DAWPDset**. This process involves preparing the COCO-formatted annotations for YOLO training and benchmarking the **YOLOv8** and **YOLOv12** architectures.

### 1. Dataset Splitting

To ensure a robust evaluation, the dataset is divided into training, validation, and test sets using the `split_data.py` script.

* **Configuration**: Set the following paths in your environment, for example:
* `IMAGES_DIR = 'Documents/DAPWH/COCO_subset/images/default'`
* `JSON_FILE = 'Documents/DAPWH/COCO_subset/annotations/instances_default.json'`
* `OUTPUT_DIR = 'Documents/DAPWH_split'`

* **Execution**: Run the splitting script with a 70/15/15 ratio:
```bash
!python scripts/split_data.py $IMAGES_DIR $JSON_FILE $OUTPUT_DIR --train_ratio 0.7 --val_ratio 0.15

```

### 2. Format Conversion (COCO to YOLO)

Since the **DAPWH** dataset uses COCO annotations by default, they must be converted to the YOLO `.txt` format for detection training.

* **Execution**: Use the `coco2yolo.py` script in detection mode:
```bash
path_detection_dataset = 'Documents/DAPWH_split'
!python scripts/coco2yolo.py $path_detection_dataset --mode detection
```

### 3. Model Training
We benchmarked two generations of YOLO models YOLOv8 and YOLOv12

#### YOLOv8
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.train(
    data=path_detection_train,
    epochs=150,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    name="yolov8_detection",
)
```
#### YOLOv12
```python

from ultralytics import YOLO
model = YOLO("yolov12s.pt")

results = model.train(
    data=path_detection_train,
    epochs=150,
    imgsz=1280,       
    batch=4,         
    device=0,
    workers=8,
    name="yolov12_detection",
)
```
