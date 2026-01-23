# Supervised Deep Multimodal Matrix Factorization for Brain Network Analysis

This repository contains a PyTorch implementation of SD3MF.

The code is provided for **anonymous peer review and reproducibility evaluation**.

---

## Requirements

* Python 3.8 or later
* PyTorch == 2.9.0
* scikit-learn == 1.6.1

Install dependencies:

```bash
pip install torch scikit-learn
```

GPU is used automatically if available.

---

## Repository Structure

```
.
├── SD3MF.py          # Main training and evaluation script
├── README.md        # This file
└── <DATASET>.pt     # Dataset file (provided separately)
```

---

## Data Availability

The datasets used in this work are **medical research datasets** and cannot be redistributed by third parties due to ethical, legal, and IRB constraints.

In accordance with standard practice in the neuroimaging and medical machine learning community, we do **not** provide the raw data in this repository.

---

## Dataset Format

The script expects a PyTorch file named:

```
<DATASET_NAME>.pt
```

The file must contain a dictionary with the following fields:

```python
{
    "A_list": [A1, A2, ..., Av],   # list of tensors, each [N, d_v, d_v]
    "y": y                        # labels tensor, shape [N]
}
```

Where:

* `N` = number of samples
* `v` = number of modalities (views)
* `d_v` = number of nodes for view `v`
* Each `A_v[i]` is a square connectivity matrix for sample `i`
* `y` contains integer class labels

All views must have the same number of samples `N`.

---

## Running the Code

Basic usage:

```bash
python SD3MF.py --dataset DATASET_NAME
```

Example:

```bash
python SD3MF.py --dataset BP
```

---

## Command Line Arguments

| Argument    | Description                           | Default    |
| ----------- | ------------------------------------- | ---------- |
| `--dataset` | Dataset name (without `.pt`)          | `BP`       |
| `--epochs`  | Number of training epochs             | `30000`    |
| `--lr`      | Learning rate                         | `1e-5`     |
| `--mu`      | Reconstruction loss weight            | `1`        |
| `--widths`  | Hidden layer widths (comma-separated) | `30,20,10` |
| `--device`  | Training device (`cuda` or `cpu`)     | automatic  |

Custom example:

```bash
python SD3MF.py \
    --dataset BP \
    --epochs 30000 \
    --lr 1e-4 \
    --mu 1 \
    --widths 30,20,10
```

---


## Output

During training, the script prints:

* Total training loss
* Reconstruction loss
* Classification loss
* Training accuracy
* Test loss and accuracy
* AUC (only for binary classification)

Example log:

```
Epoch  100 | train loss ... | train acc ... | test acc ... | AUC ...
```

At the end:

```
FINAL TEST — ACC: XXXX | AUC: XXXX
```


