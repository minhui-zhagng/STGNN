# STGNN

Source code for bearing health-state classification with six edge-feature variants.
The original XJTU-SY vibration dataset is not included. The current package contains 15 bearing label arrays, six sets of train/test graphs, and six trained models.
Evaluation results are generated on demand.

## Associated manuscript

**Title:** Modeling Full-Lifecycle Fault Evolution via State Transition Graph Neural Networks for Bearing State Identification

**Authors:** Diwang Ruan, Yifeng Shao, Minhui Zhang, Jianping Yan, and Clemens Gühmann.

This repository accompanies the manuscript listed above. Publication details and the article DOI will be added when available.

## Files

```text
STGNN_publication/
├── README.md
├── requirements.txt
├── config.json
├── code/
│   ├── common.py       # Arguments, configuration and paths
│   ├── label.py        # Generate bearing labels
│   ├── graph.py        # Extract features and build graphs
│   ├── model.py        # STGNN network
│   ├── train.py        # Train and save weights/configuration
│   └── evaluate.py     # Predict, compute metrics and draw confusion matrix
└── outputs/
    ├── labels/         # label_<condition>_<bearing>.npy
    ├── graphs/<mode>/  # train.pt, test.pt, config.json
    └── training/<mode>/
        ├── model.pt
        ├── config.json
        └── evaluation/ # Created by evaluate.py
```

`graph.py` also writes `markov.npy` in each graph-mode folder. This is a graph construction output; training and evaluation read the saved graphs directly.
`train.py` saves `model.pt` and `config.json`;  Each mode has its own model, shared by all test samples.

## Environment

The current models were trained with Python 3.11.7, PyTorch 2.7.1+cu128, PyTorch Geometric 2.6.1 and an NVIDIA RTX 5070. 
`requirements.txt` pins the six direct dependencies to the versions installed in that environment. 
From the package directory:

```bash
python -m pip install -r requirements.txt
```

For GPU training, use a CUDA-enabled PyTorch 2.7.1 build compatible with the local driver. 
The requirements use `torch==2.7.1` to allow CPU or CUDA builds of that release. Training selects CUDA when available.

## Dataset and configuration

### Raw data availability and download

The original XJTU-SY vibration dataset is **not included** in this repository. The included bearing-state label arrays, preconstructed graphs, and pretrained models are separate from the original vibration recordings.

Obtain the raw data from the dataset providers through either of these official sources:

- [XJTU-SY dataset website](https://biaowang.tech/xjtu-sy-bearing-datasets/).
- [Dataset providers' GitHub repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets), whose README lists download options, including Google Drive and Baidu Netdisk.

Follow the providers' download instructions, extract the data, and locate the directory containing `35Hz12kN`, `37.5Hz11kN`, and `40Hz10kN`. Set `data_root` to that directory as shown below. Raw vibration data are needed to regenerate labels or rebuild graphs; evaluation with the included graphs and pretrained models does not require the raw recordings.

Please acknowledge the original dataset providers and cite their requested reference when using XJTU-SY data:

> Biao Wang, Yaguo Lei, Naipeng Li, and Ningbo Li, "A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings," IEEE Transactions on Reliability, vol. 69, no. 1, pp. 401-412, 2020. DOI: [10.1109/TR.2018.2882682](https://doi.org/10.1109/TR.2018.2882682).

### Local data configuration

Set `data_root` in the root `config.json` to your extracted XJTU-SY directory:

```text
<data_root>/35Hz12kN/Bearing1_1/1.csv
<data_root>/37.5Hz11kN/Bearing2_1/1.csv
<data_root>/40Hz10kN/Bearing3_1/1.csv
```

CSV files have a header and two vibration columns. Graph construction reads the horizontal signal (column index 0) from the CSV corresponding to each graph.
Label generation defaults to the vertical signal (column index 1).

The current bearing-level split is:

| Split | Bearings | Graphs |
|---|---|---:|
| Training | 1-1, 1-3, 2-1, 2-2, 2-3, 2-4, 3-1, 3-2, 3-4, 3-5 | 8171 |
| Test | 1-2, 2-5, 3-3, 1-4, 1-5 | 1045 |

 `output_root` is resolved relative to the selected configuration file. `data_root` should point to the dataset on your machine.



## Run

If using the included graphs and weights, run evaluation directly:

```bash
python code/evaluate.py --mode exponential_markov
```

To retrain from existing graphs:

```bash
python code/train.py --mode exponential_markov
```

To rebuild graphs from existing labels and the raw vibration dataset:

```bash
python code/graph.py --mode exponential_markov
```

`label.py` is a separate optional step that overwrites the label arrays:

```bash
python code/label.py
```

All scripts accept `--config <path>`. The default is the package's root `config.json`; the default mode is `exponential_markov`.
Graph construction, training and evaluation run one mode per invocation. Repeat them with each mode below as needed.

| Mode | Scalar edge attribute |
|---|---|
| `no_edge` | Constant 1 on every existing edge |
| `markov` | Training-label Markov prior |
| `linear` | Linear temporal progression |
| `exponential` | Exponential temporal progression |
| `linear_markov` | alpha × Markov + (1 − alpha) × linear |
| `exponential_markov` | alpha × Markov + (1 − alpha) × exponential |

## Evaluation outputs

`evaluate.py` predicts each test graph once. It writes to `outputs/training/<mode>/evaluation/`:

- `metrics.json`: Accuracy, Macro-F1, Balanced Accuracy, and class-wise precision, recall, F1 and support. Macro-F1 includes all six classes.
- `predictions.csv`: sequential sample index, true label and predicted label.
- `confusion_counts.csv`: a 6 × 6 count matrix, true classes in rows.
- `confusion_percent.png`: row-normalized confusion matrix with percentages.

State IDs: 0 normal, 1 inner-race fault, 2 outer-race fault, 3 cage fault, 4 compound fault and 5 complete failure. 

