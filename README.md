# MLAD: Multi-system Log Anomaly Detection

This is an implementation of the MLAD (Multi-system Log Anomaly Detection) model as described in the paper "Towards Multi-System Log Anomaly Detection". MLAD is designed to detect anomalies in system logs across multiple systems by combining a Transformer with a Gaussian Mixture Model (GMM).

## Key Features

- **Multi-System Anomaly Detection:** Detects anomalies across multiple systems, overcoming limitations of traditional one-model-per-system methods.
- **Hybrid Transformer-GMM Architecture:** Integrates Transformers with GMMs, jointly learning semantic log representations while maintaining clear separation between normal and abnormal events.
- **Alpha-Entmax Attention:** Uses sparse attention mechanism to better identify important keywords in log sequences.
- **"Identical Shortcut" Problem Solver:** Mitigates the identical shortcut problem by transforming the vector space, effectively separating abnormal samples from normal ones.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MLAD.git
   cd MLAD
   ```

2. Create a virtual environment:
   ```bash
   conda create -n mlad python=3.8
   conda activate mlad
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Datasets

The implementation uses three public datasets:

1. **BGL:** Blue Gene/L supercomputer logs from Lawrence Livermore National Laboratory.
2. **HDFS:** Hadoop Distributed File System logs from Amazon EC2 nodes.
3. **Thunderbird:** System service messages from Sandia National Labs' Thunderbird supercomputer.

To obtain these datasets:
- BGL and Thunderbird: [USENIX CFDR Data](https://www.usenix.org/cfdr-data)
- HDFS: [LogHub on GitHub](https://github.com/logpai/loghub)

After downloading, place the log files in their respective directories:
```
data/
├── BGL/
│   └── BGL.log
├── HDFS/
│   └── HDFS.log
└── Thunderbird/
    └── Thunderbird.log
```

## Project Structure

```
MLAD/
├── data/                  # Datasets
├── models/                # Model implementations
│   ├── alpha_entmax.py    # Alpha-entmax implementation
│   ├── feed_forward.py    # Feed-forward network with CeLU
│   ├── gmm.py             # Gaussian Mixture Model
│   └── mlad.py            # Complete MLAD model
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading utilities
│   └── log_preprocessing.py # Log preprocessing functions
├── saved_models/          # Saved models directory
├── results/               # Evaluation results directory
├── main.py                # Main script to run the pipeline
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Usage

### Quick Start

To run the complete pipeline (train, evaluate, visualize) on all datasets:

```bash
python main.py --visualize
```

### Dataset Check

To check if datasets are available:

```bash
python main.py --download_only
```

### Training Only

To train on specific datasets:

```bash
python main.py --train_only --datasets BGL HDFS
```

### Evaluation Only

To evaluate on specific datasets (requires pre-trained models):

```bash
python main.py --eval_only --datasets BGL HDFS
```

### Transfer Learning

To run transfer learning experiments between BGL and Thunderbird:

```bash
python main.py --transfer_learning --datasets BGL Thunderbird
```

### Alpha Ablation Study

To run an ablation study on the alpha parameter:

```bash
python main.py --alpha_ablation --datasets BGL
```

### Model Parameters

Customize model parameters:

```bash
python main.py --d_model 100 --n_heads 4 --n_layers 2 --alpha 1.5 --n_components 5
```

### Training Parameters

Customize training parameters:

```bash
python main.py --batch_size 512 --lr 0.001 --epochs 30
```

## Results

The original paper reports the following F1 scores for MLAD across different datasets:

| Dataset     | Precision | Recall | F1 Score |
|-------------|-----------|--------|----------|
| BGL         | 0.9492    | 0.8932 | 0.9184   |
| HDFS        | 0.9296    | 0.8656 | 0.8946   |
| Thunderbird | 0.8824    | 0.9066 | 0.8962   |

Visualization examples will be saved in the `results/` directory when running with the `--visualize` flag.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@article{mlad2023,
  title={Towards Multi-System Log Anomaly Detection},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This implementation is based on the paper "Towards Multi-System Log Anomaly Detection"
- Sparse Sequence-to-Sequence Models (Peters et al., 2019) for alpha-entmax implementation 