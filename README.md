# Adaptive Few-Shot Meta-Learning with Drift Detection and Task-Aware Optimization

This repository contains the official implementation of our paper:

> **"Adaptive Few-Shot Meta-Learning with Drift Detection and Task-Aware Optimization"**  
> Yiming Yuan

## 🎯 Overview

This project implements a novel continual few-shot learning framework that adaptively handles distributional drift through:

- **Transformer-based Task Embedding**: Captures task characteristics using global statistics, class prototypes, and contextual features
- **Hybrid Drift Detection**: Combines KL divergence, mean shift, and MLP classifier for robust drift detection
- **Adaptive Controller**: Dynamically adjusts learning rates and regularization strength based on task complexity and drift signals
- **Conditional Optimization**: Routes between ProtoOnly and MetaSGD+EWC paths based on drift detection

## 📁 Project Structure

```
Design/
├── 📄 main_ablation.py          # Main training script with ablation studies
├── 📄 models.py                 # Task embedding composer + ProtoNet learner
├── 📄 meta_sgd_module.py        # MetaSGD with per-parameter learning rates + EWC
├── 📄 drift_detector.py         # Hybrid drift detection (KL + Mean Shift + MLP)
├── 📄 adaptive_controller.py    # Task-aware controller for η, λ adjustment
├── 📄 hypernet.py              # Gated hypernetwork for hyperparameter modulation
├── 📄 drift_detection.py        # Task memory and replay buffer
├── 📄 task_loader.py            # Few-shot task sampler for Mini-ImageNet
├── 📄 task_stream_simulator.py  # Simulates task stream with controlled drift
├── 📄 visualization.py          # Task embedding visualization and drift plots
├── 📄 experiment_utils.py       # Experiment logging and metrics computation
├── 📄 environment.yml           # Conda environment configuration
├── 📄 requirements.txt          # Python dependencies
├── 📁 data/                     # Dataset directory
│   └── mini-imagenet/          # Mini-ImageNet dataset
├── 📁 docs/                     # Documentation and papers
│   ├── main.tex                # Main paper
│   └── references.bib          # Bibliography
├── 📁 experiments/              # Experiment outputs
└── 📁 logs/                     # Training logs
```

## 🧠 Method Architecture

| Component | Module | Paper Section | Description |
|-----------|--------|---------------|-------------|
| **Task Embedding** | `TaskEmbeddingComposer` | III-B | `t_i = [μ_i, σ_i, {p_c}, h_ctx]` |
| **Drift Detection** | `DriftDetector` | III-C | Hybrid: KL + Mean Shift + MLP |
| **Adaptive Controller** | `AdaptiveMetaController` | III-D | Strategy selection and routing |
| **Gated Hypernetwork** | `GatedHyperNetwork` | III-D | η_i = η_0(1+αC_i+βD_i), λ_i = λ_0(1+γD_i) |
| **MetaSGD+EWC** | `MetaSGDModule` | III-E | θ' = θ - η∇L + λ/2 ΣF_ii(θ-θ*)² |
| **Task Memory** | `TaskMemory` | III-F | Replay buffer for catastrophic forgetting |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate cfsl_env

# Option 2: Using pip
pip install -r requirements.txt
```

### 2. Dataset Setup
```bash
# Setup all datasets (Mini-ImageNet, Omniglot)
python setup_datasets.py

# Or setup individually
python setup_datasets.py --dataset mini-imagenet
python setup_datasets.py --dataset omniglot
```

### 3. Run Experiments
```bash
# Run on Mini-ImageNet (default)
python main_ablation.py

# Run on Omniglot
python main_ablation.py --dataset omniglot

# Run ablation studies
python main_ablation.py --disable-controller
python main_ablation.py --disable-drift
python main_ablation.py --disable-ewc
```

### 4. Verify Implementation
```bash
# Verify alignment with paper methodology
python test_experiment_alignment.py
```

## 📊 Key Features

### Drift Detection
- **KL Divergence**: Measures distributional changes between task embeddings
- **Mean Shift**: Detects gradual drift in embedding space
- **MLP Classifier**: Learns to predict drift from task characteristics
- **Voting Mechanism**: Requires at least 2 signals for drift confirmation

### Adaptive Control
- **Learning Rate**: η_i = η_0(1 + αC_i + βD_i) where C_i is task complexity, D_i is drift score
- **Regularization**: λ_i = λ_0(1 + γD_i) for EWC strength adjustment
- **Strategy Selection**: Routes between ProtoOnly and MetaSGD+EWC based on drift signals

### Conditional Optimization
- **ProtoOnly Path**: Standard prototype learning for stable tasks
- **MetaSGD+EWC Path**: Fast adaptation with regularization for drifting tasks
- **Task Memory**: Replay buffer to prevent catastrophic forgetting

## 📈 Outputs

After training, the following files are generated:

- **Task Embeddings**: `experiments/task_embeddings.json` - Task-by-task embedding and drift logs
- **Metrics**: `experiments/metrics.json` - Performance metrics and ablation results
- **Visualizations**: 
  - `experiments/epoch_*_tsne.png` - Task embedding t-SNE plots
  - `experiments/drift_trend.png` - Drift detection over time
- **Logs**: `logs/training_*.log` - Detailed training logs

## 🔬 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `NTA_avg` | New Task Accuracy - Performance on current task |
| `OTR_avg` | Old Task Retention - Performance on replayed tasks |
| `FR_avg` | Forgetting Rate - Accuracy degradation over time |
| `DDA_acc` | Drift Detection Accuracy - Precision of drift detection |
| `CAS_η_var` | Learning Rate Variance - Adaptation variability |
| `CAS_λ_var` | Regularization Variance - EWC strength variability |
| `AT_avg_sec` | Adaptation Time - Average time per task |

## 📚 Citation

```bibtex
@misc{yuan2025adaptivecfsl,
  title={Adaptive Few-Shot Meta-Learning with Drift Detection and Task-Aware Optimization},
  author={Yiming Yuan},
  year={2025},
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mini-ImageNet dataset from [Vinyals et al. (2016)](https://arxiv.org/abs/1606.04080)
- ProtoNet implementation inspired by [Snell et al. (2017)](https://arxiv.org/abs/1703.05175)
- MetaSGD implementation based on [Li et al. (2017)](https://arxiv.org/abs/1707.09835)
