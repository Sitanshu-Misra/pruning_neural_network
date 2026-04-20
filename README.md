# Self-Pruning Neural Network 🧠✂️

This repository contains my solution for the **Tredence Analytics AI Engineering Internship Case Study**.

## 📌 Project Overview
The goal of this project is to build a feed-forward neural network for CIFAR-10 image classification that **learns to prune itself** during training. Instead of applying manual pruning heuristics after training, the network autonomously discovers which weight connections are dispensable using learnable gate parameters and L1 regularization.

### 🌟 Key Features
- **Custom `PrunableLinear` Layer**: Replaces standard fully-connected layers with learnable "gate scores" for every weight.
- **End-to-End Differentiable Pruning**: A sparsity penalty (L1 norm of the sigmoid gate values) is added to the Cross-Entropy loss.
- **Bimodal Gate Distribution**: The L1 penalty forces gates to either 0 (pruned) or 1 (retained), creating a naturally sparse architecture.
- **Automated Experimentation**: The script trains models across different sparsity thresholds (`λ`) and automatically visualizes the accuracy vs. sparsity trade-offs.

---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed. You will need the following dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

### 2. Run the Script
To start the training and evaluation loop:
```bash
python self_pruning_neural_network.py
```

### 3. Output
The script will:
- Automatically download the CIFAR-10 dataset (if not present).
- Train the network across 3 different sparsity thresholds (`λ = 1e-5`, `5e-4`, `1e-2`).
- Save the test accuracy and sparsity percentage results.
- Generate Matplotlib distribution histograms showing the spike at 0 (pruned connections) and cluster at 1.
- All plots and a text summary will be saved to the `./results/` folder.

---

## 📄 Full Case Study Report
For an in-depth mathematical intuition behind why the L1 penalty on sigmoid gates encourages sparsity, as well as a detailed analysis of the expected bimodal gate distribution, please refer to the comprehensive [REPORT.md](./REPORT.md) included in this repository.
