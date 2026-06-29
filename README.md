# AraC2AIdet 🔍

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

**Cross-Domain Stacked Ensemble Framework for AI-Generated Arabic Text Detection**

A cutting-edge machine learning framework that detects AI-generated Arabic text across diverse writing domains using advanced stacked ensemble learning with transformer-based models.

---

## 🎯 Overview

AraC2AIdet addresses the critical challenge of detecting AI-generated content in Arabic across heterogeneous domains. Unlike conventional approaches that train and evaluate on identical domains, this framework is specifically designed to generalize across different writing styles—from academic abstracts to social media posts.

### Key Innovation
Our stacked ensemble architecture combines multiple transformer-based language models as base learners, with meta-learning to enhance cross-domain robustness and detection accuracy.

---

## ✨ Features

- ✅ **Cross-Domain Detection**: Seamlessly detect AI-generated Arabic text across different writing domains
- 🏗️ **Stacked Ensemble Architecture**: Multi-stage learning with base models and meta-learners
- 🤖 **Transformer-Based Models**: 4 powerful encoder models optimized for Arabic
- 📊 **K-Fold Cross-Validation**: Robust Out-of-Fold prediction generation
- 🔄 **Multiple Meta-Learners**: Choose from Logistic Regression, SVM, Random Forest, or XGBoost
- 🌐 **Multi-LLM Support**: Evaluates across ALLaM, LLaMA, JAIS, and OpenAI
- 📦 **Modular & Extensible**: Easy to customize and integrate into your projects

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Framework Architecture](#-framework-architecture)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/kaoutarzi/AraC2AIdet.git
cd AraC2AIdet

# Install dependencies
pip install -r requirement.txt

# Run the framework
python meta_learner.py
```

---

## 📦 Installation

### Requirements
- Python 3.8 or higher
- PyTorch 2.0+
- CUDA support (optional, for GPU acceleration)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/kaoutarzi/AraC2AIdet.git
   cd AraC2AIdet
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Required Packages**
   ```bash
   pip install -r requirement.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

---

## 📁 Repository Structure

```
AraC2AIdet/
│
├── Src/                          # Source code modules
│   ├── base_models.py           # Transformer-based models
│   ├── ensemble_utils.py        # Ensemble operations
│   └── ...
│
├── traindata2stage/             # Training datasets
│   ├── social_media/
│   │   ├── human_posts/
│   │   ├── llm_generated/
│   │   └── ...
│   └── academic_abstracts/
│       ├── human_abstracts/
│       ├── llm_generated/
│       └── ...
│
├── meta_learner.py              # Meta-learning implementation
├── stacking_framework.tex       # Detailed framework documentation
├── requirement.txt              # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

---

## 🗂️ Datasets

Our framework evaluates cross-domain generalization using two comprehensive Arabic datasets:

### 1️⃣ Arabic Generated Abstracts
Academic writing domain with human-authored and AI-generated abstracts

**Content:**
- Human-written academic abstracts
- AI-generated academic abstracts

**Supported Generators:**
- ALLaM
- LLaMA
- JAIS
- OpenAI

### 2️⃣ Arabic Generated Social Media Posts
Social media domain with diverse content

**Content:**
- Human-written social media posts
- AI-generated social media posts

**Supported Generators:**
- ALLaM
- LLaMA
- JAIS
- OpenAI

---

## 🏗️ Framework Architecture

### Stage 1: Base Models
Four transformer encoders trained independently using 5-fold cross-validation:

| Model | Parameters | Architecture | Language Support |
|-------|-----------|--------------|-----------------|
| **AraBERT** | 110M | BERT-based | Arabic |
| **AraELECTRA** | 110M | ELECTRA-based | Arabic |
| **DeBERTa** | 305M | ELECTRA Variant | Multilingual |
| **XLM-RoBERTa** | 270M | RoBERTa-based | 100+ Languages |

### Stage 2: Meta-Learners
Choose from four meta-learning classifiers:

- **Logistic Regression**: Fast, interpretable baseline
- **Support Vector Machine (SVM)**: Robust non-linear classification
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting framework (Recommended for best performance)

---

## 🔄 Pipeline Overview

### Step 1: Build Balanced Datasets
```
BuildDatasets(D_human, D_llm) → Balanced binary classification data
```

### Step 2: Train Base Models
```
TrainBaseModels(D_train, B, K) → K-Fold cross-validation with K=5
```
- Generate Out-of-Fold prediction probabilities
- Concatenate predictions for meta-learner training

### Step 3: Generate Test Representation
```
TestRepresentation(D_test, B) → Base model predictions on test set
```

### Step 4: Train Meta-Learner
```
TrainMetaLearner(P_train, P_test) → Final binary predictions
```

---

## 💻 Usage

### Running the Complete Framework

```bash
python meta_learner.py
```

### Custom Configuration

```python
from Src.meta_learner import AraC2AIdetFramework

# Initialize framework
framework = AraC2AIdetFramework(
    base_models=['araBERT', 'araELECTRA', 'DeBERTa', 'xlmRoberta'],
    meta_learner='xgboost',
    n_folds=5,
    random_state=42
)

# Train on source domain
framework.train(
    human_texts='path/to/human_data',
    llm_texts='path/to/llm_data',
    domain='social_media'  # or 'academic'
)

# Evaluate on target domain
predictions = framework.predict(
    test_texts='path/to/test_data',
    target_domain='academic'  # or 'social_media'
)
```

### Available Models

```python
BASE_MODELS = {
    'araBERT': 'bert-base-arabertv2',
    'araELECTRA': 'aubmindlab/araelectra-base-discriminator',
    'DeBERTa': 'microsoft/deberta-v3-base',
    'xlmRoberta': 'xlm-roberta-base'
}

META_LEARNERS = ['logistic_regression', 'svm', 'random_forest', 'xgboost']
```

---

## 📊 Evaluation

### Cross-Domain Evaluation Scenarios

#### Direction 1: Social Media → Academic Abstracts
- **Training**: Human & AI Social Media Posts
- **Testing**: Academic Abstracts (Human + 4 LLM variants)

#### Direction 2: Academic Abstracts → Social Media
- **Training**: Human & AI Academic Abstracts
- **Testing**: Social Media Posts (Human + 4 LLM variants)

### Metrics Reported
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **Confusion Matrix**


---

## 📚 Requirements

```
transformers>=4.25.0
torch>=2.0.0
scikit-learn>=1.0.0
xgboost>=1.7.0
pandas>=1.3.0
numpy>=1.21.0
```

Install all dependencies:
```bash
pip install -r requirement.txt
```

---

## 📖 Citation

If you use AraC2AIdet in your research, please cite our paper:

```bibtex
@article{,
  title={A Cross-Domain Stacked Ensemble for Model-Aware Detection of AI-Generated Arabic Text},
  author={},
  journal={Under Review},
  year={2025}
}
```

---



### Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Kaoutar Zita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

This work builds upon these exceptional open-source projects:

- [Hugging Face Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP models
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting framework

Special thanks to the Arabic NLP community for providing Arabic-optimized models and datasets.

---

## 📧 Contact & Support

- **Author**: Kaoutar Zita
- **Issues**: [GitHub Issues](https://github.com/kaoutarzi/AraC2AIdet/issues)
- **Email**: [zita.kaoutar@univ-ghardaia.edu.dz]

---

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by [Kaoutar Zita](https://github.com/kaoutarzi)

</div>
