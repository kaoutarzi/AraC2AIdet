# AraC2AIdet

## Overview

This repository implements a **Cross‑Domain Stacked Ensemble Framework for AI‑Generated Text Detection** in Arabic. The goal is to build robust detection models that generalize across different domains of text, using ensembles of multiple base models and meta‑learners.

We evaluate cross‑domain performance using two public Arabic datasets from Hugging Face:

- **Arabic Generated Abstracts** — a dataset of academic abstracts with both human and LLM‑generated versions, created as part of the research on stylometric analysis in Arabic text generation. :contentReference[oaicite:0]{index=0}  
- **Arabic Generated Social Media Posts** — a dataset of social media posts and their polished/generated counterparts from multiple LLMs, designed to support detection and analysis of machine‑generated informal Arabic text. :contentReference[oaicite:1]{index=1}

The framework works in two directions:

1. **Social → Abstracts**: Train the ensemble on social media text and test on academic abstracts.  
2. **Abstracts → Social**: Train on academic abstracts and test on social media posts.

We leverage **K‑fold cross‑validation** to generate out‑of‑fold predictions from base models (AraBERT, AraELECTRA, DeBERTa, XLM‑RoBERTa) and then train multiple **meta‑learners** (Logistic Regression, SVM, Random Forest, XGBoost) to produce final cross‑domain predictions. This setup helps improve generalization and detection robustness across distinct Arabic text domains.



# Cross-Domain Stacked Ensemble Framework for AI-Generated Text Detection

**Input:**  
- Social media dataset `D_soc`  
- Academic abstracts dataset `D_abs`  
- Base models `B = {AraBERT, AraELECTRA, DeBERTa, XLM-RoBERTa}`  
- Meta-learners `M = {LR, SVM, RF, XGBoost}`  
- Number of folds `K = 5`  

**Output:**  
- Final cross-domain predictions `ŷ`

---

## Functions

### 1. BuildDatasets(D_human, D_llm)
- Combine `D_human` and `D_llm` to construct a balanced binary dataset  
- Return `D_comb`

### 2. TrainBaseModels(D_train, B, K)
- For each base model `b` in `B`:  
  - Apply K-fold cross-validation on `D_train`  
  - Collect out-of-fold predictions → `X_b`  
- Concatenate `{X_b}` → `X_train`  
- Return `X_train`

### 3. TestRepresentation(D_test, B)
- For each base model `b` in `B`:  
  - Train `b` on the full `D_train`  
  - Predict probabilities on `D_test` → `X_b`  
- Concatenate `{X_b}` → `X_test`  
- Return `X_test`

---

## Cross-Domain Directions

### Direction 1: Social → Abstracts
- For each LLM `g ∈ {ALLaM, LLaMA, Jais, OpenAI}`:  
  1. `D_train = BuildDatasets(Social(Human), Social(g))`  
  2. `X_train = TrainBaseModels(D_train, B, K)`  
  3. For each test subset `T ∈ {Human, ALLaM, LLaMA, Jais, OpenAI}` of `D_abs`:  
     - `X_test = TestRepresentation(T, B)`  
     - For each meta-learner `m ∈ M`:  
       - Train `m` on `X_train`  
       - Predict `ŷ = m(X_test)`  
       - Evaluate performance

### Direction 2: Abstracts → Social
- For each LLM `g ∈ {ALLaM, LLaMA, Jais, OpenAI}`:  
  1. `D_train = BuildDatasets(Abstract(Human), Abstract(g))`  
  2. `X_train = TrainBaseModels(D_train, B, K)`  
  3. For each test subset `T ∈ {Human, ALLaM, LLaMA, Jais, OpenAI}` of `D_soc`:  
     - `X_test = TestRepresentation(T, B)`  
     - For each meta-learner `m ∈ M`:  
       - Train `m` on `X_train`  
       - Predict `ŷ = m(X_test)`  
       - Evaluate performance
