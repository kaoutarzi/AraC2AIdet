# AraC2AIdet


## Overview

This algorithm implements a **cross-domain stacked ensemble framework** for detecting AI-generated text in Arabic datasets.  
It combines multiple **pretrained base models** (AraBERT, AraELECTRA, DeBERTa, XLM-RoBERTa) and uses **meta-learners** (Logistic Regression, SVM, Random Forest, XGBoost) to improve detection performance across different domains.  

The framework supports two directions:  
1. **Social → Abstracts**: Train on social media data and test on academic abstracts.  
2. **Abstracts → Social**: Train on academic abstracts and test on social media data.  

By leveraging **K-fold cross-validation** and out-of-fold predictions from base models, the ensemble reduces overfitting and enhances generalization to unseen datasets.


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
