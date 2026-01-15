"""
Cross-Domain Stacked Ensemble - Meta-Learner Stage
Author: Kaoutar Zita
Description: Combines model predictions from base models (AraBERT, AraELECTRA, DeBERTa, XLM-RoBERTa)
             to build meta-learner datasets and trains/evaluates Logistic Regression, SVM,
             Random Forest, and XGBoost classifiers on multiple test datasets.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Build dataset from base model predictions

def build_dataset(arabert_file, araelectra_file, deberta_file, xlmr_file, column_name, save_path):
    """
    Combine predictions from four base models and the true class into one CSV
    """
    df_arabert = pd.read_csv(arabert_file)
    df_araelectra = pd.read_csv(araelectra_file)
    df_deberta = pd.read_csv(deberta_file)
    df_xlmr = pd.read_csv(xlmr_file)

    df = pd.DataFrame({
        "arabert": df_arabert[column_name],
        "araelectra": df_araelectra[column_name],
        "deberta": df_deberta[column_name],
        "xlmr": df_xlmr[column_name],
        "class": df_arabert["class"].map({"human": 0, "machine": 1})
    })

    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved combined dataset to {save_path}")
    return df


#  Build all datasets

# Define file paths for base model predictions
dataset_paths = {
    "training_set": {
        "arabert": "",
        "araelectra": "",
        "deberta": "",
        "xlmr": "",
        "column": "",  # Column name for training
        "save": ""
    },
    "allam_test": {
        "arabert": "",
        "araelectra": "",
        "deberta": "",
        "xlmr": "",
        "column": "",
        "save": ""
    },
    "jais_test": {
        "arabert": "",
        "araelectra": "",
        "deberta": "",
        "xlmr": "",
        "column": "",
        "save": ""
    },
    "openai_test": {
        "arabert": "",
        "araelectra": "",
        "deberta": "",
        "xlmr": "",
        "column": "",
        "save": ""
    },
    "llama_test": {
        "arabert": "",
        "araelectra": "",
        "deberta": "",
        "xlmr": "",
        "column": "",
        "save": ""
    }
}


# Build all datasets
for key, val in dataset_paths.items():
    build_dataset(val["arabert"], val["araelectra"], val["deberta"], val["xlmr"], val["column"], val["save"])


#  Load training dataset

train_df = pd.read_csv("/kaggle/working/training_set.csv")
X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]


#  Test datasets

test_files = {
    "allam": "/kaggle/working/allam_test.csv",
    "jais": "/kaggle/working/jais_test.csv",
    "llama": "/kaggle/working/llama_test.csv",
    "openai": "/kaggle/working/openai_test.csv"
}


#  Train & evaluate models

def train_evaluate_model(model, model_name, X_train, y_train, test_files):
    model.fit(X_train, y_train)
    results = []
    for dataset_name, path in test_files.items():
        df_test = pd.read_csv(path)
        X_test = df_test.drop(columns=["class"])
        y_test = df_test["class"]
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append([dataset_name, acc, prec, rec, f1])

    results_df = pd.DataFrame(results, columns=["Dataset", "Accuracy", "Precision", "Recall", "F1"])
    results_df.to_csv(f"/kaggle/working/{model_name}_results.csv", index=False)
    print(f"\n[INFO] {model_name} Results:\n", results_df)
    return results_df


#  Meta-learners

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

for name, clf in models.items():
    train_evaluate_model(clf, name, X_train, y_train, test_files)
