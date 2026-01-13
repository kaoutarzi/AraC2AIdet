from train_fold import train_5fold

fold_paths = [
    "data/raw/sett1.csv",
    "data/raw/sett2.csv",
    "data/raw/sett3.csv",
    "data/raw/sett4.csv",
    "data/raw/sett5.csv"
]

# Train Arabert
train_5fold("aubmindlab/bert-base-arabertv2", fold_paths, "results/arabert_predictions.csv")

# Train XLM-R
train_5fold("xlm-roberta-base", fold_paths, "results/xlmr_predictions.csv")

# Train AraELECTRA
train_5fold("aubmindlab/araelectra-base-discriminator", fold_paths, "results/araelectra_predictions.csv")

# Train DeBERTa
train_5fold("microsoft/deberta-base", fold_paths, "results/deberta_predictions.csv")
