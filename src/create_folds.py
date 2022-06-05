import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

if __name__ == "__main__":
    df = pd.DataFrame(columns=["filename", "label"])
    file_names = os.listdir(config.TRAIN_DATA_PATH)
    df["filename"] = file_names
    df["label"] = list(map(lambda x: x.split(".")[0], file_names))

    kf = StratifiedKFold(n_splits=config.NUM_FOLDS)
    X = df.filename.values
    y = df.label.values
    df["kfold"] = -1

    for kf_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        df.loc[val_idx, "kfold"] = kf_idx

    df.to_csv(f"{config.FOLDS_PATH}/create_folds.csv", index=False)
