import os
import numpy as np
import json
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


class Evaluator:
    def __init__(self, name, model, X_val, y_val):
        self.name = name
        self.model = model
        self.X_val = X_val
        self.y_true = y_val.flatten()

        self.predict()

    def predict(self):
        y_score = self.model.predict(self.X_val)
        self.y_pred = np.argmax(y_score, axis=1)

    def get_classification_metrics(self):
        acc = balanced_accuracy_score(self.y_true, self.y_pred)
        prec = precision_score(self.y_true, self.y_pred, zero_division=0)
        rec = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)
        mcc = matthews_corrcoef(self.y_true, self.y_pred)

        return {
            "Accuracy": str(acc),
            "Precision": str(prec),
            "Recall": str(rec),
            "F1": str(f1),
            "MCC": str(mcc),
        }

    def get_confusion_matrix(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()

        return {"TN": str(tn), "FP": str(fp), "FN": str(fn), "TP": str(tp)}

    def print_metrics(self):
        metrics_dict = self.get_classification_metrics()
        print("Metrics:")
        for k, v in metrics_dict.items():
            print(f"{k}: {v}")

        print("\nOutcomes:")
        outcomes_dict = self.get_confusion_matrix()
        for k, v in outcomes_dict.items():
            print(f"{k}: {v}")

    def save_metrics(self, save_path, fold_idx):
        metrics_dict = self.get_classification_metrics()
        outcomes_dict = self.get_confusion_matrix()
        metrics_dict.update(outcomes_dict)

        os.makedirs(save_path, exist_ok=True)
        with open(
            f"{save_path}/fold_{fold_idx}_" + self.name + ".json", "w"
        ) as json_file:
            json.dump(metrics_dict, json_file, indent=4)
