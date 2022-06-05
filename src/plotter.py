import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, name, history, metrics):
        self.name = name
        self.history = history
        self.metrics = metrics

    def plot_fn(self, train_vals, val_vals, metric_name):
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.plot(train_vals)
        plt.title(f"Training {metric_name}")

        plt.subplot(132)
        plt.plot(val_vals)
        plt.title(f"Validation {metric_name}")

        plt.subplot(133)
        plt.plot(train_vals, label=f"Training {metric_name}")
        plt.plot(val_vals, label=f"Validation {metric_name}")
        plt.legend()
        return plt

    def plot_loss(self):
        train_loss = self.history["loss"]["train"]
        val_loss = self.history["loss"]["val"]
        return self.plot_fn(train_loss, val_loss, "Loss")

    def plot_metric(self, metric, metric_name):
        train_vals = self.history[metric]["train"]
        val_vals = self.history[metric]["val"]
        return self.plot_fn(train_vals, val_vals, metric_name)

    def save_plots(self, save_path, fold_idx):
        self.plot_loss().savefig(
            f"{save_path}/fold_{fold_idx}_" + self.name + "_loss.png"
        )
        for metric in self.metrics:
            self.plot_metric(metric, metric).savefig(
                f"{save_path}/fold_{fold_idx}_" + self.name + f"_{metric}.png"
            )
