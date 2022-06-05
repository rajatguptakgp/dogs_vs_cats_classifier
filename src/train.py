import os
import json
from keras.models import load_model


class Trainer:
    def __init__(
        self,
        model,
        model_name,
        loss_fn,
        optimizer,
        batch_size,
        learning_rate,
        n_epochs,
        training_data,
        validation_data,
        callbacks,
        metrics,
        augment,
        verbose,
    ):
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.training_data = training_data
        self.validation_data = validation_data
        self.callbacks = callbacks
        self.metrics = metrics
        self.augment = augment
        self.verbose = verbose

        self.name = f"{self.model_name}_bs_{self.batch_size}_epochs_{self.n_epochs}_lr_{self.learning_rate}"

    def compile_model(self):
        self.model.compile(
            loss=self.loss_fn, optimizer=self.optimizer, metrics=self.metrics
        )

    def fit_model(self):
        if self.augment:
            self.history = self.model.fit(
                self.training_data,
                epochs=self.n_epochs,
                shuffle=True,
                validation_data=self.validation_data,
                callbacks=self.callbacks,
            )
        else:
            self.history = self.model.fit(
                self.training_data[0],
                self.training_data[1],
                epochs=self.n_epochs,
                shuffle=True,
                validation_data=self.validation_data,
                callbacks=self.callbacks,
            )

    def save_model(self, save_path, fold_idx):
        os.makedirs(save_path, exist_ok=True)
        self.model.save(f"{save_path}/fold_{fold_idx}_" + self.name)

    def load_keras_model(self, save_path, fold_idx):
        return load_model(f"{save_path}/fold_{fold_idx}_" + self.name)

    def save_history(self, save_path, fold_idx):
        history_dict = {}
        history_dict["loss"] = {}
        history_dict["loss"]["train"] = self.history.history["loss"]
        history_dict["loss"]["val"] = self.history.history["val_loss"]

        for metric in self.metrics:
            history_dict[metric] = {}
            history_dict[metric]["train"] = self.history.history[metric]
            history_dict[metric]["val"] = self.history.history[f"val_{metric}"]

        os.makedirs(save_path, exist_ok=True)
        with open(
            f"{save_path}/fold_{fold_idx}_" + self.name + ".json", "w"
        ) as json_file:
            json.dump(history_dict, json_file, indent=4)

    def load_history(self, save_path, fold_idx):
        with open(f"{save_path}/fold_{fold_idx}_" + self.name + ".json") as json_file:
            return json.load(json_file)
