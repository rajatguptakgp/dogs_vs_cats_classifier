import os
import numpy as np
import pandas as pd
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import config
from dataset import Dataset
from model_dispatcher import MODEL_DISPATCHER
from train import Trainer
from plotter import Plotter
from evaluate import Evaluator
from augment import Augmenter

if __name__ == "__main__":
    if config.SAVE_ARRAY:
        df = pd.read_csv(f"{config.FOLDS_PATH}/create_folds.csv")
        df = df.sample(frac=config.SAMPLE_FRAC).reset_index(drop=True)

        for fold_idx in range(config.NUM_FOLDS):
            train_image_names = df[df["kfold"] != fold_idx].filename.values
            val_image_names = df[df["kfold"] == fold_idx].filename.values

            train_dataset = Dataset(
                fold_idx,
                "train",
                config.TRAIN_DATA_PATH,
                config.SAVE_ARRAY_PATH,
                train_image_names,
                config.IMG_SIZE,
                config.NORMALIZE,
                config.CONVERT_GRAY,
            )

            val_dataset = Dataset(
                fold_idx,
                "val",
                config.TRAIN_DATA_PATH,
                config.SAVE_ARRAY_PATH,
                val_image_names,
                config.IMG_SIZE,
                config.NORMALIZE,
                config.CONVERT_GRAY,
            )

    X_train = np.load(
        f"{config.SAVE_ARRAY_PATH}/train_images_fold_{config.FOLD_IDX}.npy"
    )
    y_train = np.load(
        f"{config.SAVE_ARRAY_PATH}/train_labels_fold_{config.FOLD_IDX}.npy"
    )
    X_val = np.load(f"{config.SAVE_ARRAY_PATH}/val_images_fold_{config.FOLD_IDX}.npy")
    y_val = np.load(f"{config.SAVE_ARRAY_PATH}/val_labels_fold_{config.FOLD_IDX}.npy")

    y_train = np.where(y_train == "cat", 0, 1).reshape(-1, 1)
    y_val = np.where(y_val == "cat", 0, 1).reshape(-1, 1)

    model = MODEL_DISPATCHER[config.MODEL_NAME](
        config.NUM_CLASSES, config.IMG_SIZE
    ).model

    lrop = ReduceLROnPlateau(factor=config.FACTOR_LROP, patience=config.PATIENCE_LROP)
    estop = EarlyStopping(patience=config.PATIENCE_ESTOP)
    optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)

    if config.DATA_AUGMENT:
        augmenter = Augmenter(
            config.HEIGHT_SHIFT_RANGE,
            config.WIDTH_SHIFT_RANGE,
            config.HORIZONTAL_FLIP,
            config.ROTATION_RANGE,
            config.ZOOM_RANGE,
        )

        datagen = augmenter.create_generator()
        img_generator = datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE)

        trainer = Trainer(
            model,
            config.MODEL_NAME,
            config.LOSS_FN,
            optimizer,
            config.BATCH_SIZE,
            config.LEARNING_RATE,
            config.N_EPOCHS,
            img_generator,
            (X_val, y_val),
            [lrop, estop],
            config.METRICS,
            config.DATA_AUGMENT,
            verbose=config.VERBOSITY,
        )

    else:
        trainer = Trainer(
            model,
            config.MODEL_NAME,
            config.LOSS_FN,
            optimizer,
            config.BATCH_SIZE,
            config.LEARNING_RATE,
            config.N_EPOCHS,
            (X_train, y_train),
            (X_val, y_val),
            [lrop, estop],
            config.METRICS,
            config.DATA_AUGMENT,
            verbose=config.VERBOSITY,
        )

    if config.TRAIN_MODEL:
        trainer.compile_model()
        trainer.fit_model()
        trainer.save_model(config.SAVE_MODEL_PATH, config.FOLD_IDX)
        trainer.save_history(config.SAVE_HISTORY_PATH, config.FOLD_IDX)

    trainer_name = trainer.name
    model = trainer.load_keras_model(config.SAVE_MODEL_PATH, config.FOLD_IDX)
    history = trainer.load_history(config.SAVE_HISTORY_PATH, config.FOLD_IDX)

    plotter = Plotter(trainer_name, history, config.METRICS)
    plotter.save_plots(config.SAVE_HISTORY_PATH, config.FOLD_IDX)

    evaluator = Evaluator(trainer_name, model, X_val, y_val)
    evaluator.save_metrics(config.SAVE_METRICS_PATH, config.FOLD_IDX)
