from keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from keras.models import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50


class CustomModel:
    def __init__(self, num_classes, img_size):
        self.num_classes = num_classes
        self.img_size = img_size

        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.build(input_shape=(None, self.img_size, self.img_size, 3))
        return model


class ResNetModel:
    def __init__(self, num_classes, img_size):
        self.num_classes = num_classes
        self.img_size = img_size

        self.model = self.build_model()

    def build_model(self):
        model = ResNet50(
            weights="imagenet",
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
        )
        model = Sequential(model)
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation="softmax"))

        model.build(input_shape=(None, self.img_size, self.img_size, 3))
        return model
