from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Augmenter:
    def __init__(
        self,
        height_shift_range,
        width_shift_range,
        horizontal_flip,
        rotation_range,
        zoom_range,
    ):
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.horizontal_flip = horizontal_flip
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range

    def create_generator(self):
        datagen = ImageDataGenerator(
            height_shift_range=self.height_shift_range,
            width_shift_range=self.width_shift_range,
            horizontal_flip=self.horizontal_flip,
            rotation_range=self.rotation_range,
            zoom_range=self.zoom_range,
        )

        return datagen
