import cv2
import os
import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(
        self,
        fold_idx,
        mode,
        data_path,
        save_path,
        image_names,
        img_size,
        normalize=True,
        convert_gray=False,
    ):
        self.fold_idx = fold_idx
        self.mode = mode
        self.data_path = data_path
        self.save_path = save_path
        self.image_names = image_names
        self.img_size = img_size
        self.normalize = normalize
        self.convert_gray = convert_gray

        self.save_array(save_path, mode)

    def save_array(self, save_path, mode):
        data = []
        labels = []

        for filename in tqdm(self.image_names):
            img = cv2.imread(self.data_path + "/" + filename)

            if self.convert_gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_size, self.img_size))
            if self.normalize:
                img /= 255

            label = filename.split(".")[0]
            data.append(img)
            labels.append(label)

        data = np.array(data)
        labels = np.array(labels)
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/{mode}_images_fold_{self.fold_idx}.npy", data)
        np.save(f"{save_path}/{mode}_labels_fold_{self.fold_idx}.npy", labels)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        filename1 = f"cat.{idx}.jpg"
        filename2 = f"dog.{idx}.jpg"
        path1 = f"{self.data_path}/{filename1}"
        path2 = f"{self.data_path}/{filename2}"

        if os.path.exists(path1):
            img = cv2.imread(path1)
            label = "cat"
        elif os.path.exists(path2):
            img = cv2.imread(path2)
            label = "dog"
        else:
            img = None
            label = None

        return {"img": img, "label": label}
