TRAIN_MODEL = True
FOLD_IDX = 0

TRAIN_DATA_PATH = "./input/dogs-vs-cats/train"
TEST_DATA_PATH = "./input/dogs-vs-cats/test1"
NUM_FOLDS = 5
SAMPLE_FRAC = 0.05

NORMALIZE = True
CONVERT_GRAY = False
IMG_SIZE = 64
NUM_CLASSES = 2

FOLDS_PATH = "./input"
SAVE_ARRAY = False
SAVE_ARRAY_PATH = "./input/folds"
SAVE_MODEL_PATH = "./input/models"
SAVE_HISTORY_PATH = "./input/history"
SAVE_METRICS_PATH = "./input/metrics"

MODEL_NAME = "custom"
N_EPOCHS = 5
LEARNING_RATE = 1e-2
LOSS_FN = "sparse_categorical_crossentropy"
BATCH_SIZE = 64
VERBOSITY = 1
METRICS = ["accuracy"]

DATA_AUGMENT = False
HEIGHT_SHIFT_RANGE = 0.1
WIDTH_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
ROTATION_RANGE = 30
ZOOM_RANGE = 0.1

PATIENCE_ESTOP = 10
PATIENCE_LROP = 3
FACTOR_LROP = 0.8