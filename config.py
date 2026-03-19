from pathlib import Path

EXP_NAME = "debug"
JOB_NAME = None
TEST_SUBJ = None

ABLATION_MODE = 'baseline'  # 'baseline' | 'with_visual' | 'no_pose' | 'no_geom'

MODEL_TYPE = 'DriverHOI'    # 'DriverHOI' | 'MLP-HOI' | 'TransHOI' | 'SCG-HOI'

DATA_ROOT = "/path/to/DriverHOI3D"
DEVICE_CFG_PATH = "devices_config.json"
LOG_DIR = Path("runs")
CKPT_DIR = Path("checkpoints")

DATA_SPLIT = {
    "train": ["subject1", "subject2", "subject3", "subject4", "subject5",
              "subject6", "subject7", "subject8"],
    "val":   ["subject9"],
    "test":  ["subject10"]
}
ALL_SUBJECTS = [f"subject{i}" for i in range(1, 11)]
CAMERA_VIEWS = ['MBP25030012', 'MBP25030014', 'MBP25030016', 'MBP25030017']

FRAME_SAMPLING = 'center'
NUM_FRAMES_PER_ACTION = 1
ACTION_FRAME_POLICY = {'point': 1, 'press': 1, 'push': 1, 'swing': 1}
NUM_DEVICES = 31

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
NUM_WORKERS = 4

NUM_ACT = 4
NUM_CAT = 4
NODE_DIM = 256

LAMBDA_ACT = 8.0
LAMBDA_ID = 10.0
LAMBDA_INTER = 2.0
LAMBDA_AUX = 1.0

USE_CUDA = True
SEED = 42