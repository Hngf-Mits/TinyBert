"""config.py - every path, model id and hyper-parameter in one place."""
import os

# ----------------------------------------------------------------- paths
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else _HERE
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------- models
TEACHER_ID = "bert-base-uncased"
TINYBERT_ID = "huawei-noah/TinyBERT_General_4L_312D"
TEACHER_NAME = "teacher_bert_base"          # checkpoint / result name of the teacher

# --------------------------------------------------------------- dataset
DATASET_NAME = "tomaarsen/setfit-absa-semeval-restaurants"
LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = len(LABEL_NAMES)
MAX_LENGTH = 128
BATCH_SIZE = 32
DATA_SEED = 42            # FIXED: every model/seed must see exactly the same train/dev/test split
MIN_PER_CLASS = 10        # a split with fewer samples of any class is considered broken

LABEL_COLUMNS = ("label", "polarity", "labels", "sentiment")
LABEL_SYNONYMS = {"negative": 0, "neg": 0, "neutral": 1, "neu": 1, "positive": 2, "pos": 2}
CONFLICT_NAMES = {"conflict", "conf"}

CONFLICT_POLICY = "drop"       # "drop" (standard 3-class ABSA) or "neutral" (merge into neutral)
TEST_SOURCE = "from_train"     # the OFFICIAL test split of this HF dataset is unlabeled (every label is ''),
                               # so a labelled test set is carved from train, grouped by review sentence
DEV_FRACTION = 0.1

# ------------------------------------------------------------ experiment
SEEDS = [42, 43, 44]           # student seeds
TEACHER_SEED = 42              # the teacher is trained once
TEACHER_EPOCHS = 5
STUDENT_EPOCHS = 10            # TinyBERT needs more epochs on ~3k samples; the best epoch is kept
LR_TEACHER = 2e-5
LR_STUDENT = 1e-4              # NOT tuned; CE and KD use identical hyper-parameters
WEIGHT_DECAY = 0.01
TEMPERATURE = 3.0
ALPHA = 0.5
PRECISION = "fp16"             # T4/P100: "fp16" or "fp32" ("bf16" is unsupported there)
RUN_CPU_BENCH = True
STUDENT_MODES = ("ce", "kd")   # CE-only baseline AND distillation
