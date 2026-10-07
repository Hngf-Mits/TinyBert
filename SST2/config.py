"""config.py - every constant of the SST-2 size sweep lives here (no logic)."""
import os

# ------------------------------------------------------------------ paths
BASE_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(os.path.abspath(__file__))
TAG = "sst2"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")        # fine-tuned teacher + one checkpoint per student architecture
RES_DIR = os.path.join(BASE_DIR, "results")             # per-run / per-benchmark JSON, summary.csv, kd_gain.csv, plots
CACHE_DIR = os.path.join(BASE_DIR, "cache")             # cached teacher logits, benchmark samples

# ------------------------------------------------------------------ data
HF_DATASET = ("nyu-mll/glue", "sst2")
DATA_SEED = 42            # train/dev split seed
DEV_FRACTION = 0.03       # dev is carved out of GLUE train (GLUE test labels are hidden, GLUE validation is used as TEST)
LABEL_NAMES = ["negative", "positive"]
NUM_LABELS = len(LABEL_NAMES)
MIN_PER_CLASS = 10        # a split with fewer samples of any class is considered broken
MAX_LENGTH = 128
TOKENIZER_NAME = "bert-base-uncased"

# ------------------------------------------------------------------ teacher
TEACHER_ID = "bert-base-uncased"
TEACHER_NAME = "teacher_bert_base"         # checkpoint dir name under checkpoints/
TEACHER_EPOCHS, LR_TEACHER = 3, 2e-5       # used by train_teacher.py
# None -> use the BERT-base fine-tuned by train_teacher.py (default, same as the ABSA pipeline).
# Or set a HF id (e.g. "textattack/bert-base-uncased-SST-2") to skip teacher training.
TEACHER_PRETRAINED = None

# ------------------------------------------------------------------ students
# name -> (HF id, learning rate). LRs are a rough size-based heuristic, NOT tuned; CE and KD share them.
STUDENTS = {
    "bert_tiny_L2_H128":   ("google/bert_uncased_L-2_H-128_A-2", 3e-4),
    "bert_mini_L4_H256":   ("google/bert_uncased_L-4_H-256_A-4", 2e-4),
    "tinybert_L4_H312":    ("huawei-noah/TinyBERT_General_4L_312D", 1e-4),
    "bert_small_L4_H512":  ("google/bert_uncased_L-4_H-512_A-8", 1e-4),
    "bert_medium_L8_H512": ("google/bert_uncased_L-8_H-512_A-8", 5e-5),
}
SIZES = list(STUDENTS)

# ------------------------------------------------------------------ training
SEEDS = [42, 43, 44]
MULTI_SEED_SIZES = ["bert_tiny_L2_H128", "tinybert_L4_H312", "bert_medium_L8_H512"]  # 3 seeds; others run 1 seed
EPOCHS = 3
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
TEMPERATURE, ALPHA = 3.0, 0.5       # NOT tuned; CE and KD share everything except the loss
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0

# ------------------------------------------------------------------ benchmark
BENCH_N_SAMPLES = 512
BENCH_BATCH = 32
BENCH_LATENCY_RUNS = 200
BENCH_WARMUP = 10
RUN_CPU_BENCH = True
