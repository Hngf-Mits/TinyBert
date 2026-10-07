"""model_teacher.py - build / load / freeze the teacher (bert-base-uncased)."""
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CHECKPOINT_DIR, NUM_LABELS, TEACHER_ID, TEACHER_NAME


def teacher_dir():
    return os.path.join(CHECKPOINT_DIR, TEACHER_NAME)


def build_teacher(device):
    """Fresh bert-base-uncased with a new classification head (to be fine-tuned)."""
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_ID, num_labels=NUM_LABELS).to(device)
    return model, tokenizer


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def save_teacher(model, tokenizer):
    path = teacher_dir()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def load_teacher(device, path=None):
    """Load the fine-tuned teacher saved by train_teacher.py, frozen and in eval mode."""
    path = path or teacher_dir()
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} not found - run train_teacher.py first")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    return freeze(model), tokenizer
