"""model_student.py - build the student (TinyBERT 4L-312D) and check it matches the teacher's vocabulary."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import NUM_LABELS, TINYBERT_ID


def build_student(device, model_id=TINYBERT_ID):
    """Pretrained general TinyBERT with a new (randomly initialised) classification head."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS).to(device)
    return model, tokenizer


def check_same_vocab(tokenizer_a, tokenizer_b):
    """Knowledge distillation on logits needs identical token ids for teacher and student."""
    assert tokenizer_a.get_vocab() == tokenizer_b.get_vocab(), "Tokenizer vocabularies differ!"
