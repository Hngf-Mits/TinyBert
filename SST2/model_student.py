"""model_student.py - student ladder (5 sizes, 4.4M -> 41.7M params)."""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import NUM_LABELS, STUDENTS


def build_student(size, device, tok):
    """Instantiate a student for `size` and check it shares the teacher's vocabulary."""
    hf_id, lr = STUDENTS[size]
    stok = AutoTokenizer.from_pretrained(hf_id)
    assert stok.get_vocab() == tok.get_vocab(), f"{size}: vocabulary differs from the teacher's"
    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=NUM_LABELS).to(device)
    return model, lr
