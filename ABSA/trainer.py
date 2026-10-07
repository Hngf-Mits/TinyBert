"""trainer.py - distillation loss and the training loop shared by the teacher (CE) and the students (CE / KD)."""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from utils import evaluate_model


class DistillationLoss(nn.Module):
    """alpha * CE(hard labels) + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)."""

    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature, self.alpha = temperature, alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce_loss(student_logits, labels)
        soft_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_t = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_kd = self.kl_loss(soft_s, soft_t) * (self.temperature ** 2)
        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd, loss_ce, loss_kd


def train_model(model, train_loader, dev_loader, device, *, teacher=None, epochs=10,
                lr=1e-4, weight_decay=0.01, temperature=3.0, alpha=0.5, precision="fp16"):
    """CE if teacher is None, else knowledge distillation.

    The test set is NEVER touched here. After each epoch the model is scored on the dev
    split; the best epoch (by dev macro-F1) is restored at the end.
    """
    use_kd = teacher is not None
    criterion = DistillationLoss(temperature, alpha if use_kd else 1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and precision == "fp16"))

    history, best_f1, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        running, t0 = 0.0, time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")

            with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=use_amp):
                teacher_logits = None
                if use_kd:
                    with torch.no_grad():
                        teacher_logits = teacher(**inputs).logits
                logits = model(**inputs).logits

            if use_kd:
                loss, _, _ = criterion(logits.float(), teacher_logits.float(), labels)
            else:
                loss = criterion.ce_loss(logits.float(), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        dev = evaluate_model(model, dev_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": running / len(train_loader),
                        "dev_accuracy": dev["accuracy"], "dev_macro_f1": dev["macro_f1"],
                        "seconds": time.time() - t0})
        print(f"epoch {epoch + 1:>2}: loss {running / len(train_loader):.4f} | "
              f"dev acc {dev['accuracy']:.2f} | dev macro-F1 {dev['macro_f1']:.2f}")
        if dev["macro_f1"] > best_f1:
            best_f1, best_epoch = dev["macro_f1"], epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"-> restored best epoch {best_epoch} (dev macro-F1 {best_f1:.2f})")
    return {"history": history, "best_epoch": best_epoch, "best_dev_macro_f1": best_f1}
