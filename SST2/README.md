# SST-2 — How small can the student become?

Teacher: `bert-base-uncased` (109M) fine-tuned on SST-2 by `train_teacher.py`, then frozen. Students: a ladder of 5 sizes (4.4M → 41.7M params),
each trained with **CE only** and with **KD** (T=3, α=0.5), then benchmarked on GPU and CPU.

| File | Role |
|---|---|
| `config.py` | all constants: paths, student ladder + LRs, seeds, epochs, T/α, benchmark settings |
| `dataset.py` | SST-2 loading (train / dev carved from GLUE train / test = GLUE validation), label-distribution check, loaders |
| `train_teacher.py` | fine-tunes BERT-base on SST-2 (CE, best dev epoch), saves `checkpoints/teacher_bert_base/` |
| `model_teacher.py` | loads the frozen teacher, computes + caches its train logits **once** |
| `model_student.py` | builds a student for a given size (checks vocab matches the teacher) |
| `train_distill.py` | KD loss, trainer, CE-vs-KD sweep over sizes/seeds (resumable) |
| `benchmark.py` | latency / throughput / GFLOPs / VRAM, one process per model and device |
| `report.py` | `summary.csv`, `kd_gain.csv`, `tradeoff.png`, `kd_gain.png`, "smallest size keeping ≥95% of teacher" |
| `utils.py` | seeding, JSON I/O, metrics, predict/evaluate, param counts |

## Run
```bash
cd SST-2
pip install -r ../requirements.txt     # needs: torch transformers datasets pandas matplotlib
python train_teacher.py                # fine-tune the teacher (skip: set TEACHER_PRETRAINED in config.py)
python train_distill.py                # cache teacher logits + all students (resumes if interrupted)
python benchmark.py --all              # teacher + students on GPU and CPU
python report.py --zip                 # tables, plots, results_sst2.zip
```
Useful flags: `--sizes bert_tiny_L2_H128`, `--seeds 42`, `--full-seeds`, `--epochs 3`.

Outputs go to `checkpoints/`, `results/`, `cache/` (next to the scripts, or `/kaggle/working` on Kaggle — same folder names as the ABSA pipeline, so don't run both in one Kaggle session without clearing them).
`results/label_stats.json` holds the split label distributions and the majority-class floor; `results/kd_gain.csv` the paired KD−CE gain per size and seed.

## Notes
* Teacher logits are cached because the teacher is frozen → KD runs cost about the same as CE runs.
* Hardware benchmarks run once per architecture (latency does not depend on the seed).
* Hyper-parameters (T, α, per-size LR) are **not tuned** and are shared by CE and KD, so the comparison is fair but not optimal.
* Sizes with `n=1` seed have no error bar — don't claim ~1-point differences between them.
* GPU latency at batch=1 understates the benefit of small models; CPU latency is the realistic deployment number.
