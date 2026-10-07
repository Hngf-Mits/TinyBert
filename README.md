# Knowledge Distillation: How Small Can a Neural Network Become?

This project investigates **knowledge distillation (KD)**: training a small *student* model to learn from a larger, more powerful *teacher* model, and measuring the trade-off between **model size, computational efficiency, and prediction performance**.

## Models

| Model | Role | Layers | Hidden size | Parameters | Weights (FP32) |
|---|---|---|---|---|---|
| `bert-base-uncased` | Teacher | 12 | 768 | **109.48M** | 417.7 MB |
| `TinyBERT_General_4L_312D` | Student | 4 | 312 | **14.35M** | 54.7 MB |

The student has about **7.6x fewer parameters** (86.9% smaller). Note that much of the student's size is its word-embedding table (about 9.5M of 14.35M parameters, versus about 23.4M of 109.48M for the teacher), so the transformer layers themselves are reduced by a much larger factor, from roughly 86M to roughly 4.8M parameters.

The student is trained with a combined loss: cross-entropy against the ground-truth labels plus KL divergence against the teacher's softened output distribution (temperature T = 3, alpha = 0.5).

## Experiments

The repository contains two experiments on sentiment classification.

### 1. SST-2 (sentence-level sentiment, 2 classes): `notebooka2c99cbc80.ipynb`

The teacher is the pretrained checkpoint `textattack/bert-base-uncased-SST-2`. The student is distilled for 3 epochs.

| Model | Params | Accuracy | F1 | GPU latency (bs=1) | Throughput |
|---|---|---|---|---|---|
| Teacher (BERT-base) | 109.48M | 92.43% | 92.63 | 10.11 ms | 364.6 samples/s |
| Student (TinyBERT-KD) | 14.35M | 90.14% | 90.32 | 3.83 ms | 2132.2 samples/s |

The student keeps 97.5% of the teacher's accuracy with 86.9% fewer parameters and weights, 2.64x lower latency, and about 5.8x higher throughput on a Tesla T4. This notebook has no student trained without KD, so the specific contribution of distillation cannot be isolated here.

### 2. ABSA, SemEval-2014 Restaurants (aspect-based sentiment, 3 classes): `tinybert-on-absa.ipynb`

A harder task with far less data (about 2.7k training samples). The teacher is fine-tuned inside the notebook. The test set is carved from the training data by review sentence, because the dataset's official test split is unlabeled. Results are mean over 3 seeds for the students.

| Model | Params | Accuracy | Macro-F1 | GPU latency | CPU latency |
|---|---|---|---|---|---|
| Teacher (BERT-base) | 109.48M | 76.70% | 65.93 | 7.58 ms | 66.38 ms |
| Student, CE only | 14.35M | 74.91 +/- 1.25 | 66.56 | 3.02 ms | 7.85 ms |
| Student, KD | 14.35M | 73.24 +/- 1.15 | 62.27 | 2.95 ms | 7.88 ms |

The student has the same 14.35M parameters in both settings, and is about 8.4x faster on CPU and about 2.5x faster on GPU. In this setup **KD did not improve over CE-only training**. Likely reasons are a teacher that is only modestly stronger than the student, a small dataset, and high variance across seeds.

## Shared pipeline

Both experiments use the same tokenizer and vocabulary for teacher and student, and report accuracy, F1, parameter count, model size, single-sample latency, throughput, and GPU memory. The ABSA notebook adds a sentence-grouped train/dev/test split to avoid data leakage, best-epoch selection on the dev set, multiple seeds, and CPU latency measurement.

> **Note:** SST-2 accuracy is measured on the 872-sample validation set and ABSA accuracy on a 558-sample test set, so differences smaller than about 2 percentage points may be noise. The ABSA scores are not directly comparable to published SemEval results. Embedding parameter counts are computed from vocabulary size x hidden size (30,522 x 768 and 30,522 x 312).
