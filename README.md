# LLM Agent Reliability

Master's thesis: Building and Evaluating LLM-Powered AI Agents for Business
Process Automation — Failure Modes, Reliability and Human Trust

Gagandeep Singh · MSc Data Science, AI and Digital Business
GISMA University of Applied Sciences, Berlin

## Overview

This project evaluates whether instruction-tuned language models can reliably
classify and route customer support requests in an automated business workflow.
The focus is not raw accuracy but reliability: how the system fails, whether it
knows when it is uncertain, and where the boundary should sit between automated
action and human handoff.

## Dataset

BANKING77 — 13,083 online banking customer queries across 77 intent classes.
Casanueva, Temčinas, Gerz, Henderson & Vulić (2020), *Efficient Intent Detection
with Dual Sentence Encoders*, 2nd Workshop on NLP for ConvAI, ACL. CC-BY-4.0.

An earlier candidate dataset was rejected after a supervised ceiling test: a
model trained on it scored 0.2021 against a majority-class baseline of 0.2068,
with normalised mutual information of 0.0011 between text and label.

## Results

Evaluated on 300 held-out tickets, identical for every approach.

| Approach | Accuracy | Macro F1 | Latency (s) |
|---|---|---|---|
| Supervised baseline (TF-IDF + logistic regression) | 0.8933 | 0.8841 | 0.0017 |
| Zero-shot FLAN-T5-small | 0.2300 | 0.2366 | 0.0720 |
| Zero-shot FLAN-T5-base | 0.1400 | 0.1481 | 0.1287 |
| Zero-shot FLAN-T5-large | 0.0933 | 0.0913 | 0.3760 |
| Hybrid (supervised + small) | 0.8967 | 0.8838 | 0.0364 |
| Hybrid (supervised + base) | 0.8800 | 0.8718 | 0.0599 |
| Hybrid (supervised + large) | 0.8933 | 0.8853 | 0.1836 |

Chance baseline for 77 classes: 0.0130.

## Key findings

- Zero-shot models performed well above chance but far below the supervised
  baseline, and accuracy declined with model size — consistent with surface
  form competition in likelihood-based label scoring.
- Across 900 zero-shot predictions, none was both wrong and confident. The
  supervised baseline produced four such errors.
- The hybrid router consulted the language model on roughly 140 of 300 tickets
  but overrode the supervised prediction only 3–4% of the time, with no
  measurable accuracy gain at 20–100× the latency.
- Robustness testing on the supervised baseline: accuracy fell 9.3 points under
  injected typographical errors and 35.3 points under truncation.

## Running the code

Open `llm_agent_evaluation.ipynb` in Google Colab with a GPU runtime
(Runtime → Change runtime type → T4 GPU) and run all cells. The dataset is
loaded directly from its source URL. Full evaluation takes about six minutes.

## Limitations

Trust questionnaire responses are simulated to demonstrate the measurement
instrument; no human study was conducted. Robustness testing covers the
supervised baseline only. Ticket priority is assigned heuristically, as
BANKING77 contains no priority field. A custom stratified split was used rather
than the published train/test split.
