# MLX LoRA term study: can a 4B translation model match a larger one on TR↔EN technical terms?

A small, pre-registered experiment run on a single Apple Silicon desktop. We fine-tuned
TranslateGemma 4B with LoRA (MLX) on synthetic Turkish↔English sentences from the HPC / cloud / AI
domain and compared it with the same model untouched, with a 12B model plus glossary prompt, and with
a 26B general model.

**Short answer.** On the synthetic term test sets, 4B + LoRA beat both the untouched 4B (+19.2 TAcc
points) and the 12B + glossary system (+8.9 points). On a small set of real technical text it was on par
with 12B + glossary (+1.6, not significant). General Turkish output quality in the EN→TR direction
dropped slightly (COMET −0.99), which failed the pre-registered non-inferiority threshold. The speed
target (≥ 2× the 12B system) was narrowly missed (1.9×).

Treat these as preliminary results: one language pair, one domain, one machine, synthetic training
and test data, and a very small human check.

---

## 1. Question

Can a 4B translation model with a LoRA adapter match a much larger model on technical-term accuracy
for Turkish↔English HPC/AI text, without hurting general translation quality, and run faster?

## 2. Setup

| Item | Value |
|---|---|
| Hardware | Mac mini, Apple M4 Pro, 48 GB unified memory |
| Framework | MLX / `mlx-lm` 0.29.1 (full environment: see "Reproducing") |
| Base model | `mlx-community/translategemma-4b-it-8bit` (QLoRA on the 8-bit base) |
| LoRA | rank 8, scale 20, dropout 0.05, top 16 layers, prompt masked (loss on target only) |
| Optimisation | LR 5e-5, 50 warm-up steps then cosine; batch 2 × grad-accum 4 = effective 8; 600 optimizer steps; max length 512 |
| Seeds | 3 (s0, s1, s2); checkpoint = lowest validation loss per seed (not the last step) |
| Decoding | greedy (temperature 0), max 256 new tokens, same normalisation for all systems |

Pilot run (20 optimizer steps): 3.8 min, peak memory 6.19 GB, ~27 tok/s during training.

### Data

- **Glossary:** 238 terms in three classes: kept in English and takes Turkish suffixes (66), fixed
  Turkish translation (104), kept in English (68). 190 terms were "seen" in training, 48 were held out
  as "unseen" (split seed 0).
- **Training data:** 2,000 pairs = 1,500 synthetic term-bearing pairs + 500 general "replay" pairs
  (25%). Replay targets are the untouched 4B model's own translations (self-distillation), to limit
  forgetting. Validation: 150 term pairs + 50 replay pairs.
- **Generator:** all synthetic sentences (train, valid, tests, trap) were generated locally by
  `gemma4:26b-a4b` (Ollama). Test sets were generated in a separate session with a different prompt,
  different domains and different seeds. **Limitation:** the 26B baseline (B3) is the same model that
  wrote the data, so its scores on the synthetic sets may be inflated.
- **Leakage control:** unseen terms never appear in train/valid; test items with any 8-gram overlap
  with training, or MinHash Jaccard > 0.7, were dropped (test_seen lost 6 + 3 items). Test files were
  hashed into `MANIFEST.txt` before training.

| Set | Size | Content |
|---|---|---|
| test_seen | 200 sentences, 456 term occurrences | terms seen in training, new sentences |
| test_unseen | 100 sentences, 165 term occurrences | held-out terms only are scored |
| test_insan ("human") | 64 sentences, 127 term occurrences | real technical text, no references; **not published** |
| trap | 30 sentences | a term word used in its non-technical sense (over-generalisation check) |
| FLORES-200 devtest | 200 TR→EN + 200 EN→TR | general quality / forgetting check; not shipped (see below) |

### Systems

| Code | System |
|---|---|
| B0 | TranslateGemma 4B (MLX 8-bit), official prompt only |
| B1 | B0 + glossary injected into the prompt |
| B2 | TranslateGemma 12B (Ollama) + glossary in the prompt: the production-style reference system |
| B3 | Gemma 4 26B-A4B (Ollama) with the term list in the prompt ("prompt-only" ceiling) |
| L1 | B0 + LoRA (mean of 3 seeds) |
| L2 | L1 + glossary in the prompt |

The glossary in B1/B2/L2 is added to the prompt in the same way as the live system does it; it is not
a post-processing step.

### Metrics

- **TAcc (term accuracy):** share of expected term occurrences for which the output contains an
  accepted form (regex per glossary entry; Turkish suffixes after an apostrophe allowed).
- **Forbidden form rate:** share of outputs using a listed wrong form (e.g. a literal translation of
  "head node").
- **Trap rate:** share of trap sentences where a technical term was wrongly inserted.
- **chrF++** (`sacrebleu`) and **COMET-22** (`Unbabel/wmt22-comet-da`, CPU).
- Statistics: paired bootstrap (1,000 resamples, seed 12345, 95% CI); McNemar test at occurrence
  level for TAcc.

## 3. Pre-registered hypotheses and results

Hypotheses and thresholds were frozen on 2026-10-31, before training (`PREREG.md`, in Turkish).
L1 = mean of seeds s0–s2. CI = paired bootstrap 95%.

| Hypothesis | Measure | Result | Threshold | Verdict |
|---|---|---|---|---|
| H1 LoRA improves seen-term accuracy | test_seen TAcc, L1 − B0 | +19.2 [15.4, 23.3], McNemar p = 7e-26 | ≥ +15 and CI lower bound > 0 | ✓ |
| H2 4B + LoRA is as good as 12B + glossary | test_seen TAcc, L1 − B2 | +8.9 [5.8, 12.1] | ≥ −2 | ✓ |
| H3 general quality does not drop | FLORES ΔCOMET / ΔchrF++, L1 − B0 | tr-en +0.12 / +2.19 ✓ · en-tr −0.99 [−1.51, −0.35] / +0.01 ✗ | ≥ −0.5 / ≥ −1.0 | ✗ (en-tr COMET) |
| H4 no over-generalisation | trap rate | L1 1.1 vs B0 0.0 | ≤ B0 + 3 | ✓ |
| H5 (exploratory) generalises to unseen terms | test_unseen TAcc, L1 − B0 | +8.9 [1.9, 15.5] | ≥ +5 | ✓ |
| H6 speed gain | tok/s, L1 / B2 | 20.0 / 10.5 = 1.9× | ≥ 2× | ✗ (narrowly) |

Additional comparison on the real-text set (not a pre-registered hypothesis, reported separately as
the protocol requires): test_insan TAcc L1 − B0 = +8.7 [3.5, 14.6], p = 0.0018; L1 − B2 = +1.6
[−2.9, +6.6], p = 0.55 (not significant).

### 3.1 Term accuracy tables

**test_seen** (TAcc / forbidden-form rate / chrF++)

| System | TAcc | Forbidden | chrF++ |
|---|---|---|---|
| B0 | 77.0 | 6.9 | 67.6 |
| B1 | 87.1 | 7.8 | 72.8 |
| B2 | 87.3 | 13.4 | 73.9 |
| B3 | 92.5 | 0.0 | 82.9 |
| **L1** | **96.2 ± 0.1** | **0.3** | **80.5** |
| L2 | 98.0 ± 0.2 | 1.2 | 81.5 |

**test_unseen** (TAcc on held-out terms)

| System | TAcc |
|---|---|
| B0 | 80.6 |
| B1 | 84.8 |
| B2 | 84.8 |
| B3 | 98.8 |
| **L1** | **89.5 ± 2.9** |
| L2 | 92.7 ± 1.6 |

**test_insan** (64 sentences of real technical text, no reference translations; only numbers are
published, the sentences are not)

| System | TAcc |
|---|---|
| B0 | 81.9 |
| B1 | 88.2 |
| B2 | 89.0 |
| B3 | 87.4 |
| **L1** | **90.6 ± 0.8** |
| L2 | 90.8 ± 1.2 |

On real text L1 is above B3, unlike on the synthetic sets where B3 (the data generator) leads.

**trap** (rate of wrongly inserted technical terms, lower is better)

| B0 | B1 | B2 | B3 | L1 | L2 |
|---|---|---|---|---|---|
| 0.0 | 3.3 | 3.3 | 13.3 | 1.1 ± 1.9 | 4.4 ± 1.9 |

### 3.2 General quality (FLORES-200 devtest, 200 sentences per direction)

| System | chrF++ tr-en | chrF++ en-tr | COMET tr-en | COMET en-tr |
|---|---|---|---|---|
| B0 | 56.0 | 50.6 | 88.38 | 89.33 |
| B1 | 57.6 | 51.9 | 88.77 | 89.70 |
| B2 | 60.5 | 53.9 | 89.61 | 90.37 |
| B3 | 62.6 | 61.2 | 89.76 | 90.75 |
| L1 (mean of 3 seeds) | 58.2 | 50.6 | 88.50 | 88.34 |
| L2 (mean of 3 seeds) | 58.0 | 51.0 | per seed, see `results/comet/` | per seed, see `results/comet/` |

COMET per seed (L1 en-tr: 87.53 / 88.70 / 88.79) and per-segment scores are in `results/comet/`.

### 3.3 Latency

Median wall-clock time per sentence: **B2 2.25 s · L1 1.03 s · B3 0.92 s**.

H6 was measured as wall-clock tokens per second, including prompt processing. The engines and
tokenizers differ (B2/B3 run in Ollama with GGUF weights; B0/L1 run in MLX), so the speed comparison
includes engine differences. Quality comparisons are not affected by this.

## 4. Human evaluation (very small)

A blind pairwise sheet comparing L1 and B2 on 50 test_insan sentences was prepared; only
**5 items** were judged by one person (the author), shown as A/B with system identity hidden. After
un-blinding with the answer key:

| Item | Preferred |
|---|---|
| 1 | L1 |
| 2 | L1 |
| 3 | B2 |
| 4 | L1 |
| 5 | neither (both mistranslated the same phrase) |

This is **n = 5, one rater, TR→EN only**. All 50 prepared items are TR→EN, so this check says nothing
about the EN→TR quality loss behind the H3 failure. The rater's note: read fluently, both translations
look "correct"; differences only show up when reading for terms and meaning.

## 5. Limitations

- **Synthetic data, same generator as a baseline.** Training and the synthetic test sets were written
  by `gemma4:26b-a4b`, which is also baseline B3. The synthetic reference translations are model-generated.
  The real-text set (test_insan) is the more honest measure, and it is small (64 sentences) and
  unpublished.
- **H3 failed in one direction.** EN→TR general quality dropped by about 1 COMET point. The 25% replay
  ratio was probably not enough.
- **Small n.** 30–200 sentences per set; confidence intervals are wide.
- **Speed measurement** mixes model size with engine and tokenizer differences (MLX vs Ollama).
- **Human evaluation** is 5 items, one rater, one direction.
- **Scope:** one language pair, one domain, one machine, one model family. No claim that LoRA always
  helps; LoRA here teaches terminology and style, not facts.
- **Glossary decisions** were made by the experiment's default proposals and accepted without a
  line-by-line human review (`PREREG_EK.md`).
- **Integrity note:** the SHA-256 of `PREREG.md` matches `MANIFEST.txt`. The hash of
  `PREREG_EK.md` does not match the line recorded in `MANIFEST.txt`; the appendix carries a pilot note
  dated 17:28 on 2 Nov, which was most likely added after that hash was recorded. The appendix never
  changed any hypothesis or threshold.

## 6. Next steps

- Re-train with a higher replay ratio (e.g. above 25%) and re-test H3 in the EN→TR direction.
- A proper human evaluation in the EN→TR direction, with more items and more than one rater.
- Measure speed with the same engine for all systems.

## 7. Repository contents

```
PREREG.md            pre-registration, frozen 2026-10-31 (Turkish, unchanged)
PREREG_EK.md         appendix, 2026-11-02 (Turkish, unchanged): glossary decision, matcher fix, pilot note
MANIFEST.txt         SHA-256 of test sets, glossary, FLORES subsets and PREREG, written before training
data/                synthetic sets only (see below)
  train_term.jsonl, valid_term.jsonl          term-bearing synthetic pairs
  train_replay_src.jsonl, valid_replay_src.jsonl   general (term-free) replay source sentences
  mlx/train.jsonl, mlx/valid.jsonl            final MLX chat-format training data (incl. replay targets)
  test_seen.jsonl, test_unseen.jsonl, trap.jsonl
  build_report.json                           set sizes and leakage-filter counts
  flores_ids/                                 FLORES-200 devtest line IDs used (no text)
results/
  scores.json, score_tables.txt               all tables and paired comparisons
  comet/                                      COMET-22 system and segment scores (FLORES)
  outputs/                                    per-system outputs on test_seen, test_unseen, trap
scripts/             data generation, set building, training, translation, scoring
LICENSE              Apache-2.0
```

**Not included:** the real-text set (`test_insan`) and its outputs, the human-evaluation sheets, the
glossary file (`terimler.tsv`, `split_terms.json`) and the production glossary snapshot used by
B1/B2/L2, model weights and LoRA adapters, FLORES text, training logs, and the Python environment.
Because the glossary is not shipped, `score.py` and `gen_synth.py` cannot be re-run as-is; the term IDs
in the data files (`T001`…) refer to it.

### FLORES-200 (download yourself)

FLORES-200 is licensed CC BY-SA 4.0, so it is not redistributed here.

1. Download `flores200_dataset.tar.gz` from Meta's official location
   (`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`). The file we used had SHA-256
   `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`.
2. Use `devtest/tur_Latn.devtest` and `devtest/eng_Latn.devtest`.
3. `data/flores_ids/flores_<dir>_ids.txt` lists the 200 sentences per direction as `devtest_<N>`, where
   `N` is the 0-based line number in the devtest files.
4. Write them to `data/flores/flores_tr-en.jsonl` and `data/flores/flores_en-tr.jsonl` with the fields
   `id, dir, src_text, ref`.

### Reproducing

- Models: `mlx-community/translategemma-4b-it-8bit`, `Unbabel/wmt22-comet-da`, and via Ollama
  `translategemma:12b` and `gemma4:26b-a4b`. Gemma models are under the Gemma Terms of Use.
- Two fixes were applied to the 4B base for all 4B systems: `eos_token_id = [1, 106]` in
  `config.json` and `generation_config.json` (`scripts/eos_fix.py`), and a chat-template patch on the
  assistant branch only so that training and inference prompts are byte-identical
  (`scripts/template_fix.py`, `scripts/template_check.py`).
- Order: `gen_synth.py` → `build_sets.py` → `run_baselines.sh` (replay distillation, MLX data,
  baselines) → `run_train.sh` (s0–s2) → `run_eval.sh` (L1/L2, COMET, scores).
- Scripts expect to live in `scripts/` under the repository root. An optional thermal-gate script can
  be passed with `THERMAL_GATE=/path/to/script`. Comments and log messages in the scripts are in
  Turkish.

## License

Code, synthetic data and results: Apache-2.0, Copyright 2026 PeakOps. The synthetic data was produced
with a Gemma model; check the Gemma Terms of Use before using it to train other models.

Experiments were executed with AI coding agents under the author's design and review.
