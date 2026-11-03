# Tiny Injection Guard: beginner overview

Documentation contribution by [Yahya Ali Majrashi (@iYm10)](https://github.com/iYm10), prepared with Codex assistance. Reviewed against the repository on 3 November 2026. This credit covers this overview; it does not assign authorship of earlier training or implementation work.

## The project in 30 seconds

Tiny Injection Guard is a small Transformer classifier that predicts whether text is benign or a prompt injection/jailbreak. It can check a prompt, email, document or web text before an LLM uses it. The shipped ELECTRA-small model has about 13.5 million learned parameters. Its compact int8 ONNX version runs in a browser after download. It is a capstone prototype: portable and fast, but it misses many unfamiliar attacks.

## How a prediction works

Text -> tokenizer -> ELECTRA-small -> two class scores -> injection probability -> threshold decision.

- A tokenizer turns text into numbered pieces called tokens.
- The Transformer learns contextual patterns; it is a classifier, not a chatbot generating an answer.
- The shipped decision threshold is 0.945: scores at or above it are flagged.
- A benign prediction does not prove that text is safe. A score is not a guaranteed probability of real-world harm.
- The model reads at most 512 tokens and is English only. The website has English and Arabic interface text; that does not make the model bilingual.

## What was built

| Artifact | Purpose |
| --- | --- |
| injection_guard.ipynb | Colab pipeline: data, cleaning, splits, shortcut checks, baseline, comparators, fine-tuning, threshold selection, tests, mistakes and quantization |
| handmade_test.csv | 80 examples: 40 attacks and 40 harmless texts, including difficult examples |
| PLAN.md | Data/model proposal approved 29 October, plus the run-2 changes |
| results/run1 and results/run2 | Saved metrics, predictions, mistakes and plots; run 2 adds a matched comparison |
| web/ | Landing page, browser demo, interactive architecture explorer, tokenizer and ONNX assets |
| tools/ | Export/check ONNX, rebuild site data, generate showcase QR code |
| app/ | FastAPI/PyTorch server demo kept as a backup |
| showcase/ | QR artwork for presenting the project |

The merged [PR #1](https://github.com/yzdvai/tiny-injection-guard/pull/1) synced the website on 3 November: v1.2 design, English/Arabic interface, button-triggered model download, and current site links.

## Data and experiment design

Training sources: SPML benign prompts/system prompts, jackhhao jailbreak classification, TrustAIRLab in-the-wild jailbreaks and regular prompts, Lakera Gandalf, OpenAssistant and safe XSTest prompts. The repository records their licences; sources with no stated licence or gated access were skipped.

Cleaning removes conflicting labels, exact duplicates and near-duplicates. Similarity uses word 1-3-gram TF-IDF: near-duplicate cutoff 0.8, grouping cutoff 0.5. Training-pool overlap with held-out tests is removed. Related examples stay in the same side of an approximately 80/10/10 train/validation/test split.

Train means learn weights; validation means select settings; test means estimate performance. The notebook selects the model and threshold on validation within each run. However, test findings informed run-2 design, so these reused tests are not a fresh final evaluation.

Models compared: TF-IDF + logistic regression; fine-tuned ELECTRA-small and DeBERTa-v3-xsmall; existing ProtectAI v2 and PIGuard. ELECTRA uses AdamW, learning rate 1e-4, batch 32, at most 3 epochs and early stopping. Threshold tuning maximizes validation F1 subject to validation false-positive rate at most 1%. Existing guards use their shipped 0.5 threshold, so operating points differ.

## Why run 2 happened

Run 1 performed much better on familiar-source data than unfamiliar-source attacks. Shortcut checks showed dataset style and text length could reveal labels. Run 2 dropped SPML injection labels that depended on a particular chatbot's rules, while retaining its benign texts. It added 3,000 attack-inserted paragraphs paired with clean versions, plus 1,500 harmless trigger-word paragraphs to training, with separate smaller validation augmentation.

For ELECTRA, the matched comparison shows:

| Measure | Run 1 | Run 2 |
| --- | ---: | ---: |
| NotInject false-positive rate | 16.8% | 2.4% |
| Same-source attack recall | 84.1% | 70.9% |
| Hand-made attack recall | 55.0% | 27.5% |
| deepset attack recall | 16.3% | 17.9% |
| Injected-email attack recall | 0.4% | 3.2% |

Fewer false alarms did not solve generalisation. Inserting attack openings into paragraphs did not teach reliable detection of realistic hidden instructions.

## Run-2 results in plain language

Recall means the share of attacks caught. False-positive rate (FPR) means the share of harmless texts flagged. F1 balances precision and recall; accuracy alone can hide missed attacks when harmless examples dominate.

| Test | Size | ELECTRA attack recall | ELECTRA harmless FPR |
| --- | ---: | ---: | ---: |
| Same training sources, held-out examples | 1,599 | 70.9% | 0.85% |
| deepset, unseen training source | 662 | 17.9% | 0.75% |
| Hand-made | 80 | 27.5% | 17.5% |
| LLMail emails | 703 (500 attacks, 203 benign) | 3.2% | 0% |
| NotInject trigger-word prompts | 339 benign | Not applicable | 2.36% |

Do not confuse the hand-made overall benign FPR (17.5%) with the difficult-benign subset FPR (28%). NotInject is a different test set. The README's 95.8% accuracy applies only to the same-source test; it is not an overall attack-blocking rate.

## Size, speed and browser export

ELECTRA has about 13.5M parameters versus about 184M for each larger guard, roughly 14 times fewer. The saved CPU benchmark reports about 65 ms median for ELECTRA and 274-277 ms for those guards (Colab CPU, two threads, batch one). This is about four times faster in that benchmark, not a guaranteed browser latency.

The full-precision model is about 54 MB. The notebook's PyTorch dynamic-int8 experiment reports about 25.7 MB and 21.1 ms median. The separate browser ONNX export is about 14 MB. These are different artifacts; do not mix their size/speed numbers. The README reports browser ONNX/PyTorch verdict agreement on 3,369 of 3,383 test texts, not perfect equivalence.

## Limits and fair comparisons

- Unfamiliar attacks, especially hidden email/document instructions, are often missed.
- Trigger words can still cause false alarms; examples in the notebook include harmless corrections and discussion of injection attacks.
- TF-IDF remains competitive on the held-out attack sets; a Transformer is not automatically better.
- ProtectAI trained on jackhhao and XSTest; PIGuard trained on deepset and its authors made NotInject. Dataset overlap limits comparisons.
- Shared CPU timings vary. The 512-token limit can truncate long text. Reused tests can make final estimates optimistic.
- This is one possible defence layer, not evidence of a production-ready security boundary.

## A short presentation script

We built a tiny classifier to flag attempts to redirect an AI. We cleaned licensed datasets, kept related examples together, trained two small models and compared them with a simple baseline and larger guards. ELECTRA was selected on validation and exported to run locally in a browser. Our second run greatly reduced false alarms on NotInject, but still missed most unfamiliar and email attacks. The main achievement is a reproducible, portable demo and an honest lesson: good familiar-source scores do not guarantee real-world protection.

## Sources

- [README](README.md): current project scope, deployment, results and limitations.
- [Executed notebook](injection_guard.ipynb): implementation and run-2 outputs.
- [Plan](PLAN.md): original choices and run-2 rationale; some original proposal details were superseded by implementation.
- [Matched run comparison](results/run2/comparison_vs_run1.csv).
- [Run-2 metrics](results/run2/metrics.csv).
- [Merged site sync](https://github.com/yzdvai/tiny-injection-guard/pull/1).

This overview summarizes recorded repository evidence. It does not reconstruct off-repository conversations or undocumented individual roles, and no training or benchmark was rerun for this documentation update.
