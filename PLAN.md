# Proposal: data mix, base models, comparators

Checked on the Hugging Face Hub on 2026-10-29. Status: **approved 2026-10-29**. Implemented in `injection_guard.ipynb`.

## Training pool (split into train / val / in-distribution test)

| Source | Licence | Use | Approx. rows |
|---|---|---|---|
| `reshabhs/SPML_Chatbot_Prompt_Injection` | MIT | user prompts: injection (capped ~4k) + benign (~3.5k); **system prompts as benign** (system-style formatting) | ~8k |
| `jackhhao/jailbreak-classification` | Apache-2.0 | jailbreaks + benign **role-play** prompts | 1.3k |
| `TrustAIRLab/in-the-wild-jailbreak-prompts` | MIT | real jailbreaks (`jailbreak_2023_12_25`) + **regular prompts from the same communities** as hard negatives (sample ~4k) | ~5.4k |
| `Lakera/gandalf_ignore_instructions` | MIT | injections (almost all say "ignore", so a shortcut risk) | 1k |
| `OpenAssistant/oasst2` | Apache-2.0 | first user turns in English: normal assistant prompts | ~5k sample |
| `Paul/XSTest` | CC-BY-4.0 | 250 **safe** prompts only ("how do I kill a Python process") | 250 |

Cleaning: dedupe exact and near-duplicates (char n-gram similarity), cluster near-duplicates, and split by cluster so no cluster crosses train/test. The split is 80/10/10, stratified.

## Test sets (never trained on)
1. **In-distribution test:** 10% of the pool, grouped by near-duplicate cluster.
2. **Held-out source: `deepset/prompt-injections`** (Apache-2.0, 662 rows, both labels). This set is fully held out. It is not in ProtectAI's training list, so the comparison on it is fair. Note: some rows are in German.
3. **Hard negatives: `leolee99/NotInject`** (MIT, 339 benign prompts containing trigger words such as "ignore"). Reported as false-positive rate (FPR) only.
4. **Indirect / email (recall only): `microsoft/llmail-inject-challenge`** (MIT), a 500-row sample of injected emails. It has no benign side, so it only gives recall.
5. **Hand-made set (~80):** I draft it, you edit and add your own attacks. Saved as `handmade_test.csv`.

Skipped: `xTRam1/safe-guard…`, `rubend18/…` and `jayavibhav/…` have no stated licence. `hackaprompt` and `wildjailbreak` need a gated sign-up. `qualifire` benchmark is gated and non-commercial.

## Base models (both exist, MIT/Apache)
- `microsoft/deberta-v3-xsmall`: 71M params, of which 22M are the backbone (6 layers × 384). The main candidate.
- `google/electra-small-discriminator`: 14M params, about 55 MB. The "ultra-tiny" option.

## Comparators
- `protectai/deberta-v3-base-prompt-injection-v2`: open, Apache-2.0, 184M params. **Caveat:** it was trained on `jackhhao` and XSTest, so its in-distribution score is inflated. Compare on deepset, NotInject and the hand-made set.
- `leolee99/PIGuard` (formerly InjecGuard): open, MIT, 184M params, built specifically to reduce over-blocking. **Caveat:** the same authors made NotInject, so it is tuned toward that set. Its card does not list its training data, so treat its deepset score with caution.
- Skipped: `meta-llama/Llama-Prompt-Guard-2-22M` and the other Prompt Guard models. They are gated behind a form that asks for personal details (full name, date of birth), and we chose not to submit it. `deepset/deberta-v3-base-injection` and `fmops/distilbert-prompt-injection` are also skipped, because they were trained on our held-out deepset set.
- Baseline: TF-IDF + logistic regression.

## Shortcut checks
Report how well these predict the label: length alone, presence of "ignore"/"previous instructions", and source identity (a classifier that guesses which dataset a text came from).

## Run 2 changes (2026-10-30, after run 1)
Run 1 showed shortcut learning: F1 0.95 in-distribution, but 0.38 on deepset and 0.03 on the email set.
- **SPML injection rows dropped.** SPML labels off-topic requests and bot personas as "injection" relative to one specific chatbot. Out of context they aren't attacks, and its persona rows conflicted with its system prompts, which we label benign.
- **Hidden-injection augmentation:**
  - 3,000 OpenAssistant answer paragraphs with the opening of a real training attack inserted (`injection`), each paired with the same paragraph unchanged (`benign`).
  - 1,500 paragraphs with a harmless trigger-word sentence inserted (`benign`, hard negative).
  - A matching smaller set for validation.
- The deepset, NotInject, LLMail and hand-made test sets are unchanged. The in-distribution test set only loses its SPML "injections".
- PIGuard is now loaded without any remote-code prompt.
- Found while checking PIGuard: its paper lists `deepset/prompt-injections` (546 rows, the train split) in its training data, so its deepset score is not held out.
