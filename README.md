# Tiny Injection Guard

A 13.5M-parameter Transformer that flags **prompt injection and jailbreak** text (`benign` vs `injection`) before it reaches an LLM.
The text can be typed by a user, or come from an email, document or web page. The model is small enough to run in a web browser, with
no server: it is 14× smaller than the leading open-source guards and about 4× faster on the same CPU.

| | Link |
|---|---|
| Landing page (with QR code) | https://www.yazeed.quest/injection-guard/ |
| Live demo | https://www.yazeed.quest/injection-guard/demo/ |
| Interactive architecture walkthrough | https://www.yazeed.quest/injection-guard/architecture/ |
| Model on Hugging Face (Apache-2.0) | https://huggingface.co/yzdvai/tiny-injection-guard |

*Capstone project, AI & Data Science, 2026.*

## Repository layout

```
injection_guard.ipynb      the whole pipeline (Colab): data, dedup and splits, shortcut checks, baseline, existing guards,
                           fine-tuning, thresholds, evaluation, mistakes, int8, push to Hub. Saved with its run-2 outputs.
handmade_test.csv          80 hand-written test texts (attacks, paraphrased and hidden attacks, normal and tricky harmless texts)
PLAN.md                    the agreed data mix and model choices, and what changed in run 2
results/
  run1/                    first Colab run: executed notebook, metrics, predictions, mistakes, plots
  run2/                    second run (the shipped model): the same files plus comparison_vs_run1.csv
web/                       the live site (served at www.yazeed.quest/injection-guard/)
  index.html               landing page: introduction, QR code, paper-style architecture figure
  demo/                    live demo: the int8 ONNX model runs in the browser with transformers.js
  architecture/            interactive walkthrough: real attention maps, vectors, parameters, training, maths
  model/                   tokenizer, config and the int8 ONNX models (the pages load model_explain_quantized.onnx
                           only after a button click, see model-gate.js; model_quantized.onnx is for the tools' checks)
  glass.js, style.css      shared look and the English/Arabic switch
tools/
  export_onnx.py           PyTorch model → ONNX → int8, plus the attention-revealing model; checks int8 against PyTorch
  build_site_data.py       builds web/demo/card.json and web/architecture/arch-data.json from results/run2
  make_qr.py               the styled QR code (SVG + 2048 px PNG)
app/                       server version of the demo (FastAPI + PyTorch), kept as an offline backup
showcase/                  the QR code as PNG and SVG, for slides and printing
```

## How to reproduce

**1. Train and evaluate (Google Colab, free T4 GPU, about 60 minutes)**
1. In Google Drive, create `My Drive/injection-guard/` and upload `handmade_test.csv` into it.
2. Open `injection_guard.ipynb` in Colab, then choose Runtime → Change runtime type → **T4 GPU**.
3. Choose Runtime → **Run all**, and allow Drive access when asked.

Splits, cached guard predictions and models are saved to Drive and reloaded on later runs. To start fresh, set `RETRAIN = True`.
To publish the model, section 12 explains how to add an `HF_TOKEN` Colab secret and set `PUSH_TO_HUB = True`.

**2. Export for the browser and rebuild the site data (local, Python 3.12+)**
```bash
pip install -r tools/requirements.txt
python tools/export_onnx.py --predictions results/run2/predictions.csv
python tools/build_site_data.py
```

**3. Run the site locally**
```bash
cd web && python -m http.server 8000
```
Then open http://localhost:8000/. To deploy, copy `web/` into the `yazeed.quest` repository as `injection-guard/`. Links to `/assets/` and `/apps/` point at the main site.

**Offline backup (server version):** `cd app && pip install torch transformers fastapi uvicorn && uvicorn server:app --port 7860`. Here `requirements.txt` pins Linux CPU builds for a Docker deployment.

## Data

All sources were checked on the Hugging Face Hub on 2026-10-29.

| Dataset | Licence | Role |
|---|---|---|
| [reshabhs/SPML_Chatbot_Prompt_Injection](https://huggingface.co/datasets/reshabhs/SPML_Chatbot_Prompt_Injection) | MIT | Training: benign user prompts, and system prompts as tricky benign texts. Its "injection" rows depend on one specific chatbot's rules, so they were dropped in run 2. |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Apache-2.0 | Training: jailbreaks, and benign role-play prompts |
| [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts) | MIT | Training: real jailbreaks, and regular prompts from the same communities |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | Training: injections |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | Training: normal prompts; in run 2, also assistant answers used as paragraphs for augmentation |
| [Paul/XSTest](https://huggingface.co/datasets/Paul/XSTest) | CC-BY-4.0 | Training: safe prompts that sound risky |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | **Test only:** a source never used in training |
| [leolee99/NotInject](https://huggingface.co/datasets/leolee99/NotInject) | MIT | **Test only:** 339 harmless prompts full of trigger words |
| [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | MIT | **Test only:** 500 injected emails (phase 2) and 203 normal emails |
| `handmade_test.csv` | ours | **Test only:** hand-made set |

**Not used:**
- `xTRam1/safe-guard-prompt-injection`, `rubend18/ChatGPT-Jailbreak-Prompts` and `jayavibhav/prompt-injection` state no licence.
- `hackaprompt`, `allenai/wildjailbreak` and `qualifire/prompt-injections-benchmark` are gated.

**Leakage controls:**
- Exact duplicates are removed, after lower-casing and stripping punctuation.
- Near-duplicates (cosine ≥ 0.8 on word 1–3-gram TF-IDF) are collapsed to one copy.
- Pool texts that are near-duplicates of any test text are removed.
- Related texts (cosine ≥ 0.5) stay on the same side of the 80/10/10 split.

**Run-2 augmentation** (for instructions hidden inside documents):
- 3,000 OpenAssistant answers with the opening of a training attack inserted, labelled `injection`, each paired with the same answer unchanged, labelled `benign`.
- 1,500 answers with a harmless trigger-word sentence inserted, labelled `benign`.

## Models

- **Fine-tuned:** [google/electra-small-discriminator](https://huggingface.co/google/electra-small-discriminator) (the shipped model) and [microsoft/deberta-v3-xsmall](https://huggingface.co/microsoft/deberta-v3-xsmall).
  - Training uses AdamW (learning rate 1e-4 for ELECTRA), batch size 32, up to 3 epochs, and early stopping on validation loss (patience 3 checks).
  - The threshold (0.945) is tuned on validation data for at most 1% false positives.
  - The winner is picked on validation only.
- **Baseline:** TF-IDF (word 1–2-grams) + logistic regression.
- **Existing guards** (default 0.5 threshold):
  - [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) (Apache-2.0). It was trained on jackhhao and XSTest.
  - [leolee99/PIGuard](https://huggingface.co/leolee99/PIGuard) (MIT). It was trained on deepset; its authors also made NotInject. Its architecture is rebuilt in the notebook so no remote code runs.
- **Not used:**
  - Meta Prompt Guard is gated behind a form asking for personal details.
  - `deepset/deberta-v3-base-injection` and `fmops/distilbert-prompt-injection` were trained on our deepset test set.

## Results (run 2, shipped model)

**Recall** is the share of attacks flagged; **FPR** (false-positive rate) is the share of harmless texts flagged. "Same sources" means
held-out texts from the training datasets.

| Model | Params | Download | CPU ms | Recall: same sources | FPR: same sources | Recall: deepset | Recall: emails | Recall: hand-made | FPR: NotInject | FPR: tricky |
|---|---|---|---|---|---|---|---|---|---|---|
| **ELECTRA-small (ours)** | 13.5M | 54 MB (14 MB int8) | 65 | 0.71 | 0.009 | 0.18 | 0.03 | 0.28 | **0.024** | 0.28 |
| DeBERTa-v3-xsmall (ours) | 70.8M | 283 MB | 105 | 0.66 | 0.006 | 0.16 | 0.04 | 0.40 | 0.018 | 0.20 |
| ProtectAI DeBERTa v2 | 184M | 738 MB | 277 | 0.86 | 0.109 | 0.41 | 0.41 | 0.93 | 0.434 | 0.32 |
| PIGuard | 184M | 738 MB | 274 | 0.92 | 0.087 | 0.86* | 0.47 | 0.83 | 0.115 | 0.12 |
| TF-IDF + logistic regression | 0.2M | 10 MB | 0.7 | 0.75 | 0.011 | 0.12 | 0.03 | 0.43 | 0.009 | 0.08 |

\*PIGuard was trained on deepset. CPU time is the median per message on a Colab CPU (2 threads, batch size 1); it varies between sessions.
On "same sources" the shipped model's accuracy is 95.8%. The int8 ONNX version used on the website gives the same verdict as PyTorch on
3,369 of 3,383 test texts.

**What we found:**
- **Small, fast and rarely in the way.** At 1/14 of the parameters, it wrongly blocks about 2% of tricky harmless prompts, against 12% (PIGuard) and 43% (ProtectAI).
- **Run 1 learned dataset style.** Its winner had F1 0.95 on same-source text, but 0.38 on deepset and 0.03 on the emails. The shortcut checks back this up:
  - a bag-of-words model can tell which dataset a text came from 82–86% of the time;
  - text length alone separates attacks from harmless texts in the held-out sets (AUC 0.81–0.87).
- **Run 2 cut false alarms but lost recall.** We dropped SPML's context-dependent "injections" and added hidden-injection augmentation. False alarms on NotInject fell from 17% to 2%, but the model became more conservative: recall fell on the hand-made and same-source sets and barely moved on deepset and the emails. Pasting jailbreak openings into paragraphs did not teach the model to spot realistic hidden instructions.
- **The baseline is close.** A TF-IDF baseline does about as well as the fine-tuned model on the held-out attack sets. The big guards, trained on 15–20 datasets, catch far more unfamiliar attacks.

## Limitations

- It misses most attacks written in styles it never saw, especially instructions hidden inside emails or documents. It is meant as a first line of defence, not the only one.
- It is English only and reads at most 512 tokens.
- The shipped model was chosen after seeing two runs' test results, so its numbers are slightly optimistic.
- Speed numbers come from Colab's shared CPU, which varies between sessions.
