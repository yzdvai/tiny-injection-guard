---
title: Tiny Injection Guard
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
models:
  - yzdvai/tiny-injection-guard
short_description: A 14M-parameter prompt-injection detector on CPU
---

# Tiny Injection Guard

Live demo of [`yzdvai/tiny-injection-guard`](https://huggingface.co/yzdvai/tiny-injection-guard), a fine-tuned ELECTRA-small (14M parameters) that classifies text as `benign` or `injection`. Inputs can be prompts typed by users, or text taken from documents, emails or web pages.

- The model runs on the Space's CPU, quantized to int8. Each message is checked one at a time and the latency is measured on this server.
- The decision threshold (0.945) was tuned on validation data so that at most 1% of benign texts are flagged.
- The demo calls no LLM; it only runs the guard.

The page shows how it compares with two open guards (ProtectAI DeBERTa v2 and PIGuard) on held-out test sets, along with its known limitations. The main limitation is that it misses most instructions hidden inside emails or documents.

## Files
- `server.py`: FastAPI app. `POST /api/classify` takes `{"text": "..."}`; `GET /api/info` returns model info and the comparison data.
- `static/`: the front end (HTML, CSS, JS). The glass refraction effect uses an SVG displacement filter, which only Chromium browsers render; other browsers fall back to frosted glass.
- `card.json`: the comparison table, taken from the training notebook's run-2 results.

Environment variables: `MODEL_ID` (default `yzdvai/tiny-injection-guard`), `INT8` (`1` or `0`), `THREADS` (default `2`).
