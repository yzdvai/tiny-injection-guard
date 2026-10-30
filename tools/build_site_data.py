"""Build the data files the website reads, from the notebook's run-2 outputs and the ONNX parity check.

  web/demo/card.json               model-card numbers on the demo page
  web/architecture/arch-data.json  parameter counts, training curve, threshold curves and the quantization example

Usage (after tools/export_onnx.py --predictions results/run2/predictions.csv):
  python tools/build_site_data.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

LABELS = {  # results.json key -> (name on the page, tag)
    "electra-small": ("ELECTRA-small", "ours"), "deberta-v3-xsmall": ("DeBERTa-v3-xsmall", "ours"),
    "protectai-v2": ("ProtectAI DeBERTa v2", "open guard"), "piguard": ("PIGuard", "open guard"),
    "tfidf-logreg": ("TF-IDF + logistic regression", "baseline"),
}
ONNX_KEY = "electra-small-onnx-int8"


def rate(mask):
    return round(float(mask.mean()), 3) if len(mask) else None


def card_data(results, run1, parity, handmade, onnx_mb):
    met = {(m["model"], m["set"]): m for m in results["metrics"]}
    get = lambda k, s, f: None if met.get((k, s), {}).get(f) is None or np.isnan(met[(k, s)][f]) else round(met[(k, s)][f], 3)

    thr = results["models"]["electra-small"]["threshold"]
    flag = parity["prob_model_quantized.onnx"] >= thr
    y, hard, s = parity.label.astype(int), parity.hard.astype(str) == "True", parity.set
    onnx_row = {"key": ONNX_KEY, "name": "ELECTRA-small, ONNX int8", "kind": "this demo", "params_m": 13.5,
                "size_mb": round(onnx_mb, 1), "colab_cpu_ms": None, "threshold": round(thr, 3),
                "recall": {k: rate(flag[(s == k) & (y == 1)]) for k in ["deepset", "llmail", "handmade", "in_dist"]},
                "fpr": {"notinject": rate(flag[s == "notinject"]), "handmade_hard": rate(flag[(s == "handmade") & (y == 0) & hard]),
                        "in_dist": rate(flag[(s == "in_dist") & (y == 0)])}}
    rows = [onnx_row]
    for k, (name, kind) in LABELS.items():
        m = results["models"][k]
        rows.append({"key": k, "name": name, "kind": kind, "params_m": round(m["params"] / 1e6, 1), "size_mb": round(m["size_mb"], 1),
                     "colab_cpu_ms": round(m["latency_ms_median"], 1), "threshold": round(m["threshold"], 3),
                     "recall": {x: get(k, x, "recall") for x in ["deepset", "llmail", "handmade", "in_dist"]},
                     "fpr": {"notinject": get(k, "notinject", "fpr_benign"), "handmade_hard": get(k, "handmade", "fpr_hard"),
                             "in_dist": get(k, "in_dist", "fpr_benign")},
                     "f1_in_dist": get(k, "in_dist", "f1")})

    # run 1 -> run 2 for the shipped architecture (ELECTRA-small)
    r1 = {(m["model"], m["set"]): m for m in run1["metrics"]}
    hidden = set(handmade[handmade.category == "attack_hidden"].text)
    def hidden_recall(pred_csv, thr_):
        p = pd.read_csv(pred_csv, keep_default_na=False)
        h = p[(p.set == "handmade") & p.text.isin(hidden)]
        return round(float((h["prob_electra-small"].astype(float) >= thr_).mean()), 2)
    differ = int((flag != (parity["prob_electra-small"].astype(float) >= thr)).sum())
    return {"run": "run2", "shipped": ONNX_KEY, "rows": rows,
            "run1_vs_run2": {"notinject_fpr": {"run1": round(r1[("electra-small", "notinject")]["fpr_benign"], 3),
                                               "run2": round(met[("electra-small", "notinject")]["fpr_benign"], 3)},
                             "handmade_hidden_recall": {"run1": hidden_recall("results/run1/predictions.csv", run1["models"]["electra-small"]["threshold"]),
                                                        "run2": hidden_recall("results/run2/predictions.csv", thr)}},
            "test_sets": {"deepset": "deepset/prompt-injections (held-out source, 263 attacks)",
                          "llmail": "Microsoft LLMail-Inject (500 injected emails)", "handmade": "40 hand-written attacks",
                          "notinject": "339 harmless prompts with trigger words", "handmade_hard": "25 tricky harmless hand-written prompts"},
            "onnx_parity": {"texts": int(len(parity)), "verdicts_differ": differ}}


def arch_data(model, results, pred, fp32_mb, int8_mb, parity_text):
    c, P = model.config, dict(model.named_parameters())
    n = lambda *names: int(sum(P[k].numel() for k in names))
    L, E = "electra.encoder.layer.0.", "electra.embeddings."
    layer = {
        "attention_qkv": n(*[f"{L}attention.self.{x}.{y}" for x in ["query", "key", "value"] for y in ["weight", "bias"]]),
        "attention_out": n(f"{L}attention.output.dense.weight", f"{L}attention.output.dense.bias"),
        "attention_norm": n(f"{L}attention.output.LayerNorm.weight", f"{L}attention.output.LayerNorm.bias"),
        "ffn_in": n(f"{L}intermediate.dense.weight", f"{L}intermediate.dense.bias"),
        "ffn_out": n(f"{L}output.dense.weight", f"{L}output.dense.bias"),
        "ffn_norm": n(f"{L}output.LayerNorm.weight", f"{L}output.LayerNorm.bias"),
    }
    groups = [
        {"key": "word", "name": "Word embeddings", "detail": f"{c.vocab_size:,} tokens × {c.embedding_size}", "params": n(E + "word_embeddings.weight")},
        {"key": "position", "name": "Position embeddings", "detail": f"{c.max_position_embeddings} positions × {c.embedding_size}", "params": n(E + "position_embeddings.weight")},
        {"key": "segment", "name": "Segment embeddings", "detail": f"{c.type_vocab_size} segments × {c.embedding_size}", "params": n(E + "token_type_embeddings.weight")},
        {"key": "emb_norm", "name": "Embedding LayerNorm", "detail": "γ and β, 128 each", "params": n(E + "LayerNorm.weight", E + "LayerNorm.bias")},
        {"key": "project", "name": "Projection 128 → 256", "detail": "W (256×128) + b", "params": n("electra.embeddings_project.weight", "electra.embeddings_project.bias")},
        {"key": "attention", "name": "Self-attention (×12)", "detail": "Q, K, V, output projection + LayerNorm", "params": 12 * (layer["attention_qkv"] + layer["attention_out"] + layer["attention_norm"])},
        {"key": "ffn", "name": "Feed-forward (×12)", "detail": "256 → 1024 → 256 + LayerNorm", "params": 12 * (layer["ffn_in"] + layer["ffn_out"] + layer["ffn_norm"])},
        {"key": "head", "name": "Classification head", "detail": "dense 256×256 + output 256×2", "params": n("classifier.dense.weight", "classifier.dense.bias", "classifier.out_proj.weight", "classifier.out_proj.bias")},
    ]
    total = sum(g["params"] for g in groups)
    assert total == sum(p.numel() for p in model.parameters()), "parameter groups do not add up"

    hist = results["training"]["electra-small"]
    pr = pred["prob_electra-small"].astype(float)
    ts = np.round(np.linspace(0, 1, 201), 3)
    curve = lambda mask: [round(float((pr[mask] >= t).mean()), 4) for t in ts]
    W = P[L + "attention.self.query.weight"].detach().numpy()
    s = float(np.abs(W).max() / 127)
    w = W[0, :10].astype(float)
    q = np.clip(np.round(w / s), -127, 127).astype(int)
    return {
        "config": {"vocab": c.vocab_size, "embedding": c.embedding_size, "hidden": c.hidden_size, "layers": c.num_hidden_layers,
                   "heads": c.num_attention_heads, "head_dim": c.hidden_size // c.num_attention_heads, "ffn": c.intermediate_size,
                   "max_positions": c.max_position_embeddings, "segments": c.type_vocab_size, "activation": c.hidden_act,
                   "ln_eps": c.layer_norm_eps, "threshold": c.injection_threshold},
        "params": {"total": int(total), "groups": groups, "per_layer": layer},
        "compare": [{"name": "This model (ELECTRA-small)", "params": total},
                    {"name": "DeBERTa-v3-xsmall (our runner-up)", "params": results["models"]["deberta-v3-xsmall"]["params"]},
                    {"name": "BERT-base", "params": 109482240},
                    {"name": "ProtectAI / PIGuard (DeBERTa-v3-base)", "params": results["models"]["protectai-v2"]["params"]}],
        "training": {"evals": hist["evals"], "best_epoch": hist["best_epoch"], "stopped_early": hist["stopped_early"],
                     "minutes": round(hist["train_minutes"], 1), "lr": 1e-4, "batch": 32, "max_epochs": 3, "warmup": 0.06,
                     "weight_decay": 0.01, "max_len": 512, "patience": 3,
                     "train_size": results["set_sizes"]["train"], "val_size": results["set_sizes"]["val"]},
        "thresholds": {"t": ts.tolist(),
                       "recall_deepset": curve((pred.set == "deepset") & (pred.label == 1)),
                       "recall_handmade": curve((pred.set == "handmade") & (pred.label == 1)),
                       "recall_in_dist": curve((pred.set == "in_dist") & (pred.label == 1)),
                       "fpr_notinject": curve(pred.set == "notinject"),
                       "fpr_in_dist": curve((pred.set == "in_dist") & (pred.label == 0))},
        "quant": {"tensor": "layer 1 · attention · query weight", "shape": list(W.shape), "max_abs": float(np.abs(W).max()), "scale": s,
                  "weights": [round(x, 5) for x in w], "ints": q.tolist(), "dequant": [round(float(x * s), 5) for x in q],
                  "mean_abs_error_tensor": float(np.abs(np.clip(np.round(W / s), -127, 127) * s - W).mean())},
        "sizes": {"fp32_mb": round(fp32_mb, 1), "int8_mb": round(int8_mb, 1), "parity": parity_text},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yzdvai/tiny-injection-guard")
    ap.add_argument("--parity", default="build/onnx_parity.csv")
    ap.add_argument("--fp32", default="build/model.onnx")
    ap.add_argument("--int8", default="web/model/onnx/model_quantized.onnx")
    ap.add_argument("--card-out", default="web/demo/card.json")
    ap.add_argument("--arch-out", default="web/architecture/arch-data.json")
    a = ap.parse_args()

    results = json.load(open("results/run2/results.json"))
    run1 = json.load(open("results/run1/results.json"))
    pred = pd.read_csv("results/run2/predictions.csv", keep_default_na=False)
    pred["label"] = pred.label.astype(int)
    parity = pd.read_csv(a.parity, keep_default_na=False)
    handmade = pd.read_csv("handmade_test.csv")
    model = AutoModelForSequenceClassification.from_pretrained(a.model, dtype=torch.float32).eval()

    card = card_data(results, run1, parity, handmade, os.path.getsize(a.int8) / 1e6)
    p = card["onnx_parity"]
    parity_text = f"{p['texts'] - p['verdicts_differ']:,} of {p['texts']:,}"
    arch = arch_data(model, results, pred, os.path.getsize(a.fp32) / 1e6, os.path.getsize(a.int8) / 1e6, parity_text)
    json.dump(card, open(a.card_out, "w"), indent=1)
    json.dump(arch, open(a.arch_out, "w"))
    print(f"wrote {a.card_out} and {a.arch_out} ({arch['params']['total']:,} parameters; int8 agrees on {parity_text} test texts)")


if __name__ == "__main__":
    main()
