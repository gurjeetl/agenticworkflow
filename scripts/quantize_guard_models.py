"""Build INT8-quantized ONNX copies of the LLM Guard *blocking* classifiers.

Why: on CPU the bundled FP32 models (PyTorch or the FP32 ONNX llm-guard ships)
dominate the input/output guard latency. Dynamic INT8 quantization shrinks the
weights ~4x and typically gives a 2-3x CPU speedup, at some accuracy cost — so
this is paired with a benchmark that checks detection parity (see
``security/bench_guard.py``).

Scope: only the three BLOCKING ML classifiers that gate every request —
PromptInjection, Toxicity, BanTopics. The sanitizing scanners (Anonymize /
Secrets / Sensitive) are Presidio/regex-based and left on their defaults.

Each target already publishes an FP32 ONNX on the Hub, so we download that
(plus its config.json) and run onnxruntime dynamic quantization on it — no
PyTorch→ONNX export step. Output layout (pointed at by ``LLM_GUARD_ONNX_QUANTIZED=1``):

    models/guard_onnx/<key>/
        config.json
        model.onnx            (FP32, downloaded)
        model_quantized.onnx  (INT8, produced here)

Run once (after `pip install optimum[onnxruntime]`):

    python scripts/quantize_guard_models.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models" / "guard_onnx"

# (key, tokenizer/config repo == Model.path, onnx repo == Model.onnx_path,
#  onnx_subfolder, onnx_filename) — kept in sync with the llm-guard 0.3.16
# defaults wired in security/llm_guard.py.
MODELS = [
    (
        "prompt_injection",
        "protectai/deberta-v3-base-prompt-injection-v2",
        "ProtectAI/deberta-v3-base-prompt-injection-v2",
        "onnx",
        "model.onnx",
    ),
    (
        "toxicity",
        "unitary/unbiased-toxic-roberta",
        "ProtectAI/unbiased-toxic-roberta-onnx",
        "",
        "model.onnx",
    ),
    (
        "ban_topics",
        "MoritzLaurer/roberta-base-zeroshot-v2.0-c",
        "protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx",
        "",
        "model.onnx",
    ),
]


def _download(repo: str, filename: str, subfolder: str, dest: Path) -> Path | None:
    """Best-effort hub download; returns the local copy path or None if absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        src = hf_hub_download(repo, filename, subfolder=subfolder or None)
    except EntryNotFoundError:
        return None
    local = dest / filename
    shutil.copy(src, local)
    return local


def build() -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    OUT.mkdir(parents=True, exist_ok=True)
    for key, repo, onnx_repo, sub, fname in MODELS:
        dest = OUT / key
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key} ===")

        # optimum needs config.json sitting next to the .onnx to rebuild the model.
        cfg = _download(onnx_repo, "config.json", sub, dest)
        if cfg is None:  # some onnx repos keep config at the root
            cfg = _download(onnx_repo, "config.json", "", dest)
        if cfg is None:
            print(f"  ! no config.json in {onnx_repo} — skipping {key}")
            continue

        fp32 = _download(onnx_repo, fname, sub, dest)
        if fp32 is None:
            print(f"  ! no {fname} in {onnx_repo} — skipping {key}")
            continue
        # Large models may store weights in an external-data sidecar.
        _download(onnx_repo, fname + "_data", sub, dest)
        _download(onnx_repo, fname + ".data", sub, dest)

        int8 = dest / "model_quantized.onnx"
        # Dynamic quantization: per-request activation ranges computed on the fly,
        # so no calibration dataset needed — weights to INT8, activations dynamic.
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

        fp32_mb = fp32.stat().st_size / 1e6
        int8_mb = int8.stat().st_size / 1e6
        print(f"  fp32 {fp32_mb:6.1f} MB  ->  int8 {int8_mb:6.1f} MB  ({int8_mb / fp32_mb:.0%})")

    print(f"\nDone. Quantized models in {OUT}")
    print("Enable with:  LLM_GUARD_ONNX_QUANTIZED=1  (and run security/bench_guard.py to verify)")


if __name__ == "__main__":
    build()
