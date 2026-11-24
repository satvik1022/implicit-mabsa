"""Evaluation of the three ImplicitMABSA streams with a side-by-side benchmark table.

Metrics (macro-averaged): Accuracy, Precision, Recall, F1.

* Polarity metrics are computed on *target* pairs only (candidate == gold aspect).
* Aspect-relevance metrics are computed on all pairs; the image-only stream has no
  aspect head (it never sees the candidate), so they are reported as n/a.

Usage:
    python evaluate.py --checkpoint ./saved_models/implicit_mabsa.pt --split test
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    IGNORE_INDEX,
    DataConfig,
    load_image_processor,
    load_tokenizer,
    make_dataloader,
    prepare_data,
)
from model import STREAM_DISPLAY_NAMES, VALID_MODES, ImplicitMABSAModel, ModelConfig, select_inputs
from utils import DEFAULT_CHECKPOINT_PATH, autocast_context, get_device, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def _macro_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Accuracy + macro precision / recall / F1 as plain Python floats."""
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.inference_mode()
def evaluate_stream(
    model: ImplicitMABSAModel,
    loader: DataLoader,
    mode: str,
    device: torch.device,
    use_amp: bool = False,
    show_progress: bool = False,
) -> Dict[str, Optional[float]]:
    """Evaluate one stream.

    Returns a flat dict with ``sentiment_{accuracy,precision,recall,f1}`` and
    ``aspect_{accuracy,precision,recall,f1}`` (``None`` for image-only), plus sample counts.
    """
    model.eval()
    sent_true: List[int] = []
    sent_pred: List[int] = []
    asp_true: List[int] = []
    asp_pred: List[int] = []
    has_aspect_head = False

    iterator = tqdm(loader, desc=f"eval [{mode}]", leave=False) if show_progress else loader
    for batch in iterator:
        with autocast_context(device, use_amp):
            outputs = model(mode, **select_inputs(batch, mode, device))

        sent_y = batch["sentiment_label"].to(device)
        valid = sent_y != IGNORE_INDEX
        sent_pred_batch = outputs["sentiment_logits"].float().argmax(dim=-1)
        sent_true.extend(sent_y[valid].cpu().tolist())
        sent_pred.extend(sent_pred_batch[valid].cpu().tolist())

        if outputs["aspect_logits"] is not None:
            has_aspect_head = True
            asp_true.extend(batch["aspect_label"].tolist())
            asp_pred.extend(outputs["aspect_logits"].float().argmax(dim=-1).cpu().tolist())

    results: Dict[str, Optional[float]] = {}
    for name, value in _macro_metrics(sent_true, sent_pred).items():
        results[f"sentiment_{name}"] = value
    if has_aspect_head:
        for name, value in _macro_metrics(asp_true, asp_pred).items():
            results[f"aspect_{name}"] = value
    else:
        for name in ("accuracy", "precision", "recall", "f1"):
            results[f"aspect_{name}"] = None
    results["num_sentiment_samples"] = float(len(sent_true))
    results["num_pairs"] = float(len(asp_true)) if has_aspect_head else float(len(sent_true))
    return results


def build_benchmark_table(results: Dict[str, Dict[str, Optional[float]]]) -> pd.DataFrame:
    """Side-by-side comparison of Stream A / B / C."""
    rows = []
    for mode in VALID_MODES:
        if mode not in results:
            continue
        m = results[mode]
        aspect_f1 = m.get("aspect_f1")
        rows.append(
            {
                "Stream": STREAM_DISPLAY_NAMES[mode],
                "Accuracy": m["sentiment_accuracy"],
                "Precision (macro)": m["sentiment_precision"],
                "Recall (macro)": m["sentiment_recall"],
                "Macro-F1": m["sentiment_f1"],
                "Aspect Macro-F1": float("nan") if aspect_f1 is None else aspect_f1,
            }
        )
    return pd.DataFrame(rows)


def print_benchmark_table(table: pd.DataFrame, title: str) -> None:
    line = "=" * 92
    print(f"\n{line}\n{title}\n{line}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}", na_rep="n/a"))
    print(line)
    print("Polarity metrics: target pairs only. Aspect Macro-F1: relevance of the candidate aspect "
          "(n/a = stream never sees the candidate).\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ImplicitMABSA streams.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data_path", default=None, help="override checkpoint training CSV/JSON/JSONL")
    parser.add_argument("--val_path", default=None, help="override checkpoint validation split")
    parser.add_argument("--test_path", default=None, help="override checkpoint test split")
    parser.add_argument("--image_root", default=None, help="override image root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: auto)")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    parser.add_argument("--output_dir", default=None, help="where to write benchmark.csv / benchmark.json "
                                                           "(default: checkpoint directory)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    device = get_device(args.device)
    logger.info("Device: %s", device)

    ckpt = load_checkpoint(args.checkpoint)
    model_cfg = ModelConfig(**ckpt["model_config"])
    data_cfg = DataConfig(**ckpt["data_config"])
    for key in ("data_path", "val_path", "test_path", "image_root"):
        value = getattr(args, key)
        if value is not None:
            setattr(data_cfg, key, value)

    tokenizer = load_tokenizer(model_cfg.text_model_name)
    image_processor = load_image_processor(model_cfg.vision_model_name)
    bundle = prepare_data(data_cfg, tokenizer, image_processor)  # identical splits to training
    dataset = bundle.test if args.split == "test" else bundle.val

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for mode in VALID_MODES:
        if mode not in ckpt["streams"]:
            logger.warning("Stream %s not present in checkpoint; skipping.", mode)
            continue
        model = ImplicitMABSAModel(model_cfg, modes=(mode,), pretrained=False)
        model.load_stream_state_dict(mode, ckpt["streams"][mode])
        model.to(device)
        loader = make_dataloader(dataset, mode, args.batch_size, shuffle=False, num_workers=args.num_workers)
        results[mode] = evaluate_stream(model, loader, mode, device, use_amp=args.amp, show_progress=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = build_benchmark_table(results)
    print_benchmark_table(table, f"ImplicitMABSA comparative benchmark ({args.split} split)")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"benchmark_{args.split}.csv", index=False)
    with open(out_dir / f"benchmark_{args.split}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved benchmark files to %s", out_dir)


if __name__ == "__main__":
    main()
