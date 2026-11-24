"""Train the three ImplicitMABSA streams independently on identical data splits.

For each stream a *fresh* model (fresh pre-trained encoders) is fine-tuned with AdamW,
a linear warm-up/decay schedule and class-weighted cross-entropy. The weights of the
best validation epoch (by polarity macro-F1) of every stream are written into one
checkpoint: ``./saved_models/implicit_mabsa.pt``.

Usage:
    python train.py                                   # synthetic data, all three streams
    python train.py --data_path data/twitter2015.csv --image_root data/images --epochs 5
"""
from __future__ import annotations

import argparse
import gc
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import (
    IGNORE_INDEX,
    SENTIMENT_LABELS,
    DataBundle,
    DataConfig,
    compute_class_weights,
    load_image_processor,
    load_tokenizer,
    make_dataloader,
    prepare_data,
)
from evaluate import evaluate_stream
from model import STREAM_DISPLAY_NAMES, VALID_MODES, ImplicitMABSAModel, ModelConfig, select_inputs
from utils import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_CHECKPOINT_PATH,
    autocast_context,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def compute_loss(
    outputs: Dict[str, Optional[torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    sentiment_loss_fn: nn.Module,
    aspect_loss_fn: nn.Module,
    aspect_loss_weight: float,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Weighted polarity CE (target pairs only) + weighted aspect-relevance CE (if the stream has it)."""
    total: Optional[torch.Tensor] = None
    sent_y = batch["sentiment_label"].to(device)
    if bool((sent_y != IGNORE_INDEX).any()):
        total = sentiment_loss_fn(outputs["sentiment_logits"].float(), sent_y)
    if outputs["aspect_logits"] is not None:
        aspect_term = aspect_loss_weight * aspect_loss_fn(
            outputs["aspect_logits"].float(), batch["aspect_label"].to(device)
        )
        total = aspect_term if total is None else total + aspect_term
    return total


def train_one_stream(
    mode: str,
    model_cfg: ModelConfig,
    bundle: DataBundle,
    args: argparse.Namespace,
    device: torch.device,
    sentiment_weights: torch.Tensor,
    aspect_weights: torch.Tensor,
    initial_state: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], float]:
    """Fine-tune one stream; returns (best stream state dict, per-epoch history, best val macro-F1)."""
    logger.info("=" * 70)
    logger.info("Training %s", STREAM_DISPLAY_NAMES[mode])
    model = ImplicitMABSAModel(model_cfg, modes=(mode,), pretrained=initial_state is None).to(device)
    if initial_state is not None:
        model.load_stream_state_dict(mode, initial_state)
        logger.info("Resumed %s weights from the existing checkpoint.", mode)
    logger.info("Trainable parameters: %.1fM", model.num_trainable_parameters() / 1e6)

    train_loader = make_dataloader(bundle.train, mode, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(bundle.val, mode, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = AdamW(model.parameter_groups(args.lr_encoder, args.lr_head, args.weight_decay))
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sentiment_loss_fn = nn.CrossEntropyLoss(weight=sentiment_weights.to(device), ignore_index=IGNORE_INDEX)
    aspect_loss_fn = nn.CrossEntropyLoss(weight=aspect_weights.to(device))

    best_f1, best_state = -1.0, model.stream_state_dict(mode)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, steps = 0.0, 0
        progress = tqdm(train_loader, desc=f"[{mode}] epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                outputs = model(mode, **select_inputs(batch, mode, device))
            loss = compute_loss(outputs, batch, sentiment_loss_fn, aspect_loss_fn, args.aspect_loss_weight, device)
            if loss is None:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

        val_metrics = evaluate_stream(model, val_loader, mode, device, use_amp=use_amp)
        val_f1 = float(val_metrics["sentiment_f1"] or 0.0)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, steps),
            **{k: (None if v is None else float(v)) for k, v in val_metrics.items()},
        }
        history.append(record)
        logger.info(
            "[%s] epoch %d | train_loss=%.4f | val acc=%.4f macro-F1=%.4f",
            mode, epoch, record["train_loss"], val_metrics["sentiment_accuracy"], val_f1,
        )
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, model.stream_state_dict(mode)
            logger.info("[%s] new best validation macro-F1: %.4f", mode, best_f1)

    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, history, best_f1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ImplicitMABSA streams (text / image / multimodal).")
    # data
    p.add_argument("--data_path", default=None, help="CSV/JSON/JSONL with text,image_path,aspect,sentiment "
                                                     "(omit to use synthetic data)")
    p.add_argument("--val_path", default=None, help="optional validation CSV/JSON/JSONL for an explicit split")
    p.add_argument("--test_path", default=None, help="optional test CSV/JSON/JSONL for an explicit split")
    p.add_argument("--image_root", default=None, help="prefix for relative image paths")
    p.add_argument("--num_synthetic", type=int, default=1500)
    p.add_argument("--ambiguous_prob", type=float, default=0.3)
    p.add_argument("--image_noise_prob", type=float, default=0.15)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--num_negatives", type=int, default=1, help="negative candidate aspects per record")
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--strict_images", action="store_true", help="fail on missing images instead of using a blank one")
    # model
    p.add_argument("--text_model", default="microsoft/deberta-v3-base", help="or bert-base-uncased")
    p.add_argument("--vision_model", default="google/vit-base-patch16-224")
    p.add_argument("--fusion_heads", type=int, default=8)
    p.add_argument("--head_hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_text", action="store_true")
    p.add_argument("--freeze_vision", action="store_true")
    # optimisation
    p.add_argument("--streams", nargs="+", choices=VALID_MODES, default=list(VALID_MODES))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr_encoder", type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--aspect_loss_weight", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="fp16 mixed precision (CUDA only)")
    # system
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu (default: auto)")
    p.add_argument("--resume_checkpoint", default=None, help="continue fine-tuning from a saved checkpoint")
    p.add_argument("--output_path", default=None, help="checkpoint output (defaults to a new fine-tuned file when resuming)")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    logger.info("Device: %s", device)

    resume = load_checkpoint(args.resume_checkpoint) if args.resume_checkpoint else None
    model_cfg = ModelConfig(**resume["model_config"]) if resume else ModelConfig(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        num_sentiments=len(SENTIMENT_LABELS),
        fusion_heads=args.fusion_heads,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        freeze_text=args.freeze_text,
        freeze_vision=args.freeze_vision,
    )
    data_cfg = DataConfig(
        data_path=args.data_path,
        val_path=args.val_path,
        test_path=args.test_path,
        image_root=args.image_root,
        num_synthetic=args.num_synthetic,
        ambiguous_prob=args.ambiguous_prob,
        image_noise_prob=args.image_noise_prob,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_negatives=args.num_negatives,
        max_length=args.max_length,
        seed=args.seed,
        strict_images=args.strict_images,
    )

    tokenizer = load_tokenizer(model_cfg.text_model_name)
    image_processor = load_image_processor(model_cfg.vision_model_name)
    bundle = prepare_data(data_cfg, tokenizer, image_processor)

    # Class-imbalance handling: inverse-frequency weights from the *training* split only.
    sentiment_weights = compute_class_weights(bundle.train.sentiment_labels, len(SENTIMENT_LABELS))
    aspect_weights = compute_class_weights(bundle.train.aspect_labels, model_cfg.num_aspect_classes)
    logger.info("Polarity class weights %s: %s", SENTIMENT_LABELS, [round(w, 3) for w in sentiment_weights.tolist()])
    logger.info("Aspect-relevance class weights [non-target, target]: %s", [round(w, 3) for w in aspect_weights.tolist()])

    streams: Dict[str, Dict[str, torch.Tensor]] = dict(resume["streams"]) if resume else {}
    histories: Dict[str, List[Dict[str, Any]]] = dict(resume.get("history", {})) if resume else {}
    best_scores: Dict[str, float] = dict(resume.get("best_val_macro_f1", {})) if resume else {}
    for mode in args.streams:
        if resume and mode not in resume["streams"]:
            raise ValueError(f"Checkpoint has no saved weights for stream {mode!r}.")
        state, history, best_f1 = train_one_stream(
            mode, model_cfg, bundle, args, device, sentiment_weights, aspect_weights,
            initial_state=resume["streams"][mode] if resume else None,
        )
        streams[mode], histories[mode], best_scores[mode] = state, history, best_f1

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": asdict(model_cfg),
        "data_config": asdict(data_cfg),
        "aspect_vocab": bundle.aspect_vocab,
        "sentiment_labels": SENTIMENT_LABELS,
        "class_weights": {
            "sentiment": [float(w) for w in sentiment_weights.tolist()],
            "aspect": [float(w) for w in aspect_weights.tolist()],
        },
        "best_val_macro_f1": best_scores,
        "history": histories,
        "streams": streams,
    }
    output_path = args.output_path or (
        "./saved_models/implicit_mabsa_finetuned.pt" if resume else DEFAULT_CHECKPOINT_PATH
    )
    path = save_checkpoint(payload, output_path)
    logger.info("Saved checkpoint to %s", path)
    for mode, score in best_scores.items():
        logger.info("  %-22s best val macro-F1 = %.4f", STREAM_DISPLAY_NAMES[mode], score)
    logger.info("Run `python evaluate.py --checkpoint %s` for the held-out benchmark table.", path)


if __name__ == "__main__":
    main()
