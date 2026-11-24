"""Data pipeline for ImplicitMABSA (Multimodal Aspect-Based Sentiment Analysis).

Task formulation
----------------
Every example is a *(post, image, candidate aspect)* triple. The text encoder
receives ``[CLS] post [SEP] candidate aspect [SEP]`` (tokenizer pair encoding)
and the image encoder receives the accompanying image. Two labels are produced:

* ``aspect_label``    1 if the candidate is the target implicit aspect of the post, else 0
* ``sentiment_label`` polarity (negative / neutral / positive) towards the target aspect.
                      It is ``IGNORE_INDEX`` for non-target candidates (no polarity exists).

Negative candidates are sampled from the aspect vocabulary so a model can rank
candidate aspects at inference time.

Real data (Twitter-2015 style) can be supplied as CSV / JSON / JSONL with the
columns ``text, image_path, aspect, sentiment``. If no file is given a synthetic
implicit-multimodal dataset is generated on the fly so the pipeline runs out of the box.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, ViTImageProcessor

logger = logging.getLogger(__name__)

SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]
SENTIMENT2ID: Dict[str, int] = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
IGNORE_INDEX: int = -100


# --------------------------------------------------------------------------- #
# Records & configuration
# --------------------------------------------------------------------------- #
@dataclass
class MABSASample:
    """One raw example before candidate-aspect expansion."""

    text: str
    aspect: str  # gold implicit aspect category
    sentiment: str  # gold polarity: negative / neutral / positive
    image_path: Optional[str] = None  # real data
    visual_state: Optional[int] = None  # synthetic data: sentiment id depicted by the image
    seed: int = 0  # synthetic data: rendering seed


@dataclass
class DataConfig:
    """Everything needed to rebuild identical splits (stored inside the checkpoint)."""

    data_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    image_root: Optional[str] = None
    num_synthetic: int = 1500
    ambiguous_prob: float = 0.3
    image_noise_prob: float = 0.15
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_negatives: int = 1
    max_length: int = 96
    seed: int = 42
    strict_images: bool = False


# --------------------------------------------------------------------------- #
# Synthetic implicit multimodal data
# --------------------------------------------------------------------------- #
# Texts never name the aspect ("It dies after two hours" -> battery).
SYNTHETIC_TEXTS: Dict[str, Dict[str, List[str]]] = {
    "battery": {
        "negative": [
            "It dies after two hours.",
            "Back to hunting for a charger before lunch again.",
            "Unplugged at 9am, gasping by 11.",
        ],
        "neutral": [
            "Lasts about as long as my old one.",
            "Charged it overnight like always.",
            "Usual day, usual outlet near my desk.",
        ],
        "positive": [
            "Still going strong after a whole weekend away from a socket.",
            "Forgot the charger entirely and never noticed.",
            "Made it through the flight and the layover without plugging in.",
        ],
    },
    "screen": {
        "negative": [
            "Can't see a thing when I'm outside in the sun.",
            "Fingerprints and glare everywhere, squinting all day.",
            "Half the picture just flickers whenever I tilt it.",
        ],
        "neutral": [
            "Looks about the same as every other one I've owned.",
            "Watched an episode on it last night.",
            "Brightness is set somewhere in the middle.",
        ],
        "positive": [
            "Colors pop so much I keep staring at my wallpaper.",
            "Reading in bright daylight is totally effortless.",
            "Every movie looks like it was shot right in my hands.",
        ],
    },
    "tires": {
        "negative": [
            "Pulled over on the shoulder again, third time this month.",
            "The car keeps wobbling and pulling to one side.",
            "Waiting on the roadside for a tow truck, great start to Monday.",
        ],
        "neutral": [
            "Took the car in for a routine check this morning.",
            "Drove the usual route to work today.",
            "Parked it outside the garage for the weekend.",
        ],
        "positive": [
            "Rain-soaked highway and the car never even twitched.",
            "Smooth, quiet ride all the way up the mountain.",
            "Hit a pothole and didn't feel a thing.",
        ],
    },
    "food": {
        "negative": [
            "Needed three glasses of water and still can't taste anything but salt.",
            "Pushed it around the plate and then asked for the check.",
            "Went to bed hungry after that dinner.",
        ],
        "neutral": [
            "Had dinner with the family at around eight.",
            "Tried the new place down the street.",
            "Ordered the same thing as my friend.",
        ],
        "positive": [
            "Licked the plate clean and ordered seconds.",
            "Already planning when I can go back for lunch.",
            "Every bite was better than the last.",
        ],
    },
    "delivery": {
        "negative": [
            "Sat by the door all week and nothing showed up.",
            "Tracking page hasn't moved in nine days.",
            "Arrived crushed and two days late.",
        ],
        "neutral": [
            "Got a notification that something is on the way.",
            "Left it with the neighbor like they said.",
            "Signed for it at the front desk.",
        ],
        "positive": [
            "Ordered at night, it was on my doorstep before breakfast.",
            "Arrived a day earlier than promised.",
            "Neatly packed and here before I even finished my coffee.",
        ],
    },
}

# Posts whose text carries no aspect/sentiment signal: only the image disambiguates them.
AMBIGUOUS_TEXTS: List[str] = [
    "Well, that happened.",
    "Day three of using this.",
    "Posting this so I remember.",
    "Not sure what to say honestly.",
    "Here we go.",
    "Update from today.",
    "Guess that's that.",
    "This is my life now.",
]
HASHTAG_SUFFIXES: List[str] = ["", "", "", " #mood", " #update", " #tbt", " #random"]

_STATE_COLORS: Tuple[Tuple[int, int, int], ...] = ((200, 40, 40), (230, 170, 30), (50, 160, 70))
_PaintFn = Callable[[ImageDraw.ImageDraw, int, int, int, Tuple[int, int, int], random.Random], None]


def _paint_battery(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    w, h = 120, 64
    x0, y0 = cx - w // 2, cy - h // 2
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=8, outline=(40, 40, 40), width=5)
    d.rectangle([x0 + w, cy - 12, x0 + w + 10, cy + 12], fill=(40, 40, 40))
    fraction = (0.15, 0.55, 1.0)[state]
    d.rectangle([x0 + 7, y0 + 7, x0 + 7 + int((w - 14) * fraction), y0 + h - 7], fill=color)


def _paint_screen(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    w, h = 130, 86
    fill = [(25, 25, 35), (150, 150, 160), (250, 250, 255)][state]
    d.rounded_rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], radius=8, fill=fill, outline=(30, 30, 30), width=6)
    d.rectangle([cx - 12, cy + h // 2 + 4, cx + 12, cy + h // 2 + 16], fill=(30, 30, 30))
    d.rectangle([cx - 30, cy + h // 2 + 16, cx + 30, cy + h // 2 + 22], fill=(30, 30, 30))
    if state == 0:  # cracked
        d.line([(cx - w // 4, cy - h // 2 + 6), (cx, cy), (cx - 10, cy + h // 2 - 6)], fill=(235, 235, 235), width=3)
        d.line([(cx, cy), (cx + w // 4, cy + 8)], fill=(235, 235, 235), width=3)
    elif state == 2:  # vivid colour bars
        for i, c in enumerate([(230, 60, 60), (60, 180, 90), (60, 110, 230)]):
            x = cx - w // 2 + 14 + i * 36
            d.rectangle([x, cy - h // 2 + 16, x + 26, cy + h // 2 - 16], fill=c)


def _paint_tire(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    r = 62
    squash = (0.5, 0.8, 1.0)[state]  # flat tire when negative
    ry = int(r * squash)
    cy2 = cy + (r - ry)
    d.ellipse([cx - r, cy2 - ry, cx + r, cy2 + ry], fill=(35, 35, 35))
    d.ellipse([cx - 28, cy2 - int(28 * squash), cx + 28, cy2 + int(28 * squash)], fill=(190, 190, 195))
    d.ellipse([cx - 8, cy2 - max(2, int(8 * squash)), cx + 8, cy2 + max(2, int(8 * squash))], fill=(90, 90, 95))
    d.line([(cx - r - 15, cy + r), (cx + r + 15, cy + r)], fill=(90, 90, 90), width=3)


def _paint_food(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    r = 62
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250), outline=(120, 120, 120), width=5)
    d.ellipse([cx - r + 14, cy - r + 14, cx + r - 14, cy + r - 14], outline=(200, 200, 200), width=2)
    item_color = [(60, 40, 30), color, color][state]  # burnt crumbs vs. a full plate
    for _ in range((1, 3, 6)[state]):
        px, py, rr = cx + rng.randint(-30, 30), cy + rng.randint(-30, 30), rng.randint(9, 14)
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=item_color)


def _paint_box(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    w, h = 110, 90
    x0, y0, x1, y1 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
    d.rectangle([x0, y0, x1, y1], fill=(196, 150, 95), outline=(110, 80, 45), width=5)
    d.rectangle([cx - 9, y0, cx + 9, y1], fill=(225, 200, 150))
    if state == 0:  # crushed corners
        d.line([(x0, y0), (x0 + 30, y0 + 25)], fill=(90, 60, 30), width=4)
        d.line([(x1, y1), (x1 - 30, y1 - 25)], fill=(90, 60, 30), width=4)
    for i in range((0, 1, 3)[state]):  # speed lines
        y = cy - 20 + i * 20
        d.line([(x0 - 40, y), (x0 - 14, y)], fill=color, width=5)


def _paint_generic(d: ImageDraw.ImageDraw, cx: int, cy: int, state: int, color: Tuple[int, int, int], rng: random.Random) -> None:
    d.rectangle([cx - 50, cy - 50, cx + 50, cy + 50], fill=color, outline=(40, 40, 40), width=5)


_ICON_PAINTERS: Dict[str, _PaintFn] = {
    "battery": _paint_battery,
    "screen": _paint_screen,
    "tires": _paint_tire,
    "food": _paint_food,
    "delivery": _paint_box,
}


def _paint_badge(d: ImageDraw.ImageDraw, size: int, state: int) -> None:
    """Corner badge (red X / amber dash / green check) that signals the depicted polarity."""
    cx, cy, r = size - 30, 30, 20
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_STATE_COLORS[state])
    white = (255, 255, 255)
    if state == 0:
        d.line([(cx - 8, cy - 8), (cx + 8, cy + 8)], fill=white, width=4)
        d.line([(cx - 8, cy + 8), (cx + 8, cy - 8)], fill=white, width=4)
    elif state == 1:
        d.line([(cx - 9, cy), (cx + 9, cy)], fill=white, width=4)
    else:
        d.line([(cx - 9, cy), (cx - 3, cy + 7), (cx + 9, cy - 7)], fill=white, width=4)


def render_synthetic_image(aspect: str, visual_state: int, seed: int, size: int = 224) -> Image.Image:
    """Draw an icon for ``aspect`` whose state/badge depicts polarity ``visual_state`` (0/1/2)."""
    rng = random.Random(seed)
    background = tuple(rng.randint(225, 250) for _ in range(3))
    image = Image.new("RGB", (size, size), background)  # type: ignore[arg-type]
    draw = ImageDraw.Draw(image)
    cx, cy = size // 2 + rng.randint(-12, 12), size // 2 + rng.randint(-12, 12)
    painter = _ICON_PAINTERS.get(aspect, _paint_generic)
    painter(draw, cx, cy, visual_state, _STATE_COLORS[visual_state], rng)
    _paint_badge(draw, size, visual_state)
    return image


def generate_synthetic_records(
    num_samples: int = 1500,
    seed: int = 42,
    ambiguous_prob: float = 0.3,
    image_noise_prob: float = 0.15,
) -> List[MABSASample]:
    """Create an imbalanced, implicit, multimodal toy corpus.

    * ``ambiguous_prob``: fraction of posts with an uninformative text; the image alone
      reveals both the aspect and the polarity (where multimodal fusion should help).
    * ``image_noise_prob``: for informative texts, probability that the image depicts a
      *different* polarity (where an image-only model should fail).
    """
    rng = random.Random(seed)
    aspects = list(SYNTHETIC_TEXTS)
    records: List[MABSASample] = []
    for _ in range(num_samples):
        aspect = rng.choice(aspects)
        sentiment = rng.choices(SENTIMENT_LABELS, weights=[0.5, 0.15, 0.35])[0]  # deliberate imbalance
        ambiguous = rng.random() < ambiguous_prob
        text = rng.choice(AMBIGUOUS_TEXTS) if ambiguous else rng.choice(SYNTHETIC_TEXTS[aspect][sentiment])
        text += rng.choice(HASHTAG_SUFFIXES)
        visual_sentiment = sentiment
        if not ambiguous and rng.random() < image_noise_prob:
            visual_sentiment = rng.choice([s for s in SENTIMENT_LABELS if s != sentiment])
        records.append(
            MABSASample(
                text=text,
                aspect=aspect,
                sentiment=sentiment,
                visual_state=SENTIMENT2ID[visual_sentiment],
                seed=rng.randrange(2**31),
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Real data loading & splitting
# --------------------------------------------------------------------------- #
def _normalize_sentiment(value: Any) -> str:
    text = str(value).strip().lower()
    try:
        text = str(int(float(text)))
    except ValueError:
        pass
    aliases = {
        "neg": "negative", "neu": "neutral", "pos": "positive",
        "0": "negative", "1": "neutral", "2": "positive",  # Twitter-2015 convention
    }
    text = aliases.get(text, text)
    if text not in SENTIMENT2ID:
        raise ValueError(f"Unrecognised sentiment label: {value!r}")
    return text


def load_records(path: str, image_root: Optional[str] = None) -> List[MABSASample]:
    """Load ``text, image_path, aspect, sentiment`` rows from a CSV / JSON / JSONL file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported data file type: {suffix!r} (use .csv, .json or .jsonl)")
    missing = {"text", "image_path", "aspect", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"Data file is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "aspect", "sentiment"])

    records: List[MABSASample] = []
    for row in df.to_dict("records"):
        image_path: Optional[str] = None
        raw = row.get("image_path")
        if isinstance(raw, str) and raw.strip():
            p = Path(raw.strip())
            if image_root and not p.is_absolute():
                p = Path(image_root) / p
            image_path = str(p)
        records.append(
            MABSASample(
                text=str(row["text"]),
                aspect=str(row["aspect"]).strip().lower(),
                sentiment=_normalize_sentiment(row["sentiment"]),
                image_path=image_path,
            )
        )
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def split_records(
    records: Sequence[MABSASample], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[MABSASample], List[MABSASample], List[MABSASample]]:
    """Stratified (by polarity) train/val/test split performed *before* candidate expansion."""

    def _split(items: List[MABSASample], test_size: float) -> Tuple[List[MABSASample], List[MABSASample]]:
        try:
            return train_test_split(items, test_size=test_size, random_state=seed, stratify=[r.sentiment for r in items])
        except ValueError:  # a class is too small to stratify
            return train_test_split(items, test_size=test_size, random_state=seed)

    train_val, test = _split(list(records), test_ratio)
    train, val = _split(train_val, val_ratio / (1.0 - test_ratio))
    return train, val, test


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ImplicitMABSADataset(Dataset):
    """PyTorch dataset yielding tokenised ``[CLS] text [SEP] aspect [SEP]`` + image tensors.

    Each record produces one *target* pair (candidate == gold aspect) and
    ``num_negatives`` *non-target* pairs with randomly drawn other aspects.
    """

    def __init__(
        self,
        records: Sequence[MABSASample],
        tokenizer: PreTrainedTokenizerBase,
        image_processor: ViTImageProcessor,
        aspect_vocab: Sequence[str],
        max_length: int = 96,
        num_negatives: int = 1,
        seed: int = 42,
        strict_images: bool = False,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.aspect_vocab = list(aspect_vocab)
        self.max_length = max_length
        self.strict_images = strict_images
        self._warned_missing_image = False

        rng = random.Random(seed)
        self.pairs: List[Tuple[int, str, int]] = []  # (record index, candidate aspect, is_target)
        for idx, rec in enumerate(self.records):
            self.pairs.append((idx, rec.aspect, 1))
            others = [a for a in self.aspect_vocab if a != rec.aspect]
            for cand in rng.sample(others, k=min(num_negatives, len(others))):
                self.pairs.append((idx, cand, 0))

        self.aspect_labels: List[int] = [is_target for _, _, is_target in self.pairs]
        self.sentiment_labels: List[int] = [
            SENTIMENT2ID[self.records[i].sentiment] if t else IGNORE_INDEX for i, _, t in self.pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def target_indices(self) -> List[int]:
        """Indices of pairs whose candidate is the gold aspect (used by the image-only stream)."""
        return [i for i, flag in enumerate(self.aspect_labels) if flag == 1]

    def _load_image(self, rec: MABSASample) -> Image.Image:
        if rec.image_path is not None:
            path = Path(rec.image_path)
            if path.is_file():
                return Image.open(path).convert("RGB")
            if self.strict_images:
                raise FileNotFoundError(f"Image not found: {path}")
            if not self._warned_missing_image:
                logger.warning("Image(s) missing (first: %s); substituting a blank image.", path)
                self._warned_missing_image = True
            return Image.new("RGB", (224, 224), (128, 128, 128))
        if rec.visual_state is None:
            raise ValueError("Record has neither an image_path nor a synthetic visual_state.")
        return render_synthetic_image(rec.aspect, rec.visual_state, rec.seed)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec_idx, candidate, _ = self.pairs[index]
        rec = self.records[rec_idx]

        enc = self.tokenizer(
            rec.text,
            candidate,  # pair encoding -> [CLS] text [SEP] candidate [SEP]
            truncation="only_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        pixels = self.image_processor(images=self._load_image(rec), return_tensors="pt")["pixel_values"]
        item["pixel_values"] = pixels.squeeze(0)
        item["sentiment_label"] = torch.tensor(self.sentiment_labels[index], dtype=torch.long)
        item["aspect_label"] = torch.tensor(self.aspect_labels[index], dtype=torch.long)
        return item


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    train: ImplicitMABSADataset
    val: ImplicitMABSADataset
    test: ImplicitMABSADataset
    aspect_vocab: List[str]


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)


def load_image_processor(model_name: str) -> ViTImageProcessor:
    return ViTImageProcessor.from_pretrained(model_name)


def prepare_data(
    cfg: DataConfig, tokenizer: PreTrainedTokenizerBase, image_processor: ViTImageProcessor
) -> DataBundle:
    """Load (or synthesise) records, split them, and wrap each split in a dataset."""
    explicit_splits = bool(cfg.val_path or cfg.test_path)
    if explicit_splits and not (cfg.data_path and cfg.val_path and cfg.test_path):
        raise ValueError("Explicit splits require data_path, val_path, and test_path.")

    if explicit_splits:
        train_r = load_records(cfg.data_path, cfg.image_root)
        val_r = load_records(cfg.val_path, cfg.image_root)
        test_r = load_records(cfg.test_path, cfg.image_root)
        records = train_r
    elif cfg.data_path:
        records = load_records(cfg.data_path, cfg.image_root)
        train_r, val_r, test_r = split_records(records, cfg.val_ratio, cfg.test_ratio, cfg.seed)
    else:
        logger.info("No data_path given; generating %d synthetic samples.", cfg.num_synthetic)
        records = generate_synthetic_records(cfg.num_synthetic, cfg.seed, cfg.ambiguous_prob, cfg.image_noise_prob)
        train_r, val_r, test_r = split_records(records, cfg.val_ratio, cfg.test_ratio, cfg.seed)

    aspect_vocab = sorted({r.aspect for r in train_r})
    if len(aspect_vocab) < 2:
        raise ValueError("At least two aspect categories are required to build negative candidates.")

    def make(recs: List[MABSASample], seed_offset: int) -> ImplicitMABSADataset:
        return ImplicitMABSADataset(
            recs, tokenizer, image_processor, aspect_vocab,
            max_length=cfg.max_length, num_negatives=cfg.num_negatives,
            seed=cfg.seed + seed_offset, strict_images=cfg.strict_images,
        )

    bundle = DataBundle(make(train_r, 0), make(val_r, 1), make(test_r, 2), aspect_vocab)
    logger.info(
        "Splits (records): train=%d val=%d test=%d | aspects=%s",
        len(train_r), len(val_r), len(test_r), aspect_vocab,
    )
    return bundle


def make_dataloader(
    dataset: ImplicitMABSADataset, mode: str, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    """DataLoader for a stream. Image-only sees target pairs only (it cannot use the candidate)."""
    data: Dataset = Subset(dataset, dataset.target_indices()) if mode == "image_only" else dataset
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (``IGNORE_INDEX`` entries are skipped)."""
    valid = torch.tensor([x for x in labels if x != IGNORE_INDEX], dtype=torch.long)
    counts = torch.bincount(valid, minlength=num_classes).float().clamp(min=1.0)
    return counts.sum() / (num_classes * counts)
