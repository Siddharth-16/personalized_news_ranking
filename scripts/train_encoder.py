import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from src.data.load import load_behaviors, load_news
from src.data.split import chronological_split
from src.retrieval.finetune_data import (
    build_click_triplets,
    build_title_map,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

NEWS_PATH = Path(
    "data/raw/train/news.tsv"
)

MODEL_OUTPUT_PATH = Path(
    "artifacts/models/click_adapted_minilm"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/click_adapted_minilm"
)

METADATA_PATH = Path(
    "artifacts/models/click_adapted_minilm_metadata.json"
)

BASE_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

VALIDATION_DATE = "2019-11-14"
RANDOM_SEED = 42


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-examples",
        type=int,
        default=50_000,
        help=(
            "Maximum number of click triplets "
            "used for fine-tuning."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    args = parser.parse_args()

    if args.max_examples <= 0:
        raise ValueError(
            "--max-examples must be greater than 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be greater than 0."
        )

    device = choose_device()

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------
    # Load MIND data
    # --------------------------------------------------

    print("\nLoading behaviors and news...")

    behaviors = load_behaviors(
        BEHAVIORS_PATH
    )

    news = load_news(
        NEWS_PATH
    )

    train, _ = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    print(
        f"Training impressions: "
        f"{len(train):,}"
    )

    # --------------------------------------------------
    # Build title map
    # --------------------------------------------------

    title_map = build_title_map(
        news
    )

    print(
        f"Articles with usable titles: "
        f"{len(title_map):,}"
    )

    # --------------------------------------------------
    # Build click-adaptation triplets
    # --------------------------------------------------

    print(
        "\nBuilding click triplets..."
    )

    (
        anchors,
        positives,
        negatives,
        triplet_stats,
    ) = build_click_triplets(
        behaviors=train,
        title_map=title_map,
        max_examples=args.max_examples,
        random_seed=RANDOM_SEED,
    )

    if not anchors:
        raise RuntimeError(
            "No training triplets were generated."
        )

    print(
        f"Triplets:             "
        f"{len(anchors):,}"
    )

    print(
        f"Skipped no history:   "
        f"{triplet_stats['skipped_no_history']:,}"
    )

    print(
        f"Skipped no positive:  "
        f"{triplet_stats['skipped_no_positive']:,}"
    )

    print(
        f"Skipped no negative:  "
        f"{triplet_stats['skipped_no_negative']:,}"
    )

    # --------------------------------------------------
    # Hugging Face Dataset
    # --------------------------------------------------

    train_dataset = Dataset.from_dict(
        {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
        }
    )

    # --------------------------------------------------
    # Load frozen baseline model as initialization
    # --------------------------------------------------

    print(
        f"\nLoading base encoder: "
        f"{BASE_MODEL}"
    )

    model = SentenceTransformer(
        BASE_MODEL,
        device=device,
    )

    # --------------------------------------------------
    # Contrastive retrieval objective
    # --------------------------------------------------

    train_loss = (
        losses.MultipleNegativesRankingLoss(
            model=model
        )
    )

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    CHECKPOINT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = (
        SentenceTransformerTrainingArguments(
            output_dir=str(
                CHECKPOINT_PATH
            ),

            num_train_epochs=(
                args.epochs
            ),

            per_device_train_batch_size=(
                args.batch_size
            ),

            learning_rate=(
                args.learning_rate
            ),

            warmup_ratio=0.1,

            batch_sampler=(
                BatchSamplers.NO_DUPLICATES
            ),

            # Keep reproducible.
            seed=RANDOM_SEED,
            data_seed=RANDOM_SEED,

            # We only need the final model for this
            # controlled one-epoch experiment.
            save_strategy="no",
            eval_strategy="no",

            logging_steps=100,

            # Avoid automatically activating W&B or
            # another external experiment tracker.
            report_to="none",

            # Avoid multiprocessing complications on
            # local development machines.
            dataloader_num_workers=0,

            # Keep precision simple and stable,
            # particularly when running on MPS.
            fp16=False,
            bf16=False,
        )
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    print("\nStarting encoder fine-tuning...")
    print(
        f"Examples:      {len(train_dataset):,}"
    )
    print(
        f"Batch size:    {args.batch_size}"
    )
    print(
        f"Epochs:        {args.epochs}"
    )
    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )

    train_result = trainer.train()

    print(
        "\nEncoder fine-tuning complete."
    )

    # --------------------------------------------------
    # Save adapted encoder
    # --------------------------------------------------

    MODEL_OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        str(MODEL_OUTPUT_PATH)
    )

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    training_loss = None

    if (
        train_result.training_loss
        is not None
    ):
        training_loss = float(
            train_result.training_loss
        )

    metadata = {
        "experiment": (
            "click_adapted_encoder"
        ),
        "base_model": BASE_MODEL,
        "objective": (
            "MultipleNegativesRankingLoss"
        ),
        "triplet_definition": {
            "anchor": (
                "random previously clicked "
                "article title"
            ),
            "positive": (
                "clicked article from "
                "current impression"
            ),
            "negative": (
                "displayed but unclicked article "
                "from current impression"
            ),
        },
        "training": {
            "examples": len(
                train_dataset
            ),
            "epochs": args.epochs,
            "batch_size": (
                args.batch_size
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "warmup_ratio": 0.1,
            "random_seed": (
                RANDOM_SEED
            ),
            "training_loss": (
                training_loss
            ),
        },
        "data": {
            "dataset": "MINDsmall",
            "training_impressions": (
                len(train)
            ),
            "validation_cutoff": (
                VALIDATION_DATE
            ),
            "triplet_stats": (
                triplet_stats
            ),
        },
        "device": device,
    }

    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved adapted encoder to: "
        f"{MODEL_OUTPUT_PATH}"
    )

    print(
        f"Saved metadata to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    main()