"""Data cleaning and deduplication for generated datasets."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field

from .config import Config
from .generator import QAPair

logger = logging.getLogger(__name__)


class DatasetSplit(BaseModel):
    """A split of the dataset (train/val/test)."""

    name: str
    pairs: list[QAPair] = Field(default_factory=list)

    def save(self, output_path: Path) -> None:
        """Save the split to a JSONL file.

        Args:
            output_path: Path to save the JSONL file.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in self.pairs:
                f.write(pair.to_jsonl() + "\n")

        logger.info(f"Saved {len(self.pairs)} pairs to {output_path}")


class CleaningResult(BaseModel):
    """Result of cleaning and splitting operation."""

    train: DatasetSplit = Field(default_factory=lambda: DatasetSplit(name="train"))
    validation: DatasetSplit = Field(default_factory=lambda: DatasetSplit(name="validation"))
    test: DatasetSplit = Field(default_factory=lambda: DatasetSplit(name="test"))
    original_count: int = 0
    cleaned_count: int = 0
    duplicates_removed: int = 0
    invalid_removed: int = 0

    def save(self, output_dir: Path, base_name: str = "dataset") -> dict[str, Path]:
        """Save all splits to JSONL files.

        Args:
            output_dir: Directory to save files.
            base_name: Base name for output files.

        Returns:
            Dict mapping split names to file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        if self.train.pairs:
            train_path = output_dir / f"{base_name}_train.jsonl"
            self.train.save(train_path)
            paths["train"] = train_path

        if self.validation.pairs:
            val_path = output_dir / f"{base_name}_val.jsonl"
            self.validation.save(val_path)
            paths["validation"] = val_path

        if self.test.pairs:
            test_path = output_dir / f"{base_name}_test.jsonl"
            self.test.save(test_path)
            paths["test"] = test_path

        return paths

    def save_combined(self, output_path: Path) -> None:
        """Save all pairs to a single JSONL file.

        Args:
            output_path: Path to save the combined JSONL file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_pairs = self.train.pairs + self.validation.pairs + self.test.pairs

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in all_pairs:
                f.write(pair.to_jsonl() + "\n")

        logger.info(f"Saved {len(all_pairs)} pairs to {output_path}")


class Cleaner:
    """Dataset cleaner and splitter."""

    def __init__(self, config: Config) -> None:
        """Initialize the cleaner.

        Args:
            config: Application configuration.
        """
        self.config = config

    def clean_and_split(
        self,
        pairs: list[QAPair],
        shuffle: bool = True,
    ) -> CleaningResult:
        """Clean, deduplicate, and split pairs into train/val/test.

        Args:
            pairs: List of Q&A pairs to process.
            shuffle: Whether to shuffle pairs before splitting.

        Returns:
            CleaningResult: Cleaned and split dataset.
        """
        result = CleaningResult(original_count=len(pairs))

        # Step 1: Validate pairs
        valid_pairs = self._validate_pairs(pairs)
        result.invalid_removed = len(pairs) - len(valid_pairs)

        # Step 2: Remove duplicates
        unique_pairs = self._deduplicate(valid_pairs)
        result.duplicates_removed = len(valid_pairs) - len(unique_pairs)

        result.cleaned_count = len(unique_pairs)

        # Step 3: Shuffle if requested
        if shuffle:
            random.shuffle(unique_pairs)

        # Step 4: Split into train/val/test
        splits = self._split(unique_pairs)
        result.train = DatasetSplit(name="train", pairs=splits[0])
        result.validation = DatasetSplit(name="validation", pairs=splits[1])
        result.test = DatasetSplit(name="test", pairs=splits[2])

        logger.info(
            f"Cleaning complete: {result.original_count} -> {result.cleaned_count} pairs "
            f"({result.duplicates_removed} duplicates, {result.invalid_removed} invalid)"
        )
        logger.info(
            f"Split: train={len(result.train.pairs)}, "
            f"val={len(result.validation.pairs)}, "
            f"test={len(result.test.pairs)}"
        )

        return result

    def _validate_pairs(self, pairs: list[QAPair]) -> list[QAPair]:
        """Validate Q&A pairs.

        Args:
            pairs: List of pairs to validate.

        Returns:
            List of valid pairs.
        """
        valid = []

        for pair in pairs:
            # Check minimum lengths
            if len(pair.instruction.strip()) < 10:
                logger.debug(f"Skipping pair: instruction too short")
                continue

            if len(pair.output.strip()) < 20:
                logger.debug(f"Skipping pair: output too short")
                continue

            # Check for placeholder/template text
            placeholder_patterns = [
                "[insert",
                "[your",
                "[example",
                "lorem ipsum",
                "TODO",
                "FIXME",
            ]
            has_placeholder = any(
                p.lower() in pair.instruction.lower() or p.lower() in pair.output.lower()
                for p in placeholder_patterns
            )
            if has_placeholder:
                logger.debug(f"Skipping pair: contains placeholder text")
                continue

            valid.append(pair)

        return valid

    def _deduplicate(self, pairs: list[QAPair]) -> list[QAPair]:
        """Remove duplicate Q&A pairs.

        Args:
            pairs: List of pairs to deduplicate.

        Returns:
            List of unique pairs.
        """
        seen_hashes = set()
        unique = []

        for pair in pairs:
            # Create hash based on instruction and output
            content = f"{pair.instruction.lower().strip()}|{pair.output.lower().strip()}"
            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique.append(pair)

        return unique

    def _split(self, pairs: list[QAPair]) -> tuple[list[QAPair], list[QAPair], list[QAPair]]:
        """Split pairs into train/val/test sets.

        Args:
            pairs: List of pairs to split.

        Returns:
            Tuple of (train, validation, test) lists.
        """
        ratios = self.config.output.split_ratio
        n = len(pairs)

        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        train = pairs[:train_end]
        val = pairs[train_end:val_end]
        test = pairs[val_end:]

        return train, val, test


def load_jsonl(file_path: Path) -> list[QAPair]:
    """Load Q&A pairs from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of QAPair objects.
    """
    pairs = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    pairs.append(
                        QAPair(
                            instruction=data.get("instruction", ""),
                            input=data.get("input", ""),
                            output=data.get("output", ""),
                        )
                    )
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")

    return pairs


def merge_datasets(*file_paths: Path) -> list[QAPair]:
    """Merge multiple JSONL datasets.

    Args:
        file_paths: Paths to JSONL files to merge.

    Returns:
        Combined list of QAPair objects.
    """
    all_pairs = []

    for path in file_paths:
        if path.exists():
            pairs = load_jsonl(path)
            all_pairs.extend(pairs)
            logger.info(f"Loaded {len(pairs)} pairs from {path}")

    return all_pairs
