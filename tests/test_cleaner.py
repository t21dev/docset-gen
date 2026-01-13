"""Tests for cleaner module."""

import json
import tempfile
from pathlib import Path

import pytest

from docset_gen.cleaner import Cleaner, CleaningResult, DatasetSplit, load_jsonl, merge_datasets
from docset_gen.config import Config
from docset_gen.generator import QAPair


class TestDatasetSplit:
    """Tests for DatasetSplit model."""

    def test_save(self):
        """Test saving split to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.jsonl"

            split = DatasetSplit(
                name="test",
                pairs=[
                    QAPair(instruction="Q1", input="", output="A1"),
                    QAPair(instruction="Q2", input="", output="A2"),
                ],
            )

            split.save(output_path)

            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            parsed = json.loads(lines[0])
            assert parsed["instruction"] == "Q1"


class TestCleaningResult:
    """Tests for CleaningResult model."""

    def test_save_combined(self):
        """Test saving combined dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.jsonl"

            result = CleaningResult(
                train=DatasetSplit(
                    name="train",
                    pairs=[QAPair(instruction="T1", input="", output="A1")],
                ),
                validation=DatasetSplit(
                    name="validation",
                    pairs=[QAPair(instruction="V1", input="", output="A1")],
                ),
                test=DatasetSplit(
                    name="test",
                    pairs=[QAPair(instruction="E1", input="", output="A1")],
                ),
            )

            result.save_combined(output_path)

            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_save_splits(self):
        """Test saving split files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = CleaningResult(
                train=DatasetSplit(
                    name="train",
                    pairs=[QAPair(instruction="T1", input="", output="A1")],
                ),
                validation=DatasetSplit(
                    name="validation",
                    pairs=[QAPair(instruction="V1", input="", output="A1")],
                ),
                test=DatasetSplit(
                    name="test",
                    pairs=[QAPair(instruction="E1", input="", output="A1")],
                ),
            )

            paths = result.save(output_dir)

            assert "train" in paths
            assert "validation" in paths
            assert "test" in paths
            assert paths["train"].exists()


class TestCleaner:
    """Tests for Cleaner class."""

    def test_validate_pairs(self):
        """Test pair validation."""
        config = Config()
        cleaner = Cleaner(config)

        pairs = [
            QAPair(instruction="Valid question here?", input="", output="This is a valid answer with enough content."),
            QAPair(instruction="Q", input="", output="A"),  # Too short
            QAPair(instruction="What is [insert topic]?", input="", output="Valid answer here."),  # Placeholder
        ]

        valid = cleaner._validate_pairs(pairs)
        assert len(valid) == 1
        assert valid[0].instruction == "Valid question here?"

    def test_deduplicate(self):
        """Test deduplication."""
        config = Config()
        cleaner = Cleaner(config)

        pairs = [
            QAPair(instruction="What is Python?", input="", output="Python is a language."),
            QAPair(instruction="What is Python?", input="", output="Python is a language."),  # Duplicate
            QAPair(instruction="What is JavaScript?", input="", output="JavaScript is a language."),
            QAPair(instruction="what is python?", input="", output="python is a language."),  # Case-different duplicate
        ]

        unique = cleaner._deduplicate(pairs)
        assert len(unique) == 2

    def test_split(self):
        """Test train/val/test splitting."""
        config = Config()
        config.output.split_ratio = [0.8, 0.1, 0.1]
        cleaner = Cleaner(config)

        pairs = [QAPair(instruction=f"Q{i}", input="", output=f"A{i}") for i in range(100)]

        train, val, test = cleaner._split(pairs)

        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_clean_and_split(self):
        """Test full clean and split process."""
        config = Config()
        config.output.split_ratio = [0.6, 0.2, 0.2]
        cleaner = Cleaner(config)

        pairs = [
            QAPair(instruction=f"Valid question number {i}?", input="", output=f"This is answer {i} with enough content.")
            for i in range(50)
        ]
        # Add some duplicates
        pairs.extend(pairs[:10])

        result = cleaner.clean_and_split(pairs)

        assert result.original_count == 60
        assert result.cleaned_count == 50
        assert result.duplicates_removed == 10
        assert len(result.train.pairs) + len(result.validation.pairs) + len(result.test.pairs) == 50


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_jsonl(self):
        """Test loading JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            # Write test data
            with open(filepath, "w") as f:
                f.write(json.dumps({"instruction": "Q1", "input": "", "output": "A1"}) + "\n")
                f.write(json.dumps({"instruction": "Q2", "input": "I2", "output": "A2"}) + "\n")

            pairs = load_jsonl(filepath)

            assert len(pairs) == 2
            assert pairs[0].instruction == "Q1"
            assert pairs[1].input == "I2"

    def test_load_jsonl_with_errors(self):
        """Test loading JSONL with invalid lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.jsonl"

            with open(filepath, "w") as f:
                f.write(json.dumps({"instruction": "Q1", "input": "", "output": "A1"}) + "\n")
                f.write("invalid json\n")
                f.write(json.dumps({"instruction": "Q2", "input": "", "output": "A2"}) + "\n")

            pairs = load_jsonl(filepath)

            # Should skip invalid line
            assert len(pairs) == 2


class TestMergeDatasets:
    """Tests for merge_datasets function."""

    def test_merge_datasets(self):
        """Test merging multiple datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "data1.jsonl"
            file2 = Path(tmpdir) / "data2.jsonl"

            with open(file1, "w") as f:
                f.write(json.dumps({"instruction": "Q1", "input": "", "output": "A1"}) + "\n")

            with open(file2, "w") as f:
                f.write(json.dumps({"instruction": "Q2", "input": "", "output": "A2"}) + "\n")
                f.write(json.dumps({"instruction": "Q3", "input": "", "output": "A3"}) + "\n")

            pairs = merge_datasets(file1, file2)

            assert len(pairs) == 3
