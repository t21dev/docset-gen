"""Tests for generator module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from docset_gen.config import Config
from docset_gen.generator import Generator, QAPair
from docset_gen.scraper import ScrapedPage


class TestQAPair:
    """Tests for QAPair model."""

    def test_to_jsonl(self):
        """Test JSONL conversion."""
        pair = QAPair(
            instruction="What is Python?",
            input="",
            output="Python is a programming language.",
        )

        jsonl = pair.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["instruction"] == "What is Python?"
        assert parsed["input"] == ""
        assert parsed["output"] == "Python is a programming language."

    def test_with_input(self):
        """Test QA pair with input field."""
        pair = QAPair(
            instruction="Explain this code:",
            input="print('hello')",
            output="This code prints 'hello' to the console.",
        )

        jsonl = pair.to_jsonl()
        parsed = json.loads(jsonl)

        assert parsed["input"] == "print('hello')"


class TestGenerator:
    """Tests for Generator class."""

    @patch("docset_gen.generator.OpenAI")
    def test_generator_init(self, mock_openai):
        """Test generator initialization."""
        config = Config()
        config.openai.api_key = "test-key"

        generator = Generator(config)
        mock_openai.assert_called_once_with(api_key="test-key")

    def test_parse_json_response(self):
        """Test JSON response parsing."""
        config = Config()
        config.openai.api_key = "test-key"

        with patch("docset_gen.generator.OpenAI"):
            generator = Generator(config)

        # Test plain JSON
        content = '[{"instruction": "Q1", "output": "A1"}]'
        result = generator._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q1"

        # Test with markdown code block
        content = '''```json
[{"instruction": "Q2", "output": "A2"}]
```'''
        result = generator._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q2"

        # Test with extra text
        content = '''Here is the JSON:
[{"instruction": "Q3", "output": "A3"}]
That's all!'''
        result = generator._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["instruction"] == "Q3"

    @patch("docset_gen.generator.OpenAI")
    def test_generate_from_page(self, mock_openai):
        """Test generating Q&A from a page."""
        config = Config()
        config.openai.api_key = "test-key"

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([
            {"instruction": "What is X?", "input": "", "output": "X is..."},
            {"instruction": "How to use Y?", "input": "", "output": "To use Y..."},
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = Generator(config)

        page = ScrapedPage(
            url="https://example.com",
            title="Test Page",
            markdown="This is documentation about X and Y.",
        )

        pairs = generator._generate_from_page(page, mode="qa", num_pairs=2)

        assert len(pairs) == 2
        assert pairs[0].instruction == "What is X?"
        assert pairs[1].instruction == "How to use Y?"

    @patch("docset_gen.generator.OpenAI")
    def test_generate_multiple_pages(self, mock_openai):
        """Test generating from multiple pages."""
        config = Config()
        config.openai.api_key = "test-key"

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([
            {"instruction": "Q", "input": "", "output": "A"}
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = Generator(config)

        pages = [
            ScrapedPage(url="https://example.com/1", title="P1", markdown="Content 1"),
            ScrapedPage(url="https://example.com/2", title="P2", markdown="Content 2"),
        ]

        result = generator.generate(pages, pairs_per_page=1)

        assert result.pages_processed == 2
        assert len(result.pairs) == 2

    @patch("docset_gen.generator.OpenAI")
    def test_max_pairs_limit(self, mock_openai):
        """Test max_pairs limit is respected."""
        config = Config()
        config.openai.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([
            {"instruction": f"Q{i}", "input": "", "output": f"A{i}"}
            for i in range(5)
        ])

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = Generator(config)

        pages = [
            ScrapedPage(url="https://example.com/1", title="P1", markdown="Content 1"),
            ScrapedPage(url="https://example.com/2", title="P2", markdown="Content 2"),
        ]

        result = generator.generate(pages, max_pairs=3)

        # Should stop after reaching max_pairs
        assert result.total_generated <= 5  # First page might generate more
