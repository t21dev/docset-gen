"""Tests for scraper module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docset_gen.config import Config
from docset_gen.scraper import ScrapedPage, ScrapeResult, Scraper, load_scraped_content


class TestScrapedPage:
    """Tests for ScrapedPage model."""

    def test_word_count(self):
        """Test word count calculation."""
        page = ScrapedPage(
            url="https://example.com",
            title="Test Page",
            markdown="This is a test page with some content.",
        )
        assert page.word_count() == 8

    def test_empty_markdown(self):
        """Test page with empty markdown."""
        page = ScrapedPage(
            url="https://example.com",
            title="Empty",
            markdown="",
        )
        assert page.word_count() == 0


class TestScrapeResult:
    """Tests for ScrapeResult model."""

    def test_save(self):
        """Test saving scrape results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            result = ScrapeResult(
                pages=[
                    ScrapedPage(
                        url="https://example.com/page1",
                        title="Page 1",
                        markdown="# Page 1\n\nContent for page 1.",
                    ),
                    ScrapedPage(
                        url="https://example.com/page2",
                        title="Page 2",
                        markdown="# Page 2\n\nContent for page 2.",
                    ),
                ],
                total_pages=2,
            )

            result.save(output_dir)

            # Check files were created
            assert output_dir.exists()
            md_files = list(output_dir.glob("*.md"))
            assert len(md_files) == 2

            # Check metadata file
            metadata_path = output_dir / "_metadata.json"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                metadata = json.load(f)
            assert metadata["total_pages"] == 2
            assert metadata["scraped_pages"] == 2


class TestScraper:
    """Tests for Scraper class."""

    @patch("docset_gen.scraper.FirecrawlApp")
    def test_scraper_init(self, mock_firecrawl):
        """Test scraper initialization."""
        config = Config()
        config.firecrawl.api_key = "test-key"

        scraper = Scraper(config)
        mock_firecrawl.assert_called_once_with(api_key="test-key")

    @patch("docset_gen.scraper.FirecrawlApp")
    def test_process_page(self, mock_firecrawl):
        """Test page processing."""
        config = Config()
        config.firecrawl.api_key = "test-key"
        config.generation.min_content_length = 10

        scraper = Scraper(config)

        page_data = {
            "metadata": {
                "sourceURL": "https://example.com/test",
                "title": "Test Page",
            },
            "markdown": "This is test content that is long enough.",
        }

        page = scraper._process_page(page_data)
        assert page is not None
        assert page.url == "https://example.com/test"
        assert page.title == "Test Page"

    @patch("docset_gen.scraper.FirecrawlApp")
    def test_skip_short_content(self, mock_firecrawl):
        """Test skipping pages with too little content."""
        config = Config()
        config.firecrawl.api_key = "test-key"
        config.generation.min_content_length = 1000

        scraper = Scraper(config)

        page_data = {
            "metadata": {"sourceURL": "https://example.com/test"},
            "markdown": "Short content",
        }

        page = scraper._process_page(page_data)
        assert page is None


class TestLoadScrapedContent:
    """Tests for loading scraped content."""

    def test_load_scraped_content(self):
        """Test loading scraped markdown files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test markdown files
            md_content = """# Test Page

Source: https://example.com/test

---

This is the markdown content.
"""
            (output_dir / "0001_test.md").write_text(md_content)

            # Create metadata
            metadata = {
                "pages": [{"url": "https://example.com/test", "title": "Test Page"}]
            }
            (output_dir / "_metadata.json").write_text(json.dumps(metadata))

            pages = load_scraped_content(output_dir)
            assert len(pages) == 1
            assert pages[0].url == "https://example.com/test"
            assert "markdown content" in pages[0].markdown

    def test_load_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = load_scraped_content(Path(tmpdir))
            assert len(pages) == 0
