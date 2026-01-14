"""Firecrawl integration for documentation scraping."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from firecrawl import Firecrawl
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config

logger = logging.getLogger(__name__)


class ScrapedPage(BaseModel):
    """Represents a scraped documentation page."""

    url: str
    title: str = ""
    markdown: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def word_count(self) -> int:
        """Return the word count of the markdown content."""
        return len(self.markdown.split())


class ScrapeResult(BaseModel):
    """Result of a scraping operation."""

    pages: list[ScrapedPage] = Field(default_factory=list)
    total_pages: int = 0
    failed_urls: list[str] = Field(default_factory=list)

    def save(self, output_dir: Path) -> None:
        """Save scraped pages to output directory.

        Args:
            output_dir: Directory to save scraped content.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each page as markdown
        for i, page in enumerate(self.pages):
            # Create a safe filename from URL
            safe_name = page.url.replace("://", "_").replace("/", "_").replace("?", "_")[:100]
            filename = f"{i:04d}_{safe_name}.md"
            filepath = output_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {page.title}\n\n")
                f.write(f"Source: {page.url}\n\n")
                f.write("---\n\n")
                f.write(page.markdown)

        # Save metadata
        metadata_path = output_dir / "_metadata.json"
        metadata = {
            "total_pages": self.total_pages,
            "scraped_pages": len(self.pages),
            "failed_urls": self.failed_urls,
            "pages": [
                {"url": p.url, "title": p.title, "word_count": p.word_count()}
                for p in self.pages
            ],
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(self.pages)} pages to {output_dir}")


class Scraper:
    """Documentation scraper using Firecrawl API."""

    def __init__(self, config: Config) -> None:
        """Initialize the scraper.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.app = Firecrawl(api_key=config.firecrawl.api_key)

    def scrape(
        self,
        url: str,
        depth: int | None = None,
        verbose: bool = False,
    ) -> ScrapeResult:
        """Scrape a documentation site.

        Args:
            url: Base URL to scrape.
            depth: Maximum crawl depth. Uses config default if None.
            verbose: Whether to show detailed progress.

        Returns:
            ScrapeResult: The scraped pages and metadata.
        """
        depth = depth or self.config.firecrawl.max_depth

        logger.info(f"Starting crawl of {url} with depth {depth}")

        result = ScrapeResult()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Crawling {url}...", total=None)

            try:
                # Use Firecrawl's crawl method (v2 API)
                crawl_result = self.app.v2.crawl(
                    url,
                    limit=100,
                    max_discovery_depth=depth,
                    exclude_paths=self.config.firecrawl.exclude_patterns or None,
                    include_paths=self.config.firecrawl.include_patterns or None,
                    poll_interval=5,
                )

                progress.update(task, description="Processing results...")

                # Process crawl results - v2 API returns data directly or in 'data' key
                pages_data = []
                if isinstance(crawl_result, dict):
                    pages_data = crawl_result.get("data", [])
                elif hasattr(crawl_result, "data"):
                    pages_data = crawl_result.data or []

                for page_data in pages_data:
                    page = self._process_page(page_data)
                    if page and page.markdown.strip():
                        result.pages.append(page)

                result.total_pages = len(pages_data)

                if verbose:
                    logger.info(f"Crawled {result.total_pages} pages")

            except Exception as e:
                logger.error(f"Crawl failed: {e}")
                result.failed_urls.append(url)
                raise

        logger.info(
            f"Scraping complete: {len(result.pages)} pages scraped, "
            f"{len(result.failed_urls)} failed"
        )

        return result

    def _process_page(self, page_data: dict[str, Any] | Any) -> ScrapedPage | None:
        """Process raw page data from Firecrawl.

        Args:
            page_data: Raw page data from Firecrawl API.

        Returns:
            ScrapedPage or None if page should be skipped.
        """
        try:
            # Handle both dict and object responses
            if isinstance(page_data, dict):
                metadata = page_data.get("metadata", {})
                url = metadata.get("sourceURL", "") or metadata.get("url", "")
                title = metadata.get("title", "")
                markdown = page_data.get("markdown", "")
            else:
                # Object-style response (v2 API returns Document objects)
                markdown = getattr(page_data, "markdown", "") or ""
                metadata_obj = getattr(page_data, "metadata", None)

                if metadata_obj:
                    # DocumentMetadata is an object with attributes, not a dict
                    url = getattr(metadata_obj, "source_url", None) or getattr(metadata_obj, "sourceURL", None) or getattr(metadata_obj, "url", None) or ""
                    title = getattr(metadata_obj, "title", None) or ""
                else:
                    url = getattr(page_data, "url", "") or ""
                    title = ""

            # Skip pages with too little content
            if len(markdown) < self.config.generation.min_content_length:
                logger.debug(f"Skipping {url}: content too short")
                return None

            return ScrapedPage(
                url=url,
                title=title,
                markdown=markdown,
                metadata={},
            )

        except Exception as e:
            logger.warning(f"Failed to process page: {e}")
            return None


def load_scraped_content(input_dir: Path) -> list[ScrapedPage]:
    """Load previously scraped content from directory.

    Args:
        input_dir: Directory containing scraped markdown files.

    Returns:
        List of ScrapedPage objects.
    """
    pages = []

    # Load metadata if available
    metadata_path = input_dir / "_metadata.json"
    metadata_map = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
            metadata_map = {p["url"]: p for p in metadata.get("pages", [])}

    # Load markdown files
    for md_file in sorted(input_dir.glob("*.md")):
        if md_file.name.startswith("_"):
            continue

        with open(md_file, encoding="utf-8") as f:
            content = f.read()

        # Parse the markdown file
        lines = content.split("\n")
        title = ""
        url = ""
        markdown_start = 0

        for i, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("Source: "):
                url = line[8:].strip()
            elif line.strip() == "---":
                markdown_start = i + 2
                break

        markdown = "\n".join(lines[markdown_start:])

        if markdown.strip():
            pages.append(
                ScrapedPage(
                    url=url,
                    title=title,
                    markdown=markdown,
                    metadata=metadata_map.get(url, {}),
                )
            )

    logger.info(f"Loaded {len(pages)} pages from {input_dir}")
    return pages
