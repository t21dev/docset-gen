"""Tests for llms.txt generation."""

import tempfile
from pathlib import Path

import pytest

from docset_gen.config import Config, LLMsTxtConfig
from docset_gen.llms_generator import (
    LLMsTxtGenerator,
    LLMsTxtLink,
    LLMsTxtResult,
    LLMsTxtSection,
)
from docset_gen.scraper import ScrapedPage


class TestLLMsTxtLink:
    """Tests for LLMsTxtLink model."""

    def test_link_creation(self):
        """Test creating a link."""
        link = LLMsTxtLink(
            title="Getting Started",
            url="https://example.com/docs/getting-started",
            description="Quick start guide",
        )
        assert link.title == "Getting Started"
        assert link.url == "https://example.com/docs/getting-started"
        assert link.description == "Quick start guide"

    def test_link_without_description(self):
        """Test creating a link without description."""
        link = LLMsTxtLink(
            title="Home",
            url="https://example.com",
        )
        assert link.title == "Home"
        assert link.description == ""

    def test_link_with_content(self):
        """Test creating a link with full content (for full mode)."""
        link = LLMsTxtLink(
            title="Getting Started",
            url="https://example.com/docs/start",
            description="Quick start guide",
            content="# Getting Started\n\nThis is the full content of the page.",
        )
        assert link.title == "Getting Started"
        assert link.content == "# Getting Started\n\nThis is the full content of the page."


class TestLLMsTxtSection:
    """Tests for LLMsTxtSection model."""

    def test_section_creation(self):
        """Test creating a section with links."""
        links = [
            LLMsTxtLink(title="Page 1", url="https://example.com/1"),
            LLMsTxtLink(title="Page 2", url="https://example.com/2"),
        ]
        section = LLMsTxtSection(name="Docs", links=links)

        assert section.name == "Docs"
        assert len(section.links) == 2
        assert section.is_optional is False

    def test_optional_section(self):
        """Test creating an optional section."""
        section = LLMsTxtSection(name="Optional", links=[], is_optional=True)
        assert section.is_optional is True


class TestLLMsTxtResult:
    """Tests for LLMsTxtResult model."""

    def test_result_creation(self):
        """Test creating a result."""
        result = LLMsTxtResult(
            project_name="Test Project",
            summary="A test project for testing.",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(
                            title="Getting Started",
                            url="https://example.com/docs/start",
                            description="Quick start guide",
                        )
                    ],
                )
            ],
        )

        assert result.project_name == "Test Project"
        assert result.summary == "A test project for testing."
        assert len(result.sections) == 1
        assert result.total_links == 1

    def test_to_markdown(self):
        """Test markdown rendering."""
        result = LLMsTxtResult(
            project_name="Test Project",
            summary="A test project for testing.",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(
                            title="Getting Started",
                            url="https://example.com/docs/start",
                            description="Quick start guide",
                        ),
                        LLMsTxtLink(
                            title="Config",
                            url="https://example.com/docs/config",
                        ),
                    ],
                ),
                LLMsTxtSection(
                    name="API Reference",
                    links=[
                        LLMsTxtLink(
                            title="Auth",
                            url="https://example.com/api/auth",
                            description="Authentication endpoints",
                        ),
                    ],
                ),
            ],
        )

        markdown = result.to_markdown()

        # Check H1 heading
        assert "# Test Project" in markdown

        # Check blockquote
        assert "> A test project for testing." in markdown

        # Check section headings
        assert "## Docs" in markdown
        assert "## API Reference" in markdown

        # Check links with descriptions
        assert "- [Getting Started](https://example.com/docs/start): Quick start guide" in markdown
        assert "- [Auth](https://example.com/api/auth): Authentication endpoints" in markdown

        # Check links without descriptions
        assert "- [Config](https://example.com/docs/config)" in markdown

    def test_to_markdown_without_summary(self):
        """Test markdown rendering without summary."""
        result = LLMsTxtResult(
            project_name="Test Project",
            sections=[],
        )

        markdown = result.to_markdown()

        assert "# Test Project" in markdown
        assert ">" not in markdown

    def test_to_markdown_full_mode(self):
        """Test markdown rendering in full mode with content."""
        result = LLMsTxtResult(
            project_name="Test Project",
            summary="A test project.",
            mode="full",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(
                            title="Getting Started",
                            url="https://example.com/docs/start",
                            description="Quick start guide",
                            content="# Getting Started\n\nWelcome to the guide!",
                        ),
                    ],
                ),
            ],
        )

        markdown = result.to_markdown()

        # Check H1 heading
        assert "# Test Project" in markdown

        # Check section heading
        assert "## Docs" in markdown

        # Check H3 link heading (full mode uses ### for links)
        assert "### [Getting Started](https://example.com/docs/start)" in markdown

        # Check description blockquote
        assert "> Quick start guide" in markdown

        # Check full content is included
        assert "# Getting Started" in markdown
        assert "Welcome to the guide!" in markdown

    def test_to_markdown_minimal_mode(self):
        """Test markdown rendering in minimal mode (default)."""
        result = LLMsTxtResult(
            project_name="Test Project",
            mode="minimal",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(
                            title="Getting Started",
                            url="https://example.com/docs/start",
                            description="Quick start guide",
                            content="Full content here",  # Should be ignored in minimal mode
                        ),
                    ],
                ),
            ],
        )

        markdown = result.to_markdown()

        # Check minimal format (list item, not H3)
        assert "- [Getting Started](https://example.com/docs/start): Quick start guide" in markdown

        # Content should not be included
        assert "Full content here" not in markdown

    def test_save(self):
        """Test saving to file."""
        result = LLMsTxtResult(
            project_name="Test Project",
            summary="Test summary",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(title="Home", url="https://example.com"),
                    ],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "llms.txt"
            result.save(output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "# Test Project" in content

    def test_total_links(self):
        """Test total links calculation."""
        result = LLMsTxtResult(
            project_name="Test",
            sections=[
                LLMsTxtSection(
                    name="Docs",
                    links=[
                        LLMsTxtLink(title="Page 1", url="https://example.com/1"),
                        LLMsTxtLink(title="Page 2", url="https://example.com/2"),
                    ],
                ),
                LLMsTxtSection(
                    name="API",
                    links=[
                        LLMsTxtLink(title="Page 3", url="https://example.com/3"),
                    ],
                ),
            ],
        )

        assert result.total_links == 3


class TestLLMsTxtConfig:
    """Tests for LLMsTxtConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMsTxtConfig()
        assert config.include_optional_section is True
        assert config.max_links_per_section == 20
        assert config.generate_full_version is False

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            LLMsTxtConfig(max_links_per_section=0)

        with pytest.raises(ValueError):
            LLMsTxtConfig(max_links_per_section=101)


class TestLLMsTxtGenerator:
    """Tests for LLMsTxtGenerator."""

    def test_fallback_categorization(self):
        """Test fallback categorization when GPT fails."""
        config = Config()
        generator = LLMsTxtGenerator(config)

        pages = [
            ScrapedPage(
                url="https://example.com/page1",
                title="Page 1",
                markdown="Content 1",
            ),
            ScrapedPage(
                url="https://example.com/page2",
                title="Page 2",
                markdown="Content 2",
            ),
        ]

        sections = generator._fallback_categorization(pages)

        assert len(sections) == 1
        assert sections[0].name == "Docs"
        assert len(sections[0].links) == 2

    def test_fallback_categorization_with_content_map(self):
        """Test fallback categorization with content map for full mode."""
        config = Config()
        generator = LLMsTxtGenerator(config)

        pages = [
            ScrapedPage(
                url="https://example.com/page1",
                title="Page 1",
                markdown="Content for page 1",
            ),
            ScrapedPage(
                url="https://example.com/page2",
                title="Page 2",
                markdown="Content for page 2",
            ),
        ]

        content_map = {
            "https://example.com/page1": "Content for page 1",
            "https://example.com/page2": "Content for page 2",
        }

        sections = generator._fallback_categorization(pages, content_map)

        assert len(sections) == 1
        assert sections[0].links[0].content == "Content for page 1"
        assert sections[0].links[1].content == "Content for page 2"

    def test_parse_categorization_response_valid(self):
        """Test parsing valid categorization JSON response."""
        config = Config()
        generator = LLMsTxtGenerator(config)

        response = '''
        {
            "sections": [
                {
                    "name": "Docs",
                    "is_optional": false,
                    "pages": [
                        {
                            "url": "https://example.com/docs",
                            "title": "Documentation",
                            "description": "Main docs"
                        }
                    ]
                },
                {
                    "name": "Optional",
                    "is_optional": true,
                    "pages": [
                        {
                            "url": "https://example.com/changelog",
                            "title": "Changelog",
                            "description": "Version history"
                        }
                    ]
                }
            ]
        }
        '''

        sections = generator._parse_categorization_response(response)

        assert len(sections) == 2
        assert sections[0].name == "Docs"
        assert sections[0].is_optional is False
        assert len(sections[0].links) == 1
        assert sections[0].links[0].title == "Documentation"
        assert sections[1].name == "Optional"
        assert sections[1].is_optional is True

    def test_parse_categorization_response_with_code_blocks(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        config = Config()
        generator = LLMsTxtGenerator(config)

        response = '''```json
        {
            "sections": [
                {
                    "name": "Docs",
                    "pages": [
                        {
                            "url": "https://example.com/docs",
                            "title": "Documentation",
                            "description": "Main docs"
                        }
                    ]
                }
            ]
        }
        ```'''

        sections = generator._parse_categorization_response(response)

        assert len(sections) == 1
        assert sections[0].name == "Docs"

    def test_parse_categorization_response_invalid(self):
        """Test parsing invalid JSON response."""
        config = Config()
        generator = LLMsTxtGenerator(config)

        response = "This is not valid JSON"

        sections = generator._parse_categorization_response(response)

        assert sections == []
