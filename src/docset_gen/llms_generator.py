"""llms.txt generation from scraped documentation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Config
from .scraper import ScrapedPage

logger = logging.getLogger(__name__)


class LLMsTxtLink(BaseModel):
    """A single link entry in llms.txt."""

    title: str
    url: str
    description: str = ""


class LLMsTxtSection(BaseModel):
    """A section in the llms.txt file."""

    name: str
    links: list[LLMsTxtLink] = Field(default_factory=list)
    is_optional: bool = False


class LLMsTxtResult(BaseModel):
    """Result of llms.txt generation."""

    project_name: str
    summary: str = ""
    sections: list[LLMsTxtSection] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the result as llms.txt markdown format.

        Returns:
            Formatted llms.txt content.
        """
        lines = []

        # H1 heading (required)
        lines.append(f"# {self.project_name}")
        lines.append("")

        # Blockquote summary (optional)
        if self.summary:
            lines.append(f"> {self.summary}")
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append(f"## {section.name}")
            lines.append("")
            for link in section.links:
                if link.description:
                    lines.append(f"- [{link.title}]({link.url}): {link.description}")
                else:
                    lines.append(f"- [{link.title}]({link.url})")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save the llms.txt to a file.

        Args:
            path: Output file path.
        """
        content = self.to_markdown()
        path.write_text(content, encoding="utf-8")
        logger.info(f"Saved llms.txt to {path}")

    @property
    def total_links(self) -> int:
        """Get total number of links across all sections."""
        return sum(len(section.links) for section in self.sections)


class LLMsTxtGenerator:
    """Generate llms.txt from scraped documentation."""

    CATEGORIZATION_PROMPT = """Analyze these documentation pages and categorize them into logical sections for an llms.txt file.

Pages to categorize:
{pages_info}

Return a JSON object with this structure:
{{
    "sections": [
        {{
            "name": "Section Name",
            "is_optional": false,
            "pages": [
                {{
                    "url": "page url",
                    "title": "Page Title",
                    "description": "Brief 5-10 word description of what this page covers"
                }}
            ]
        }}
    ]
}}

Guidelines:
- Common section names: "Docs", "API Reference", "Guides", "Examples", "Tutorials", "Optional"
- Use "Optional" section for changelog, contributing, and other supplementary content
- Each page should only appear in one section
- Order sections by importance (most important first)
- Limit to 5-7 sections maximum
- Write concise descriptions that explain what the page is about

Return ONLY valid JSON, no other text."""

    SUMMARY_PROMPT = """Based on these documentation page titles and URLs, write a 1-2 sentence summary that explains what this project does.

Project name: {project_name}

Pages:
{pages_info}

Requirements:
- Be concise and focus on key capabilities
- Don't start with "This is" or "This project"
- Focus on what users can do with the project
- Maximum 150 characters

Return ONLY the summary text, no quotes or other formatting."""

    def __init__(self, config: Config) -> None:
        """Initialize the generator.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai.api_key)

    def generate(
        self,
        pages: list[ScrapedPage],
        site_name: str,
    ) -> LLMsTxtResult:
        """Generate llms.txt from scraped pages.

        Args:
            pages: List of scraped documentation pages.
            site_name: Name of the project/site.

        Returns:
            LLMsTxtResult: Generated llms.txt content.
        """
        logger.info(f"Generating llms.txt for {site_name} from {len(pages)} pages")

        # Generate summary
        summary = self._generate_summary(pages, site_name)

        # Categorize pages into sections
        sections = self._categorize_pages(pages)

        # Apply max_links_per_section limit
        max_links = self.config.llms_txt.max_links_per_section
        for section in sections:
            if len(section.links) > max_links:
                section.links = section.links[:max_links]

        # Filter out Optional section if not configured
        if not self.config.llms_txt.include_optional_section:
            sections = [s for s in sections if not s.is_optional]

        result = LLMsTxtResult(
            project_name=site_name,
            summary=summary,
            sections=sections,
        )

        logger.info(
            f"Generated llms.txt with {len(result.sections)} sections, "
            f"{result.total_links} total links"
        )

        return result

    def _generate_summary(self, pages: list[ScrapedPage], project_name: str) -> str:
        """Generate a concise project summary.

        Args:
            pages: List of scraped pages.
            project_name: Name of the project.

        Returns:
            Summary string for the blockquote.
        """
        # Build pages info for the prompt
        pages_info = "\n".join(
            f"- {p.title or 'Untitled'}: {p.url}" for p in pages[:30]
        )

        prompt = self.SUMMARY_PROMPT.format(
            project_name=project_name,
            pages_info=pages_info,
        )

        try:
            response = self._call_openai(prompt)
            summary = response.strip().strip('"\'')
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return ""

    def _categorize_pages(self, pages: list[ScrapedPage]) -> list[LLMsTxtSection]:
        """Categorize pages into logical sections.

        Args:
            pages: List of scraped pages.

        Returns:
            List of categorized sections.
        """
        # Build pages info for the prompt
        pages_info = "\n".join(
            f"- URL: {p.url}\n  Title: {p.title or 'Untitled'}"
            for p in pages
        )

        prompt = self.CATEGORIZATION_PROMPT.format(pages_info=pages_info)

        try:
            response = self._call_openai(prompt, max_tokens=4096)
            categories = self._parse_categorization_response(response)
            return categories
        except Exception as e:
            logger.warning(f"Failed to categorize pages: {e}")
            # Fallback: put all pages in a single "Docs" section
            return self._fallback_categorization(pages)

    def _parse_categorization_response(self, response: str) -> list[LLMsTxtSection]:
        """Parse the categorization response from OpenAI.

        Args:
            response: JSON response string.

        Returns:
            List of LLMsTxtSection objects.
        """
        # Clean up the response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            response = response.strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    data = json.loads(response[start:end])
                except json.JSONDecodeError:
                    logger.warning("Failed to parse categorization JSON")
                    return []
            else:
                return []

        sections = []
        for section_data in data.get("sections", []):
            links = []
            for page_data in section_data.get("pages", []):
                links.append(
                    LLMsTxtLink(
                        title=page_data.get("title", "Untitled"),
                        url=page_data.get("url", ""),
                        description=page_data.get("description", ""),
                    )
                )

            if links:
                sections.append(
                    LLMsTxtSection(
                        name=section_data.get("name", "Documentation"),
                        links=links,
                        is_optional=section_data.get("is_optional", False),
                    )
                )

        return sections

    def _fallback_categorization(
        self, pages: list[ScrapedPage]
    ) -> list[LLMsTxtSection]:
        """Create a simple fallback categorization.

        Args:
            pages: List of scraped pages.

        Returns:
            Single "Docs" section with all pages.
        """
        links = [
            LLMsTxtLink(
                title=p.title or "Untitled",
                url=p.url,
                description="",
            )
            for p in pages
        ]

        return [LLMsTxtSection(name="Docs", links=links)]

    def _call_openai(self, prompt: str, max_tokens: int | None = None) -> str:
        """Call OpenAI API with the given prompt.

        Args:
            prompt: User prompt to send.
            max_tokens: Optional max tokens override.

        Returns:
            Response content string.
        """
        tokens = max_tokens or self.config.openai.max_tokens

        # Reasoning models (o1, o3, o4) and gpt-5.x require max_completion_tokens
        model_name = self.config.openai.model
        model_lower = model_name.lower()
        use_new_param = any(
            x in model_lower for x in ["o1", "o3", "o4", "gpt-5", "gpt-4.1"]
        )

        api_params = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.openai.temperature,
        }

        if use_new_param:
            api_params["max_completion_tokens"] = tokens
        else:
            api_params["max_tokens"] = tokens

        response = self.client.chat.completions.create(**api_params)
        content = response.choices[0].message.content

        return content or ""
