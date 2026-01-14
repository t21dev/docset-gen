"""OpenAI-powered Q&A pair generation."""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import Config
from .scraper import ScrapedPage

logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    """A question-answer pair for training."""

    instruction: str
    input: str = ""
    output: str

    def to_jsonl(self) -> str:
        """Convert to JSONL format string."""
        return json.dumps(self.model_dump(), ensure_ascii=False)


class GenerationResult(BaseModel):
    """Result of Q&A generation."""

    pairs: list[QAPair] = Field(default_factory=list)
    total_generated: int = 0
    pages_processed: int = 0
    errors: list[str] = Field(default_factory=list)


class Generator:
    """Q&A pair generator using OpenAI API."""

    SYSTEM_PROMPTS = {
        "qa": """You are an expert at creating high-quality question-answer pairs from documentation.
Given a piece of documentation, generate clear, specific questions that users might ask,
along with accurate, helpful answers based solely on the provided content.

Requirements:
- Questions should be natural and diverse (how, what, why, when, etc.)
- Answers should be accurate and derived from the documentation
- Each Q&A pair should be self-contained and useful
- Avoid generic or obvious questions
- Focus on practical, actionable information""",
        "completion": """You are an expert at creating completion-style training data from documentation.
Given a piece of documentation, create context-completion pairs where the context sets up
a scenario or partial information, and the completion provides the natural continuation.

Requirements:
- Context should be meaningful and set up clear expectations
- Completions should follow naturally from the context
- Focus on technical accuracy and practical information""",
        "chat": """You are an expert at creating multi-turn conversation training data.
Given a piece of documentation, create realistic conversations between a user seeking help
and an assistant providing guidance based on the documentation.

Requirements:
- Conversations should feel natural and helpful
- Include follow-up questions when appropriate
- Answers should be accurate and based on the documentation
- Focus on practical, actionable information""",
    }

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
        mode: Literal["qa", "completion", "chat"] | None = None,
        pairs_per_page: int | None = None,
        max_pairs: int | None = None,
        verbose: bool = False,
    ) -> GenerationResult:
        """Generate Q&A pairs from scraped pages.

        Args:
            pages: List of scraped documentation pages.
            mode: Generation mode (qa, completion, chat).
            pairs_per_page: Number of pairs to generate per page.
            max_pairs: Maximum total pairs to generate.
            verbose: Whether to show detailed progress.

        Returns:
            GenerationResult: Generated Q&A pairs and metadata.
        """
        mode = mode or self.config.generation.mode
        pairs_per_page = pairs_per_page or self.config.generation.pairs_per_page

        # If max_pairs requires more pairs per page, adjust dynamically
        if max_pairs and len(pages) > 0:
            needed_per_page = (max_pairs + len(pages) - 1) // len(pages)  # Ceiling division
            pairs_per_page = max(pairs_per_page, min(needed_per_page, 20))  # Cap at 20 per page

        result = GenerationResult()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Generating Q&A pairs...", total=len(pages))

            for page in pages:
                # Check if we've hit the max
                if max_pairs and result.total_generated >= max_pairs:
                    break

                try:
                    remaining = None
                    if max_pairs:
                        remaining = max_pairs - result.total_generated
                        if remaining <= 0:
                            break

                    pairs = self._generate_from_page(
                        page,
                        mode=mode,
                        num_pairs=min(pairs_per_page, remaining) if remaining else pairs_per_page,
                    )

                    result.pairs.extend(pairs)
                    result.total_generated += len(pairs)
                    result.pages_processed += 1

                    if verbose:
                        logger.info(f"Generated {len(pairs)} pairs from {page.url}")

                except Exception as e:
                    error_msg = f"Failed to generate from {page.url}: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)

                progress.update(task, advance=1)

        logger.info(
            f"Generation complete: {result.total_generated} pairs from "
            f"{result.pages_processed} pages"
        )

        return result

    def _generate_from_page(
        self,
        page: ScrapedPage,
        mode: Literal["qa", "completion", "chat"],
        num_pairs: int,
    ) -> list[QAPair]:
        """Generate Q&A pairs from a single page.

        Args:
            page: The scraped page to generate from.
            mode: Generation mode.
            num_pairs: Number of pairs to generate.

        Returns:
            List of generated Q&A pairs.
        """
        system_prompt = self.SYSTEM_PROMPTS[mode]

        user_prompt = f"""Based on the following documentation, generate exactly {num_pairs} high-quality question-answer pairs.

Documentation Title: {page.title}
Source URL: {page.url}

Content:
{page.markdown[:8000]}  # Limit content to avoid token limits

Return the pairs as a JSON array with objects containing "instruction" (the question) and "output" (the answer) fields.
The "input" field should be empty for Q&A pairs.

Example format:
[
    {{"instruction": "How do I configure X?", "input": "", "output": "To configure X, you need to..."}},
    {{"instruction": "What is the purpose of Y?", "input": "", "output": "Y is used for..."}}
]

Return ONLY the JSON array, no other text."""

        try:
            # Reasoning models (o1, o3, o4) and gpt-5.x require max_completion_tokens
            # Older models (gpt-3.5, gpt-4, gpt-4o, gpt-4-turbo) use max_tokens
            model_name = self.config.openai.model
            model_lower = model_name.lower()
            use_new_param = any(x in model_lower for x in ["o1", "o3", "o4", "gpt-5", "gpt-4.1"])

            api_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.openai.temperature,
            }

            if use_new_param:
                api_params["max_completion_tokens"] = self.config.openai.max_tokens
            else:
                api_params["max_tokens"] = self.config.openai.max_tokens

            response = self.client.chat.completions.create(**api_params)

            content = response.choices[0].message.content
            if not content:
                return []

            # Parse JSON response
            pairs_data = self._parse_json_response(content)

            # Convert to QAPair objects
            pairs = []
            for pair_data in pairs_data:
                if "instruction" in pair_data and "output" in pair_data:
                    pairs.append(
                        QAPair(
                            instruction=pair_data["instruction"],
                            input=pair_data.get("input", ""),
                            output=pair_data["output"],
                        )
                    )

            return pairs

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _parse_json_response(self, content: str) -> list[dict]:
        """Parse JSON from OpenAI response.

        Args:
            content: Response content from OpenAI.

        Returns:
            List of parsed dictionaries.
        """
        # Try to find JSON array in the response
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find array within the content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {content[:200]}...")
            return []
