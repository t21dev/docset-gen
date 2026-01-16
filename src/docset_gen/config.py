"""Configuration handling for DocSet Gen."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()

DEFAULT_CONFIG_FILE = "docset-gen.yaml"


class FirecrawlConfig(BaseModel):
    """Firecrawl API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("FIRECRAWL_API_KEY", ""))
    max_depth: int = Field(default=3, ge=1, le=5)
    exclude_patterns: list[str] = Field(default_factory=lambda: ["/api/*", "/changelog/*", "/blog/*"])
    include_patterns: list[str] = Field(default_factory=list)
    timeout: int = Field(default=30000, ge=1000, le=120000)

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable references like ${VAR_NAME}."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var, "")
        return v


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-5.1"))
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=100, le=16000)

    @field_validator("api_key", mode="before")
    @classmethod
    def resolve_env_var(cls, v: str) -> str:
        """Resolve environment variable references like ${VAR_NAME}."""
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_var = v[2:-1]
            return os.getenv(env_var, "")
        return v


class GenerationConfig(BaseModel):
    """Dataset generation configuration."""

    mode: Literal["qa", "completion", "chat"] = Field(default="qa")
    pairs_per_page: int = Field(default=5, ge=1, le=20)
    min_content_length: int = Field(default=100, ge=10)


class OutputConfig(BaseModel):
    """Output configuration."""

    format: Literal["jsonl"] = Field(default="jsonl")
    split_ratio: list[float] = Field(default=[0.8, 0.1, 0.1])
    min_quality_score: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("split_ratio")
    @classmethod
    def validate_split_ratio(cls, v: list[float]) -> list[float]:
        """Validate that split ratio sums to 1.0."""
        if len(v) != 3:
            raise ValueError("split_ratio must have exactly 3 values (train, val, test)")
        if abs(sum(v) - 1.0) > 0.001:
            raise ValueError("split_ratio must sum to 1.0")
        return v


class LLMsTxtConfig(BaseModel):
    """llms.txt generation configuration."""

    include_optional_section: bool = Field(default=True)
    max_links_per_section: int = Field(default=20, ge=1, le=100)
    generate_full_version: bool = Field(default=False)


class Config(BaseModel):
    """Main configuration model."""

    firecrawl: FirecrawlConfig = Field(default_factory=FirecrawlConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    llms_txt: LLMsTxtConfig = Field(default_factory=LLMsTxtConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        """Load configuration from YAML file and environment variables.

        Args:
            config_path: Path to config file. If None, looks for docset-gen.yaml
                        in current directory.

        Returns:
            Config: Loaded configuration.
        """
        config_data = {}

        # Determine config file path
        if config_path is None:
            config_path = Path.cwd() / DEFAULT_CONFIG_FILE

        # Load from YAML if exists
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

    def save(self, config_path: Path | None = None) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path to save config file. If None, saves to
                        docset-gen.yaml in current directory.
        """
        if config_path is None:
            config_path = Path.cwd() / DEFAULT_CONFIG_FILE

        # Convert to dict, excluding API keys for security
        config_dict = self.model_dump()

        # Replace actual API keys with environment variable references
        config_dict["firecrawl"]["api_key"] = "${FIRECRAWL_API_KEY}"
        config_dict["openai"]["api_key"] = "${OPENAI_API_KEY}"

        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are set.

        Returns:
            List of missing API key names.
        """
        missing = []
        if not self.firecrawl.api_key:
            missing.append("FIRECRAWL_API_KEY")
        if not self.openai.api_key:
            missing.append("OPENAI_API_KEY")
        return missing


def create_default_config(config_path: Path | None = None) -> Config:
    """Create and save a default configuration file.

    Args:
        config_path: Path to save config file.

    Returns:
        Config: The created default configuration.
    """
    config = Config()
    config.save(config_path)
    return config
