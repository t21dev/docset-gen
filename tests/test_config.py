"""Tests for configuration handling."""

import os
import tempfile
from pathlib import Path

import pytest

from docset_gen.config import Config, FirecrawlConfig, OpenAIConfig, OutputConfig


class TestFirecrawlConfig:
    """Tests for FirecrawlConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FirecrawlConfig()
        assert config.max_depth == 3
        assert config.timeout == 30000
        assert "/api/*" in config.exclude_patterns

    def test_env_var_resolution(self):
        """Test environment variable resolution."""
        os.environ["TEST_API_KEY"] = "test-key-123"
        try:
            config = FirecrawlConfig(api_key="${TEST_API_KEY}")
            assert config.api_key == "test-key-123"
        finally:
            del os.environ["TEST_API_KEY"]

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            FirecrawlConfig(max_depth=0)

        with pytest.raises(ValueError):
            FirecrawlConfig(max_depth=100)


class TestOpenAIConfig:
    """Tests for OpenAIConfig."""

    def test_default_model(self):
        """Test default model setting."""
        config = OpenAIConfig()
        assert config.model == "gpt-4o-mini"

    def test_temperature_bounds(self):
        """Test temperature validation."""
        config = OpenAIConfig(temperature=0.5)
        assert config.temperature == 0.5

        with pytest.raises(ValueError):
            OpenAIConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            OpenAIConfig(temperature=2.5)


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_split_ratio_validation(self):
        """Test split ratio must sum to 1.0."""
        config = OutputConfig(split_ratio=[0.8, 0.1, 0.1])
        assert sum(config.split_ratio) == pytest.approx(1.0)

        with pytest.raises(ValueError):
            OutputConfig(split_ratio=[0.5, 0.5, 0.5])

        with pytest.raises(ValueError):
            OutputConfig(split_ratio=[0.8, 0.2])


class TestConfig:
    """Tests for main Config class."""

    def test_load_default(self):
        """Test loading default configuration."""
        config = Config.load()
        assert config.firecrawl is not None
        assert config.openai is not None
        assert config.generation is not None
        assert config.output is not None

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test-config.yaml"

            # Create and save config
            config = Config()
            config.save(config_path)

            assert config_path.exists()

            # Load the saved config
            loaded = Config.load(config_path)
            assert loaded.firecrawl.max_depth == config.firecrawl.max_depth
            assert loaded.openai.model == config.openai.model

    def test_validate_api_keys(self):
        """Test API key validation."""
        config = Config()

        # Clear API keys
        config.firecrawl.api_key = ""
        config.openai.api_key = ""

        missing = config.validate_api_keys()
        assert "FIRECRAWL_API_KEY" in missing
        assert "OPENAI_API_KEY" in missing

        # Set one key
        config.firecrawl.api_key = "test-key"
        missing = config.validate_api_keys()
        assert "FIRECRAWL_API_KEY" not in missing
        assert "OPENAI_API_KEY" in missing
