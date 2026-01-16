"""DocSet Gen - Transform documentation into LLM training datasets."""

__version__ = "1.0.0"
__author__ = "TriptoAfsin"

from .cleaner import Cleaner, CleaningResult, DatasetSplit, QAPair
from .config import Config, LLMsTxtConfig, create_default_config
from .generator import Generator, GenerationResult
from .llms_generator import LLMsTxtGenerator, LLMsTxtLink, LLMsTxtResult, LLMsTxtSection
from .scraper import ScrapedPage, ScrapeResult, Scraper

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "Config",
    "LLMsTxtConfig",
    "create_default_config",
    # Scraper
    "Scraper",
    "ScrapedPage",
    "ScrapeResult",
    # Generator
    "Generator",
    "GenerationResult",
    "QAPair",
    # Cleaner
    "Cleaner",
    "CleaningResult",
    "DatasetSplit",
    # llms.txt Generator
    "LLMsTxtGenerator",
    "LLMsTxtResult",
    "LLMsTxtSection",
    "LLMsTxtLink",
]
