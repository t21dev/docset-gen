"""Typer CLI entry point for DocSet Gen."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .cleaner import Cleaner
from .config import Config, create_default_config
from .generator import Generator
from .scraper import Scraper, load_scraped_content

# Initialize Typer app
app = typer.Typer(
    name="docset-gen",
    help="Transform documentation into LLM training datasets",
    add_completion=False,
)

console = Console()

# Type aliases for CLI options
OutputPath = Annotated[
    Optional[Path],
    typer.Option("--output", "-o", help="Output directory or file path"),
]
Depth = Annotated[
    int,
    typer.Option("--depth", "-d", help="Maximum crawl depth", min=1, max=10),
]
Model = Annotated[
    Optional[str],
    typer.Option("--model", "-m", help="OpenAI model to use"),
]
Pairs = Annotated[
    Optional[int],
    typer.Option("--pairs", "-p", help="Number of Q&A pairs to generate"),
]
Verbose = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable verbose output"),
]
Quiet = Annotated[
    bool,
    typer.Option("--quiet", "-q", help="Minimal output"),
]


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Override with environment variable if set
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), level)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"docset-gen version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """DocSet Gen - Transform documentation into LLM training datasets."""
    pass


@app.command()
def init(
    output: OutputPath = None,
    verbose: Verbose = False,
) -> None:
    """Initialize configuration file with defaults."""
    setup_logging(verbose)

    config_path = output or Path.cwd() / "docset-gen.yaml"

    if config_path.exists():
        if not typer.confirm(f"{config_path} already exists. Overwrite?"):
            raise typer.Abort()

    create_default_config(config_path)

    console.print(Panel(
        f"[green]Created configuration file:[/green] {config_path}\n\n"
        "Next steps:\n"
        "1. Set your API keys in .env file\n"
        "2. Customize the configuration as needed\n"
        "3. Run: docset-gen scrape <url>",
        title="Configuration Initialized",
    ))


@app.command()
def scrape(
    url: Annotated[str, typer.Argument(help="Documentation URL to scrape")],
    output: OutputPath = None,
    depth: Depth = 3,
    verbose: Verbose = False,
    quiet: Quiet = False,
) -> None:
    """Scrape documentation site to markdown files."""
    setup_logging(verbose, quiet)

    output_dir = output or Path.cwd() / "scraped"

    # Load configuration
    config = Config.load()

    # Check API key
    missing_keys = config.validate_api_keys()
    if "FIRECRAWL_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] FIRECRAWL_API_KEY is not set.")
        console.print("Set it in your .env file or as an environment variable.")
        raise typer.Exit(1)

    # Create scraper and run
    scraper = Scraper(config)

    try:
        with console.status(f"[bold blue]Scraping {url}...[/bold blue]"):
            result = scraper.scrape(url, depth=depth, verbose=verbose)

        result.save(output_dir)

        # Show summary
        table = Table(title="Scraping Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Pages Scraped", str(len(result.pages)))
        table.add_row("Total Pages Found", str(result.total_pages))
        table.add_row("Failed URLs", str(len(result.failed_urls)))
        table.add_row("Output Directory", str(output_dir))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    input_path: Annotated[Path, typer.Argument(help="Path to scraped content directory")],
    output: OutputPath = None,
    model: Model = None,
    pairs: Pairs = None,
    verbose: Verbose = False,
    quiet: Quiet = False,
) -> None:
    """Generate Q&A pairs from scraped content."""
    setup_logging(verbose, quiet)

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        raise typer.Exit(1)

    output_path = output or Path.cwd() / "dataset.jsonl"

    # Load configuration
    config = Config.load()

    # Override model if specified
    if model:
        config.openai.model = model

    # Check API key
    missing_keys = config.validate_api_keys()
    if "OPENAI_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        console.print("Set it in your .env file or as an environment variable.")
        raise typer.Exit(1)

    # Load scraped content
    pages = load_scraped_content(input_path)

    if not pages:
        console.print("[red]Error:[/red] No scraped content found in input directory.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(pages)} pages from {input_path}")

    # Generate Q&A pairs
    generator = Generator(config)

    try:
        result = generator.generate(
            pages,
            max_pairs=pairs,
            verbose=verbose,
        )

        if not result.pairs:
            console.print("[yellow]Warning:[/yellow] No Q&A pairs were generated.")
            raise typer.Exit(1)

        # Clean and split the dataset
        cleaner = Cleaner(config)
        cleaned = cleaner.clean_and_split(result.pairs)

        # Save output
        if output_path.suffix == ".jsonl":
            cleaned.save_combined(output_path)
        else:
            # Save as split files
            output_path.mkdir(parents=True, exist_ok=True)
            cleaned.save(output_path)

        # Show summary
        table = Table(title="Generation Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Pages Processed", str(result.pages_processed))
        table.add_row("Total Generated", str(result.total_generated))
        table.add_row("After Cleaning", str(cleaned.cleaned_count))
        table.add_row("Train Set", str(len(cleaned.train.pairs)))
        table.add_row("Validation Set", str(len(cleaned.validation.pairs)))
        table.add_row("Test Set", str(len(cleaned.test.pairs)))
        table.add_row("Output", str(output_path))

        console.print(table)

        if result.errors:
            console.print(f"\n[yellow]Warnings:[/yellow] {len(result.errors)} errors during generation")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    url: Annotated[str, typer.Argument(help="Documentation URL to process")],
    output: OutputPath = None,
    depth: Depth = 3,
    model: Model = None,
    pairs: Pairs = None,
    verbose: Verbose = False,
    quiet: Quiet = False,
) -> None:
    """Run full pipeline: scrape + generate in one command."""
    setup_logging(verbose, quiet)

    output_path = output or Path.cwd() / "dataset.jsonl"

    # Load configuration
    config = Config.load()

    # Override model if specified
    if model:
        config.openai.model = model

    # Check API keys
    missing_keys = config.validate_api_keys()
    if missing_keys:
        console.print(f"[red]Error:[/red] Missing API keys: {', '.join(missing_keys)}")
        console.print("Set them in your .env file or as environment variables.")
        raise typer.Exit(1)

    try:
        # Step 1: Scrape
        console.print(Panel("[bold blue]Step 1/3: Scraping Documentation[/bold blue]"))
        scraper = Scraper(config)
        scrape_result = scraper.scrape(url, depth=depth, verbose=verbose)

        if not scrape_result.pages:
            console.print("[red]Error:[/red] No pages were scraped.")
            raise typer.Exit(1)

        console.print(f"[green]Scraped {len(scrape_result.pages)} pages[/green]\n")

        # Step 2: Generate
        console.print(Panel("[bold blue]Step 2/3: Generating Q&A Pairs[/bold blue]"))
        generator = Generator(config)

        # Convert ScrapedPage to format expected by generator
        gen_result = generator.generate(
            scrape_result.pages,
            max_pairs=pairs,
            verbose=verbose,
        )

        if not gen_result.pairs:
            console.print("[red]Error:[/red] No Q&A pairs were generated.")
            raise typer.Exit(1)

        console.print(f"[green]Generated {gen_result.total_generated} Q&A pairs[/green]\n")

        # Step 3: Clean and save
        console.print(Panel("[bold blue]Step 3/3: Cleaning and Saving Dataset[/bold blue]"))
        cleaner = Cleaner(config)
        cleaned = cleaner.clean_and_split(gen_result.pairs)

        # Save output
        if output_path.suffix == ".jsonl":
            cleaned.save_combined(output_path)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            cleaned.save(output_path)

        # Final summary
        table = Table(title="Pipeline Complete")
        table.add_column("Step", style="cyan")
        table.add_column("Result", style="green")
        table.add_row("Pages Scraped", str(len(scrape_result.pages)))
        table.add_row("Pairs Generated", str(gen_result.total_generated))
        table.add_row("Final Dataset Size", str(cleaned.cleaned_count))
        table.add_row("Train / Val / Test", f"{len(cleaned.train.pairs)} / {len(cleaned.validation.pairs)} / {len(cleaned.test.pairs)}")
        table.add_row("Output", str(output_path))

        console.print(table)

        console.print(Panel(
            f"[green]Dataset saved to {output_path}[/green]\n\n"
            "Your dataset is ready for fine-tuning!",
            title="Success",
        ))

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
