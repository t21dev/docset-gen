"""Typer CLI entry point for DocSet Gen."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from . import __version__
from .cleaner import Cleaner
from .config import Config, create_default_config
from .generator import Generator
from .llms_generator import LLMsTxtGenerator
from .scraper import Scraper, load_scraped_content

# Initialize Typer app
app = typer.Typer(
    name="docset-gen",
    help="Transform documentation into LLM training datasets",
    add_completion=False,
)

console = Console()

LOGO = """[bold cyan]
 ██████╗  ██████╗  ██████╗███████╗███████╗████████╗
 ██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝╚══██╔══╝
 ██║  ██║██║   ██║██║     ███████╗█████╗     ██║
 ██║  ██║██║   ██║██║     ╚════██║██╔══╝     ██║
 ██████╔╝╚██████╔╝╚██████╗███████║███████╗   ██║
 ╚═════╝  ╚═════╝  ╚═════╝╚══════╝╚══════╝   ╚═╝
  ██████╗ ███████╗███╗   ██╗
 ██╔════╝ ██╔════╝████╗  ██║
 ██║  ███╗█████╗  ██╔██╗ ██║
 ██║   ██║██╔══╝  ██║╚██╗██║
 ╚██████╔╝███████╗██║ ╚████║
  ╚═════╝ ╚══════╝╚═╝  ╚═══╝
[/bold cyan]                        [dim]by t21.dev[/dim]"""


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging based on verbosity settings."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

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


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, is_eager=True),
    ] = False,
) -> None:
    """DocSet Gen - Transform documentation into LLM training datasets."""
    # If no command provided, run interactive mode
    if ctx.invoked_subcommand is None:
        run_interactive()


def run_interactive() -> None:
    """Run the interactive pipeline."""
    setup_logging()

    console.print(LOGO)
    console.print("[dim]Transform documentation into LLM training datasets[/dim]\n")

    # Load configuration
    config = Config.load()

    # Check API keys
    missing_keys = config.validate_api_keys()
    if missing_keys:
        console.print(f"[red]Error:[/red] Missing API keys: {', '.join(missing_keys)}")
        console.print("\nSet them in your .env file:")
        console.print("  FIRECRAWL_API_KEY=fc-your-key")
        console.print("  OPENAI_API_KEY=sk-your-key")
        raise typer.Exit(1)

    # Main loop
    while True:
        success = _run_pipeline(config)

        # Ask to continue
        console.print()
        choice = Prompt.ask(
            "[cyan]What would you like to do?[/cyan]",
            choices=["new", "exit"],
            default="exit",
        )

        if choice == "exit":
            console.print("\n[dim]Goodbye![/dim]")
            break
        else:
            console.print("\n" + "─" * 50 + "\n")


def _run_pipeline(config: Config) -> bool:
    """Run the scrape and generate pipeline.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Step 1: Get URL
        console.print()
        url = Prompt.ask("[cyan]Enter documentation URL[/cyan]")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Step 2: Get crawl depth
        depth = IntPrompt.ask(
            "[cyan]Crawl depth (max 5)[/cyan]",
            default=3,
            show_default=True,
        )

        # Step 3: Scrape
        console.print()
        console.print(Panel("[bold blue]Step 1/3: Scraping Documentation[/bold blue]"))

        scraper = Scraper(config)

        with console.status(f"[bold]Crawling {url}...[/bold]"):
            scrape_result = scraper.scrape(url, depth=depth)

        if not scrape_result.pages:
            console.print("[red]Error:[/red] No pages were scraped.")
            return False

        # Show what we found
        total_words = sum(p.word_count() for p in scrape_result.pages)
        console.print(f"\n[green]Found {len(scrape_result.pages)} pages[/green] ({total_words:,} words total)")

        # Step 4: Ask for output format
        console.print()
        format_choice = Prompt.ask(
            "[cyan]Output format[/cyan]",
            choices=["dataset", "llms.txt", "both"],
            default="dataset",
        )

        # Handle llms.txt generation
        if format_choice in ["llms.txt", "both"]:
            # Extract site name from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            default_name = parsed.netloc.replace("www.", "").split(".")[0].title()

            site_name = Prompt.ask(
                "[cyan]Site/Project name[/cyan]",
                default=default_name,
                show_default=True,
            )

            # Ask for mode
            llms_mode = Prompt.ask(
                "[cyan]llms.txt mode[/cyan]",
                choices=["minimal", "full"],
                default="minimal",
            )

            # Set default filename based on mode
            default_filename = "llms-full.txt" if llms_mode == "full" else "llms.txt"
            llms_output = Prompt.ask(
                "[cyan]llms.txt output file[/cyan]",
                default=default_filename,
                show_default=True,
            )
            llms_output_path = Path(llms_output)

            console.print()
            mode_label = "full" if llms_mode == "full" else "minimal"
            console.print(Panel(f"[bold blue]Generating llms.txt ({mode_label} mode)[/bold blue]"))

            llms_gen = LLMsTxtGenerator(config)

            with console.status("[bold]Analyzing and categorizing pages...[/bold]"):
                llms_result = llms_gen.generate(scrape_result.pages, site_name=site_name, mode=llms_mode)

            llms_result.save(llms_output_path)

            # Show llms.txt summary
            console.print()
            table = Table(title="llms.txt Complete")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Pages Analyzed", str(len(scrape_result.pages)))
            table.add_row("Sections Generated", str(len(llms_result.sections)))
            table.add_row("Total Links", str(llms_result.total_links))
            table.add_row("Mode", llms_mode)
            table.add_row("Output", str(llms_output_path))
            console.print(table)

            if format_choice == "llms.txt":
                console.print(Panel(
                    f"[green]llms.txt saved to {llms_output_path}[/green]",
                    title="Success",
                ))
                return True

        # Handle dataset generation
        if format_choice in ["dataset", "both"]:
            # Step: Ask how many pairs
            console.print()
            suggested_pairs = min(len(scrape_result.pages) * 5, 500)
            pairs_count = IntPrompt.ask(
                "[cyan]How many Q&A pairs to generate?[/cyan]",
                default=suggested_pairs,
                show_default=True,
            )

            # Step: Get output path
            output_path = Prompt.ask(
                "[cyan]Dataset output file[/cyan]",
                default="dataset.jsonl",
                show_default=True,
            )
            output_path = Path(output_path)

            # Step: Generate
            console.print()
            console.print(Panel("[bold blue]Generating Q&A Pairs[/bold blue]"))

            generator = Generator(config)
            gen_result = generator.generate(
                scrape_result.pages,
                max_pairs=pairs_count,
            )

            if not gen_result.pairs:
                console.print("[red]Error:[/red] No Q&A pairs were generated.")
                return False

            console.print(f"[green]Generated {gen_result.total_generated} Q&A pairs[/green]")

            # Step: Clean and save
            console.print()
            console.print(Panel("[bold blue]Cleaning and Saving[/bold blue]"))

            cleaner = Cleaner(config)
            cleaned = cleaner.clean_and_split(gen_result.pairs)

            if output_path.suffix == ".jsonl":
                cleaned.save_combined(output_path)
            else:
                output_path.mkdir(parents=True, exist_ok=True)
                cleaned.save(output_path)

            # Final summary
            console.print()
            table = Table(title="Dataset Complete")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Pages Scraped", str(len(scrape_result.pages)))
            table.add_row("Q&A Pairs Generated", str(gen_result.total_generated))
            table.add_row("After Cleaning", str(cleaned.cleaned_count))
            table.add_row("Train / Val / Test", f"{len(cleaned.train.pairs)} / {len(cleaned.validation.pairs)} / {len(cleaned.test.pairs)}")
            table.add_row("Output", str(output_path))
            console.print(table)

            console.print(Panel(
                f"[green]Dataset saved to {output_path}[/green]\n\n"
                "Your dataset is ready for fine-tuning!",
                title="Success",
            ))

        return True

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        return False
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return False


@app.command()
def init(
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
) -> None:
    """Initialize configuration file with defaults."""
    setup_logging()

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
        "3. Run: python -m docset_gen",
        title="Configuration Initialized",
    ))


@app.command()
def scrape(
    url: Annotated[str, typer.Argument(help="Documentation URL to scrape")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    depth: Annotated[int, typer.Option("--depth", "-d")] = 3,
) -> None:
    """Scrape documentation site to markdown files."""
    setup_logging()

    output_dir = output or Path.cwd() / "scraped"
    config = Config.load()

    missing_keys = config.validate_api_keys()
    if "FIRECRAWL_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] FIRECRAWL_API_KEY is not set.")
        raise typer.Exit(1)

    scraper = Scraper(config)

    try:
        with console.status(f"[bold]Scraping {url}...[/bold]"):
            result = scraper.scrape(url, depth=depth)

        result.save(output_dir)

        table = Table(title="Scraping Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Pages Scraped", str(len(result.pages)))
        table.add_row("Output Directory", str(output_dir))
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    input_path: Annotated[Path, typer.Argument(help="Path to scraped content directory")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    pairs: Annotated[Optional[int], typer.Option("--pairs", "-p")] = None,
) -> None:
    """Generate Q&A pairs from scraped content."""
    setup_logging()

    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input path does not exist: {input_path}")
        raise typer.Exit(1)

    output_path = output or Path.cwd() / "dataset.jsonl"
    config = Config.load()

    missing_keys = config.validate_api_keys()
    if "OPENAI_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        raise typer.Exit(1)

    pages = load_scraped_content(input_path)

    if not pages:
        console.print("[red]Error:[/red] No scraped content found.")
        raise typer.Exit(1)

    console.print(f"Loaded {len(pages)} pages")

    generator = Generator(config)

    try:
        result = generator.generate(pages, max_pairs=pairs)

        if not result.pairs:
            console.print("[yellow]Warning:[/yellow] No Q&A pairs generated.")
            raise typer.Exit(1)

        cleaner = Cleaner(config)
        cleaned = cleaner.clean_and_split(result.pairs)

        if output_path.suffix == ".jsonl":
            cleaned.save_combined(output_path)
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            cleaned.save(output_path)

        table = Table(title="Generation Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total Generated", str(result.total_generated))
        table.add_row("After Cleaning", str(cleaned.cleaned_count))
        table.add_row("Output", str(output_path))
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("llms-txt")
def llms_txt(
    url: Annotated[str, typer.Argument(help="Documentation URL to scrape")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o")] = None,
    depth: Annotated[int, typer.Option("--depth", "-d")] = 3,
    name: Annotated[Optional[str], typer.Option("--name", "-n", help="Project/site name")] = None,
    mode: Annotated[str, typer.Option("--mode", "-m", help="Generation mode: minimal (links only) or full (with content)")] = "minimal",
) -> None:
    """Generate llms.txt from documentation site.

    Modes:
      minimal - Links with brief descriptions (default)
      full    - Full page content included
    """
    setup_logging()

    # Validate mode
    if mode not in ("minimal", "full"):
        console.print(f"[red]Error:[/red] Invalid mode '{mode}'. Use 'minimal' or 'full'.")
        raise typer.Exit(1)

    # Set default output based on mode
    if output is None:
        output_path = Path.cwd() / ("llms-full.txt" if mode == "full" else "llms.txt")
    else:
        output_path = output

    config = Config.load()

    # Validate API keys
    missing_keys = config.validate_api_keys()
    if "FIRECRAWL_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] FIRECRAWL_API_KEY is not set.")
        raise typer.Exit(1)
    if "OPENAI_API_KEY" in missing_keys:
        console.print("[red]Error:[/red] OPENAI_API_KEY is not set.")
        raise typer.Exit(1)

    # Normalize URL
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Extract site name from URL if not provided
    if not name:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        name = parsed.netloc.replace("www.", "").split(".")[0].title()

    # Scrape the site
    scraper = Scraper(config)

    try:
        with console.status(f"[bold]Scraping {url}...[/bold]"):
            scrape_result = scraper.scrape(url, depth=depth)

        if not scrape_result.pages:
            console.print("[red]Error:[/red] No pages were scraped.")
            raise typer.Exit(1)

        console.print(f"[green]Scraped {len(scrape_result.pages)} pages[/green]")

        # Generate llms.txt
        llms_gen = LLMsTxtGenerator(config)

        with console.status(f"[bold]Generating llms.txt ({mode} mode)...[/bold]"):
            result = llms_gen.generate(scrape_result.pages, site_name=name, mode=mode)

        result.save(output_path)

        table = Table(title="llms.txt Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Pages Analyzed", str(len(scrape_result.pages)))
        table.add_row("Sections Generated", str(len(result.sections)))
        table.add_row("Total Links", str(result.total_links))
        table.add_row("Mode", mode)
        table.add_row("Output", str(output_path))
        console.print(table)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
