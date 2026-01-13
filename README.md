# DocSet Gen

> Transform documentation into LLM training datasets

DocSet Gen scrapes documentation websites and generates high-quality Q&A training datasets for fine-tuning LLMs.

## Features

- **Smart Scraping** - Uses Firecrawl to handle JS-rendered sites, anti-bot measures, and content cleaning
- **AI-Powered Generation** - Generates Q&A pairs using GPT-4o/GPT-4o-mini
- **Quality Controls** - Automatic deduplication, validation, and filtering
- **Ready-to-Use Output** - JSONL format with automatic train/val/test splits

## Installation

```bash
# Clone the repository
git clone https://github.com/t21dev/docset-gen.git
cd docset-gen

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```bash
FIRECRAWL_API_KEY=fc-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

3. (Optional) Create a config file for advanced settings:
```bash
python -m docset_gen init
```

## Usage

### Quick Start - Full Pipeline

```bash
# Generate dataset in one command
python -m docset_gen pipeline https://docs.example.com --output dataset.jsonl
```

### Step-by-Step

```bash
# Step 1: Scrape documentation
python -m docset_gen scrape https://docs.example.com --depth 3 --output ./scraped

# Step 2: Generate Q&A pairs
python -m docset_gen generate ./scraped --pairs 500 --output dataset.jsonl
```

### Commands

| Command | Description |
|---------|-------------|
| `init` | Create config file with defaults |
| `scrape` | Scrape documentation site to markdown |
| `generate` | Generate Q&A pairs from scraped content |
| `pipeline` | Run full scrape + generate workflow |

### Options

| Flag | Description |
|------|-------------|
| `--output, -o` | Output directory/file |
| `--depth` | Crawl depth (default: 3) |
| `--model` | OpenAI model to use |
| `--pairs` | Number of Q&A pairs to generate |
| `--verbose, -v` | Verbose output |
| `--quiet, -q` | Minimal output |

## Output Format

```json
{"instruction": "What is dependency injection?", "input": "", "output": "Dependency injection is..."}
{"instruction": "How do I configure logging?", "input": "", "output": "To configure logging..."}
```

## Configuration File

Create `docset-gen.yaml` for advanced settings:

```yaml
firecrawl:
  max_depth: 3
  exclude_patterns:
    - "/changelog/*"
    - "/blog/*"

openai:
  model: gpt-4o-mini
  temperature: 0.7

generation:
  mode: qa
  pairs_per_page: 5

output:
  split_ratio: [0.8, 0.1, 0.1]
```

## Requirements

- Python 3.10+
- [Firecrawl API key](https://firecrawl.dev)
- [OpenAI API key](https://platform.openai.com)

## License

MIT License - see [LICENSE](LICENSE)

## Author

Created by [@TriptoAfsin](https://github.com/TriptoAfsin) | [t21dev](https://github.com/t21dev)
