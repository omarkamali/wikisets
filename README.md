# Wikisets

[![PyPI version](https://badge.fury.io/py/wikisets.svg)](https://badge.fury.io/py/wikisets)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Flexible Wikipedia dataset builder with sampling and pretraining support. Built on top of [wikipedia-monthly](https://huggingface.co/datasets/omarkamali/wikipedia-monthly), providing fresh, clean Wikipedia dumps updated monthly.

## Features

- 🌍 **Multi-language support** - Access Wikipedia in any language
- 📊 **Flexible sampling** - Use exact sizes, percentages, or prebuilt samples (1k/5k/10k)
- ⚡ **Memory efficient** - Reservoir sampling for large datasets
- 🔄 **Reproducible** - Deterministic sampling with seeds
- 📦 **HuggingFace compatible** - Subclasses `datasets.Dataset`
- ✂️ **Pretraining ready** - Built-in text chunking with tokenizer support
- 📝 **Auto-generated cards** - Comprehensive dataset documentation

## Installation

```bash
pip install wikisets
```

Or with uv:
```bash
# Preferred: Add to your project
uv add wikisets

# Or just install
uv pip install wikisets
```

## Quick Start

```python
from wikisets import Wikiset, WikisetConfig

# Create a multi-language dataset
config = WikisetConfig(
    languages=[
        {"lang": "en", "size": 10000},      # 10k sample
        {"lang": "fr", "size": "50%"},      # 50% of French Wikipedia
        {"lang": "ar", "size": 0.1},        # 10% of Arabic Wikipedia
    ],
    seed=42
)

dataset = Wikiset.create(config)

# Access like any HuggingFace dataset
print(len(dataset))
print(dataset[0])

# View dataset card
print(dataset.get_card())
```

## Configuration Options

### WikisetConfig Parameters

- **languages** (required): List of `{lang: str, size: int|float|str}` dictionaries
  - `lang`: Language code (e.g., "en", "fr", "ar", "simple")
  - `size`: Can be:
    - Integer (e.g., `1000`, `5000`, `10000`) - Uses prebuilt samples when available
    - Percentage string (e.g., `"50%"`) - Samples that percentage
    - Float 0-1 (e.g., `0.5`) - Samples that fraction
- **date** (optional, default: `"latest"`): Wikipedia dump date in yyyymmdd format
- **use_train_split** (optional, default: `False`): Force sampling from full "train" split, ignoring prebuilt samples
- **shuffle** (optional, default: `False`): Proportionally interleave languages
- **seed** (optional, default: `42`): Random seed for reproducibility
- **num_proc** (optional): Number of parallel processes

## Usage Examples

### Basic Usage

```python
from wikisets import Wikiset, WikisetConfig

config = WikisetConfig(
    languages=[{"lang": "en", "size": 5000}]
)
dataset = Wikiset.create(config)

# Wikiset is just an HF Dataset
dataset.push_to_hub("my-wiki-dataset")
```

### Pretraining with Chunking

```python
# Create base dataset
config = WikisetConfig(
    languages=[
        {"lang": "en", "size": 10000},
        {"lang": "ar", "size": 5000},
    ]
)
dataset = Wikiset.create(config)

# Convert to pretraining format with 2048 token chunks
pretrain_dataset = dataset.to_pretrain(
    split_token_len=2048,
    tokenizer="gpt2",
    nearest_delimiter="newline",
    num_proc=4
)

# Do whatever you want with it
pretrain_dataset.map(lambda x: x["text"].upper())

# It's still just a HuggingFace Dataset
pretrain_dataset.push_to_hub("my-wiki-pretraining-dataset")
```

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Examples](docs/examples.md)** - Common usage patterns
- **[Technical Specification](SPEC.md)** - Design and implementation details

## Builds on wikipedia-monthly

Wikisets is built on top of [omarkamali/wikipedia-monthly](https://huggingface.co/datasets/omarkamali/wikipedia-monthly), which provides:

- Fresh Wikipedia dumps updated monthly
- Clean, preprocessed text
- 300+ languages
- Prebuilt 1k/5k/10k samples for large languages

Wikisets adds:
- Simple configuration-based building
- Intelligent sampling strategies
- Multi-language mixing
- Pretraining utilities
- Comprehensive dataset cards

## Citation

```bibtex
@software{wikisets2025,
  author = {Omar Kamali},
  title = {Wikisets: Flexible Wikipedia Dataset Builder},
  year = {2025},
  url = {https://github.com/omarkamali/wikisets}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- **GitHub**: https://github.com/omarkamali/wikisets
- **PyPI**: https://pypi.org/project/wikisets/
- **Documentation**: https://github.com/omarkamali/wikisets/tree/main/docs
- **Wikipedia Monthly**: https://huggingface.co/datasets/omarkamali/wikipedia-monthly
