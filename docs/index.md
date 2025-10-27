# Wikisets Documentation

Welcome to Wikisets - a flexible Wikipedia dataset builder with sampling and pretraining support.

## Quick Links

- [Quick Start](quickstart.md)
- [API Reference](api.md)
- [Examples](examples.md)
- [Technical Spec](../SPEC.md)

## Features

- Multi-language support (300+ languages)
- Flexible sampling (int, float, percentage)
- Memory-efficient reservoir sampling
- Pretraining utilities with chunking
- Real-time progress tracking
- Auto-generated dataset cards

## Installation

```bash
pip install wikisets
```

## Basic Example

```python
from wikisets import Wikiset, WikisetConfig

config = WikisetConfig(
    languages=[
        {"lang": "en", "size": 10000},
        {"lang": "fr", "size": "50%"},
    ]
)

dataset = Wikiset.create(config)
```
