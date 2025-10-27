# Quick Start Guide

## Installation

```bash
pip install wikisets
```

## Create Your First Dataset

```python
from wikisets import Wikiset, WikisetConfig

config = WikisetConfig(
    languages=[{"lang": "en", "size": 5000}]
)

dataset = Wikiset.create(config)
print(f"Created dataset with {len(dataset)} items")
```

## Multi-Language

```python
config = WikisetConfig(
    languages=[
        {"lang": "en", "size": 10000},
        {"lang": "es", "size": 5000},
        {"lang": "fr", "size": 5000},
    ],
    shuffle=True
)
dataset = Wikiset.create(config)
```

## Pretraining

```python
pretrain = dataset.to_pretrain(
    split_token_len=2048,
    tokenizer="gpt2"
)
```

See [Examples](examples.md) for more patterns.
