# Examples

## Single Language

```python
from wikisets import Wikiset, WikisetConfig

config = WikisetConfig(
    languages=[{"lang": "en", "size": 5000}]
)
dataset = Wikiset.create(config)
```

## Multiple Languages (Interleaved)

```python
config = WikisetConfig(
    languages=[
        {"lang": "en", "size": 10000},
        {"lang": "es", "size": 10000},
        {"lang": "fr", "size": 10000},
    ],
    shuffle=True,
    seed=42
)
dataset = Wikiset.create(config)
```

## Percentage Sampling

```python
config = WikisetConfig(
    languages=[
        {"lang": "en", "size": "10%"},
        {"lang": "fr", "size": 0.05},
    ]
)
dataset = Wikiset.create(config)
```

## Pretraining with Chunking

```python
pretrain = dataset.to_pretrain(
    split_token_len=2048,
    tokenizer="gpt2",
    nearest_delimiter="newline",
    num_proc=4
)
```

## Custom Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
pretrain = dataset.to_pretrain(
    split_token_len=4096,
    tokenizer=tokenizer
)
```

See [API Reference](api.md) for complete details.
