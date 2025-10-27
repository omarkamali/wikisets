# API Reference

## WikisetConfig

```python
WikisetConfig(
    languages: List[Dict],
    date: str = "latest",
    use_train_split: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    num_proc: Optional[int] = None
)
```

### Parameters

- **languages**: List of `{"lang": str, "size": int|float|str}`
    - `lang`: Language code (e.g., "en", "fr", "ar", "simple")
    - `size`: Can be:
        - Integer (e.g., `1000`, `5000`, `10000`) - Uses prebuilt samples when available
        - Percentage string (e.g., `"50%"`) - Samples that percentage
        - Float 0-1 (e.g., `0.5`) - Samples that fraction
- **date**: yyyymmdd or "latest"
- **use_train_split**: Force train split, ignoring prebuilt samples
- **shuffle**: Interleave languages
- **seed**: Random seed
- **num_proc**: Parallel processes

## Wikiset

### Wikiset.create()

```python
dataset = Wikiset.create(config, num_proc=None)
```

Build dataset from configuration.

### Wikiset.to_pretrain()

```python
pretrain = dataset.to_pretrain(
    split_token_len=None,
    tokenizer=None,
    nearest_delimiter="newline",
    num_proc=None,
    batch_size=1000
)
```

Convert to pretraining format with optional chunking.

### Wikiset.get_card()

```python
card = dataset.get_card()
```

Get dataset card markdown.

### Wikiset.get_warnings()

```python
warnings = dataset.get_warnings()
```

Get construction warnings.

## Schema

Base: `{id, url, title, text, lang}`
Pretrain: Adds `{chunk_index, total_chunks}`
