# Wikisets Technical Specification

**Version:** 1.0  
**Date:** 2025-10-27  
**Author:** Omar Kamali

## Overview

Wikisets is a Python package providing a flexible interface for building customized Wikipedia datasets from [wikipedia-monthly](https://huggingface.co/datasets/omarkamali/wikipedia-monthly) with comprehensive progress tracking.

## Core Features

- Declarative configuration via WikisetConfig
- Intelligent split selection (1k/5k/10k samples or train)
- Memory-efficient reservoir sampling
- Multi-language support with proportional interleaving
- Pretraining utilities with token-based chunking
- Auto-generated dataset cards
- **Real-time progress tracking with tqdm**

## Configuration

```python
WikisetConfig(
    languages=[{"lang": str, "size": int|float|str}],
    date="latest",  # or yyyymmdd
    use_train_split=False,
    shuffle=False,
    seed=42,
    num_proc=None
)
```

## Sampling Logic

- **Exact 1k/5k/10k**: Uses prebuilt sample split
- **Below 10k**: Uses smallest sample that fits (ceil)
- **Above 10k or percentages**: Reservoir sampling from train
- **100%**: Full train split without sampling

## Schema

Base: `{id, url, title, text, lang}`

After `to_pretrain()`: Adds `{chunk_index, total_chunks}`

## Progress Tracking

### Wikiset.create()
- tqdm progress bar showing language loading
- Current language and loaded item counts
- Dataset combination messages
- Final statistics

### Wikiset.to_pretrain()
- Tokenizer loading notification
- Article count and token limit display
- HuggingFace datasets progress bar
- Final chunk/article statistics

### Example Output

```
Loading languages: 100%|██████████| 3/3 [00:15<00:00, current: ar, loaded: 5000]
Combining datasets...
✓ Created dataset with 25,000 items

Converting to pretraining format...
Loading tokenizer: gpt2
Chunking 25,000 articles (max 2048 tokens)...
Chunking articles: 100%|██████████| 25/25 [01:30<00:00, 0.28ba/s]
✓ Created 38,234 chunks from 25,000 articles
```

## Dependencies

- datasets ≥2.14.0
- numpy ≥1.20.0
- transformers ≥4.30.0
- tokenizers ≥0.13.0
- **tqdm ≥4.65.0** (for progress tracking)

## Citation

```bibtex
@software{wikisets2025,
  author = {Omar Kamali},
  title = {Wikisets: Flexible Wikipedia Dataset Builder},
  year = {2025},
  url = {https://github.com/omarkamali/wikisets}
}
```
