"""Main Wikiset dataset class."""

from typing import Any, Optional, Union

from datasets import (
    Dataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_dataset_builder,
)
from tqdm.auto import tqdm

from .card_generator import generate_dataset_card
from .config import WikisetConfig
from .pretrain import apply_pretrain_chunking
from .sampler import (
    compute_interleave_probabilities,
    reservoir_sample,
    reservoir_sample_streaming,
)
from .utils import WarningTracker, parse_size, select_split_for_size


class Wikiset(Dataset):
    """Extended Dataset class with Wikipedia-specific functionality.

    This class subclasses HuggingFace Dataset and adds methods for
    building customized Wikipedia datasets with sampling and pretraining support.
    """

    _config: Optional[WikisetConfig] = None
    _warnings: Optional[WarningTracker] = None
    _language_stats: Optional[list[dict[str, Any]]] = None

    @classmethod
    def create(
        cls,
        config: Union[dict[str, Any], WikisetConfig],
        num_proc: Optional[int] = None,
    ) -> "Wikiset":
        """Create a Wikiset from configuration.

        Args:
            config: WikisetConfig instance or dictionary.
            num_proc: Number of processes for parallel operations.

        Returns:
            Wikiset instance.
        """
        # Parse config
        if isinstance(config, dict):
            cfg = WikisetConfig.from_dict(config)
        else:
            cfg = config

        # Override num_proc if provided
        if num_proc is not None:
            cfg.num_proc = num_proc

        # Initialize warning tracker
        tracker = WarningTracker()

        # Build per-language datasets
        language_datasets = []
        language_stats = []

        # Progress bar for language loading
        pbar = tqdm(cfg.languages, desc="Loading languages", unit="lang", leave=True)

        for entry in pbar:
            lang = entry["lang"]
            size = entry["size"]

            pbar.set_postfix({"current": lang})

            try:
                ds, stat = cls._load_language(
                    lang=lang,
                    size=size,
                    date=cfg.date,
                    use_train_split=cfg.use_train_split,
                    seed=cfg.seed,
                    tracker=tracker,
                )
                language_datasets.append(ds)
                language_stats.append(stat)
                pbar.set_postfix({"current": lang, "loaded": len(ds)})

            except Exception as e:
                tracker.warn(f"Failed to load language '{lang}': {e}")
                pbar.set_postfix({"current": lang, "status": "failed"})
                continue

        pbar.close()

        if not language_datasets:
            raise ValueError("No valid languages loaded. Check warnings.")

        # Combine datasets
        print("Combining datasets...")
        if cfg.shuffle:
            # Proportional interleaving
            sizes = [len(ds) for ds in language_datasets]
            probabilities = compute_interleave_probabilities(sizes)

            combined = interleave_datasets(
                language_datasets,
                probabilities=probabilities,
                seed=cfg.seed,
                stopping_strategy="first_exhausted",
            )
            # Convert to Dataset
            print("Materializing interleaved dataset...")
            combined = Dataset.from_dict(combined[:])
        else:
            # Simple concatenation
            combined = concatenate_datasets(language_datasets)

        print(f"✓ Created dataset with {len(combined):,} items")

        # Create Wikiset instance
        wikiset = cls(combined._data)
        wikiset._info = combined._info
        wikiset._split = combined._split
        wikiset._indices = combined._indices
        wikiset._fingerprint = combined._fingerprint

        # Store metadata
        wikiset._config = cfg
        wikiset._warnings = tracker
        wikiset._language_stats = language_stats

        # Generate and attach dataset card
        card = generate_dataset_card(
            config=cfg,
            language_stats=language_stats,
            warnings=tracker.get_warnings(),
            total_size=len(combined),
        )
        wikiset._info.description = card

        return wikiset

    @classmethod
    def _load_language(
        cls,
        lang: str,
        size: Union[int, float, str],
        date: str,
        use_train_split: bool,
        seed: int,
        tracker: WarningTracker,
    ) -> tuple[Dataset, dict[str, Any]]:
        """Load a single language dataset.

        Args:
            lang: Language code.
            size: Size specification.
            date: Date string.
            use_train_split: Force train split.
            seed: Random seed.
            tracker: Warning tracker.

        Returns:
            Tuple of (dataset, statistics_dict).
        """
        # Build subset name
        subset = f"{date}.{lang}"

        # Determine if percentage/fraction
        is_percentage = isinstance(size, (float, str))

        # Helper: try to load a split with streaming, but fall back to non-streaming
        def _load_split_with_streaming(path: str, name: str, split: str) -> tuple[Any, bool]:
            try:
                ds_any = load_dataset(path, name, split=split, streaming=True)  # type: ignore[arg-type]
                # Some test doubles won't be iterable; detect and treat as non-streaming
                if hasattr(ds_any, "__iter__"):
                    return ds_any, True
                return ds_any, False
            except TypeError:
                # load_dataset may not accept streaming kwarg (tests monkeypatch with a narrower signature)
                ds_any = load_dataset(path, name, split=split)  # type: ignore[arg-type]
                return ds_any, False

        # Helper: get split size via builder, else via len(dataset)
        def _get_split_size(path: str, name: str, split: str) -> Optional[int]:
            try:
                builder = load_dataset_builder(path, name)  # type: ignore[arg-type]
                if builder.info.splits and split in builder.info.splits:
                    info = builder.info.splits[split]
                    if getattr(info, "num_examples", None) is not None:
                        return int(info.num_examples)
            except Exception:
                pass
            try:
                ds_tmp = load_dataset(path, name, split=split)  # non-streaming fallback
                return len(ds_tmp)  # type: ignore[arg-type]
            except Exception:
                return None

        if is_percentage:
            # Load train split for percentage sampling (streaming)
            ds_any, is_streaming = _load_split_with_streaming(
                "omarkamali/wikipedia-monthly", subset, "train"
            )

            total_size = _get_split_size(
                "omarkamali/wikipedia-monthly", subset, "train"
            )
            if total_size is None:
                raise ValueError(f"Failed to load train split for {lang}: unknown size")

            target_size, size_desc = parse_size(size, total_size)

            # Check if 100%
            if target_size >= total_size:
                # Materialize full dataset via streaming
                if is_streaming:
                    train_items: list[dict[str, Any]] = []
                    for ex in ds_any:  # type: ignore[assignment]
                        train_items.append(ex)
                    ds = Dataset.from_list(train_items)
                else:
                    ds = ds_any  # type: ignore[assignment]
                ds = ds.add_column("lang", [lang] * len(ds))
                return ds, {
                    "language": lang,
                    "requested_size": size_desc,
                    "split_used": "train",
                    "actual_size": len(ds),
                }

            # Reservoir sample from stream and materialize
            if is_streaming:
                ds = reservoir_sample_streaming(ds_any, target_size, seed)  # type: ignore[arg-type]
            else:
                ds = reservoir_sample(ds_any, target_size, seed, total_size)  # type: ignore[arg-type]
            ds = ds.add_column("lang", [lang] * len(ds))

            return ds, {
                "language": lang,
                "requested_size": size_desc,
                "split_used": "train (sampled)",
                "actual_size": len(ds),
            }

        else:
            # Integer size
            target_size = int(size)
            split_name = select_split_for_size(target_size, use_train_split)

            # Try to load the selected split (prefer streaming, fallback compatible with tests)
            try:
                ds_any, is_streaming = _load_split_with_streaming(
                    "omarkamali/wikipedia-monthly", subset, split_name
                )
            except Exception:
                # Fallback to train
                tracker.warn(
                    f"Split '{split_name}' not found for {lang}, falling back to train"
                )
                ds_any, is_streaming = _load_split_with_streaming(
                    "omarkamali/wikipedia-monthly", subset, "train"
                )
                split_name = "train"

            if split_name == "train":
                # Sample down to target_size from stream
                if is_streaming and hasattr(ds_any, "__iter__"):
                    ds = reservoir_sample_streaming(ds_any, target_size, seed)  # type: ignore[arg-type]
                else:
                    # Non-streaming fallback for tests
                    actual_size = len(ds_any)  # type: ignore[arg-type]
                    ds = reservoir_sample(ds_any, target_size, seed, actual_size)  # type: ignore[arg-type]
                split_used = "train (sampled)"
            else:
                # For non-train sample splits, materialize full split
                if is_streaming and hasattr(ds_any, "__iter__"):
                    split_items: list[dict[str, Any]] = []
                    for ex in ds_any:  # type: ignore[assignment]
                        split_items.append(ex)
                    ds = Dataset.from_list(split_items)
                else:
                    ds = ds_any  # type: ignore[assignment]
                split_used = split_name

                # Always downsample to the requested target_size if smaller than the split size
                # This ensures the amount returned equals the amount requested for non-exact sizes
                try:
                    actual_split_size = len(ds)  # type: ignore[arg-type]
                except Exception:
                    actual_split_size = None

                if actual_split_size is not None and target_size < actual_split_size:
                    ds = reservoir_sample(ds, target_size, seed, actual_split_size)  # type: ignore[arg-type]

            # Add lang column
            ds = ds.add_column("lang", [lang] * len(ds))

            return ds, {
                "language": lang,
                "requested_size": f"{target_size} items",
                "split_used": split_used,
                "actual_size": len(ds),
            }

    def to_pretrain(
        self,
        split_token_len: Optional[int] = None,
        tokenizer: Optional[Union[str, Any]] = None,
        nearest_delimiter: str = "newline",
        num_proc: Optional[int] = None,
        batch_size: int = 1000,
    ) -> "Wikiset":
        """Convert to pretraining format with optional chunking.

        Args:
            split_token_len: Maximum tokens per chunk (None = no chunking).
            tokenizer: Tokenizer instance or HuggingFace model name.
            nearest_delimiter: Delimiter for splitting ("space", "newline", or regex).
            num_proc: Number of processes.
            batch_size: Batch size for processing.

        Returns:
            New Wikiset with pretraining format.
        """
        # Validate parameters and establish tokenizer to use
        if split_token_len is not None:
            if split_token_len <= 0:
                raise ValueError("split_token_len must be positive")
        # Determine tokenizer name/value to use throughout (default to GPT-2)
        tokenizer_to_use = tokenizer if tokenizer is not None else "gpt2"
        tokenizer_name = str(tokenizer_to_use)

        # Use config num_proc if not specified
        if num_proc is None and self._config is not None:
            num_proc = self._config.num_proc

        # Create new warning tracker for this operation
        tracker = WarningTracker()

        print("Converting to pretraining format...")

        # Apply chunking
        chunked = apply_pretrain_chunking(
            dataset=self,
            split_token_len=split_token_len,
            tokenizer=tokenizer_to_use,
            nearest_delimiter=nearest_delimiter,
            num_proc=num_proc,
            batch_size=batch_size,
            tracker=tracker,
        )

        print(f"✓ Created {len(chunked):,} chunks from {len(self):,} articles")
        print(f"Tokenizer used: {tokenizer_name}")

        # Compute token statistics
        token_stats: dict[str, Any] | None = None
        if "token_len" in chunked.column_names and "lang" in chunked.column_names:
            token_lens = chunked["token_len"]  # type: ignore[index]
            langs = chunked["lang"]  # type: ignore[index]
            total_tokens = int(sum(int(t) for t in token_lens))
            per_lang: dict[str, int] = {}
            for l, t in zip(langs, token_lens):
                per_lang[l] = per_lang.get(l, 0) + int(t)
            per_language_rows = [
                {"language": lang, "tokens": count} for lang, count in sorted(per_lang.items())
            ]
            token_stats = {
                "total_tokens": total_tokens,
                "per_language": per_language_rows,
            }

            # Print concise summary
            print(f"Total tokens: {total_tokens:,}")
            for row in per_language_rows:
                print(f"  - {row['language']}: {row['tokens']:,} tokens")

        # Create new Wikiset
        wikiset = Wikiset(chunked._data)
        wikiset._info = chunked._info
        wikiset._split = chunked._split
        wikiset._indices = chunked._indices
        wikiset._fingerprint = chunked._fingerprint

        # Preserve original config and stats
        wikiset._config = self._config
        wikiset._language_stats = self._language_stats

        # Merge warnings
        if self._warnings is not None:
            all_warnings = self._warnings.get_warnings() + tracker.get_warnings()
        else:
            all_warnings = tracker.get_warnings()

        merged_tracker = WarningTracker()
        merged_tracker.warnings = all_warnings
        wikiset._warnings = merged_tracker

        # Generate updated card with pretrain config
        if self._config is not None and self._language_stats is not None:
            pretrain_config = {
                "split_token_len": split_token_len,
                "tokenizer": tokenizer_name,
                "nearest_delimiter": nearest_delimiter,
            }

            card = generate_dataset_card(
                config=self._config,
                language_stats=self._language_stats,
                warnings=all_warnings,
                total_size=len(chunked),
                pretrain_config=pretrain_config,
                token_stats=token_stats,
            )
            wikiset._info.description = card

        return wikiset

    def get_card(self) -> str:
        """Get the dataset card.

        Returns:
            Dataset card markdown string.
        """
        return self._info.description or "No card available"

    def get_warnings(self) -> list[str]:
        """Get all warnings from dataset construction.

        Returns:
            List of warning messages.
        """
        if self._warnings is None:
            return []
        return self._warnings.get_warnings()
