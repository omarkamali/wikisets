"""Main Wikiset dataset class."""

from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset
from tqdm.auto import tqdm

from .card_generator import generate_dataset_card
from .config import WikisetConfig
from .pretrain import apply_pretrain_chunking
from .sampler import compute_interleave_probabilities, reservoir_sample
from .utils import WarningTracker, parse_size, select_split_for_size


class Wikiset(Dataset):
    """Extended Dataset class with Wikipedia-specific functionality.

    This class subclasses HuggingFace Dataset and adds methods for
    building customized Wikipedia datasets with sampling and pretraining support.
    """

    _config: Optional[WikisetConfig] = None
    _warnings: Optional[WarningTracker] = None
    _language_stats: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def create(
        cls,
        config: Union[Dict[str, Any], WikisetConfig],
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
        pbar = tqdm(
            cfg.languages,
            desc="Loading languages",
            unit="lang",
            leave=True
        )

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
                pbar.set_postfix({
                    "current": lang,
                    "loaded": len(ds)
                })

            except Exception as e:
                tracker.warn(f"Failed to load language '{lang}': {e}")
                pbar.set_postfix({
                    "current": lang,
                    "status": "failed"
                })
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
                stopping_strategy="first_exhausted"
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
    ) -> tuple[Dataset, Dict[str, Any]]:
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

        if is_percentage:
            # Load train split for percentage sampling
            try:
                ds = load_dataset(
                    "omarkamali/wikipedia-monthly",
                    subset,
                    split="train"
                )
            except Exception as e:
                raise ValueError(f"Failed to load train split for {lang}: {e}")

            total_size = len(ds)
            target_size, size_desc = parse_size(size, total_size)

            # Check if 100%
            if target_size >= total_size:
                # Return full dataset
                ds = ds.add_column("lang", [lang] * len(ds))
                return ds, {
                    "language": lang,
                    "requested_size": size_desc,
                    "split_used": "train",
                    "actual_size": len(ds),
                }

            # Reservoir sample
            ds = reservoir_sample(ds, target_size, seed, total_size)
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

            # Try to load the selected split
            try:
                ds = load_dataset(
                    "omarkamali/wikipedia-monthly",
                    subset,
                    split=split_name
                )
            except Exception:
                # Fallback to train
                tracker.warn(
                    f"Split '{split_name}' not found for {lang}, falling back to train"
                )
                ds = load_dataset(
                    "omarkamali/wikipedia-monthly",
                    subset,
                    split="train"
                )
                split_name = "train"

            actual_size = len(ds)

            # If we got more than needed, sample down
            if actual_size > target_size and split_name == "train":
                ds = reservoir_sample(ds, target_size, seed, actual_size)
                split_used = "train (sampled)"
            else:
                split_used = split_name

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
        # Validate parameters
        if split_token_len is not None:
            if tokenizer is None:
                raise ValueError("tokenizer required when split_token_len is set")
            if split_token_len <= 0:
                raise ValueError("split_token_len must be positive")

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
            tokenizer=tokenizer,
            nearest_delimiter=nearest_delimiter,
            num_proc=num_proc,
            batch_size=batch_size,
            tracker=tracker,
        )

        print(f"✓ Created {len(chunked):,} chunks from {len(self):,} articles")

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
                "tokenizer": str(tokenizer) if tokenizer else None,
                "nearest_delimiter": nearest_delimiter,
            }

            card = generate_dataset_card(
                config=self._config,
                language_stats=self._language_stats,
                warnings=all_warnings,
                total_size=len(chunked),
                pretrain_config=pretrain_config,
            )
            wikiset._info.description = card

        return wikiset

    def get_card(self) -> str:
        """Get the dataset card.

        Returns:
            Dataset card markdown string.
        """
        return self._info.description or "No card available"

    def get_warnings(self) -> List[str]:
        """Get all warnings from dataset construction.

        Returns:
            List of warning messages.
        """
        if self._warnings is None:
            return []
        return self._warnings.get_warnings()
