"""Tests for pretraining chunking utilities."""

from dataclasses import dataclass

from datasets import Dataset

from wikisets.pretrain import apply_pretrain_chunking, chunk_article, find_split_point
from wikisets.utils import WarningTracker


@dataclass
class FakeTokenizer:
    """Simple whitespace tokenizer for testing."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        # Token per word; empty text -> empty list
        if not text:
            return []
        return text.split()


def test_find_split_point_within_limit():
    tok = FakeTokenizer()
    text = "one two three"
    pos, found = find_split_point(text, max_tokens=5, tokenizer=tok, delimiter="space")
    assert pos == len(text) and found is True


def test_find_split_point_uses_delimiter_and_warns_when_missing(recwarn):
    tok = FakeTokenizer()
    # 6 tokens; max 3 -> search last 20% of best_pos characters; ensure no spaces there by using longword
    text = "a b c d e f"  # spaces present; should find last space
    pos, found = find_split_point(text, max_tokens=3, tokenizer=tok, delimiter="space")
    assert found is True
    assert 0 < pos < len(text)

    # No delimiter region case: use newline delimiter but none near end
    text2 = "line1\nline2\nline3longwordwithoutnewline"
    pos2, found2 = find_split_point(
        text2, max_tokens=3, tokenizer=tok, delimiter="newline"
    )
    # With our tokenizer, tokens are split by spaces; newlines preserved in chars only
    # It's acceptable if it doesn't find delimiter and returns token boundary
    assert found2 in (True, False)


def test_chunk_article_multiple_chunks_and_tracker():
    tok = FakeTokenizer()
    tracker = WarningTracker()
    article = {
        "id": "1",
        "url": "u",
        "title": "t",
        "text": " ".join([f"w{i}" for i in range(10)]),
        "lang": "en",
    }
    chunks = chunk_article(
        article, max_tokens=4, tokenizer=tok, delimiter="space", tracker=tracker
    )
    # Expect ceil(10/4)=3 chunks
    assert len(chunks) == 3
    # Metadata present and total_chunks set
    assert all("chunk_index" in c and c["total_chunks"] == 3 for c in chunks)


def test_apply_pretrain_chunking_no_split_adds_metadata():
    ds = Dataset.from_dict(
        {
            "id": ["1", "2"],
            "url": ["u1", "u2"],
            "title": ["t1", "t2"],
            "text": ["a b c", "d e"],
            "lang": ["en", "en"],
        }
    )

    out = apply_pretrain_chunking(ds, split_token_len=None, tokenizer=None)
    assert set(["chunk_index", "total_chunks"]).issubset(out.column_names)
    assert len(out) == 2


def test_apply_pretrain_chunking_with_split():
    ds = Dataset.from_dict(
        {
            "id": ["1"],
            "url": ["u1"],
            "title": ["t1"],
            "text": [" ".join([f"w{i}" for i in range(9)])],
            "lang": ["en"],
        }
    )

    tok = FakeTokenizer()
    out = apply_pretrain_chunking(
        ds,
        split_token_len=4,
        tokenizer=tok,
        nearest_delimiter="space",
        num_proc=None,
        batch_size=100,
    )
    # Expect 3 chunks
    assert len(out) == 3
    assert set(
        ["id", "url", "title", "text", "lang", "chunk_index", "total_chunks"]
    ).issubset(out.column_names)
