"""Document and snippet filtering, deduplication, and salient passage extraction."""

import hashlib
import re
from typing import List, Sequence, Set

from qna_bot.rag.crag.models import CRAGDocument

_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+|\n\n+")


def _sentence_hash(text: str) -> str:
    """Hash normalized sentence text for fast deduplication."""
    norm = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


class DocumentFilter:
    """Extracts salient passages, strips noise, and deduplicates content across documents."""

    def __init__(self, min_sentence_len: int = 15, max_snippets_per_doc: int = 6) -> None:
        self.min_sentence_len = min_sentence_len
        self.max_snippets_per_doc = max_snippets_per_doc

    def filter_and_extract_snippets(
        self,
        query: str,
        documents: Sequence[CRAGDocument],
        query_keywords: Sequence[str] = (),
    ) -> List[str]:
        """Extract deduplicated salient sentences matching query concepts."""
        q_tokens = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 2}
        for kw in query_keywords:
            q_tokens.update(w.lower() for w in re.findall(r"\w+", kw) if len(w) > 2)

        seen_hashes: Set[str] = set()
        ranked_snippets: List[tuple[float, str]] = []

        for doc in documents:
            sentences = _SPLIT_REGEX.split(doc.content)
            doc_snippets = 0
            for sent in sentences:
                cleaned = sent.strip()
                if len(cleaned) < self.min_sentence_len:
                    continue
                h = _sentence_hash(cleaned)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                s_tokens = {w.lower() for w in re.findall(r"\w+", cleaned) if len(w) > 2}
                overlap = len(q_tokens & s_tokens)
                if overlap > 0 or doc.score >= 0.70:
                    # Score based on overlap density and parent chunk score
                    density = overlap / max(1, len(s_tokens))
                    composite_score = 0.6 * doc.score + 0.4 * min(1.0, density * 3.0)
                    ranked_snippets.append((composite_score, cleaned))
                    doc_snippets += 1
                    if doc_snippets >= self.max_snippets_per_doc:
                        break

        # Sort by composite score descending
        ranked_snippets.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in ranked_snippets]

    def deduplicate_documents(
        self,
        documents: Sequence[CRAGDocument],
        similarity_threshold: float = 0.85,
    ) -> List[CRAGDocument]:
        """Deduplicate overlapping document chunks using n-gram containment."""
        unique_docs: List[CRAGDocument] = []
        seen_token_sets: List[Set[str]] = []

        for doc in documents:
            tokens = {w.lower() for w in re.findall(r"\w+", doc.content) if len(w) > 3}
            if not tokens:
                continue

            is_dup = False
            for prev_tokens in seen_token_sets:
                inter = len(tokens & prev_tokens)
                union = len(tokens | prev_tokens)
                jaccard = inter / max(1, union)
                if jaccard >= similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                seen_token_sets.append(tokens)
                unique_docs.append(doc)

        return unique_docs
