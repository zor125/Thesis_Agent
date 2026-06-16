from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TOP_K = 20


@dataclass(frozen=True)
class RankedItem:
    rank: int
    score: float
    text: str
    index: int | None
    model: str | None


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension.")
    if not left:
        raise ValueError("Vectors cannot be empty.")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    return dot_product / (left_norm * right_norm)


def rank_embeddings(
    query_embedding: Sequence[float],
    items: Sequence[dict[str, Any]],
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[RankedItem]:
    if top_k < 1:
        raise ValueError("top_k must be greater than 0.")

    scored_items = []
    for item in items:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Each item must contain an embedding list.")

        scored_items.append(
            (
                cosine_similarity(query_embedding, embedding),
                item,
            )
        )

    scored_items.sort(key=lambda scored_item: scored_item[0], reverse=True)
    return [
        RankedItem(
            rank=rank,
            score=score,
            text=str(item.get("text", "")),
            index=item.get("index") if isinstance(item.get("index"), int) else None,
            model=item.get("model") if isinstance(item.get("model"), str) else None,
        )
        for rank, (score, item) in enumerate(scored_items[:top_k], start=1)
    ]


def load_embeddings(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Embeddings file must contain a JSON array.")
    return data


def load_query_embedding(args: argparse.Namespace) -> list[float]:
    if args.query_embedding:
        return parse_vector(args.query_embedding)
    if args.query_embedding_file:
        data = json.loads(args.query_embedding_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("embedding")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0].get("embedding")
        return parse_vector(data)
    raise ValueError("Provide --query-embedding or --query-embedding-file.")


def parse_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Query embedding must be a JSON array of numbers.")

    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("Query embedding must contain only numbers.") from exc


def write_results(results: Sequence[RankedItem], output_path: Path | None) -> None:
    json_text = json.dumps(
        [asdict(result) for result in results],
        ensure_ascii=False,
        indent=2,
    )
    if output_path is None:
        print(json_text)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank embeddings by cosine similarity and return the top results.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("embeddings.json"),
        help="JSON file created by embedding.py.",
    )
    parser.add_argument(
        "--query-embedding",
        help='Query embedding as a JSON array, for example "[0.1, 0.2]".',
    )
    parser.add_argument(
        "--query-embedding-file",
        type=Path,
        help="JSON file containing a query embedding, an embedding object, or embedding.py output.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return. Defaults to {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Omit to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        query_embedding = load_query_embedding(args)
        embeddings = load_embeddings(args.embeddings)
        results = rank_embeddings(query_embedding, embeddings, top_k=args.top_k)
        write_results(results, args.output)
    except Exception as exc:
        print(f"Ranking failed: {exc}", file=sys.stderr)
        return 1

    destination = "stdout" if args.output is None else args.output
    print(f"Ranked {len(embeddings)} embeddings -> top {len(results)} to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
