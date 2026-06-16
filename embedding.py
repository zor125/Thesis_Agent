from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv


DEFAULT_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class EmbeddingResult:
    text: str
    embedding: list[float]
    model: str
    index: int


def create_embedding(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    dimensions: int | None = None,
    client: Any | None = None,
) -> EmbeddingResult:
    return create_embeddings(
        [text],
        model=model,
        dimensions=dimensions,
        client=client,
    )[0]


def create_embeddings(
    texts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    dimensions: int | None = None,
    client: Any | None = None,
) -> list[EmbeddingResult]:
    cleaned_texts = [normalize_text(text) for text in texts]
    if not cleaned_texts:
        raise ValueError("At least one text value is required.")
    if any(not text for text in cleaned_texts):
        raise ValueError("Embedding input cannot be empty.")

    openai_client = client if client is not None else build_client()
    request: dict[str, Any] = {
        "input": cleaned_texts,
        "model": model,
        "encoding_format": "float",
    }
    if dimensions is not None:
        request["dimensions"] = dimensions

    response = openai_client.embeddings.create(**request)
    return [
        EmbeddingResult(
            text=cleaned_texts[item.index],
            embedding=list(item.embedding),
            model=response.model,
            index=item.index,
        )
        for item in response.data
    ]


def build_client() -> Any:
    load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install it with: pip install -r requirements.txt"
        ) from exc

    return OpenAI()


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def read_texts(args: argparse.Namespace) -> list[str]:
    if args.text:
        return args.text
    if args.input_file:
        content = args.input_file.read_text(encoding="utf-8")
        if args.jsonl:
            return [line for line in content.splitlines() if line.strip()]
        return [content]
    raise ValueError("Provide --text or --input-file.")


def write_results(results: Sequence[EmbeddingResult], output_path: Path | None) -> None:
    payload = [asdict(result) for result in results]
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)

    if output_path is None:
        print(json_text)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create text embeddings with the OpenAI Embedding API.",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Text to embed. Repeat this option to embed multiple texts.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to a UTF-8 text file to embed.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Treat --input-file as one text input per non-empty line.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        help="Optional output dimensions for text-embedding-3 models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embeddings.json"),
        help="Output JSON path. Use '-' to print to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dimensions is not None and args.dimensions < 1:
        print("--dimensions must be greater than 0.", file=sys.stderr)
        return 2

    try:
        texts = read_texts(args)
        results = create_embeddings(
            texts,
            model=args.model,
            dimensions=args.dimensions,
        )
        output_path = None if str(args.output) == "-" else args.output
        write_results(results, output_path)
    except Exception as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1

    destination = "stdout" if output_path is None else output_path
    print(f"Created {len(results)} embeddings with {args.model} -> {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
