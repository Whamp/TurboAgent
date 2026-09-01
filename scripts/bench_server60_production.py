"""Run the production server60 inference study and write JSON evidence."""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx
from server60_benchmark.server60_capacity import run_capacity_and_cancellation
from server60_benchmark.server60_turbo import (
    run_turbo_quality_corpus,
    run_turbo_selection_scale,
)
from server60_benchmark.server60_vllm import (
    run_direct_concurrency,
    run_n4_comparison,
)
from server60_benchmark.server60_workloads import (
    BOUNDED_ANSWER_PROMPT,
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DIRECT_PROMPT,
)


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run all server60 production benchmark phases."""
    endpoint = args.base_url.rstrip("/")
    root_url = endpoint.removesuffix("/v1")
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    async with httpx.AsyncClient(limits=limits) as client:
        model_response = await client.get(f"{endpoint}/models", timeout=10)
        model_response.raise_for_status()
        report: dict[str, Any] = {
            "environment": {
                "base_url": endpoint,
                "models": model_response.json(),
                "samples": args.samples,
                "direct_max_tokens": args.direct_max_tokens,
            },
            "direct_concurrency": await run_direct_concurrency(
                client,
                endpoint,
                args.model,
                args.samples,
                args.direct_max_tokens,
            ),
            "independent_vs_n4": {
                "reasoning_to_cap": await run_n4_comparison(
                    client,
                    endpoint,
                    args.model,
                    args.samples,
                    args.direct_max_tokens,
                    DIRECT_PROMPT,
                ),
                "bounded_answer": await run_n4_comparison(
                    client,
                    endpoint,
                    args.model,
                    args.samples,
                    16384,
                    BOUNDED_ANSWER_PROMPT,
                ),
            },
        }
        with TemporaryDirectory(prefix="turbo-server60-") as directory:
            tmp = Path(directory)
            report["turbo_selection_scale"] = await run_turbo_selection_scale(
                tmp,
                endpoint,
                args.model,
                args.embedding_base_url,
                args.embedding_model,
                args.samples,
            )
            report["turbo_quality_corpus"] = await run_turbo_quality_corpus(
                tmp,
                endpoint,
                args.model,
                args.embedding_base_url,
                args.embedding_model,
                args.quality_max_tokens,
            )
        report["capacity_and_cancellation"] = await run_capacity_and_cancellation(
            client,
            endpoint,
            root_url,
            args.model,
        )
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-base-url", default=DEFAULT_EMBEDDING_BASE_URL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--direct-max-tokens", type=int, default=2048)
    parser.add_argument("--quality-max-tokens", type=int, default=65536)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = asyncio.run(run_benchmark(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "written": str(args.out),
                "direct_n1_s": report["direct_concurrency"]["1"]["wall_mean_s"],
                "direct_n4_s": report["direct_concurrency"]["4"]["wall_mean_s"],
                "semantic_majority_tasks": report["turbo_quality_corpus"][
                    "semantic_majority_tasks"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
