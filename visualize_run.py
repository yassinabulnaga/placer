#!/usr/bin/env python3
"""
Run the PartCL placer on a benchmark and save a visualization PNG.

Examples:
    python3 submissions/partcl/visualize_run.py --benchmark ibm01
    python3 submissions/partcl/visualize_run.py --benchmark ibm01 --profile-mode outline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from macro_place.evaluate import evaluate_benchmark
from macro_place.utils import visualize_placement
from submissions.partcl.runner import PartitionSeededPortfolioPlacer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", "-b", default="ibm01", help="Benchmark name, e.g. ibm01")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional output PNG path. Defaults to submissions/partcl/vis/<benchmark>.png",
    )
    parser.add_argument(
        "--profile-mode",
        default="outline",
        help="Optional PARTCL_PROFILE_MODE to use for faster visualization runs. Default: outline",
    )
    args = parser.parse_args()

    if args.profile_mode:
        os.environ["PARTCL_PROFILE_MODE"] = args.profile_mode

    placer = PartitionSeededPortfolioPlacer()
    testcase_root = "external/MacroPlacement/Testcases/ICCAD04"
    result = evaluate_benchmark(placer, args.benchmark, testcase_root)

    vis_dir = Path("submissions/partcl/vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.output) if args.output else vis_dir / f"{args.benchmark}.png"
    visualize_placement(
        result["placement"],
        result["benchmark"],
        save_path=str(save_path),
        plc=result.get("plc"),
    )

    print(
        f"{args.benchmark}: proxy={result['proxy_cost']:.4f} "
        f"(wl={result['wirelength']:.4f} den={result['density']:.4f} cong={result['congestion']:.4f})"
    )
    print(save_path.resolve())


if __name__ == "__main__":
    main()
