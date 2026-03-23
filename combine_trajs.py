"""
Merge multiple SFT trajectory JSONL files, de-duplicate per instance,
and shuffle.

When the same instance appears in multiple runs (e.g. different teacher
temperatures), ``max_per_inst`` controls how many trajectories to keep
per instance to prevent over-fitting on popular issues.

Usage::

    python -m traj_mgr.combine_trajs \\
        --sft_dir trajectories_sft/ \\
        --max_per_inst 3

Reference:
    Adapted from swesmith/train/traj_mgr/combine_trajs.py.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def merge_and_shuffle_jsonl(
    sft_dir: Path = Path("trajectories_sft/"),
    max_per_inst: int = 3,
    output_file: Path | None = None,
    seed: int = 42,
):
    """Merge JSONL files, cap per-instance count, and shuffle."""

    jsonl_files = sorted(sft_dir.glob("ft_*.jsonl"))
    if not jsonl_files:
        print(f"No ft_*.jsonl files found in {sft_dir}")
        return

    print("Available JSONL files:")
    for idx, f in enumerate(jsonl_files):
        with open(f, encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        print(f"  [{idx}] {f.name}  ({n} trajectories)")

    selected = input(
        "\nIndices to merge (e.g. '0 1' or '0-2', Enter for all): "
    ).strip()

    if not selected:
        files = jsonl_files
    else:
        def expand(token: str) -> list[int]:
            if "-" in token:
                lo, hi = token.split("-", 1)
                return list(range(int(lo), int(hi) + 1))
            return [int(token)]
        indices = [i for tok in selected.split() for i in expand(tok)]
        files = [jsonl_files[i] for i in indices]

    if output_file is None:
        name = input("Output filename (without extension): ").strip() + ".jsonl"
        output_file = sft_dir / name

    # Read and group by instance_id
    inst_to_trajs: dict[str, list[dict]] = defaultdict(list)
    for fp in files:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                traj = json.loads(line)
                inst_id = traj.get("instance_id", "unknown")
                inst_to_trajs[inst_id].append(traj)

    # Sample and shuffle
    rng = random.Random(seed)
    all_trajs = []
    for inst_id, trajs in inst_to_trajs.items():
        k = min(len(trajs), max_per_inst)
        all_trajs.extend(rng.sample(trajs, k))
    rng.shuffle(all_trajs)

    # Stats
    repo_counts: Counter = Counter()
    for t in all_trajs:
        repo = t.get("instance_id", "").rsplit("-", 1)[0]
        repo_counts[repo] += 1

    # Write
    with open(output_file, "w", encoding="utf-8") as fh:
        for traj in all_trajs:
            fh.write(json.dumps(traj, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_trajs)} trajectories to {output_file}")
    print(f"  Unique instances: {len(inst_to_trajs)}")
    print(f"  max_per_inst:     {max_per_inst}")
    print(f"  seed:             {seed}")

    # Write metadata
    meta_path = output_file.with_name(f"metadata__{output_file.stem}.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "output_file": str(output_file),
                "source_files": [str(f) for f in files],
                "num_trajs": len(all_trajs),
                "num_instances": len(inst_to_trajs),
                "max_per_inst": max_per_inst,
                "seed": seed,
                "repo_counts": dict(
                    repo_counts.most_common()
                ),
            },
            fh,
            indent=2,
        )
    print(f"  Metadata:         {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d", "--sft_dir",
        type=Path,
        default=Path("trajectories_sft/"),
    )
    parser.add_argument(
        "-m", "--max_per_inst",
        type=int,
        default=3,
    )
    parser.add_argument(
        "-o", "--output_file",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    merge_and_shuffle_jsonl(**vars(args))
