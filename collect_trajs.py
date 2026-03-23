"""
Collect mini-swe-agent v2 trajectories and transform them into SFT-ready JSONL.

Reads trajectory files produced by ``mini-extra swebench`` and evaluation
results from ``swebench.harness.run_evaluation`` (or sb-cli), then outputs
one JSONL file per style where each line is a resolved trajectory in the
target model's chat format.

Output schema (one JSON object per line)::

    {
        "messages":    [{"role": ..., "content": ..., ...}, ...],
        "instance_id": "django__django-14493",
        "resolved":    true,
        "model":       "openai/glm-5",
        "traj_id":     "django__django-14493.<run_hash>",
        "patch":       "diff --git a/..."
    }

Usage::

    python -m traj_mgr.collect_trajs \\
        --traj_dir  ./trajs/ \\
        --eval_dir  ./logs/run_evaluation/glm5_verified_10/ \\
        --style     nemotron \\
        --out_dir   trajectories_sft/

Styles: nemotron | qwen35 | qwen35_think
    See utils.py for detailed descriptions of each style.

Reference:
    Adapted from swesmith/train/traj_mgr/collect_trajs.py to work with
    mini-swe-agent v2 trajectory format (messages-based) instead of
    SWE-agent v1 format (trajectory-based).
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from tqdm.auto import tqdm

from traj_mgr.utils import MAP_STYLE_TO_FUNC


def _generate_hash(text: str, length: int = 8) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def load_resolved_ids(eval_dir: Path) -> set[str]:
    """Load the set of resolved instance IDs from an evaluation directory.

    Supports two formats:
      1. SWE-bench harness ``results.json`` (top-level ``resolved_ids`` list)
      2. Per-instance ``report.json`` inside ``eval_dir/<instance_id>/report.json``
    """
    resolved: set[str] = set()

    # Format 1: single results.json (produced by swebench.harness or sb-cli)
    for candidate in eval_dir.glob("*.json"):
        try:
            data = json.loads(candidate.read_text())
            if "resolved_ids" in data:
                resolved.update(data["resolved_ids"])
                return resolved
        except (json.JSONDecodeError, KeyError):
            continue

    # Format 2: per-instance report.json (SWE-smith style)
    for report_path in eval_dir.rglob("report.json"):
        try:
            report = json.loads(report_path.read_text())
            instance_id = report_path.parent.name
            is_resolved = report.get("resolved", False)
            if isinstance(is_resolved, dict):
                # report.json may nest under instance_id key
                is_resolved = is_resolved.get(instance_id, {}).get("resolved", False)
            if is_resolved:
                resolved.add(instance_id)
        except (json.JSONDecodeError, KeyError):
            continue

    return resolved


def _find_traj_files(traj_dir: Path) -> list[Tuple[str, Path]]:
    """Discover trajectory files and return (instance_id, path) pairs.

    Supports two layouts:
      1. Flat:   traj_dir/<instance_id>.traj.json
      2. Nested: traj_dir/<instance_id>/<instance_id>.traj.json
    """
    results = []

    # Flat layout (mini-swe-agent default)
    for p in traj_dir.glob("*.traj.json"):
        instance_id = p.name.replace(".traj.json", "")
        results.append((instance_id, p))

    # Nested layout (some mini-swe-agent versions or manual reorganisation)
    for d in traj_dir.iterdir():
        if d.is_dir():
            traj_file = d / f"{d.name}.traj.json"
            if traj_file.exists() and d.name not in {r[0] for r in results}:
                results.append((d.name, traj_file))

    return sorted(results, key=lambda x: x[0])


def process_single_trajectory(
    instance_id: str,
    traj_path: Path,
    resolved_ids: set[str],
    transform_traj,
    run_hash: str,
    resolved_only: bool = True,
) -> Optional[Tuple[str, dict]]:
    """Process a single trajectory file and return the transformed result."""
    try:
        traj_orig = json.loads(traj_path.read_text(encoding="utf-8"))

        is_resolved = instance_id in resolved_ids
        if resolved_only and not is_resolved:
            return None

        info = traj_orig.get("info", {})
        exit_status = info.get("exit_status", "unknown")
        if exit_status != "Submitted":
            return None

        traj = transform_traj(traj_orig)

        # Attach metadata (same schema as SWE-smith output)
        traj["instance_id"] = instance_id
        traj["resolved"] = is_resolved
        traj["traj_id"] = f"{instance_id}.{run_hash}"
        traj["patch"] = info.get("submission", "")

        # Extract model name from traj info
        model_name = ""
        config = info.get("config", {})
        if isinstance(config, dict):
            model_name = config.get("model", "")
        if not model_name:
            model_stats = info.get("model_stats", {})
            model_name = model_stats.get("model", "")
        traj["model"] = model_name

        return (instance_id, traj)
    except Exception as e:
        print(f"Error processing {instance_id}: {e}")
        return None


def main(
    traj_dir: Path,
    eval_dir: Path,
    style: str,
    out_dir: Path,
    workers: int,
    resolved_only: bool = True,
):
    if style not in MAP_STYLE_TO_FUNC:
        raise ValueError(
            f"Style {style!r} not supported. Options: {list(MAP_STYLE_TO_FUNC.keys())}"
        )
    transform_traj = MAP_STYLE_TO_FUNC[style]

    # Load resolved IDs
    resolved_ids = load_resolved_ids(eval_dir)
    print(f"Found {len(resolved_ids)} resolved instances in {eval_dir}")

    # Discover trajectory files
    traj_files = _find_traj_files(traj_dir)
    print(f"Found {len(traj_files)} trajectory files in {traj_dir}")

    if not traj_files:
        print("No trajectory files found. Check --traj_dir path.")
        return

    run_hash = _generate_hash(str(traj_dir.resolve()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ft_{style}_{eval_dir.name}.jsonl"

    # Process trajectories in parallel
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                process_single_trajectory,
                instance_id,
                traj_path,
                resolved_ids,
                transform_traj,
                run_hash,
                resolved_only,
            ): instance_id
            for instance_id, traj_path in traj_files
        }

        for future in tqdm(
            as_completed(future_to_id),
            total=len(future_to_id),
            desc="Processing trajectories",
        ):
            result = future.result()
            if result is not None:
                results.append(result)

    # Sort by instance_id for reproducibility
    results.sort(key=lambda x: x[0])

    # Write output
    num_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, traj in results:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"\nWrote {num_written} trajectories to {out_path.absolute()}")
    print(f"  Style:    {style}")
    print(f"  Resolved: {len(resolved_ids)}")
    print(f"  Skipped:  {len(traj_files) - num_written}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-t", "--traj_dir",
        type=Path,
        required=True,
        help="Directory containing mini-swe-agent trajectory files (.traj.json)",
    )
    parser.add_argument(
        "-e", "--eval_dir",
        type=Path,
        required=True,
        help="Directory containing evaluation results (results.json or per-instance report.json)",
    )
    parser.add_argument(
        "-s", "--style",
        type=str,
        default="nemotron",
        choices=list(MAP_STYLE_TO_FUNC.keys()),
        help="Target chat template style (default: nemotron)",
    )
    parser.add_argument(
        "-o", "--out_dir",
        type=Path,
        default=Path("trajectories_sft/"),
        help="Output directory for SFT JSONL files (default: trajectories_sft/)",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 4) + 4),
        help="Max worker threads (default: min(32, cpu_count+4))",
    )
    parser.add_argument(
        "--include_unresolved",
        action="store_true",
        help="Include unresolved trajectories (default: resolved only)",
    )
    args = parser.parse_args()
    main(
        traj_dir=args.traj_dir,
        eval_dir=args.eval_dir,
        style=args.style,
        out_dir=args.out_dir,
        workers=args.workers,
        resolved_only=not args.include_unresolved,
    )
