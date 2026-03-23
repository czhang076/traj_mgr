"""
Remove temporary / debug files from the trajectory directory.

mini-swe-agent produces log files alongside trajectories that are not
needed for SFT data processing. This script removes them to save disk
space before archiving or uploading trajectory directories.

Usage::

    python -m traj_mgr.clean_trajs ./trajs/

Reference:
    Adapted from swesmith/train/traj_mgr/clean_trajs.py for
    mini-swe-agent v2 directory layout.
"""

import argparse
import os
from pathlib import Path

# File patterns to remove
_REMOVE_SUFFIXES = {
    ".log",
    ".debug.log",
    ".info.log",
    ".trace.log",
    ".config.yaml",
}

# Files to always keep
_KEEP_FILES = {
    "preds.json",
}

# File extensions to always keep
_KEEP_SUFFIXES = {
    ".traj.json",
    ".patch",
}


def main(traj_dir: str, dry_run: bool = False):
    traj_path = Path(traj_dir)
    if not traj_path.exists():
        print(f"Directory not found: {traj_dir}")
        return

    removed = 0
    kept = 0

    for root, dirs, files in os.walk(traj_path):
        for fname in files:
            fpath = Path(root) / fname

            # Always keep trajectory and prediction files
            if fname in _KEEP_FILES:
                kept += 1
                continue
            if any(fname.endswith(s) for s in _KEEP_SUFFIXES):
                kept += 1
                continue

            # Remove known temporary patterns
            if any(fname.endswith(s) for s in _REMOVE_SUFFIXES):
                if dry_run:
                    print(f"  [dry-run] would remove: {fpath}")
                else:
                    fpath.unlink()
                removed += 1
            # Remove YAML exit status files
            elif fname.startswith("exit_statuses_") and fname.endswith(".yaml"):
                if dry_run:
                    print(f"  [dry-run] would remove: {fpath}")
                else:
                    fpath.unlink()
                removed += 1
            else:
                kept += 1

    action = "Would remove" if dry_run else "Removed"
    print(f"{action} {removed} files, kept {kept} files in {traj_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "traj_dir",
        type=str,
        help="Path to the trajectory directory",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be removed without actually deleting",
    )
    args = parser.parse_args()
    main(**vars(args))
