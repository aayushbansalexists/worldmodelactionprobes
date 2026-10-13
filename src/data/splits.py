from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def _frac(demo_id: str) -> float:
    h = hashlib.md5(str(demo_id).encode("utf-8")).hexdigest()
    return (int(h[:16], 16) % 10_000_000) / 10_000_000


def split_demo_ids(demo_ids, train=0.7, val=0.15, test=0.15):
    train_ids, val_ids, test_ids = [], [], []
    for did in sorted(set(demo_ids)):
        f = _frac(str(did))
        if f < train:
            train_ids.append(did)
        elif f < train + val:
            val_ids.append(did)
        else:
            test_ids.append(did)
    return train_ids, val_ids, test_ids


def build_and_save_splits(processed_dir, demo_meta):
    out_root = os.environ.get("DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data"))
    splits_dir = Path(out_root) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    ids = list(demo_meta.keys())
    tr, va, te = split_demo_ids(ids)
    out = {
        "split_rule": "deterministic md5 hash of demo_id; train<0.70, val<0.85, else test; frames never shuffled across splits",
        "n_demos": len(ids),
        "train": tr, "val": va, "test": te,
        "demo_meta": demo_meta,
    }
    with open(splits_dir / "libero_splits.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


__all__ = ["split_demo_ids", "build_and_save_splits"]
