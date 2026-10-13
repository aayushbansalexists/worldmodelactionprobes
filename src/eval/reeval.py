from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import build_eval_trajectories, build_eval_trajectories_latent
from models.world_model import WorldModel
from eval.run_eval import run_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    res_root = os.environ["RESULTS_ROOT"]
    run_dir = Path(res_root) / "raw" / args.run_id
    cfg = yaml.safe_load(open(run_dir / "config_resolved.yaml"))
    ck_path = Path(os.environ["CHECKPOINT_ROOT"]) / f"{args.run_id}.pt"
    model = WorldModel(cfg, cfg["model"]["obs_dim"], cfg["model"]["action_dim"]).to(args.device)
    ck = torch.load(ck_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    if cfg.get("data", {}).get("obs_mode", "lowdim") == "latents":
        eval_trajs, _, _ = build_eval_trajectories_latent("test")
    else:
        eval_trajs, _, _ = build_eval_trajectories("test")
    ev = run_eval(model, None, eval_trajs, cfg, device=args.device,
                  max_seqs_pred=cfg["eval"]["n_test_seqs"],
                  max_seqs_ctrl=min(64, cfg["eval"]["n_test_seqs"]))
    mp = run_dir / "metrics.json"
    m = json.load(open(mp)) if mp.exists() else {}
    m["prediction"] = ev["prediction"]
    m["controllability"] = ev["controllability"]
    json.dump(m, open(mp, "w"), indent=2)
    print(f"[reeval] updated {mp}")
    print(json.dumps({"prediction": ev["prediction"], "controllability": ev["controllability"]}, indent=2))


if __name__ == "__main__":
    main()
