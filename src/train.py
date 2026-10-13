from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.dataset import build_datasets, build_datasets_latent
from data.dataset import build_eval_trajectories, build_eval_trajectories_latent
from models.world_model import WorldModel
from eval.run_eval import run_eval
from analysis.param_match import count_params


def _interpolate_env(obj):
    import re
    pat = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def sub(s):
        return pat.sub(lambda m: os.environ.get(m.group(1), m.group(0)), s)

    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    if isinstance(obj, str):
        return sub(obj)
    return obj


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.get("base")
    if base:
        base_path = Path(path).parent / base
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "base"})
    return _interpolate_env(cfg)


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_cli_overrides(cfg, args):
    if args.R: cfg["injector"]["R"] = args.R
    if args.F: cfg["injector"]["F"] = args.F
    if args.T: cfg["injector"]["T"] = args.T
    if args.W: cfg["injector"]["W"] = args.W
    if args.steps: cfg["train"]["steps"] = args.steps
    if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
    if args.seed is not None: cfg["train"]["seed"] = args.seed
    if args.overfit: cfg["run"]["overfit"] = args.overfit
    if args.action_noise_std is not None:
        cfg["train"]["action_noise_std"] = args.action_noise_std
    if args.dab:
        cfg.setdefault("model", {})["decoder_action_bypass"] = True
    if args.stochastic_action_noise:
        cfg["train"]["stochastic_action_noise"] = True
    if args.action_contrastive is not None:
        cfg["train"]["action_contrastive"] = args.action_contrastive
    if args.action_contrastive_K is not None:
        cfg["train"]["action_contrastive_K"] = args.action_contrastive_K
    if args.ac_temperature is not None:
        cfg["train"]["ac_temperature"] = args.ac_temperature
    if args.agm_weight is not None:
        cfg["train"]["agm_weight"] = args.agm_weight
    if args.shuffle_actions:
        cfg["train"]["shuffle_actions"] = True
    if args.shuffle_mode is not None:
        cfg["train"]["shuffle_mode"] = args.shuffle_mode
    if args.action_blind:
        cfg.setdefault("model", {})["action_blind"] = True
    if args.inverse_dynamics is not None:
        cfg["train"]["inverse_dynamics"] = args.inverse_dynamics
    if args.delta_prediction:
        cfg.setdefault("model", {})["delta_prediction"] = True
    if args.matched_direction is not None:
        cfg["train"]["matched_direction"] = args.matched_direction
    if args.md_partner is not None:
        cfg["train"]["md_partner"] = args.md_partner
    if args.md_shuffle:
        cfg["train"]["md_shuffle"] = True
    if args.md_impute is not None:
        cfg["train"]["md_impute"] = args.md_impute
    return cfg


def set_seed(seed, deterministic=True):
    torch.manual_seed(seed); np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def make_optim(model, cfg):
    t = cfg["train"]
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    import math
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def env_block():
    import torch as _t
    lines = [
        f"python={platform.python_version()}",
        f"torch={_t.__version__}",
        f"cuda_available={_t.cuda.is_available()}",
        f"cuda_device={_t.cuda.get_device_name(0) if _t.cuda.is_available() else 'cpu'}",
        f"cuda_build={_t.version.cuda}",
        f"platform={platform.platform()}",
    ]
    for k in ["HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "DATA_ROOT", "CHECKPOINT_ROOT", "RESULTS_ROOT"]:
        lines.append(f"{k}={os.environ.get(k)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--R", default=None); ap.add_argument("--F", default=None)
    ap.add_argument("--T", default=None); ap.add_argument("--W", default=None)
    ap.add_argument("--action-noise-std", type=float, default=None,
                    help="action-denoising training: Gaussian std added to the action INPUT "
                         "during training (target stays clean). 0 = off (standard).")
    ap.add_argument("--dab", action="store_true",
                    help="Decoder Action Bypass: add a direct action->prediction residual.")
    ap.add_argument("--stochastic-action-noise", action="store_true",
                    help="Sample action-noise std ~ U(0, action_noise_std) per batch (for FAC-MS-SG "
                         "training: gate sees both clean and jittery actions).")
    ap.add_argument("--action-contrastive", type=float, default=None,
                    help="weight for the action-contrastive (info-NCE) objective (0=off). "
                         "Directly rewards the true action's prediction matching the true next-state.")
    ap.add_argument("--action-contrastive-K", type=int, default=None,
                    help="number of wrong-action negatives for the contrastive loss (default 4).")
    ap.add_argument("--agm-weight", type=float, default=None,
                    help="AGM (Action-Gradient Maximization) weight: maximize ||d pred/d a||^2 "
                         "via a first-order finite difference, to counteract MSE action-smoothing. 0=off.")
    ap.add_argument("--inverse-dynamics", type=float, default=None,
                    help="weight for the inverse-dynamics auxiliary: recover a_t from "
                         "(s_t, predicted s_{t+1}). Forces action information into the "
                         "prediction without rewarding raw sensitivity. 0=off.")
    ap.add_argument("--matched-direction", type=float, default=None,
                    help="weight of the matched direction loss: 1 - cos between the model's response "
                         "to swapping in the action of the nearest logged transition from another "
                         "training demonstration and the logged estimate of that swap's effect "
                         "(data/matched_bank.py). Uses the training split only.")
    ap.add_argument("--ac-temperature", type=float, default=None,
                    help="temperature dividing the contrastive logits. The logits are negative squared errors "
                         "about 1e-3 apart, so without a temperature near that scale the softmax stays uniform "
                         "and the loss sits at log(K+1)")
    ap.add_argument("--md-partner", default=None, choices=["nearest", "random"],
                    help="ablation: pair with a random transition of another demonstration instead of the nearest")
    ap.add_argument("--md-shuffle", action="store_true",
                    help="ablation: permute the logged estimates across pairs, breaking their link to the swap")
    ap.add_argument("--md-impute", type=float, default=None,
                    help="ablation: weight of a squared error loss that imputes the swapped next state as x_p plus "
                         "the partner's logged state change")
    ap.add_argument("--delta-prediction", action="store_true",
                    help="predict the state delta s_{t+1}-s_t instead of s_{t+1} "
                         "(Nagabandi et al. 2017), the standard remedy for a target "
                         "dominated by the observation auto-regression.")
    ap.add_argument("--shuffle-mode", default=None, choices=["sequence", "derangement", "timestep"],
                    help="how the negative control re-pairs actions (default: sequence)")
    ap.add_argument("--action-blind", action="store_true",
                    help="zero the action input everywhere, train and eval: the architectural "
                         "no-causal-link reference that every probe should score at its floor")
    ap.add_argument("--shuffle-actions", action="store_true",
                    help="Action-alignment negative control: shuffle action labels across "
                         "the batch, which breaks the action->state link while preserving the "
                         "action marginal. A faithfulness gain that survives this did not come "
                         "from the causal link. Passing the control is necessary, not sufficient.")
    args = ap.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["run"]["device"] = device
    run_id = args.run_id
    out_dir = Path(cfg["run"]["out_root"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = cfg["data"]["seq_len"]
    overfit = cfg["run"].get("overfit", 0)
    obs_mode = cfg["data"].get("obs_mode", "lowdim")
    if obs_mode == "latents":
        ds_tr, ds_va, ds_te, nz, obs_dim, action_dim = build_datasets_latent(seq_len=seq_len, overfit=overfit)
        build_eval = build_eval_trajectories_latent
    else:
        ds_tr, ds_va, ds_te, nz, obs_dim, action_dim = build_datasets(seq_len=seq_len, overfit=overfit)
        build_eval = build_eval_trajectories
    cfg["model"]["obs_dim"] = obs_dim
    cfg["model"]["action_dim"] = action_dim
    print(f"[train] run_id={run_id} seed={seed} device={device} "
          f"obs_dim={obs_dim} action_dim={action_dim} "
          f"|train|={len(ds_tr)} |val|={len(ds_va)} |test|={len(ds_te)} overfit={overfit}")

    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], drop_last=False)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=128, shuffle=False, num_workers=0)

    model = WorldModel(cfg, obs_dim, action_dim).to(device)
    pc = model.param_counts()
    flops = model.estimate_flops()
    print(f"[train] factors R/F/T/W = {model.factor_tuple()}")
    print(f"[train] params total={pc['total']} injector={pc['injector']} "
          f"encoder={pc['encoder']} backbone={pc['backbone']} decoder={pc['decoder']} "
          f"flops~{flops:.3e}")

    mdw = float(cfg["train"].get("matched_direction", 0.0) or 0.0)
    imw = float(cfg["train"].get("md_impute", 0.0) or 0.0)
    md_bank, md_gen = None, None
    if mdw > 0 or imw > 0:
        from data.matched_bank import build_bank, impute_loss, matched_direction_loss
        b = build_bank(nz, seq_len=seq_len, device=device, partner=cfg["train"].get("md_partner", "nearest"))
        if cfg["train"].get("md_shuffle", False):
            b["e"] = b["e"][np.random.default_rng(seed).permutation(len(b["e"]))]
        md_bank = {k: torch.from_numpy(v).to(device) for k, v in b.items() if k != "gap"}
        md_gen = torch.Generator(device="cpu").manual_seed(seed)
        print(f"[train] matched direction bank: {len(b['e'])} pairs, median state gap {np.median(b['gap']):.3f}")

    optim = make_optim(model, cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, cfg["train"]["warmup_steps"], cfg["train"]["steps"]))
    use_amp = cfg["train"]["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cfg_str = yaml.safe_dump(cfg, sort_keys=True)
    cfg_hash = hashlib.sha1(cfg_str.encode()).hexdigest()[:12]
    (out_dir / "config_resolved.yaml").write_text(cfg_str)
    (out_dir / "environment.txt").write_text(env_block())
    (out_dir / "stdout.log").open("w")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    best_val = float("inf"); best_state = None
    log = {"loss": []}
    step = 0
    it = iter(dl_tr)
    accum = cfg["train"]["accum"]
    log_every = cfg["train"]["log_every"]
    val_every = cfg["train"]["val_every"]
    total_steps = cfg["train"]["steps"]
    pbar_step = 0
    while step < total_steps:
        optim.zero_grad()
        for _ in range(accum):
            try:
                obs, act = next(it)
            except StopIteration:
                it = iter(dl_tr); obs, act = next(it)
            obs = obs.to(device); act = act.to(device)
            if cfg["train"].get("shuffle_actions", False):
                B = act.shape[0]
                mode = cfg["train"].get("shuffle_mode", "sequence")
                if mode == "sequence":
                    act = act[torch.randperm(B, device=act.device)]
                elif mode == "derangement":
                    k = int(torch.randint(1, B, (1,)).item())
                    act = torch.roll(act, shifts=k, dims=0)
                elif mode == "timestep":
                    idx = torch.arange(B, device=act.device)
                    ks = torch.randint(1, B, (act.shape[1],), device=act.device)
                    act = torch.stack([act[(idx + ks[t]) % B, t] for t in range(act.shape[1])], 1)
                else:
                    raise ValueError(f"unknown shuffle_mode {mode!r}")
            ans = float(cfg["train"].get("action_noise_std", 0.0) or 0.0)
            if ans > 0 and cfg["train"].get("stochastic_action_noise", False):
                import numpy as _np
                std = float(_np.random.uniform(0.0, ans))
            else:
                std = ans
            if std > 0:
                act_in = act + torch.randn_like(act) * std
            else:
                act_in = act
            L = act.shape[1]
            agm = float(cfg["train"].get("agm_weight", 0.0) or 0.0)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                out = model(obs[:, :L], act_in)
                loss = model.next_step_loss(out["pred"], obs) / accum
                acw = float(cfg["train"].get("action_contrastive", 0.0) or 0.0)
                if acw > 0:
                    K = int(cfg["train"].get("action_contrastive_K", 4))
                    target = obs[:, 1:L + 1]
                    s_true = -((out["pred"] - target) ** 2).mean(-1)
                    B = act.shape[0]
                    a_neg = act.unsqueeze(1).expand(B, K, L, -1).reshape(B * K, L, -1)
                    a_neg = a_neg + torch.randn_like(a_neg) * 0.3
                    o_rep = obs[:, :L].unsqueeze(1).expand(B, K, L, -1).reshape(B * K, L, -1)
                    out_neg = model(o_rep, a_neg)
                    s_neg = -((out_neg["pred"] - target.unsqueeze(1).expand(B, K, L, -1)
                               .reshape(B * K, L, -1)) ** 2).mean(-1).view(B, K, L)
                    logits = torch.cat([s_true.unsqueeze(1), s_neg], dim=1)
                    logits = logits / float(cfg["train"].get("ac_temperature", 1.0) or 1.0)
                    loss = loss + acw * (-F.log_softmax(logits, dim=1)[:, 0]).mean() / accum
                idw = float(cfg["train"].get("inverse_dynamics", 0.0) or 0.0)
                if idw > 0 and "a_hat" in out:
                    loss = loss + idw * ((out["a_hat"] - act_in) ** 2).mean() / accum
                if agm > 0:
                    eps = float(cfg["train"].get("agm_eps", 0.1))
                    out_p = model(obs[:, :L], act_in + torch.randn_like(act_in) * eps)
                    response = out_p["pred"] - out["pred"]
                    loss = loss - agm * (response.float() ** 2).mean() / accum
                if mdw > 0:
                    loss = loss + mdw * matched_direction_loss(model, md_bank, 128, md_gen) / accum
                if imw > 0:
                    loss = loss + imw * impute_loss(model, md_bank, 128, md_gen) / accum
            scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        scaler.step(optim); scaler.update()
        sched.step()
        step += 1
        log["loss"].append(float(loss.item() * accum))
        if step % log_every == 0:
            recent = np.mean(log["loss"][-log_every:])
            line = (f"[train] step {step}/{total_steps} loss={recent:.5f} "
                    f"lr={sched.get_last_lr()[0]:.2e}")
            print(line)
            (out_dir / "stdout.log").open("a").write(line + "\n")
        if step % val_every == 0 or step == total_steps:
            v = _validate(model, dl_va, device, use_amp)
            if v < best_val:
                best_val = v; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            line = f"[val] step {step} val_mse={v:.5f} (best={best_val:.5f})"
            print(line); (out_dir / "stdout.log").open("a").write(line + "\n")

    train_time = time.time() - t0
    peak_vram = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else 0.0

    if best_state is not None:
        model.load_state_dict(best_state)
    if cfg["run"].get("save_ckpt"):
        ck_dir = Path(os.environ.get("CHECKPOINT_ROOT", "checkpoints"))
        ck_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "cfg": cfg, "val_mse": best_val,
                    "params": pc["total"]},
                   ck_dir / f"{run_id}.pt")

    print("[eval] running frozen prediction + controllability suite ...")
    eval_trajs, _, _ = build_eval(split="test")
    try:
        ev = run_eval(model, dl_va, eval_trajs, cfg, device=device,
                      max_seqs_pred=cfg["eval"]["n_test_seqs"],
                      max_seqs_ctrl=min(64, cfg["eval"]["n_test_seqs"]))
    except Exception as e:
        ev = {"eval_error": repr(e)}

    metrics = {
        "run_id": run_id,
        "seed": seed,
        "config_hash": cfg_hash,
        "factor_R": model.factor_tuple()["R"],
        "factor_F": model.factor_tuple()["F"],
        "factor_T": model.factor_tuple()["T"],
        "factor_W": model.factor_tuple()["W"],
        "params_total": pc["total"], "params_injector": pc["injector"],
        "params_encoder": pc["encoder"], "params_backbone": pc["backbone"], "params_decoder": pc["decoder"],
        "flops_est": flops,
        "peak_vram_mb": peak_vram,
        "train_time_s": train_time,
        "best_val_mse": best_val,
        "train_loss_final": float(np.mean(log["loss"][-50:])),
        "obs_dim": obs_dim, "action_dim": action_dim,
        "overfit": overfit,
        "action_noise_std": float(cfg["train"].get("action_noise_std", 0.0) or 0.0),
        "inverse_dynamics": float(cfg["train"].get("inverse_dynamics", 0.0) or 0.0),
        "delta_prediction": bool(cfg.get("model", {}).get("delta_prediction", False)),
        "agm_weight": float(cfg["train"].get("agm_weight", 0.0) or 0.0),
        "matched_direction": float(cfg["train"].get("matched_direction", 0.0) or 0.0),
        "md_partner": cfg["train"].get("md_partner", "nearest"),
        "md_shuffle": bool(cfg["train"].get("md_shuffle", False)),
        "md_impute": float(cfg["train"].get("md_impute", 0.0) or 0.0),
        "ac_temperature": float(cfg["train"].get("ac_temperature", 1.0) or 1.0),
        "shuffle_actions": bool(cfg["train"].get("shuffle_actions", False)),
        "shuffle_mode": cfg["train"].get("shuffle_mode", "sequence"),
        "action_blind": bool(cfg.get("model", {}).get("action_blind", False)),
        "prediction": ev.get("prediction", {}),
        "controllability": ev.get("controllability", {}),
        "trained_gate": (float(torch.sigmoid(getattr(model.injector, "gate", torch.tensor(0.0))).item())
                          if hasattr(model.injector, "gate") and isinstance(getattr(model.injector, "gate"), torch.nn.Parameter)
                          else None),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[done] wrote {out_dir/'metrics.json'}")
    print(json.dumps({
        "run_id": metrics["run_id"],
        "R/F/T/W": [metrics["factor_R"], metrics["factor_F"], metrics["factor_T"], metrics["factor_W"]],
        "params_total": metrics["params_total"], "params_injector": metrics["params_injector"],
        "best_val_mse": metrics["best_val_mse"],
        "prediction": metrics.get("prediction", {}),
        "controllability": metrics.get("controllability", {}),
    }, indent=2))
    print(f"result: {run_id} trained | best_val_mse={best_val:.5f} "
          f"params={pc['total']} vram_mb={peak_vram:.1f} time_s={train_time:.1f}")


@torch.no_grad()
def _validate(model, dl, device, use_amp):
    model.eval()
    losses = []
    for obs, act in dl:
        obs = obs.to(device); act = act.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            out = model(obs[:, :act.shape[1]], act)
            losses.append(model.next_step_loss(out["pred"], obs).item())
    model.train()
    return float(np.mean(losses))


if __name__ == "__main__":
    main()
