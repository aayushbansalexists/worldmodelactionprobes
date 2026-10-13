# Supplementary code

Code for the submission "Can We Check Whether a World Model Follows Actions Without a Simulator?".
This copy is anonymized for double blind review.

The paper asks whether probes that score a world model's response to actions, without running a
simulator, notice when that response points the wrong way. The code measures the true response with
simulators that can be reset to a saved state, builds models whose response is reflected, rotated,
rescaled or removed, scores every probe on them, and trains models on each probe's criterion.

## Layout

| path | contents |
|---|---|
| `wmprobe/` | a small library you can apply to your own model: the matched probe (`matched_probe`), the response transform audit (`audit_direction`), zero action divergence and other size probes, logged action retrieval, adversaries and negative controls |
| `src/` | every experiment in the paper: training, simulator ground truth, the probes, the released model studies, the analyses and the table scripts |
| `proofs/` | Lean 4 proofs of every proposition and the corollary (`PaperResults.lean`, with lemmas in `WamProofs.lean`) and an axiom check (`AxiomCheck.lean`) |
| `examples/` | three short scripts that run without any dataset |

## Quick check, no data needed

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python examples/matched.py          # the matched probe on a small world with a known answer
python examples/adversary.py        # a model with no dynamics that raises retrieval
cd src
python -m analysis.verify_math      # 37 checks, including symbolic and numerical checks of Propositions 3, 6, 7 and 8
python -m analysis.audit_critical   # 11 checks of the library functions
```

`examples/matched.py` takes about a minute on a laptop. It prints the probe for a trained model,
its reflected copy (the sign flips), a rescaled copy (unchanged) and a copy with no response (0).

## Names in the paper and in the code

| paper | code |
|---|---|
| direction under random perturbations, D | `direction` in the result files |
| direction on logged swaps, D_log | `dir_on` (`dir_on_k1` is the truth on the matched probe's own pairs) |
| size ratio, M | `mag_ratio` |
| zero action divergence, C1 | `C1` |
| calibration score, WorldSimProbe Task 1 | `task1` |
| logged action retrieval, Gaussian or real alternatives | `ret_gauss_meanrank`, `ret_marg_meanrank` |
| inverse model check, cycle error or direction form | `inv_err_rand`, `inv_err_swap`, `inv_dir_rand`, `inv_dir_swap` |
| matched probe | `matched` (`matched_k1`, `matched_k3`, `matched_k5`) |
| matched swap alignment | `train.py --matched-direction W`, loss in `src/data/matched_bank.py` |
| its ablations | `--md-partner random`, `--md-shuffle`, `--md-impute W` |
| contrastive loss, retrieval's criterion | `--action-contrastive W --ac-temperature 1e-3` |
| AGM, inverse dynamics, delta prediction | `--agm-weight 0.5`, `--inverse-dynamics W`, `--delta-prediction` |
| negative controls | `--shuffle-actions --shuffle-mode {sequence,derangement,timestep}`, `--action-blind` |
| response transforms | `identity`, `reflect`, `rotate`, `scale_0.25`, `scale_4`, `blind` |

## Data and third party code

Run everything from `src/`. The scripts look for `data/` and `third_party/` next to them, and
`scripts/env.sh` keeps every cache inside the project.

```bash
bash scripts/setup.sh && source scripts/env.sh
pip install "mujoco==3.2.7" "robosuite==1.4.1" bddl cloudpickle
pip install --no-deps gym==0.25.2 gym_notices
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git clone https://github.com/pxxq25/WorldSimProbe.git third_party/WorldSimProbe
export PYTHONPATH=$PWD/third_party/LIBERO:$PYTHONPATH
export LIBERO_CONFIG_PATH=$PWD/.cache/libero
```

Place the LIBERO hdf5 files for the Goal, Object and Spatial suites in `data/raw/libero/`, then run
`python -m data.download_libero`, which reads them and writes the proprioceptive trajectories and the
splits. It does not download anything.

For the released TD-MPC2 models, install `dm_control==1.0.27`, clone
`https://github.com/nicklashansen/tdmpc2` into `third_party/tdmpc2` (only its task definitions are
used) and put the single task checkpoints from `https://huggingface.co/nicklashansen/tdmpc2`
(`dmcontrol/*.pt`) in `data/raw/tdmpc2/dmcontrol/`. The loader needs neither hydra nor tensordict. Run
dm_control with `MUJOCO_GL=disabled`, in a separate process from LIBERO, which needs `glx`.

For the released DIAMOND models, clone `https://github.com/eloialonso/diamond` into
`third_party/diamond` and use a separate environment with its requirements and `gymnasium[atari]`.
`src/eval/diamond_probe.py` explains the setup in its docstring.

`src/eval/libero_compat.py` must be imported before anything touches robosuite. It fixes a joint
lookup that robosuite 1.4.1 gets wrong on mujoco 3 and selects a GL backend that works headless.

## Reproducing each table

Run from `src/` in this order. GPU jobs use one 8 GB GPU. Run one simulator job at a time on a
machine with 8 GB of RAM.

| paper | commands |
|---|---|
| the 92 LIBERO models | `bash scripts/sweep_cf.sh`, then `bash scripts/parallel_sweep.sh 1` for the jobs in `scripts/validity_jobs.txt` (1 to 3 minutes per model) |
| simulator ground truth | `python -m eval.counterfactual_lowdim --tag v3`, `python -m eval.counterfactual_wsp --tag v3`, `python -m eval.counterfactual_matched --k 5 --tag v3` (about 35 minutes) |
| Tables 1, 2 (LIBERO columns), 7, 8, 9, 10 and 15 | `python -m analysis.probe_validity --tag v3 --runs $(sed 's/ .*//' scripts/validity_jobs.txt)`, `python -m analysis.matched_study`, `python -m analysis.inverse_probe`, `python -m eval.closed_loop --tasks 0 2 5 7 10 12 15 17 20 22 25 27`, `python -m analysis.probe_landscape`, then `python -m analysis.make_tables` |
| Tables 2 (TD-MPC2 columns), 4 and 5 | `python eval/tdmpc2_verify.py`, `python eval/tdmpc2_probe.py`, then `python -m analysis.make_tables` |
| Atari seed replication (Appendix G) | in the DIAMOND environment, `python eval/diamond_probe.py --positions 100 --seed 1 --games BankHeist BattleZone Breakout Freeway Pong Qbert --out results/tables/diamond_validity_seed1.json` |
| Tables 2 (Atari columns) and 6 | in the DIAMOND environment, `python eval/diamond_probe.py --positions 100 --games Alien Amidar Assault Asterix BankHeist BattleZone Boxing Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite Gopher Hero Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner Seaquest UpNDown`, then `python -m analysis.diamond_tables` |
| Tables 11 and 12 | `python -m analysis.inverse_probe`, then `python -m analysis.inverse_tables` |
| Tables 3 and 13 | `bash scripts/sweep_md.sh`, `bash scripts/sweep_ac.sh`, `bash scripts/sweep_ablate.sh`, then `python -m analysis.objectives_md` |
| control with matched swap alignment (Appendix K) | `python -m eval.closed_loop --runs cf_md01_causal_s0 cf_md01_causal_s1 cf_md01_causal_s2 cf_md01_causal_s3 cf_md01_causal_s4 --tasks 0 2 5 7 10 12 15 17 20 22 25 27 --out results/tables/closed_loop_md_v3.json` |
| Table 14 | `python eval/tdmpc2_finetune.py`, `python eval/tdmpc2_estimate.py`, then `python -m analysis.finetune_tables` |
| retrieval's permutation null (Appendix O) | `python -m analysis.null_levels` |
| the Proposition 7 prediction for the matched probe (Appendix C) | `python -m analysis.attenuation` |
| Table 16 | `python -m analysis.adversary_study`, `python -m analysis.adversary_matched`, then `python -m analysis.make_tables` |
| Figures 2 to 9 | `python -m analysis.fig_matched landscape eef3d validity_bars tdmpc2 closed_loop` and `python -c "import analysis.fig_validity as f; f.fig_verdicts(); f.fig_scatter(); f.fig_task1()"`, written to `figures/` |
| Propositions 1 to 8, Corollary 5 and Appendix D | in `proofs/`, `lake build` then `lake env lean AxiomCheck.lean` (Lean 4, Mathlib v4.15.0); `proofs/README.md` maps each result to its theorems |

Every table script writes LaTeX to `tables/` and prints the numbers the text quotes. `GT_TAG`
selects the ground truth version, and the paper uses `v3`.

## License

MIT
