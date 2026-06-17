# VQ Codebook Regularization Plan

Last updated: 2026-06-17

## Motivation

The strongest latent-space evidence so far is the `vq1024+decoder` model:
token ablation showed that it actually uses game-rule conditioning, while the
continuous-slot and encoder-only VQ variants mostly behave like memorizing
baselines. The remaining problem is codebook collapse: the useful VQ+decoder
run activated only a small fraction of the 1024-entry codebook.

The next experiment should test whether a codebook-usage regularizer can make
the latent space less collapsed without destroying the semantic conditioning
signal.

## High-Level Track: Latent Regularization for Adaptation

This track is intentionally scoped to the game-latent representation rather
than the broader data-scaling or architecture-expansion agenda. The central
question is whether the learned game latent can become a practical adaptation
interface for novel PuzzleScript environments, not merely an auxiliary
conditioning vector that helps fit the training set.

The current evidence suggests this is plausible but not yet established:
VQ+token-decoder training made the model measurably depend on rule tokens under
token ablation, while continuous-slot and encoder-only VQ variants were easier
for the NCA body to bypass or memorize around. The weakness is that the useful
VQ run still collapsed to a small number of hard codebook entries. The first
goal is therefore to make the discrete slot space less collapsed while
preserving the semantic signal that token ablation exposed.

Evaluation should stay adaptation-centered. A regularized VQ checkpoint is
interesting only if it keeps transition and rollout quality, remains sensitive
to token ablation, and improves held-out or few-shot slot fitting. A checkpoint
with higher code usage but no token-ablation sensitivity is not solving the
right problem; it is only spreading mass across codes.

If usage regularization works, the next step is to make adaptation more
structured. Two natural follow-ups are:

- discrete code search or relaxed code-assignment fitting against a frozen
  codebook, instead of unconstrained continuous slot optimization;
- a history-to-latent encoder that amortizes inverse fitting by mapping a small
  set of observed `(state, action, next_state)` transitions into rule slots.

This keeps the VQ/regularization work complementary to the separate tracks on
more data, history-conditioned world models, and code-conditioning dropout.

## New Training Knobs

The current branch adds these flags to `nca_wm.train`:

- `--vq_usage_loss_weight`: multiplier on a differentiable codebook usage
  penalty, `log(K) - H(mean soft assignment)`.
- `--vq_entropy_temp`: temperature for the soft assignment distribution used
  by that penalty and its diagnostics.

Training curves now include:

- `vq_usage_losses`
- `vq_soft_perplexities`
- existing hard `vq_utils`

## Primary Sweep

Run the existing VQ+token-decoder recipe and vary only the usage regularizer.
Use `scaling_14` first because it is large enough to expose multi-game latent
collapse but still much cheaper than the larger n-per-rule sweeps.

```bash
GPU=0 \
GAMES=scaling_14 \
WEIGHTS="0 0.001 0.01 0.05" \
N_UPDATES=200000 \
nca_wm/scripts/run_vq_usage_ablation.sh
```

Default output:

```text
nca_wm/logs/vq_usage_ablation_scaling_14_s0/
  w0/
  w0p001/
  w0p01/
  w0p05/
```

The `w0` run is the no-usage-regularizer control under the same code path.

## Current Observations

Initial `scaling_14` results suggest the usage regularizer is useful, but not
for the originally hoped-for reason.

Completed runs:

- `w0`: no usage regularizer.
- `w0p001`: `--vq_usage_loss_weight 0.001`.
- `w0p01`: `--vq_usage_loss_weight 0.01`, launched in parallel on GPU 1.

The `w0p05` run is still in progress. Around 120k/200k steps it remained
stable, with decoder accuracy at 1.000, soft perplexity near 1024, and hard
batch code utilization around 4 active entries.

Readout so far:

- `0.001` and `0.01` both push soft assignment perplexity to about the full
  1024-entry codebook.
- They do not solve hard nearest-code collapse: hard `vq_util` is still about
  3 active entries for the completed nonzero-weight runs.
- Transition loss and decoder accuracy do not visibly regress at `0.001` or
  `0.01`.
- Token ablation still matters more for the regularized runs than for `w0`,
  especially under zeroed tokens, so the regularizer has not destroyed semantic
  conditioning.
- Few-shot inverse-fit improves versus `w0` after the loss-weighting fix in
  `inverse_fit_slot.py`, which makes the regularizer worth keeping even though
  hard code collapse remains.

Working interpretation: the current regularizer broadens soft code assignment
mass and improves adaptation behavior, but it is not a complete hard-code
usage fix. The next VQ-specific follow-up should test either a stronger
straight-through/code-assignment pressure or discrete code fitting, not just a
larger soft entropy coefficient.

## Machine Setup Checklist

On a fresh machine:

```bash
git pull --ff-only
git submodule update --init PuzzleScript
source .venv/bin/activate
.venv/bin/python setup_cpp.py build_ext --inplace
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
```

If the virtualenv does not exist, follow the root `AGENTS.md` setup notes:
create a Python 3.13 venv with `uv`, install requirements, install CUDA JAX
`jax[cuda12]==0.7.1`, install `pybind11`, then build the C++ extension.

Before launching, check GPU memory:

```bash
nvidia-smi
```

The default sweep expects a mostly free 24GB GPU. If the GPU is shared, run in
`tmux` and let the script wait for free memory:

```bash
tmux new-session -s vq_usage_ablation
cd /path/to/script-doctor
GPU=0 WAIT_FOR_GPU=1 GPU_FREE_THRESHOLD_MB=4000 \
    nca_wm/scripts/run_vq_usage_ablation.sh
```

## Monitoring

During training:

```bash
tail -f nca_wm/logs/vq_usage_ablation_scaling_14_s0/w0p01.log
```

Key fields in the train log:

- `vq_util`: hard number of codebook entries used by the current batch.
- `vq_usage`: entropy penalty value; lower is more uniform soft usage.
- `vq_perp`: soft assignment perplexity; higher means broader effective usage.
- `err` and `change_err`: transition prediction error.
- `dec_loss` and `dec_acc`: whether the latent still preserves game-token
  information.

## Follow-up Evaluation

After the sweep, compare:

1. Final and best validation transition error.
2. Autoregressive rollout error from `eval_multigame.npz`.
3. Hard code utilization and soft perplexity from `curves.npz`.
4. Token-ablation sensitivity using `nca_wm/scripts/token_ablation_eval.py`.
5. Few-shot inverse-fit behavior on the best checkpoint.

The best checkpoint is not simply the one with the largest codebook usage. The
regularizer is useful only if it increases effective code usage while keeping
or improving transition accuracy and preserving token-ablation sensitivity.

## Decision Criteria

Proceed with the regularizer if one of the nonzero weights:

- increases hard `vq_util` and soft perplexity over `w0`,
- does not regress teacher-forced or autoregressive transition error,
- still shows large degradation under token zeroing/permutation,
- improves held-out inverse-fit or at least gives a better initialization for
  slot fitting.

Reject or downweight it if:

- code usage increases but token ablation stops mattering,
- decoder accuracy improves while world-model rollout quality regresses,
- the regularizer produces unstable training or worse identity-relative
  behavior on held-out games.

## Fallback Sweep

If the primary sweep is too expensive on the available machine, run a short
pilot first:

```bash
GPU=0 \
GAMES=scaling_14 \
WEIGHTS="0 0.01" \
N_UPDATES=20000 \
SAVE_ROOT=nca_wm/logs/vq_usage_ablation_scaling_14_pilot_s0 \
nca_wm/scripts/run_vq_usage_ablation.sh
```

Use the pilot only to confirm that the loss scale is sane and that nonzero
`--vq_usage_loss_weight` actually raises `vq_perp`; do not treat it as the
final result.
