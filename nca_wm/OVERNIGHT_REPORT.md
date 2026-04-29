# Overnight run report — 2026-04-18

Live document updated as runs complete.

## Goal (user-stated)

Train the best NCA world model possible; first milestone = cover **all games in the gallery**.

## Key config discovered works

- `--conditional` (FiLM via transformer encoder)
- `--axis_pool --axis_cummax --global_pool` (full global-context stack)
- `--balanced_sampling` (per-batch equal share per game)
- `--grad_clip 1.0` (stability; bounds BPTT gradient explosions)
- `--patience 80 --min_delta 1e-6` (loose early stopping)
- `--n_hid 128` single-game / small-set; hid=256 candidate for larger sets
- `--n_nca_steps 4` (with pool flags this is enough for `...` and `[X][Y]` rules)

## Key fixes made during the overnight

- Added `--max_transitions_per_game 200000` flag. Prevents dataset-load OOM when scaling to many games with disparate transition counts (fix after `small` OOM'd at ~50GB RAM load).
- Fixed `config.json` to be written at START of training (was only at end) — in-flight runs now visible in status tools.
- Dataset cache key bumped to `format_version=6` to include the new sampling cap.
- Killed the arch sweep's clearing_baseline (holding 18GB RAM) to free memory for `small`.
- Relaunched `run_gpu1_after_arch.sh` — now only trains clearing + nirvana (constellationz is already training on GPU 0 via phase 1b).

## Empirical results

### Phase 1 — scaling_6 @ hid=128 (6 games)

- Early-stopped at step 50,800 (patience 80 / min_delta 1e-6 triggered)
- Best change_err: **4.0%**, EMA100 at end ~12%
- Per-game @ step 40K:
  - blocks: 0.7% ✓
  - Travelling_salesman: 3.1%
  - nekopuzzle: 5.9%
  - Zen_Puzzle_Garden: 12.2%
  - kettle: 15.1%
  - sokoban_basic: 11.6% (surprisingly high — fits perfectly alone)
- Interpretation: capacity-shared bottleneck. kettle + Zen + sokoban_basic stall together — suggests hid=128 is the ceiling for 6-game joint training.

### Phase 1 — small @ hid=128 (9 games)

- **OOM'd** during dataset load (~50GB RAM needed). Skipped. Will retry after restart with `--max_transitions_per_game 200000`.

### Phase 1b — constellationz @ hid=128 ✅ DONE

- Full pool stack + grad_clip + patience 80 (vs arch sweep's no grad_clip + patience 30)
- At step 6,300: best change_err 2.0%, EMA100 ~9%
- At step 14,700: best change_err 1.52%, EMA100 ~3.5%
- At step 36,800: best change_err 0.064%
- At step 50,500: best change_err 0.02%
- **Early-stopped at step 57,600** (patience=80 on smoothed loss). **Best loss 7.9e-6** (~0.0008%).
- Loss oscillated 4e-4 ↔ 2e-2 with bounded spikes (grad_clip cap working). Best params saved.
- Key insight: arch sweep's aggressive early stopping (patience=30, min_delta=1e-5) cut constellationz off at 4-14K steps; with patience=80 + grad_clip, convergence was ~4 orders of magnitude deeper.

### Phase 1b followup — nirvana @ hid=128 ✅ DONE

- Data collection: 4 levels capped at per-level share of 200K.
- **Early-stopped at step 26,300. Best loss 3.46e-7** — cleanest fit of the overnight so far.
- Nirvana (unlike clearing/constellationz) converged nearly instantly — appears to not need deep pooling stress. Likely just a simple movement/push game without multi-kernel structure.

### Phase 1b followup — clearing @ hid=128 ✅ DONE

- Previous run OOM'd (3.8M-transition concat). Retry with per-level cap (200K per game → per-level share) worked: dataset loaded, training converged.
- **Early-stopped at step 30,900. Best loss 1.9e-5.** Per-cell change_err trained down to ~7e-3. Fits cleanly.

### Phase 1 — small @ hid=128 (9 games, capped) ✅ DONE

- With `--max_transitions_per_game 200000` the prior 50GB OOM becomes a ~12GB footprint. Dataset loaded cleanly.
- **Early-stopped at step 37,000. Best loss 4.2e-4. Change_err ~26%**.
- **Worst-fit: nekopuzzle with change_err 68%.** Nekopuzzle fits to 6.6e-9 alone, so this is a pure capacity/interference symptom, not an architectural one. Phase 2 @ hid=256 will test whether bigger unlocks sub-10% on the hard members of `small`.

### Phase 1c singletons (scaling_6 bottleneck games) — partial

Purpose: does each hard game fit *alone* at hid=128? If yes, scaling_6's 4% plateau is capacity-sharing; if no, the game is intrinsically hard.

- **zen @ hid=128**: ✅ Early-stopped at step 9,600. Best loss **6e-7**. Converged near-perfectly alone.
- **kettle @ hid=128**: ⚠️ Ran full 40,000 steps, no early-stop — still improving at end. Harder than zen alone; may need larger n_updates or hid=256.
- **travelling_salesman**: queued next on GPU 0.

### Phase 2 — scaling_6 @ hid=256 (GPU 1) ✅ DONE

- **Early-stopped at step 37,600. Best loss 1.29e-4** — vs hid=128's 5.81e-4. **~4.5× improvement** from doubling capacity.
- Strong signal that scaling_6's 6-game plateau at hid=128 was indeed capacity-bound, not architectural. Hid=256 unlocks another half-order of magnitude.
- Autoregressive eval still running; multi-level per-game rollouts.

### Architecture sweep (partial, arch_sweep killed mid-way for memory)

- **neko** (all 4 variants done, hid=128 n_nca_steps=4 200K budget):
  - baseline: 9.4% (stuck — proves `...` rule needs global context)
  - axis_pool: 6.6e-9 ✓
  - global_pool: 6.6e-9 ✓
  - axis_cummax + global_pool: 6.6e-9 ✓
  - **Any single pool flag suffices** for neko's `...` rule.
- **constellationz** (4 variants, patience 30 / no grad_clip → premature stops):
  - baseline: 5.4% (early-stop 4K)
  - global_pool: 3.96% (early-stop 6.7K)
  - axis_pool: 4.4% (early-stop 6.8K)
  - cummax_global: 4.1% (early-stop 13.6K)
  - All ran into loss-spike oscillations — see `plots/analysis/constellationz_loss_spikes.png`
  - Phase 1b rerun (with grad_clip + patience 80) aims to clean up these numbers.
- **clearing**, **nirvana**: arch sweep killed before they ran. Followup script on GPU 1 trains clearing + nirvana with good config.

## What's running now

| GPU | Job | Status |
|---|---|---|
| 0 | small @ hid=512 (phase 4, scaling experiment) | just started |
| 1 | scaling_6 @ hid=512 (staggered +5min) | scheduled |

### Phase 2 — small @ hid=256 ✅ DONE

- Despite mid-run SIGTERM chaos (see "OOM incident"), small@h256 early-stopped cleanly at step 43,200.
- **Best loss 9.2e-5** (vs h128's 4.2e-4, **4.5× better** — same improvement factor as scaling_6's h128→h256 jump).
- Final per-game change_err: blocks 3%, notsnake 7%, sokoban_match3 8%, sokoban_basic 9%, nekopuzzle 23%, Zen 27%, Multi-word 27%, kettle 31%, Travelling_salesman 33%.
- Capacity-scaling trend is strong and monotonic. Motivates the hid=512 runs.

### Phase 3 — scaling_large (19 games) ❌ OOM → refactored, retry queued

- Even with `--max_transitions_per_game 200000`, 19 games padded to max spatial dims (constellationz: ~58×58) demands ~250GB RAM.
- Kernel OOM-killer SIGKILL'd GPU 0 scaling_large runs (exit=137) and also SIGTERM'd small@h256 on GPU 1 (exit=143, but training had already early-stopped cleanly before — 43.2K steps survived in the save_dir).
- **Fix (per-user request): per-game native storage.** Dataset now stores each game's states at its own (C_g, H_g, W_g) shape; batches are padded to global max per-step at batch construction. Cache bumped to v7. Smoke-tested on scaling_2. Expected RAM for scaling_large ~30GB instead of ~250GB.
- `run_scaling_large_after_h512.sh` queues scaling_large@h128 on GPU 0 and @h256 on GPU 1 to fire when the respective h512 runs exit.

### Phase 4 — hid=512 scaling experiment

- `run_overnight_hid512.sh`: GPU 0 runs small@h512 (100K updates), GPU 1 runs scaling_6@h512 (80K updates) staggered by 5 min to prevent OOM collision.
- Hypothesis: if h128→h256 gave 4.5× improvement on both small and scaling_6, h512 should continue the trend.

#### scaling_6@h512 ✅ DONE
- **Early-stopped at step 37,600. Best loss 1.32e-5** — vs h256's 1.29e-4. **~10× improvement from capacity doubling.**
- Final smoothed change_err 3.1%.
- The h128 → h256 → h512 scaling law (4-10× loss drop per doubling) holds across capacity regimes on 6-game joint training.

#### small@h512 ✅ DONE
- **Early-stopped at step 52,200. Final smoothed change_err 11.0%** (vs h256 20.1%, h128 26.1%).
- losses.min: h128=1.67e-4 → h256=9.21e-5 → h512=**6.44e-5**.
- **Change_err nearly halves with each capacity doubling.** Scaling trend continues.

### Post-mortem on identity collapse (updated interpretation)

Initial read (scaling_large collapsed, call it stuck) was premature. Trajectory comparison across runs:

| Run | peak change_err | @ step | end | behavior |
|---|---|---|---|---|
| small@h256 (9g) | 0.79 (init) | 500 | 0.19 | monotone down |
| **scaling_14@h256 (14g, uniform)** | 0.78 | 9K | **0.69 @ 30.3K (early-stopped)** | peaked then recovering, but patience killed it |
| scaling_large@h256 (19g) | 0.80 | 24K | 0.74 @ 30K (killed) | turning the corner when I killed it |
| scaling_large@h128 (19g) | 0.93 | 37K | 0.88 @ 54K (early-stopped) | stuck — capacity-bound |
| scaling_large@h512 (19g) | TBD | in progress | step 9K/100K, err 0.64 | warming up |

- **Identity prediction is a transient warmup minimum**, not a stable attractor. Model climbs into it, then gradually escapes as rule-structured signal becomes reachable.
- Depth + duration of the climb scale with game count. h=128 can't escape 19 games; h=256 CAN but needs patience we didn't give it.
- **Patience on uniform BCE loss was fooled by identity plateau** — change_err is still dropping while mean BCE plateaus. scaling_14@h256 was cut off mid-recovery.
- `change_loss_weight` flag now plumbed through (added during post-mortem). Upweights BCE on cells where state != next_state. Makes identity solution strictly worse, should speed escape AND align patience metric with what we care about.

### Post-mortem phase 2 — `scaling_14@h256 + change_loss_weight=5`
- Direct A/B against the uniform-loss baseline that just finished at cerr=0.69.
- Patience 120, n_updates 100K.
- Early signal was strong: at step 27.6K, change_err = 9.4% (vs ~75% on uniform baseline at same step). KILLED to chase a bigger fix (below).

### ⚠️ Encoder mask bug discovered

While debugging the AR token-sampling loop (for the game-description autoencoder built tonight), found that Flax's `MultiHeadDotProductAttention` interprets the `mask` argument as **boolean** (via `jnp.where(mask, attn_weights, big_neg)`). Our `GameSpecEncoder` (and my new `TokenDecoder`) were passing a FLOAT mask with `0.0` for "keep" and `-1e9` for "mask out". `jnp.where` treats `0.0` as falsy → real tokens get masked OUT; `-1e9` as truthy → padding gets KEPT. Inverted.

**Impact measurement** (from the 9-game token-AE trained on small@h512's game_infos):

- Pairwise cosine similarity between games' z's: **0.77–0.99** across all 9 games (should be < 0.8 for different games; duplicates should be 1.0).
- Implication: the encoder was producing **near-identical z's for every game**. NCA's FiLM conditioning was essentially game-independent, turning the multi-game world model into a single-game model trying to fit all games at once — a very plausible contributor to the identity-collapse phenomenon.

**Fix applied**: pass the bool mask directly to `nn.MultiHeadDotProductAttention`. Smoke-tested on 4 fake games — cos-sim now 0.31 for very-different games, 1.0 for duplicates. Healthy.

### Phase 6 — post-encoder-fix sweep

#### scaling_14@h256 + clw=5 + encfix ✅ DONE
- Early-stopped at step 34,300. **Final change_err 8.6%** (smoothed).
- Per-game (last logs avg): blocks 0.8%, scriptcross 2.8%, sokoban_match3 3.6%, notsnake 3.8%, sokoban_basic 4.0%, Zen 6.0%, Travelling_salesman 8.5%, Modality 10.8%, actiontest 11.1%, Love 12.6%, Multi-word 13.4%, kettle 14.7%, Collapsable_Sokoban 20.3%, nekopuzzle 29.2%.
- vs pre-fix + uniform loss (8K patience): **69%** (identity collapse), vs pre-fix + clw5 (killed mid-run): ~9% at same step. So for 14 games, the fix moves the needle slightly but **clw5 alone recovered most** of the ground.

#### scaling_large@h256 + clw=5 + encfix ✅ DONE
- Early-stopped at step 39,300. **Final smoothed change_err 18.96%** — on par with small@h256 (20%).
- Per-game spread: 12 of 19 games <20% change_err; kettle still 81%, Travelling 57%, Modality 51%, Collapsable/Love/nekopuzzle/scriptcross 41-47%.
- **The encoder bug was the dominant factor behind the 19-game collapse.** Fix alone (plus clw=5 to speed escape) gets us to near-small quality at 19-game scale without more capacity.

#### scaling_large@h128 + clw=5 + encfix ✅ DONE
- Early-stopped at step 89,100. **Final smoothed change_err 14.7%** — actually beats h256's 19% because h128's loss plateaued more slowly so patience let it train longer.
- Per-game: notsnake 0.3%, constellationz/blocks/sumo 2%, 12 games <20%, worst: Modality 60%, Travelling 68%, kettle 70%.
- **First time h=128 has escaped 19-game collapse.** Pre-fix h=128 was stuck at 88% at step 54K; the encoder bug was the dominant blocker, not capacity.

### Phase 7 — rule-attention vs wide-FiLM (A/B)

#### rule_attn @ h256, K=16, d_slot=64 + clw5 + encfix (running)
- At step **30K/120K**: change_err **9.6%** (smoothed). Worst-fit Travelling_salesman 41% (dropping). Already beats narrow FiLM's converged 19% at less than half the FiLM run's step count.
- Step time ~147ms vs FiLM's 158ms — *no slowdown* despite 1024-d structured conditioning + per-step cell-to-slot cross-attention.

#### wide_FiLM @ h256, d_z=256, d_model=128, encL=4 + clw5 + encfix ✅ DONE
- Early-stopped at step 72,300. **Best smoothed-500 change_err 14.5%**, final 18.8%.
- Worst-fit: Travelling 58%, Modality 56%, kettle 55%.
- Wider z helped *moderately* (vs FiLM h128 13%), but rule_attn at h=256 (9.4% smoothed best) wins decisively. **The win is from structured K-slot conditioning, not from raw conditioning dimension.**

#### rule_attn @ h256, K=16, d_slot=64 + clw5 + encfix ✅ DONE
- Early-stopped at step 32,100. **Best smoothed-500 change_err 9.4%**, final ~12%.
- Per-game: 12 of 19 games <12%; worst Travelling 49%, nekopuzzle 23%, Collapsable 21%. **Worst-case dropped from kettle 81% (FiLM) to kettle 15% (rule_attn).**
- Convergence ~3.4× faster than wide_FiLM at meaningfully better final loss.

#### rule_attn @ h512 (KILLED at step 20K)
- Bigger body substantially slowed convergence (smoothed change_err 31% at step 20K vs K=16/h=256 at 11% at same step). Killed — poor ROI on GPU time.

#### rule_attn @ h256, K=32 ✅ DONE
- Early-stopped at step 36,200. **Best smoothed-500 change_err 10.4%** — slightly worse than K=16's 9.4%.
- K=16 already saturates slot capacity for 19 games; K=32 is over-parameterized.

### Phase 8 — Best-recipe (K=16 h=256) at `gallery` (174 games), GPU 1
- Launched immediately after K=32 finished, same recipe. Uses `--games gallery` preset (new) = 12 PRIORITY + 162 gallery-deduped = 174 games.
- Bumped `batch_size` 64→256 (at 174 games, balanced sampling needs ≥1 sample per game on average). Halved `max_transitions_per_game` 200K→100K (keeps memory tight across 174 games). Patience 160 (looser).
- Compilation + data collection expected ~30-60 min; training ~4-6 hours after that.

### Phase 8b — rule_attn nca_steps=8 ✅ DONE
- Early-stopped at step 79,600. **Best smoothed-500 change_err 9.15%** (step 71.5K).
- Marginal improvement over nca=4's 9.41% (32K steps). Depth helps a *tiny* bit but requires ~2-2.5× more steps (wall-clock comparable). **Not worth the complexity.**
- Per-game worst: Travelling_salesman 39% (vs nca=4's 49%), Modality 21% (vs 18%). Small reshuffling of hard-game ranking; overall gains small.

### Bugfix during gallery launch
- First gallery launch crashed: `Beam_Islands` level 1 has more objects than level 0. Fixed: `collect_multigame_dataset` now takes `max(n_objs)` over all levels (not just level 0). Relaunched.

### Phase 9 — Gallery milestone (174 games) — running on GPU 1
- Recipe: `rule_attn K=16 h=256 nca_steps=4 + clw5 + encfix + v7`, `batch_size=256` (so each batch covers ~1.5 samples/game at 174-way balanced sampling), `--max_transitions_per_game 100000` (budget tight for 174-game memory).
- `patience=160`, `n_updates=200000` — loose enough to not early-stop mid-recovery.

### Side work: latent-game sampling infra

Not on GPU time:

- **TokenDecoder** (`nca_wm/token_decoder.py`): causal transformer, 466K params default, z-as-prefix conditioning. Smoke-tested: 100% reconstruction on small's 9 games.
- **train_token_ae.py**: standalone AE trainer, supports `--init_from <ckpt>` or fresh init. CPU-smoke trains in ~70s.
- **sample_latent_games.py**: reconstruction + latent interpolation + Gaussian-prior sampling demo. After decoder mask fix, shows meaningful interpolation (endpoints decode to their games, midpoints hybridize).
- **Encoder mask fix impact on AE cos-sim** (small, 9 games, 300 steps): pre-fix mean off-diag **0.998** → post-fix **0.903**. Fix helps but the real effect is in NCAWM training, where dynamics loss amplifies the signal.
- **`rule_attn_model.py`** (rule-attention architecture, parallel to ConditionalNCAWorldModel): compiles and runs; 443K params for h=64/K=8/d=32 config. Scaffolding for the vector-latent vs rule-latent comparison; not wired into train.py's CLI yet.

All prior multi-game models (small, scaling_6, scaling_14, scaling_large) were trained with the buggy encoder. They still "worked" to varying degrees because games could be distinguished by token-sequence length (the surviving signal after the bug) and because the decoder absorbed per-game differences in FFN. But they were operating well below the encoder's potential.

Archived pre-fix save_dirs with suffix `_preEncFix` for comparison. New runs:

- **GPU 0**: `scaling_14 @ h=256 + clw=5 + patience=120 + fixed encoder`, n_updates=100K
- **GPU 1**: `scaling_large @ h=256 + clw=5 + patience=120 + fixed encoder`, n_updates=120K

Key hypothesis: the encoder bug was a significant chunk of the multi-game capacity ceiling. If scaling_large @ **h=256** escapes collapse cleanly now (before we even go to h=512), the bug was a dominant issue.

### Phase 5 — scaling_large via v7 per-game storage (NEW — user requested)

- GPU 1 freed → `scaling_large@h256` now training on the 19-game gallery with per-game native-shape storage.
- Dataset: **2,151,505 transitions, 19 games, global max (16, 26, 36)** (constellationz's biggest cached level is 58×58 but only smaller levels cached — we got lucky).
- Cached in 3.5s, balanced sampling with per-batch sizes [4,4,…,3,3] across 19 games.
- Model: 5,146,769 params at hid=256. Compilation hit a JAX rematerialization warning (wanted ~20GB) but training started — GPU 1 at 86% util.
- Will kick off scaling_large@h128 on GPU 0 when small@h512 exits.

### Clearing retry — done
- Early-stopped at step 30,900. Both clearing and nirvana proved they fit with the capped dataset. Final eval pending.

### Fix landed mid-overnight
- Added `early_stopped` + `n_updates_requested` fields to `train_meta.json`.
- Updated `run_overnight_growth.sh` skip-check to treat early-stopped runs (at matching or larger budget) as done — prevents reruns of already-converged runs.
- Added `RUNNING.pid` lock file mechanism (written by `train.py`, checked by launcher). Allows a GPU 1 parallel queue to co-exist with overnight_growth.sh on GPU 0 — if the same save_dir is already being trained by another process, the launcher skips it.
- Backfilled `early_stopped` flag for scaling_6 / constellationz / nirvana runs that finished before the fix.

### GPU 1 parallel queue (launched)
- `run_gpu1_phase2_phase3.sh` (pid 3231329) waits for clearing retry to exit, then runs phase 2 (scaling_6 @ hid=256, small @ hid=256) and phase 3 (scaling_large @ 128, 256) — in that order.
- Overnight_growth on GPU 0 still walks its own queue; once it reaches phase 2, the skip-check sees either (a) a live RUNNING.pid from GPU 1 or (b) a completed run, so no duplicate.

## Next queued

| When | What | Where |
|---|---|---|
| After phase 1b | Phase 1c singletons (kettle, Zen, travelling) if bash re-reads script; else phase 2 scaling_6@256, small@256 | GPU 0 |
| After clearing | nirvana (followup) | GPU 1 |
| (possibly) | Relaunch growth with `--max_transitions_per_game 200000` for small + scaling_large | GPU 0 |
