# Mario 2-World WM: RecurrentNCA vs Transformer (mature pipeline, param-matched)

**Date:** 2026-06-28. **Goal:** decide which architecture to scale for the
info-gain (IG) active-learning experiment on the two Mario worlds
(`mario` vs `mario_breakable`, which differ in exactly the platform-break jump).

## Setup (identical for both)
- Data: mature `collect_unique_transitions` A* caches (realtime-tick-aware, slot 5),
  capped 60k transitions/world, `val_frac=0.1`. Shared merged dataset
  `rollout_data/_merged/recurrent_ds_389a88137e99bdec.pkl` (mario 54001tr/6000val,
  mario_breakable 54000/6000). NO active-learning data collection.
- k=8 (L=9 trajectory), batch 32, 50k updates, lr 3e-4 cosine, seed 0.
- Both **unconditional** (must infer world from history), ~5.1-5.4M params.

## Worlds differ in exactly 12 transitions (measured)
Of 60k transitions/world, only **6,030** `(state,action)` keys are shared; of those
exactly **12 are ambiguous** (same state+action -> different next-state), and **all
12 are action id 0 (UP)** = the platform-break jumps. Ambiguous fraction = **2.0e-4**.
=> Any global cell metric is dominated by the 99.98% shared, unambiguous dynamics;
the 12 breaks are invisible in it.

## Results
| arch | params | final change_err | q0NLL / loss |
|------|--------|------------------|--------------|
| RecurrentNCA (`nca_wm.train_recurrent`, n_hid160/k8/n_steps8, pool+cummax+global+skip) | 5,132,181 | ~0.0 (6.2e-5 blip @ step45k) | loss 4.7e-8 |
| Transformer (`mario_transformer_baseline`, AttnBeliefModel d208/d_model272/L5/d_cond168) | 5,363,736 | mario 0.0035 / breakable 0.0021 | q0NLL 0.104 / 0.071 |

Ckpts: NCA `nca_wm/logs/mario2_recurrent/params.pkl`; Transformer
`nca_wm/active_learning/ckpts/mario2_transformer/params.pkl` (rsynced back from 210).

## Interactive viewer
`mario_compare_serve.py` (port 8772): 3 panels engine | RecurrentNCA | Transformer,
real sprites, both fed the same true last-9-frame history, one-step PREDICT (default)
or DREAM (autoregressive) mode, realtime ▶ play, reset BASE/BREAKABLE, per-model
diff-vs-engine readout. JAX(CPU)+torch(CPU); fixed L=9 zero-left-pad input (compiles
once ~33s, then ~1.5s/step). Launch:
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python3 -u -m nca_wm.active_learning.mario_compare_serve --port 8772 --device cpu`

## Reading
- **RecurrentNCA is fully competitive — marginally cleaner on the global metric** at
  matched params on identical data. The NCA's 6.2e-5 change_err blip at step 45k is
  the fingerprint of an eval batch sampling one of the 12 ambiguous breaks (1 wrong
  cell / ~16k changed cells ~ 6e-5).
- **Neither number tests the breaks.** A global ~0.0 says the model mastered the
  shared dynamics; it says nothing about whether it resolves the 12 UP-break
  transitions (the IG-relevant ones). That requires the per-break probe = predictive
  entropy at those 12 UP-cells, NOT yet run.
- Verdict: NCA clears the bar -> justifies the next step of making the NCA IG-capable
  (add the haoo' q1 head; the note's IG needs NO latent) and/or stripping the
  transformer's latent mixture for a like-for-like IG comparison.
