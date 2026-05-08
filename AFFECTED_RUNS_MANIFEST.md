# Affected Runs Manifest

Generated: 2026-05-08

No run directories were deleted. This manifest lists runs affected, or conservatively treated as affected, by the level-size padding bug.

## Criteria

- **Authored mixed-size all-level run**: trained with `level=null` and no `train_levels`, and saved `game_infos.pkl` shows at least one game with authored levels of more than one `(H,W)`. These runs had loss/metrics computed over per-game max-size padding, not the sampled level's true extent.
- **Synthetic multi-grid mixed-size run**: trained with `synthetic_multi_grid=true` and more than one generated grid size. These runs had loss/metrics computed over per-game max-size padding after merging generated sizes.
- **In-progress unknown authored run**: has `RUNNING.pid`, trains authored all-level data, but `game_infos.pkl` was unavailable at manifest time. Treat as affected until rebuilt under the fix.
- **Hidden-state leakage**: FiLM conditional or U-Net baseline runs would be affected because those model paths did not consistently mask hidden activations. No current run in this manifest matched that criterion.

## Summary

- Total listed: **96**
- Currently in progress: **6**
- Deleted: **0**

## Runs

| Run directory | Status | Reason |
|---|---:|---|
| `nca_wm/logs/multi_scaling_14_cond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_cond_match_s2` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_mask_v1` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_mask_v2_perstep` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s0_h288` | RUNNING | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s0_legacy_pool_in_conv` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s1` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s2` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_14_uncond_match_s3` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v2_cond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v2_cond_match_s1` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s0_legacy_pool_in_conv` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v2_uncond_match_s1` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v3` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v3_aborted_n97` | RUNNING | In-progress unknown authored all-level run |
| `nca_wm/logs/multi_scaling_gallery_v3_decoder` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v3_decoder_sprites_eos` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_cond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_cond_match_s1` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_decoder` | RUNNING | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_decoder_sprites_eos` | RUNNING | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s0` | complete | Authored mixed-size all-level run |
| `nca_wm/logs/multi_scaling_gallery_v4_uncond_match_s1` | RUNNING | Authored mixed-size all-level run |
| `nca_wm/logs/scaling_large_joint_v1` | RUNNING | In-progress unknown authored all-level run |
| `nca_wm/logs_canary/varislide_postfixA_d16_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d16_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d16_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d32_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d32_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d32_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d8_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d8_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixA_d8_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L16R1_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L16R1_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L16R1_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L1R16_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L1R16_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L1R16_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L2R8_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L2R8_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L2R8_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L4R4_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L4R4_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L4R4_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L8R2_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L8R2_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixB_L8R2_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d16_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d16_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d16_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d32_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d32_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d32_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d8_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d8_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixC_nopool_d8_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfixCp_nopool_noskip_d16_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_canary/varislide_postfix_mask_h128_d16_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d2_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d2_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d2_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d4_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d4_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d4_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d8_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d8_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_d8_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_uniform_T32_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_uniform_T32_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/nopool_uniform_T32_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/pool_d2_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/pool_d2_s1` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_depth_extrap/pool_d2_s2` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_nopool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_nopool_shared_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_evolve_n256_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_evolve_n64_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n256_30k_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n256_5sizes_30k_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n256_5sizes_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n256_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n512_5sizes_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_perstep_tpe_n512_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d16_pool_shared_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d32_nopool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d32_nopool_shared_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d32_pool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d32_pool_shared_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d8_nopool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d8_nopool_shared_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d8_pool_perstep_s0` | complete | Synthetic multi-grid mixed-size run |
| `nca_wm/logs_neko_arch/neko_d8_pool_shared_s0` | complete | Synthetic multi-grid mixed-size run |

## Notes

- Runs with a single explicit synthetic grid size, such as `synthetic_grid_sizes=8x7`, are not listed unless another criterion applies.
- Authored all-level runs without `game_infos.pkl` are listed only when currently in progress, because their exact level-size mix cannot be audited from saved metadata yet.
