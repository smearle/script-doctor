# use_layernorm ablation — results

(`val change_err` = held-out-transition 1-step changed-cell error; `conv@` = first val step with val change_err <= 0.0001; rollout = mean over levels of AR cell-error.)

| group | depth | LN | best val cerr | final train cerr | conv@ | best_loss | bfs | astar | rand-AR |
|---|---|---|---|---|---|---|---|---|---|
| TSM pool-ON+skip | 8 | off | 0.000% | 0.000% | 3000 | 1.43e-07 | 0.000% | 0.000% | 0.001% |
| TSM pool-ON+skip | 8 | **on** | 0.000% | 0.000% | 8500 | 9.15e-08 | 0.034% | 0.000% | 0.040% |
| TSM pool-ON+skip | 32 | off | 0.000% | 0.000% | 3500 | 1.01e-07 | 0.000% | 0.000% | 0.000% |
| TSM pool-ON+skip | 32 | **on** | 0.000% | 0.000% | 4000 | 6.39e-08 | 0.011% | 0.000% | 0.065% |
| THL pool-ON+skip | 8 | off | -0.000% | 0.000% | 19500 | 3.48e-07 | 0.435% | 0.552% | 0.134% |
| THL pool-ON+skip | 8 | **on** | 0.000% | 0.000% | 14500 | 2.88e-07 | 4.882% | 5.534% | 2.970% |
| THL pool-ON+skip | 32 | off | -0.000% | 0.000% | 17000 | 7.30e-07 | 2.995% | 3.002% | 0.920% |
| THL pool-ON+skip | 32 | **on** | -0.000% | 0.000% | 22000 | 3.53e-07 | 2.908% | 4.024% | 5.269% |
| TSM pool-OFF+skip (d32) | 32 | off | 0.000% | 0.000% | 3500 | 6.08e-06 | 1.066% | 0.000% | 1.397% |
| TSM pool-OFF+skip (d32) | 32 | **on** | 0.000% | 0.000% | 2000 | 1.62e-05 | 0.000% | 0.000% | 0.000% |
| TSM pool-OFF,no-skip (d32) | 32 | off | 0.000% | 0.000% | 3500 | 3.23e-06 | 0.000% | 0.000% | 1.647% |
| TSM pool-OFF,no-skip (d32) | 32 | **on** | 0.000% | 0.000% | 3500 | 1.44e-06 | 0.000% | 0.000% | 0.727% |
