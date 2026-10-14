# Stage2 checkpoint training-window screening — 2026-09-18

This report screens saved models using their preceding training windows. It does **not** claim that a checkpoint is best on hardware or a fixed evaluation set. The main selection work will evaluate candidates on a fixed H2-only motion set.

## Candidates

Prefer the original run **70,000** and **73,150**, with **30,000** as an earlier reference. The last 1,000-iteration window of 70,000 has the smallest root error across all saved checkpoints; 73,150 has the smallest local-foot error and highest reward. Their root difference is only 0.0197 cm and reverses when using 500 iterations, so it should not decide the result. The 73,000 checkpoint is almost equivalent and has the lowest physical-joint error in a 1,000-iteration window.

The new sonic_transfer run ends at 6,000 and remains materially worse in tracking than the mature original run. It is not a preferred deployment candidate based on these logs.

## Last 1,000 iterations, ending at the saved checkpoint

All positional errors below are converted from meters to centimeters. Joint error is radians. Action-rate values are the logged L2 penalty statistic, not degrees per second.

| Run / checkpoint | Root cm | Global feet cm | Local feet cm | Local hands cm | Local body cm | Joint rad | Reward | Action rate L2 | Ankle saturation % | Motor saturation % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original / 30,000 | 31.5081 | 31.9797 | 6.8582 | 6.3666 | 5.0752 | 0.119494 | 70.0920 | 0.088474 | 0.5545 | 0.0836 |
| original / 70,000 | 26.8249 | 27.2582 | 6.4941 | 6.1000 | 4.8710 | 0.113652 | 71.5492 | 0.090547 | 0.9965 | 0.1402 |
| original / 73,000 | 26.8432 | 27.2746 | 6.4877 | 6.0938 | 4.8640 | 0.113621 | 71.5772 | 0.091658 | 1.0982 | 0.1534 |
| original / 73,150 | 26.8446 | 27.2764 | 6.4874 | 6.0926 | 4.8654 | 0.113757 | 71.6389 | 0.091671 | 1.0803 | 0.1512 |
| transfer / 6,000 | 40.7111 | 41.0450 | 7.4273 | 6.5947 | 5.5476 | 0.135313 | 66.8632 | 0.089482 | 0.4842 | 0.0733 |

## Stability and failure-related evidence

The mature models improve positional tracking while policy action-rate L2 and actuator saturation rise modestly. This is a tradeoff to check in the fixed-motion evaluation, not evidence of hardware instability by itself. Logged action clipping is zero in the displayed windows.

`Env/Episode_Termination/*` holds, for each environment, flags from its latest completed episode, then averages across environments. Terms may overlap. Its timeout statistic is **not** a formal full-clip success rate; starts are randomized, and a timeout can coincide with another termination. `Env/adp_samp/failure_rate_mean` averages a historical ratio over adaptive bins and is not an event-counted recent failure rate. Raw values are retained in the CSV/JSON without relabeling them as success/failure rates.

| Checkpoint | Root cm, preceding 500 | Root cm, preceding 1,000 | Last-episode timeout flag mean | Episode mean seconds |
|---|---:|---:|---:|---:|
| original_stage2 / 30,000 | 31.5051 | 31.5081 | 0.964135 | 5.3475 |
| original_stage2 / 70,000 | 27.1103 | 26.8249 | 0.967141 | 5.3128 |
| original_stage2 / 73,000 | 27.0237 | 26.8432 | 0.967507 | 5.3123 |
| original_stage2 / 73,150 | 27.0203 | 26.8446 | 0.967445 | 5.3161 |
| transfer_stage2 / 6,000 | 40.4298 | 40.7111 | 0.951771 | 5.4090 |

The best 500-iteration root window belongs to 62,000 (26.6297 cm), whereas the best 1,000-iteration window belongs to 70,000 (26.8249 cm). This reinforces that small root differences between mature training windows do not establish a best fixed policy.

## Scope and extraction integrity

- Original local store: `/home/qingyul/workspace/GR00T-WBC-H2TRAINBASE/swanlog/run-20260907_011159-xrkq3m5g/run-xrkq3m5g.swanlab`.
- Snapshot size 4,945,658,669 bytes; maximum logged step 73,165. The saved paused checkpoint is step **73,150**; records after 73,150 are excluded from its windows.
- CRC-checked extraction recovered isolated malformed fragments: 2,463 recovery events, 151,716 skipped bytes, 46 orphan fragments, 2 invalid records. It retained 3,804,496 scalar records covering 52 metrics. These recovery counts are disclosure, not an assumption that every metric/step is present.
- All values are deduplicated by `(metric, step)`; the 19 ranking metrics have **zero duplicate/conflicting records**. Missing steps and exact `n` for each metric/window are in `metric_window_details.csv`; no interpolation or filling is used.
- Raw step errors come from rank 0's training environments and include H2 robot encoding, G1 teacher encoding, and SMPL encoding in the configured mixture. The robot body is H2 throughout. They do not separately measure H2-encoder-only deployment behavior.
- ADP changes the sampled motion distribution as training progresses. The policies also update within each historical window. Consequently these are screening statistics, not a controlled comparison of saved policies.
- Windows are stage2 iterations, with each iteration corresponding to 24 control steps at 50 Hz. Adding stage1 gives cumulative iteration = stage2 iteration + 2,000.

## Artifacts

- `checkpoint_windows.csv` / `.json`: all saved checkpoints, 500- and 1,000-iteration windows, means/counts and per-metric ranks.
- `metric_window_details.csv`: long-form metric keys and exact missing-step disclosure.
- `selected_training_windows.json`: selected candidates plus 30k and new 6k.
- `scalar_coverage.json`: coverage, duplication and conflict audit.
- `old_stage2_selected_scalars.json.gz`: extracted original run scalars.
- `old_stage2_extraction_metadata.json`: extraction provenance/recovery counts.
- `rank_training_windows.py`: exact CPU-only reproduction of window aggregation and per-metric ranks.
