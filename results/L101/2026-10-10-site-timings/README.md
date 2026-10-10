# L101 solved-task timings at sites 1, 10, 20 and 30

All 36 requested task outputs are complete: four sites x three initial states x three channels (zz, pm, mp), 1474 recurrence vectors per task. They share GS SHA-256 `af0ca9dd645a1884c3eee6351f2f92c83f25aed098bf62b19fdd3f28cddbb16b`. Read-only source snapshot: completed spectral segment UUID `4b4117de-7c29-4a5b-a1fe-78f438c854eb`. L101 remains terminated; this is analysis of preserved outputs, not a new simulation.

The final file field task_seconds measures only the final invocation, so it can understate work across restarts. Reconstruction traversed the resume-state ancestry to first segment UUID `f33f136d-12b5-4e9c-9477-f0331be45c16`, selected manifests for the 36 task IDs and parsed 114 task invocations. Final logged task timers match all 36 final JLD2 metadata records; identities, GS hash, target 1474 and resumed checkpoint bounds were checked.

For a single-task worker with a terminal worker timer, its active elapsed timer is used. Completed invocations otherwise use task_seconds. Interrupted invocations without a terminal timer use the last logged recurrence elapsed, a lower bound. This excludes queue time and startup outside the worker timer and includes repeated work after interruption/rollback. All selected tasks used block-sparse mode with two Julia threads; these historical timings are not BLAS8 predictions. Worker counts/node loads changed across segments.

| Site | Mean recorded active hours per task | Min–max across nine tasks (hours) | Lower-bound tasks |
|---:|---:|---:|---:|
| 1 | 1.693 | 1.418–2.132 | 0 / 9 |
| 10 | 13.045 | 7.794–17.822 | 0 / 9 |
| 20 | 19.570 | 11.785–27.477 | 4 / 9 |
| 30 | 21.718 | 13.050–31.012 | 4 / 9 |

The site 1/10 reconstructed totals have terminal timers for every invocation. Four tasks each at sites 20 and 30 have interrupted invocations without terminal timers; their reconstructed times, and consequently those site means/ranges, can understate the full active runtime. These are registered time lower bounds, not precise end-to-end completion times. Preserve that qualification when quoting the table.

| Site | Median zz (hours) | Median pm (hours) | Median mp (hours) |
|---:|---:|---:|---:|
| 1 | 2.041 | 1.601 | 1.459 |
| 10 | 16.235 | 13.958 | 9.888 |
| 20 | 26.623 | 21.397 | 16.148 |
| 30 | 29.546 | 23.844 | 19.456 |

`per_task_times.csv` preserves the state/channel breakdown and lower-bound flags. `cumulative_times.json` contains every contributing invocation and timer source. `final_task_timers.toml` is a scalar-only extraction from the final task files; no MPS vectors were loaded. The scripts reproduce the extraction/reconstruction. The original small final JLD2 copies remain at `/home/jovyan/work/l101_site_timings_20261010/tasks/` and the immutable source remains on Daint.

Some historical worker steps were interrupted and resumed from their last saved checkpoint, replaying up to several logged iterations. This is accounted for as actually consumed work. It does not demonstrate a defect in the currently deployed safe-stop path, which explicitly saves progress before its intentional stop.
