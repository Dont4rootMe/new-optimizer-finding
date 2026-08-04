# Live cluster progress snapshots

These are explicitly partial, read-only snapshots of the running canonical
EvolutionLoop job. They are derived from the existing Cloud.ru scheduler log;
creating them does not submit or mutate a cluster job.

`circle_packing_progress_g57_20260804T1059Z.json` contains the source points for
`circle_packing_progress_g57_20260804T1059Z.png`. Cloud.ru log retention had
already dropped exact per-organism score lines for generations 0–28, so the
plot shows the rounded generation-0 seed anchor, leaves generations 1–28 as an
explicit gap, and plots retained finite `simple_score` observations from
generation 29 onward. It must not be cited as the final generation-300 result.
