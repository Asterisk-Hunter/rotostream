# RotoStream engineering assessment

Updated 2026-10-03. The earlier review remains in the
[evaluation plan](DAVIS-EVALUATION-PLAN.md).

## What the evidence supports

The complete workflow covers upload, prompts, tracking, saved sessions, inspection
and six exports. Bounded compute, immutable artifacts, restart recovery, private
media and an authenticated gateway address concrete service failures. Lint, types,
build, 13 frontend tests, 258 Python tests and a 73-check native HTTP smoke passed
during this update. One Python case is an expected failure. Latest CI is the
container release gate.

This is an independent SAM 2 memory reproduction over reused pretrained Hiera and
released SAM 2.1 weights. Parity probes cover raw decoder outputs, presence,
pointer selection, memory writes and conditioned features. They establish
numerical module compatibility, not identical end-to-end reference policy.

DAVIS 2017 validation measures 89.40 J&F over 30 clips and 61 objects using
first-frame ground-truth masks, interior-frame scoring and object averaging.
Objects are independent without joint label arbitration. This is a reproducible
measurement, not an official leaderboard entry or one-click result. Released
weights can be evaluated without new training. [Raw evidence](RESULTS.md).

## Limits

- Neural propagation measured 2.88 FPS on an RTX 4050 Laptop GPU; real-time neural
  tracking is not demonstrated.
- One object per session; joint multi-object overlaps and identity conflicts are unresolved.
- One editor and API worker; no per-user ownership, quotas or tenant isolation.
- Jobs are in memory. Restart marks interrupted work failed rather than resuming computation.
- Cancellation is cooperative; ffmpeg may finish or reach its deadline first.
- Gradient wiring is verified; real-data convergence and trained-weight generalization are unproven.
- Synthetic ablations are component regressions, not real-video explanations.
- Full reference-tracker policy comparison remains useful beyond module parity.

## Useful next work

Review the weakest DAVIS tracks from footage before inferring failure causes.
Joint-object work should define overlap arbitration and score joint label maps
with the official evaluator. Shared deployment needs durable job ownership and
explicit per-user storage authorization.

The [case study](CASE_STUDY.md) gives resume wording within measured claims.
The [deployment guide](DEPLOYMENT.md) defines the supported operating envelope.
