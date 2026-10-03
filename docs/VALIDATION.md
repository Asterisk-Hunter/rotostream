# Release validation

Validated 2026-10-03. This is a record of executed checks, not a claim that every
possible input or deployment has been tested.

| Check | Result |
| --- | --- |
| Windows API/ML suite | 258 passed, 1 expected failure; 390 seconds |
| Ubuntu API/ML CI suite | 260 passed, 1 expected failure, including memory-storage regressions |
| Web lint, TypeScript, production build | Passed locally and in CI |
| Production JavaScript dependencies | Audited; Next.js upgraded to 16.3.8 for GHSA-vcvr-r3jv-pc5j |
| API container Python dependencies | Resolved runtime requirements audited; no known vulnerabilities reported |
| Frontend behavior suite | 13 passed |
| Native real HTTP smoke | 73 checks passed |
| Authenticated container smoke | 73 checks passed |
| Gateway access boundary | Unauthenticated UI, API, media and SSE returned 401 |
| Production image build and readiness | API, web and gateway started successfully in Linux CI |
| Neural causality | 30 probes, five scenarios, both directions, zero changed pixels |
| Serial Windows tracker contract after memory fix | 16 passed; 427 seconds |
| Module parity | Every probe within 1e-5; zero missing/unexpected released weight keys |
| Synthetic ablations | Eight completed variants, five clips per variant; bfloat16 CUDA |
| DAVIS coverage | 30 validation clips, 61 independently tracked objects, 3,862 scored object-frames |

The completed Linux run for implementation commit `35d19f1` is
[CI run 37122556817](https://github.com/Asterisk-Hunter/rotostream/actions/runs/37122556817).
The latest branch CI should also be green before deployment. Subsequent prompt
error wording and causality CLI precision changes passed their focused API,
baseline and leakcheck tests; the neural CLI was executed with the recorded flags.
The final memory-storage regression compares 100-frame forward and backward sweeps
against an unpruned bank, including current-frame retries, while enforcing the
retained-history bound. Prompted anchors remain intact.

The dated dependency audit outputs are retained as
[API runtime JSON](../reports/security/api-runtime.json) and
[web runtime JSON](../reports/security/web-runtime.json). They describe known
advisories at validation time, not a guarantee against undiscovered vulnerabilities.
CI repeats both audits. The Next.js patch addresses
[GHSA-vcvr-r3jv-pc5j](https://github.com/advisories/GHSA-vcvr-r3jv-pc5j).

## Browser workflow

A production Next.js build and native API served the deterministic included clip.
Keyboard foreground prompting, 60-frame tracking, confidence inspection and
transparent WebM export completed successfully. A background-only correction
exercised the baseline validation error; the existing completed session and export
remained available. That error now instructs the editor to mark foreground first.
No browser errors were reported. At a 375px viewport, document width remained
within the viewport. The README screenshot is from this running studio.

## Evidence and reproduction

[Results](RESULTS.md) links the raw benchmark JSON/CSV, parity probes, gradient
dry run, ablations and causality report with their precision/protocol caveats.
`python scripts/render_results.py` regenerates results and the README score table.
`pnpm verify` runs the checks; `pnpm smoke` needs a running API. CI additionally
builds the production stack and exercises it through gateway authentication.

Local Windows Docker image-list and pull commands hung after Desktop startup.
Container execution is therefore established by Linux CI, not a local Docker pass.
The shipped image supports the CPU baseline; native GPU inference was measured
separately. No hosted production deployment, real-data fine-tuning or real-time
neural throughput is claimed.
