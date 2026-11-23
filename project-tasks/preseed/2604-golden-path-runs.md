# DOCBOT-2604 golden-path run log

Script: `python3 scripts/golden_path.py --base-url https://docbot-production-c4c0.up.railway.app --runs 10`
Steps: demo init (quickbite) -> hybrid tie-out question -> planted discrepancy flagged with doc and DB citations, no reasoning-tag leak -> tie-out XLSX export -> autopilot question.
A run passes only if every step passes. Latency is wall-clock seconds from the client.

## Set A: 2026-11-23 14:51-15:00 UTC, prod at dff2211 then c4222f7

Invalid as a clean set: PR #146 merged and Railway redeployed at 14:53:27 UTC, during runs 2-4.

| Run | demo_init | tieout_chat | xlsx_export | autopilot | Result | Note |
|---|---|---|---|---|---|---|
| 1 | 6.26 | 28.58 | 0.49 | 43.79 | PASS | |
| 2 | 2.39 | 21.16 | 0.33 FAIL | - | FAIL | 404 on export, redeploy window |
| 3 | 3.52 | 13.79 FAIL | - | - | FAIL | stream cut by redeploy |
| 4 | 16.46 FAIL | - | - | - | FAIL | 502 during container swap |
| 5 | 3.62 | 25.63 | 0.53 | 25.08 | PASS | |
| 6 | 2.96 | 23.33 | 0.32 | 44.22 | PASS | |
| 7 | 1.80 | 21.71 | 0.33 | 37.60 | PASS | |
| 8 | 1.78 | 20.99 | 0.37 | 47.67 | PASS | |
| 9 | 1.92 | 18.44 | 0.32 | 26.42 | PASS | |
| 10 | 1.67 | 23.78 | 0.31 | 53.88 | PASS | |

## Set B: 2026-11-23 15:04-15:14 UTC, prod at c4222f7 (no deploy during the set)

| Run | demo_init | tieout_chat | xlsx_export | autopilot | Result | Note |
|---|---|---|---|---|---|---|
| 1 | 17.10 | 20.89 | 0.53 | 35.61 | PASS | cold demo init |
| 2 | 2.12 | 25.60 | 0.32 FAIL | - | FAIL | export 404, lineage write race |
| 3 | 4.92 | 26.74 | 0.34 | 43.47 | PASS | |
| 4 | 2.34 | 30.03 | 0.34 | 51.33 | PASS | |
| 5 | 2.67 | 26.52 | 0.36 FAIL | - | FAIL | export 404, lineage write race |
| 6 | 3.17 | 26.88 | 0.35 | 42.08 | PASS | |
| 7 | 2.66 | 25.14 | 0.36 | 50.21 | PASS | |
| 8 | 2.05 | 27.27 | 0.35 | 56.57 | PASS | |
| 9 | 2.18 | 24.92 | 0.32 FAIL | - | FAIL | export 404, lineage write race |
| 10 | 2.83 | 26.90 | 0.33 | 47.64 | PASS | |

7/10. p50 / p95 seconds (passing steps only, nearest-rank p95):

| Step | n | p50 | p95 |
|---|---|---|---|
| demo_init | 10 | 2.66 | 17.10 |
| tieout_chat | 10 | 26.63 | 30.03 |
| xlsx_export | 7 | 0.35 | 0.53 |
| autopilot | 7 | 47.64 | 56.57 |

Every tie-out answer flagged Restaurant Advertising, doc $890M vs DB $885M, with both citations. No reasoning-tag leaks.

## Failures and fixes (in PR #131, not yet on prod)

1. Export 404 (3/10 in set B). The lineage row is written by a background task after the SSE lineage event is sent, so an export requested within about 0.3s can arrive before the insert commits. Confirmed on prod: same session got `404 Lineage not found` immediately and 200 two seconds later. Fix: the tie-out route waits for in-flight lineage writes (`wait_pending_writes`). Test: `tests/unit/test_tieout_export.py::test_wait_pending_writes_lets_inflight_lineage_land`. A person clicking Export is slower than the script, so the demo rarely hits this, but the script does.
2. Autopilot tool router and verifier fail on Bedrock (every autopilot run). ChatGroq always sends `service_tier=on_demand` and Bedrock rejects it with a 400, so `call_llm` callers fell back to the heuristic tool picker and skipped the contradiction verifier. The answer still arrives, so the run passes, but with less checking. Fix: Bedrock LangChain clients drop `service_tier`. Tests: `tests/unit/test_bedrock_primary.py`.

Per the AC, the 10-run set is repeated from zero after PR #131 is merged and deployed.
