# DOCBOT-2812 Sandbox egress off (SANDBOX_PIP_INTERNET=false + custom E2B template)
| Epic | Phase | Sprint | Priority | Points | Lane | Due | Status |
|---|---|---|---|---|---|---|---|
| EPIC-28 | 1 | S1 | P0 | 3 | senior-dev | - | New (scheduled from parking lot) |

## User story
As a fractional CFO uploading a client's general ledger, I want the code DocBot runs on my data to have no internet access at all, so that a malicious cell or document text cannot make the sandbox send my client's numbers anywhere.

## Why now
Design-partner ledgers start arriving in S1 through the CSV path. Today `api/utils/sandbox_policy.py` grants internet to internal LLM-codegen runs that import a non-prebaked package (`SANDBOX_PIP_INTERNET` defaults on, line 64-66), so an injected `import statsmodels; requests.post(...)` can exfiltrate. Prod flip happens on or after 2026-12-02 (prod freeze 11-27 to 12-01).

## Scope
In:
- A custom E2B template that bakes `statsmodels`, `xgboost`, `lightgbm`, `tabulate` (pinned versions) on top of the code-interpreter base.
- `E2B_SANDBOX_TEMPLATE=<template id>` and `SANDBOX_PIP_INTERNET=false` in dev and prod (Railway now, AWS env when it exists).
- Policy fix so template packages count as prebaked.
- Fail-closed check that the SDK accepts `allow_internet_access`.
- An external regression test that proves egress is blocked and statsmodels still works.
Out:
- Full exfiltration suite (env secrets, typosquat, resource abuse) (3511).
- Self-hosted sandbox (3203); egress allowlist proxy for API tasks (2808).

## Acceptance criteria
- [ ] Given the template built from `e2b/template/e2b.Dockerfile`, when a sandbox is created from it, then `import statsmodels, xgboost, lightgbm, tabulate` succeeds with no pip install.
- [ ] Given `SANDBOX_PIP_INTERNET=false` and the template set, when generated code imports statsmodels, then `sandbox_policy.internet_allowed()` returns False, no pip install runs, and a SARIMAX fit on a 36-point monthly series prints a forecast.
- [ ] Given the same config, when generated code calls `requests.post` (and `urllib`, raw `socket`) to a reachable public canary host, then the call fails with a network error; in a control run with `allow_internet=True` the same request succeeds, so the negative result is not caused by DNS or a dead host.
- [ ] Given an E2B SDK whose `Sandbox.create` does not accept `allow_internet_access`, when the app boots or a sandbox is created, then it raises instead of logging "lockdown degraded" and continuing (`sandbox_policy.create_kwargs`, lines 120-127).
- [ ] Given Railway dev and prod, when env vars are inspected on or after 2026-12-02, then both have `E2B_SANDBOX_TEMPLATE` and `SANDBOX_PIP_INTERNET=false`, and a CSV forecast question on prod returns a chart.

## Deliverable and approach
- Template: `e2b/template/e2b.Dockerfile` (`FROM` the code-interpreter base image at a pinned tag, then `pip install` pinned versions) and `e2b/template/e2b.toml`; built with the E2B CLI (`e2b template build`) by the founder (E2B account login). Record the template id in the runbook, not in code.
- `api/utils/sandbox_policy.py`: `PREBAKED_PACKAGES` (line 54) is static, so with a template set and `SANDBOX_PIP_INTERNET` left on, statsmodels code still gets egress. When `E2B_SANDBOX_TEMPLATE` is set, treat `TEMPLATE_PACKAGES = {"statsmodels", "xgboost", "lightgbm", "tabulate"}` as prebaked in `packages_to_install` (line 69). Make `create_kwargs` raise if `allow_internet_access` is dropped.
- No change needed for the kill switch: `pip_egress_enabled()` (line 64) and the template kwarg (line 117) already exist and are unit-tested (`tests/unit/test_sandbox_lockdown.py:336, 343`).
- `api/sandbox_service.py`: no logic change; `_detect_needed_packages` (line 192) keeps mapping imports, which now resolve to prebaked.
- Env: `E2B_SANDBOX_TEMPLATE`, `SANDBOX_PIP_INTERNET` added to the `CLAUDE.md` env table.
- Branch `feature/DOCBOT-2812-sandbox-egress-off`.

## Test / verification
- Unit (CI): extend `tests/unit/test_sandbox_lockdown.py` with (a) template set + pip egress on + statsmodels: no internet, no install; (b) `allow_internet_access` unsupported: raises.
- External: new `tests/external/test_sandbox_egress.py` marked `@pytest.mark.external` (needs `E2B_API_KEY`, template id): egress negative + positive control + statsmodels positive. Use a real reachable host (for example `https://example.com`), never a `.invalid` domain, which fails DNS regardless and would pass vacuously.
- `pytest tests/unit/test_sandbox_lockdown.py -v` and `pytest tests/external/test_sandbox_egress.py -m external -v`. Cost: 3 short sandbox runs, a few cents of E2B time, no LLM calls.

## Dependencies
Blocked by: none.  Blocks: 3511, 3201, 3203; first DP ledger upload via 2577.

## Definition of done
- [ ] AC met  - [ ] tests pass in CI  - [ ] Plane issue updated  - [ ] tracker row updated  - [ ] no secrets / no banned claims

## Open questions
- Points: 3 holds only if the template build is smooth. It needs the E2B CLI, Docker, an E2B account login and possibly a paid E2B plan for custom templates; 5 is safer. The founder lane owns the build and the env flip.
- Packages outside the template (Prophet is mentioned in the forecast prompt in `sandbox_service.py`) will now fail with `ModuleNotFoundError`. Per the backlog risk table, add them to the template rather than re-enabling egress, and consider removing Prophet from the prompt.
- AWS env var location (Secrets Manager vs task def) is owned by the AWS lane; confirm with 2576.
