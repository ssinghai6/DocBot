# E2B Code Interpreter SDK — Integration Notes (DOCBOT-103)

## Overview

E2B provides cloud-hosted Python sandboxes via `e2b-code-interpreter`. Each sandbox is an isolated
microVM. DocBot uses it to execute LLM-generated pandas/matplotlib code without risking the host server.

Package: `e2b-code-interpreter`

---

## Key SDK Patterns

### 1. Creating a Sandbox

```python
from e2b_code_interpreter import AsyncSandbox

# SDK reads E2B_API_KEY from environment automatically — never hardcode it.
sbx = await AsyncSandbox.create()
```

### 2. Executing Code

```python
execution = await sbx.run_code("print('hello')")
```

The `execution` object has:
- `execution.logs.stdout` — `list[str]` of stdout lines
- `execution.logs.stderr` — `list[str]` of stderr lines
- `execution.error` — `ExecutionError | None` with `.name`, `.value`, `.traceback`
- `execution.results` — `list[Result]` (rich output: images, DataFrames, etc.)

Join stdout: `"\n".join(execution.logs.stdout)`

### 3. Chart / PNG Extraction

Each `Result` in `execution.results` may have:
- `result.png`  — base64-encoded PNG string (primary for matplotlib figures)
- `result.jpeg` — base64-encoded JPEG string
- `result.svg`  — SVG string
- `result.html` — HTML string (for interactive plots like plotly)
- `result.text` — plain text representation

```python
charts: list[str] = []
for result in execution.results:
    if result.png:
        charts.append(result.png)
```

**Important:** `result.png` is a raw base64 string — no `data:image/png;base64,` prefix.
The frontend must prepend that before using it as an `<img>` src.

### 4. Timeout Handling

```python
import asyncio

try:
    execution = await asyncio.wait_for(sbx.run_code(code), timeout=25.0)
except asyncio.TimeoutError:
    return SandboxResult(
        stdout="", stderr="", charts=[],
        error="Execution timed out after 25 seconds.",
        execution_time_ms=25_000,
    )
```

### 5. Cleanup — Always Use `finally`

```python
sbx = await AsyncSandbox.create()
try:
    execution = await sbx.run_code(code)
finally:
    await sbx.kill()  # unclosed sandboxes leak quota and billing
```

### 6. Error Object Shape

`execution.error` is a data object, NOT a Python exception:
- `.name` — exception class name (e.g. `"ValueError"`)
- `.value` — exception message string
- `.traceback` — full traceback string

```python
if execution.error:
    error_str = f"{execution.error.name}: {execution.error.value}"
```

---

## Gotchas and Edge Cases

### Matplotlib requires the Agg backend

Always prepend to generated matplotlib code:

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

`sandbox_service.py` should inject this header automatically for any LLM-generated chart code.

### `plt.show()` triggers chart capture — `plt.savefig()` alone does NOT

`result.png` entries come from IPython's display hook, triggered by `plt.show()`.
`plt.savefig('chart.png')` saves to the sandbox filesystem but produces no `result.png`.

Correct pattern:
```python
fig, ax = plt.subplots()
ax.bar(...)
plt.show()  # triggers display hook → result.png appears in execution.results
```

### `stdout` is a list, not a string

`execution.logs.stdout` is `list[str]`. Join with `"\n"` before returning.

### Cold Start Latency

First sandbox creation takes 2–5 seconds (microVM boot). Incompatible with Vercel's 30s timeout —
Railway persistent container (DOCBOT-101) is a hard dependency.

### Network Access

E2B sandboxes have internet access by default. Disable for untrusted user code via sandbox config if needed.

### Pre-installed Packages

The default `e2b-code-interpreter` template includes pandas, numpy, and matplotlib.
For statsmodels or scipy (future forecasting stories), a custom E2B template is required.

---

## SandboxResult Model

```python
class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    charts: list[str]       # base64 PNG strings — no data-URI prefix
    error: Optional[str]    # None on success; "ExceptionType: message" on failure
    execution_time_ms: int
```

---

## Execution Sequence

```
await AsyncSandbox.create()                      # microVM boot (~2-5s)
asyncio.wait_for(sbx.run_code(code), 25.0)       # 25s wall-clock timeout
stdout = "\n".join(execution.logs.stdout)
stderr = "\n".join(execution.logs.stderr)
charts = [r.png for r in execution.results if r.png]
error  = f"{execution.error.name}: {execution.error.value}" if execution.error else None
await sbx.kill()                                 # always, in finally block
return SandboxResult(...)
```
