# Troubleshooting

Known local-environment failure modes for the Compute layer (`rag/pipeline.py`),
kept here because the same generic error message covers several unrelated
root causes and re-diagnosing from scratch each time wastes an afternoon.

## "Error: Compute layer unavailable. Provider 'anthropic' is misconfigured."

This wrapper text (`pipeline.py`'s `_generate` / `_generate_with_system`) is
printed for *any* exception raised while calling the LLM. Read the `Detail:`
line at the end to tell which of these you actually have:

| `Detail:` text | Meaning | Fix |
|---|---|---|
| `Connection error.` | The SDK never got an HTTP response at all (`anthropic.APIConnectionError`). Not a key/billing/model problem — something failed before or during the network call. | See below. |
| `Error code: 401 ...` / `invalid x-api-key` | Request reached Anthropic; the key is wrong. | See "Truncated API key" below. |
| `Error code: 404 ...` | Model name typo/retired model. | Check `ANTHROPIC_MODEL` / `Settings.anthropic_model` in `config.py`. |
| `Error code: 429 ...` | Billing/quota. | Check console.anthropic.com usage. |

### Diagnosing a bare "Connection error."

Don't assume network/proxy/firewall — verify each layer before changing
anything, in this order:

1. **Confirm basic reachability** with `curl.exe` (PowerShell, not `cmd.exe` —
   `$env:VAR` syntax silently doesn't expand in `cmd.exe` and produces a
   garbled header that looks like an auth failure, not a connection one):
   ```powershell
   curl.exe -v https://api.anthropic.com/v1/models -H "x-api-key: $env:ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"
   ```
   A clean HTTP response (even a 401) means the network path is fine and the
   bug is in the Python SDK stack, not connectivity.

2. **Reproduce with the real `anthropic` package**, not bare `httpx` — a raw
   `httpx.get()` can succeed while `anthropic.Anthropic().messages.create()`
   still fails, because the SDK's response decoder is a separate code path:
   ```powershell
   python -c "from acheron.config import get_settings; import anthropic; s=get_settings(); c=anthropic.Anthropic(api_key=s.anthropic_api_key); r=c.messages.create(model=s.resolved_llm_model,max_tokens=10,messages=[{'role':'user','content':'hi'}]); print(r.content)"
   ```

3. **If that throws `TypeError: process() takes no keyword arguments` inside
   `httpx2/_decoders.py`**: this is a real dependency of the current
   `anthropic` SDK (verify against `https://pypi.org/pypi/anthropic/json` and
   `https://pypi.org/pypi/httpx2/json` before assuming a fake/typosquatted
   package — the unfamiliar name alone is not evidence of tampering). The
   actual bug is a version mismatch between `httpx2` and an old `brotli`
   already in site-packages (commonly pulled in transitively by something
   like `gradio` or `mitmproxy`, not by this project). Fix:
   ```powershell
   pip show brotli
   pip install --upgrade --force-reinstall "brotli>=1.2.0"
   ```

4. **Rule out an unrelated local MITM/dev tool** if the above doesn't apply
   and `curl`/schannel behaves oddly (e.g. `remote party requests
   renegotiation`). Check for an injected root CA and stale startup entries
   from any locally-built proxy/interception tool:
   ```powershell
   Get-ChildItem Cert:\CurrentUser\Root, Cert:\LocalMachine\Root
   Get-CimInstance Win32_StartupCommand | Select Name, Command
   ```
   This project has no dependency on any such tool — if one shows up, it's
   leftover from an unrelated local project and should be fully removed
   (kill process, remove Root cert, remove Run key, delete/quarantine its
   install folder), since a half-uninstalled interception tool makes every
   other symptom in this list harder to read correctly.

### Truncated / wrong API key in `.env`

`anthropic.AuthenticationError: invalid x-api-key` with a real HTTP 401 means
the key itself is bad, most often because `.env` has a truncated value from
a bad copy/paste, not because the key was revoked. Verify the length before
assuming anything else:
```powershell
(Select-String -Path .env -Pattern "ANTHROPIC_API_KEY").Line.Split("=",2)[1].Trim().Length
```
A real key is ~108 characters (`sk-ant-api03-...`). Anything much shorter
means the paste was cut off. When rewriting the line, avoid `-replace` with
the raw key inline in a single command — a dropped `ANTHROPIC_API_KEY=`
prefix silently produces a line with no variable name at all (next
`Select-String` lookup returns null, which looks like a different failure).
Rebuild the line explicitly instead:
```powershell
$lines = Get-Content .env | Where-Object { $_ -notmatch "ANTHROPIC_API_KEY" -and $_ -notmatch "^sk-ant-" }
$lines += "ANTHROPIC_API_KEY=<full key>"
$lines | Set-Content .env
Get-Content .env | Select-String "ANTHROPIC_API_KEY"
```
Always confirm by printing the resulting line before rerunning the app.

## Prevention

- Give this project its own virtualenv instead of a shared global Python
  install — a global `pip install` here can pick up (or be shadowed by)
  packages belonging to unrelated local projects, which is what turned a
  10-minute fix into a multi-hour investigation the first time this
  happened.
- After ever editing `.env` by hand or by script, print the resulting key's
  length before assuming the edit worked.
- Any time a package's dependency list looks unfamiliar, check its real
  PyPI listing (`https://pypi.org/pypi/<name>/json`) before concluding it's
  malicious — a name you don't recognize is not evidence by itself.
