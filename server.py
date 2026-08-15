"""Stage 7 deployment entry point — bring-your-own-key HTTP server.

`adk web` (used for local development, see README.md) starts ADK's own Dev
UI and reads exactly one Gemini key from argus/.env for the whole process
lifetime. That's wrong for a public deployment: this file is what actually
runs in the container (see Dockerfile), and it deliberately does NOT carry
any Gemini key of its own — `argus/.env` is excluded from the image
entirely (.dockerignore). Instead, each visitor supplies their own key via
the `X-User-Api-Key` header on every request; the middleware below stores it
in `argus.model_provider.current_api_key` for the lifetime of that one
request. See `argus/model_provider.py` for the full mechanism (and the real
bug it took to get there — read that file's module docstring).

Why `get_fast_api_app(..., web=False)` and not `adk web`'s own Dev UI:
ADK's own `--with_ui` flag documents itself as "for development and testing
only — do not use in production" (verified in
google/adk/cli/cli_tools_click.py). The Dev UI also exposes session/eval
inspection endpoints this deployment has no reason to expose publicly.

Why `session_service_uri="memory://"` / `use_local_storage=False`: forces
ADK's session AND artifact services to pure in-memory backends instead of
the local-SQLite-file default. A container's filesystem is not guaranteed
to persist across restarts on a free-tier host (Render's free web services
spin down after 15 minutes idle and lose local disk state on the next
cold start), so relying on local disk would be a silent lie about
durability. This is a disclosed limitation, not an oversight — see the
plan's "Known limitations" section: a restart loses session history and
any saved memo artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse

from google.adk.cli.fast_api import get_fast_api_app

from argus.model_provider import current_api_key

_REPO_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _REPO_ROOT / "static"

app = get_fast_api_app(
    agents_dir=str(_REPO_ROOT),
    web=False,
    session_service_uri="memory://",
    memory_service_uri="memory://",
    use_local_storage=False,
    auto_create_session=True,
)


@app.middleware("http")
async def inject_byo_api_key(request: Request, call_next):
    """Stash this request's visitor-supplied key for model_provider to read.

    A `contextvars.ContextVar` is scoped to the current async task, so this
    is safe under concurrent requests — one visitor's key can never leak
    into another visitor's in-flight agent run. `.set()` returns a token
    whose `.reset()` clears the var again once the response is done, so a
    key never lingers in this task's context past its own request.
    """
    key = request.headers.get("X-User-Api-Key") or None
    token = current_api_key.set(key)
    try:
        return await call_next(request)
    finally:
        current_api_key.reset(token)


@app.get("/")
async def serve_frontend():
    return FileResponse(_STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    # Bind 0.0.0.0, not 127.0.0.1 — a container's own external interface
    # would otherwise refuse all outside traffic. PORT is read from the
    # environment (Render sets it to 10000 by default) with 7860 as a
    # fallback for local `python server.py` runs, so this same file runs
    # unmodified on Render, Cloud Run, or anywhere else that assigns its
    # own port.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
