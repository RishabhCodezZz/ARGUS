"""Per-request Gemini API key support for BYO-key deployment (Stage 7).

`argus/config.py`'s `MODEL_FLASH` is a model-ID *string* that every agent
passes to `LlmAgent(model=...)`. ADK resolves a bare string into a `Gemini`
instance whose `api_client` is a `@cached_property` (see
`google/adk/models/google_llm.py:336`) — built once, from `GOOGLE_API_KEY`
in the process environment, and reused for that object's whole lifetime.
That's correct for one person on one laptop. It's wrong for a deployed
server handling many visitors: a cached client bakes in ONE key for the
whole process, and every agent module under `argus/agents/` constructs its
`LlmAgent`s at IMPORT TIME — before any request, let alone any visitor's
key, exists.

Fix, verified against ADK 2.5.0 source rather than assumed: `Gemini`'s own
docstring (`google_llm.py:97-114`) documents subclassing and overriding
`api_client`, explicitly noting a plain `@property` (not `@cached_property`)
is the right choice "if you hit asyncio lock contention" — i.e. exactly when
the value must vary per call. Paired with a `contextvars.ContextVar`, each
in-flight async request gets its own key with no cross-request leakage, and
critically, no mutation of any shared object: the `LlmAgent` singletons in
gather.py/quant.py/synthesis.py/etc. are never touched, so this file is the
ONLY thing that needed to know about per-request keys.

Deployment (`server.py`, Stage 7 Phase 2) reads an `X-User-Api-Key` header
per request and does `current_api_key.set(key)` before invoking the agent.
Local `adk web` never touches this context var at all, so it stays `None`
and falls through to `GOOGLE_API_KEY` from `argus/.env` exactly as before —
`argus/.env` is therefore still required for local development; it is the
DEPLOYED container that carries no key at all (see `Dockerfile`,
`.dockerignore`).

CAUGHT LIVE, FIXED HERE: a first version of this file made `api_client` a
plain `@property` that called `Client(**kwargs)` fresh on every single
access, reasoning (from the ADK docstring) that `@cached_property` would
freeze in whichever visitor's key happened to be active first. That broke
the very first live pipeline run with `aiohttp`'s
`AssertionError: assert self._connector is not None`. Root cause, confirmed
by reading `google/genai/_api_client.py:2284`: `Client.__del__` closes the
client's HTTP session when the object is garbage-collected — a documented,
deliberate cleanup path. `Gemini._api_backend` (a separate `@cached_property`
on the base class, untouched here) reads `self.api_client.vertexai` once,
just to check a flag; with an uncached property that call alone constructs
a throwaway `Client`, immediately eligible for GC once the expression
finishes, which tore its session down out from under the real
`generate_content_async` call moments later. `@cached_property` avoids this
by keeping ONE client alive for the object's whole lifetime — but that's
exactly the behavior a multi-key deployment can't use.
The fix below keeps both properties: cache one `Client` per distinct API
key (a `None` key — local dev's env-var fallback — gets its own cached
slot), so nothing is ever thrown away mid-request, while different visitors
still transparently get different, isolated clients.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any, Optional

from google.adk.models import Gemini
from google.genai import Client, types

# Cache of one Client per distinct API key (None = local env-var fallback),
# so a client is never garbage-collected mid-request — see module docstring.
_clients_by_key: dict[Optional[str], Client] = {}
_clients_lock = threading.Lock()

# Set by server.py's request middleware for a deployed visitor's request;
# left unset (None) for local `adk web`, in which case google-genai's own
# Client() falls back to the GOOGLE_API_KEY environment variable — identical
# to an unmodified Gemini model's behavior.
current_api_key: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_api_key", default=None
)


class BYOKeyGemini(Gemini):
    """Gemini model whose client varies by the active `current_api_key`.

    Mirrors `Gemini.api_client` (`google/adk/models/google_llm.py:336-363`)
    field-for-field — same base_url/api_version resolution, tracking
    headers, retry_options, and client_kwargs override — so nothing about
    ADK's retry/tracking behavior is silently lost. Unlike the base class's
    `@cached_property` (one client for the object's whole process lifetime),
    this is a plain `@property` that re-reads `current_api_key` on every
    access — but it still returns a STABLE, cached `Client` per distinct key
    (see `_clients_by_key` and the module docstring's "CAUGHT LIVE" note for
    why an uncached-per-access version broke the first live run).
    """

    @property
    def api_client(self) -> Client:
        key = current_api_key.get()

        cached = _clients_by_key.get(key)
        if cached is not None:
            return cached

        with _clients_lock:
            # Re-check: another coroutine may have built it while we waited.
            cached = _clients_by_key.get(key)
            if cached is not None:
                return cached

            base_url, api_version = self._base_url_and_api_version
            kwargs_for_http_options: dict[str, Any] = {
                "headers": self._tracking_headers(),
                "retry_options": self.retry_options,
                "base_url": base_url,
            }
            if api_version:
                kwargs_for_http_options["api_version"] = api_version

            kwargs: dict[str, Any] = {
                "http_options": types.HttpOptions(**kwargs_for_http_options),
            }
            if self.model.startswith("projects/"):
                kwargs["enterprise"] = True

            if key:
                kwargs["api_key"] = key
            # else: no explicit api_key -> google-genai's Client() falls
            # back to the GOOGLE_API_KEY environment variable itself, same
            # as local dev.

            if self.client_kwargs:
                kwargs.update(self.client_kwargs)

            client = Client(**kwargs)
            _clients_by_key[key] = client
            return client


def get_model(model_id: str) -> BYOKeyGemini:
    """Build the one model type every ARGUS agent should use.

    Every agent under argus/agents/ should call this instead of passing a
    bare model-ID string, so that a visitor's Gemini key (set into
    `current_api_key` by server.py's request middleware) reaches every
    agent in the pipeline without mutating any shared agent object.
    """
    return BYOKeyGemini(model=model_id)
