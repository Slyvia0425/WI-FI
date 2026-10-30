"""Backend routes. This file is the router: every URL the website can call is listed in ROUTES.

Request flow:
    browser -> backend/api/server.py -> routes.dispatch -> handler here
            -> backend/modules/<module>/api.py replay() -> outputs/replay.json

Adding a module never edits this file: any backend/modules/<module>/api.py with a replay() function is found by discover().
"""
import re
from importlib import import_module

from backend.shared.paths import MODULES_DIR


def discover(only: list[str] | None = None) -> dict:
    """Map module folder name -> its api module. `only` limits it to the named modules."""
    found = {}
    for path in sorted(MODULES_DIR.iterdir()):
        if only and path.name not in only:
            continue
        if (path / "api.py").exists():
            found[path.name] = import_module(f"backend.modules.{path.name}.api")
    return found


MODULES = discover()

# A handler takes the path parameters and returns (http status, JSON-serialisable body).
def health(params):
    return 200, {"ok": True}


def list_modules(params):
    return 200, sorted(MODULES)


def module_replay(params):
    name = params["module"]
    module = MODULES.get(name)
    if module is None:
        return 404, {"error": f"unknown module {name!r}"}
    try:
        payload = module.replay()
    except Exception as exc:  # a broken module must not take the others down
        return 500, {"error": f"{type(exc).__name__}: {exc}"}
    if payload is None:
        return 404, {"error": "no output yet", "module": name}
    return 200, payload


# ---- the route table ---------------------------------------------------------------
ROUTES = [
    ("GET", "/api/health", health),
    ("GET", "/api/modules", list_modules),
    ("GET", "/api/{module}/replay", module_replay),
]
# ------------------------------------------------------------------------------------


def _compile(pattern: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$")


_COMPILED = [(method, _compile(pattern), handler) for method, pattern, handler in ROUTES]


def dispatch(method: str, path: str):
    """Find the handler for a request. Returns (status, body)."""
    path = path.split("?")[0].rstrip("/") or "/"
    for route_method, regex, handler in _COMPILED:
        match = regex.match(path)
        if match and route_method == method:
            return handler(match.groupdict())
    return 404, {"error": "not found"}
