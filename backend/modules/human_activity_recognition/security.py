"""Path and role policy. These checks must run outside the frontend."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import AuthorizationError, InputValidationError

READ_ROLES = {"system", "researcher"}
REPLAY_ROLES = READ_ROLES | {"viewer"}
WRITE_ROLES = {"system", "researcher"}


def resolve_under(root: Path, candidate: str | Path, *, must_exist: bool = False) -> Path:
    root_resolved = root.expanduser().resolve()
    path = (
        (root_resolved / candidate).resolve()
        if not Path(candidate).is_absolute()
        else Path(candidate).expanduser().resolve()
    )
    if path != root_resolved and root_resolved not in path.parents:
        raise InputValidationError(
            "path escapes the configured root", {"root": str(root_resolved), "candidate": str(candidate)}
        )
    if must_exist and not path.exists():
        raise InputValidationError("path does not exist", {"path": str(path)})
    return path


@dataclass(frozen=True)
class AccessPolicy:
    roles: frozenset[str]
    readable_roots: tuple[Path, ...]
    writable_roots: tuple[Path, ...]

    @classmethod
    def build(
        cls, *, roles: Iterable[str], readable_roots: Iterable[Path], writable_roots: Iterable[Path]
    ) -> AccessPolicy:
        return cls(
            roles=frozenset(roles),
            readable_roots=tuple(path.expanduser().resolve() for path in readable_roots),
            writable_roots=tuple(path.expanduser().resolve() for path in writable_roots),
        )

    def authorize_read(self, actor: str, path: Path, *, replay: bool = False) -> Path:
        allowed = REPLAY_ROLES if replay else READ_ROLES
        if actor not in allowed:
            raise AuthorizationError("actor may not read this resource", {"actor": actor})
        resolved = path.expanduser().resolve()
        if not any(root == resolved or root in resolved.parents for root in self.readable_roots):
            raise AuthorizationError("resource is outside readable roots", {"actor": actor, "path": str(resolved)})
        return resolved

    def authorize_write(self, actor: str, path: Path) -> Path:
        if actor not in WRITE_ROLES:
            raise AuthorizationError("actor may not write this resource", {"actor": actor})
        resolved = path.expanduser().resolve()
        if not any(root == resolved or root in resolved.parents for root in self.writable_roots):
            raise AuthorizationError("resource is outside writable roots", {"actor": actor, "path": str(resolved)})
        return resolved
