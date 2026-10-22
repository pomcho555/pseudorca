"""Helpers shared by the command modules."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

import click

from pseudorca.model import Effective
from pseudorca.store import Store, StoreError

SYMBOLS = {"present": "✓", "partial": "◐", "absent": "✗", "unknown": "?", "conflict": "!"}
NOT_AUDITED = "·"
CLAIMED = "c"  # in the notebook's `applies` list but not audited yet
LEGEND = (
    "✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  c claimed, not audited  · none  * stale"
)


@dataclass
class Ctx:
    store_path: Path | None = None
    actor: str | None = None
    _store: Store | None = field(default=None, repr=False)

    def store(self) -> Store:
        if self._store is None:
            try:
                self._store = Store.at(self.store_path) if self.store_path else Store.find()
            except StoreError as e:
                raise click.ClickException(str(e)) from e
            self._store.actor_override = self.actor
        return self._store


pass_ctx = click.make_pass_decorator(Ctx, ensure=True)


def fail(message: str) -> NoReturn:
    raise click.ClickException(message)


def warn(message: str) -> None:
    click.secho(f"warning: {message}", fg="yellow", err=True)


@contextmanager
def user_errors() -> Iterator[None]:
    """Show expected errors (bad references, store problems, bad input) as one line instead of a traceback."""
    try:
        yield
    except KeyError:
        raise
    except (StoreError, LookupError, ValueError) as e:
        raise click.ClickException(str(e)) from e


def show_warnings(store: Store) -> None:
    for w in store.state().warnings:
        warn(w)


def verdict_symbol(eff: Effective | None, claimed: bool = False) -> str:
    if eff is None:
        return CLAIMED if claimed else NOT_AUDITED
    return SYMBOLS.get(eff.status, "?") + ("*" if eff.stale else "")
