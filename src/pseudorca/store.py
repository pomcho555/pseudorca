"""The store: append-only JSON events plus content-addressed blobs, kept in a team's private git repository.

    .pseudorca/
      config.toml                          settings; commit it, and never put secrets in it
      events/YYYY-MM/<ulid>-<actor>.json   one immutable event per file, so git merges never conflict
      blobs/<sha256[:2]>/<sha256>          notebooks, patches and result tables (checked against the name on read)
      cache/                               local only (extracted trees, LLM transcripts); gitignored
      .gitattributes                       `blobs/** -text -diff`, so git never rewrites blob bytes

Everything a command shows is rebuilt by replaying the events in id order (see `pseudorca.model.State`).
"""

from __future__ import annotations

import getpass
import json
import os
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pseudorca import util

if TYPE_CHECKING:
    from pseudorca.model import State

STORE_DIR = ".pseudorca"
SCHEMA = 1
EVENT_TYPES = frozenset(
    {"idea.add", "idea.update", "nb.add", "nb.update", "sub.upsert", "run.add", "audit.verdict"}
)

CONFIG_TEMPLATE = """\
# Pseudorca store settings. Commit this file; never put secrets in it.

[store]
name = {name}
# Kaggle competition slug used by `pseudorca kaggle sync`.
competition = {competition}

[audit]
# `pseudorca check` expects ideas with these statuses to be in the notebook...
expect_status = ["adopted"]
# ...and flags ideas with these statuses that are in it.
forbid_status = ["rejected"]
# When to ask the LLM: "auto" (only when the deterministic checks are inconclusive), "always" or "never".
llm = "auto"

[llm]
# provider = "openai"              # any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, OpenAI, ...) or "anthropic"
# model = "your-model-name"
# base_url = "http://localhost:8000/v1"
# api_key_env = "OPENAI_API_KEY"   # the NAME of an environment variable; the key itself never goes here
# allow_remote = false             # must be true before notebook code is sent to a non-local endpoint
"""


_INCOMPLETE = (
    "{path} has no events/ folder (not pulled yet, or created by hand); run `pseudorca init` in its parent "
    "to complete it (init never overwrites existing files such as config.toml)"
)


class StoreError(Exception):
    """A problem with the store itself: missing, unreadable or misused."""


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    time: str
    actor: str
    data: dict[str, Any]
    schema: int = SCHEMA
    path: Path | None = field(default=None, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "type": self.type,
            "time": self.time,
            "actor": self.actor,
            "data": self.data,
        }


def _git_user(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "user.name"], cwd=cwd, capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Store:
    """A Pseudorca store rooted at a `.pseudorca` directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.actor_override: str | None = None
        self.load_warnings: list[str] = []
        self.created: list[str] = []  # what `init` created (relative to the store root)
        self._events: list[Event] | None = None
        self._state: State | None = None
        self._config: dict[str, Any] | None = None
        self._git_actor: str | None = None

    # ----- finding and creating stores -------------------------------------------------------------

    @classmethod
    def at(cls, path: Path) -> Store:
        """The store at `path`, which may be the `.pseudorca` directory or the directory holding it."""
        p = Path(path).expanduser().resolve()
        if p.name != STORE_DIR and (p / STORE_DIR).is_dir():
            p = p / STORE_DIR
        if not (p / "events").is_dir():
            if p.name == STORE_DIR and p.is_dir():
                raise StoreError(_INCOMPLETE.format(path=p))
            raise StoreError(f"{path} is not a Pseudorca store (run `pseudorca init` there first)")
        return cls(p)

    @classmethod
    def find(cls, start: Path | None = None) -> Store:
        """The nearest store in `start` (default: the current directory) or one of its parents."""
        env = os.environ.get("PSEUDORCA_STORE")
        if env:
            return cls.at(Path(env))
        cur = (start or Path.cwd()).resolve()
        for d in (cur, *cur.parents):
            if (d / STORE_DIR / "events").is_dir():
                return cls(d / STORE_DIR)
            if (d / STORE_DIR).is_dir():
                raise StoreError(_INCOMPLETE.format(path=d / STORE_DIR))
        raise StoreError(
            "no Pseudorca store here or in any parent directory; run `pseudorca init` in your team's private repository"
        )

    @classmethod
    def init(cls, directory: Path, name: str = "", competition: str = "") -> Store:
        """Create a store, or complete a partial one. Existing files (a committed config.toml) are never
        overwritten; `store.created` lists what was added. Raises StoreError if nothing was missing."""
        root = Path(directory).expanduser().resolve() / STORE_DIR
        files = {
            "events/.gitkeep": "",
            "blobs/.gitkeep": "",
            ".gitignore": "cache/\n",
            ".gitattributes": "# blobs are stored byte for byte; their file names are their sha256\nblobs/** -text -diff\n",
            "config.toml": CONFIG_TEMPLATE.format(
                name=json.dumps(name or root.parent.name, ensure_ascii=False),
                competition=json.dumps(competition, ensure_ascii=False),
            ),
        }
        created: list[str] = []
        try:
            for sub in ("events", "blobs", "cache"):
                if not (root / sub).is_dir():
                    (root / sub).mkdir(parents=True, exist_ok=True)
                    created.append(sub + "/")
            for rel, text in files.items():
                try:
                    with open(root / rel, "x", encoding="utf-8") as f:
                        f.write(text)
                except FileExistsError:
                    continue
                if not rel.endswith(".gitkeep"):
                    created.append(rel)
        except OSError as e:
            raise StoreError(f"cannot create the store at {root}: {e.strerror or e}") from e
        if not created:
            raise StoreError(f"a store already exists at {root}")
        store = cls(root)
        store.created = created
        return store

    # ----- settings --------------------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            p = self.root / "config.toml"
            try:
                self._config = tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
                raise StoreError(f"cannot read {p}: {e}") from e
            except OSError as e:
                raise StoreError(f"cannot read {p}: {e.strerror or e}") from e
        return self._config

    def setting(self, section: str, key: str, default: Any = None) -> Any:
        value = self.config.get(section, {})
        return value.get(key, default) if isinstance(value, dict) else default

    def actor(self) -> str:
        """Who new events are recorded as: --as, then $PSEUDORCA_ACTOR, then git user.name, then the OS user."""
        for candidate in (self.actor_override, os.environ.get("PSEUDORCA_ACTOR")):
            if candidate and candidate.strip():
                return candidate.strip()
        if self._git_actor is None:  # one git call per store, not per event
            self._git_actor = (_git_user(self.root) or "").strip()
        if self._git_actor:
            return self._git_actor
        try:
            return getpass.getuser()
        except (OSError, KeyError):
            return "unknown"

    # ----- events ----------------------------------------------------------------------------------

    def events(self) -> list[Event]:
        """All events, sorted by id (= creation time)."""
        if self._events is None:
            events: list[Event] = []
            self.load_warnings = []
            for p in self._event_files():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    events.append(
                        Event(
                            id=str(raw["id"]),
                            type=str(raw["type"]),
                            time=str(raw["time"]),
                            actor=str(raw["actor"]),
                            data=dict(raw.get("data") or {}),
                            schema=int(raw.get("schema", SCHEMA)),
                            path=p,
                        )
                    )
                except (OSError, ValueError, KeyError, TypeError) as e:
                    self.load_warnings.append(f"skipped unreadable event file {p.name}: {e}")
            events.sort(key=lambda e: e.id)
            self._events = events
        return self._events

    def _event_files(self) -> list[Path]:
        """Every event file, failing loudly on an unreadable folder (a silent empty ledger invites duplicates)."""
        events_dir = self.root / "events"
        files: list[Path] = []
        try:
            months = sorted(d for d in events_dir.iterdir() if d.is_dir())
            for month in months:
                files.extend(month / name for name in sorted(os.listdir(month)) if name.endswith(".json"))
        except OSError as e:
            raise StoreError(f"cannot read {e.filename or events_dir}: {e.strerror or e}") from e
        return files

    def append(self, type: str, data: dict[str, Any]) -> Event:
        """Write one new event file and return the event.

        The id sorts after every event this store already holds (a hybrid logical clock), so an edit made
        after pulling someone's change wins over it even if this machine's clock is behind.
        """
        if type not in EVENT_TYPES:
            raise StoreError(f"unknown event type {type!r}")
        existing = self.events()
        newest_ms = util.ulid_ms(existing[-1].id) if existing else 0
        now_ms = time.time_ns() // 1_000_000
        if newest_ms - now_ms > 60_000:
            note = (
                f"this machine's clock is {(newest_ms - now_ms) / 60_000:.0f} min behind the newest event in the "
                "ledger; new events are ordered after it anyway, but check the clock"
            )
            if note not in self.load_warnings:
                self.load_warnings.append(note)
                if self._state is not None:
                    self._state.warnings.append(note)
        actor = self.actor()
        event = Event(id=util.new_ulid(newest_ms + 1), type=type, time=util.now_iso(), actor=actor, data=data)
        folder = self.root / "events" / event.time[:7]
        path = folder / f"{event.id}-{util.slug(actor)}.json"
        tmp = path.with_name(path.name + ".tmp")
        try:
            folder.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(event.to_json(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            raise StoreError(f"cannot write {path}: {e.strerror or e}") from e
        existing.append(Event(**{**event.__dict__, "path": path}))
        # Rebuilt on the next state() call. (Applying the event in place would be cheaper, but callers keep
        # references to entities from before the append and compare them with the new state.)
        self._state = None
        return event

    def state(self) -> State:
        """The ledger as of now, rebuilt from the events (cached until the next append)."""
        if self._state is None:
            from pseudorca.model import State

            self._state = State.replay(self.events())
            self._state.warnings[:0] = self.load_warnings
        return self._state

    # ----- blobs and cache -------------------------------------------------------------------------

    def blob_path(self, sha: str) -> Path:
        """Where a blob lives. Only a 64-hex sha256 is accepted: ids come from shared events, and anything
        else could point outside the store."""
        if not util.is_sha256(sha):
            raise StoreError(f"{str(sha)[:80]!r} is not a blob id (a sha256)")
        return self.root / "blobs" / sha[:2] / sha

    def has_blob(self, sha: str) -> bool:
        return util.is_sha256(sha) and self.blob_path(sha).is_file()

    def put_blob(self, data: bytes) -> str:
        sha = util.sha256_bytes(data)
        p = self.blob_path(sha)
        if not p.exists():
            tmp = p.with_name(p.name + ".tmp")
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data)
                os.replace(tmp, p)
            except OSError as e:
                raise StoreError(f"cannot write {p}: {e.strerror or e}") from e
        return sha

    def get_blob(self, sha: str) -> bytes:
        """A blob's bytes, checked against its sha256 (a Git LFS pointer or a line-ending conversion would
        otherwise be audited as if it were the notebook)."""
        p = self.blob_path(sha)
        if not p.is_file():
            raise StoreError(
                f"blob {sha[:12]} is not in this store (was it added with --no-store, or not pulled yet?)"
            )
        try:
            data = p.read_bytes()
        except OSError as e:
            raise StoreError(f"cannot read {p}: {e.strerror or e}") from e
        if util.sha256_bytes(data) != sha:
            raise StoreError(
                f"blob {sha[:12]} does not match its sha256: it is probably a Git LFS pointer (run `git lfs pull`) "
                "or was changed by a line-ending conversion (.pseudorca/.gitattributes should contain "
                "`blobs/** -text -diff`; run `pseudorca init` to add it)"
            )
        return data

    def cache_dir(self, *parts: str) -> Path:
        p = self.root.joinpath("cache", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p
