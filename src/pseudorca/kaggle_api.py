"""Read-only Kaggle access: list a team's submissions and pull kernel sources.

Pseudorca never submits, pushes or creates anything on Kaggle. Every client call goes through `_call`,
which allows only the methods in ALLOWED_CALLS. Credentials are the `kaggle` package's own (kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY); this module never reads or prints them. Pulled sources are read and hashed,
never run.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import math
import re
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pseudorca import util
from pseudorca.model import Notebook
from pseudorca.notebooks import register_bytes
from pseudorca.store import Store

# The only client methods this module may call: listing submissions and pulling kernel sources.
ALLOWED_CALLS = ("authenticate", "competition_submissions", "kernels_pull")

AUTH_DOCS = "https://www.kaggle.com/docs/api#authentication"
CREDENTIALS_HELP = (
    f"put your API token in ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY (see {AUTH_DOCS})"
)
SOURCE = "kaggle"
PAGE_SIZE = 100  # submissions asked for per page; the server may send fewer
MAX_PAGES = 500
MIN_NAME_CHARS = 4  # shorter notebook names never link a submission
METADATA_FILE = "kernel-metadata.json"
CODE_SUFFIXES = (".ipynb", ".py")
EXTRA_FIELDS = ("team_name", "file_name", "url", "error_description")  # kept in Submission.extra

_SLUG = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}"
KERNEL_REF_RE = re.compile(rf"({_SLUG})/({_SLUG})")
COMPETITION_RE = re.compile(_SLUG)
_SHA_PREFIX_RE = re.compile(r"(?<![0-9A-Za-z])[0-9A-Fa-f]{8,64}(?![0-9A-Za-z])")
_NAME_EDGE = "0-9A-Za-z_-"  # a notebook name found in a description may not touch these characters

_LIST_HINTS = {
    401: "check your Kaggle credentials",
    403: "has your account joined the competition and accepted its rules?",
    404: "check the competition slug",
}
_PULL_HINTS = {
    401: "check your Kaggle credentials",
    403: "is the kernel private to another account?",
    404: "check OWNER/SLUG; private kernels need their owner's credentials",
}


class KaggleError(Exception):
    """Kaggle cannot be used: client missing, no credentials, a failed call or an unexpected download."""


# ----- the client ------------------------------------------------------------------------------------


def _method(api: Any, name: str) -> Any:
    if name not in ALLOWED_CALLS:
        raise KaggleError(f"refusing to call Kaggle API method {name!r}: Pseudorca only reads from Kaggle")
    return getattr(api, name)


def _call(api: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Call one of ALLOWED_CALLS on the client; anything else is refused before it reaches the client."""
    return _method(api, name)(*args, **kwargs)


def _accepts(api: Any, name: str, param: str) -> bool:
    """Whether the client's method takes a keyword `param` (client versions differ)."""
    try:
        return param in inspect.signature(_method(api, name)).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def _reason(e: BaseException, hints: Mapping[int, str] | None = None) -> str:
    """A one-line description of a failed client call, with a hint for common HTTP statuses."""
    if isinstance(e, SystemExit):
        return "the Kaggle client gave up"
    text = " ".join(str(e).split())[:200] or type(e).__name__
    status = getattr(getattr(e, "response", None), "status_code", None)
    hint = (hints or {}).get(status) if isinstance(status, int) else None
    return f"{text} ({hint})" if hint else text


def get_api() -> Any:
    """An authenticated `kaggle` client (KaggleApi), using the kaggle package's own credentials."""
    chatter = io.StringIO()  # the client prints help and warnings to stdout; keep them out of our output
    try:
        with contextlib.redirect_stdout(chatter):
            from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise KaggleError(
            f'cannot import the Kaggle client ({e}); install it with: pip install "pseudorca[kaggle]"'
        ) from None
    except Exception as e:  # importing kaggle also creates its config directory, which can fail
        raise KaggleError(f"cannot load the Kaggle client: {_reason(e)}") from None
    try:
        with contextlib.redirect_stdout(chatter):
            api = KaggleApi()
            _call(api, "authenticate")
    except SystemExit:  # how the client reports missing credentials
        raise KaggleError(f"no Kaggle credentials found: {CREDENTIALS_HELP}") from None
    except Exception as e:
        raise KaggleError(f"Kaggle authentication failed ({type(e).__name__}): {CREDENTIALS_HELP}") from None
    for line in dict.fromkeys(x for x in chatter.getvalue().splitlines() if x.strip()):
        print(line, file=sys.stderr)  # e.g. the client's warning about a world-readable kaggle.json
    return api


# ----- submissions -----------------------------------------------------------------------------------


def _field(raw: Any, name: str) -> Any:
    """`raw.name`, also trying the camelCase spelling and dict keys that older clients use."""
    camel = re.sub(r"_(\w)", lambda m: m.group(1).upper(), name)
    for key in dict.fromkeys((name, camel)):
        value = raw.get(key) if isinstance(raw, Mapping) else getattr(raw, key, None)
        if value is not None:
            return value
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _status(value: Any) -> str:
    """The status as a lowercase word ("complete", "pending", "error") from the client's enum or string."""
    if value is None:
        return ""
    name = getattr(value, "name", None)
    text = name if isinstance(name, str) else str(value)
    return text.rpartition(".")[2].strip().lower()


def _iso_date(value: Any) -> str:
    """An ISO date-time in UTC, to the second. The client's naive datetimes are UTC."""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return text
    if isinstance(value, datetime):
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return value.isoformat(timespec="seconds")
    return _text(value)


def normalize_submission(raw: Any) -> dict[str, Any] | None:
    """One submission from the client as a plain dict, or None if it has no id.

    Accepts kagglesdk's ApiSubmission as well as older clients' camelCase objects or dicts. Scores become
    floats (None when empty or not a finite number) and the status a lowercase word.
    """
    if raw is None:
        return None
    ref = _field(raw, "ref")
    if ref in (None, "", 0):
        return None
    item = {
        "id": str(ref).strip(),
        "date": _iso_date(_field(raw, "date")),
        "description": _text(_field(raw, "description")),
        "status": _status(_field(raw, "status")),
        "public_score": _score(_field(raw, "public_score")),
        "private_score": _score(_field(raw, "private_score")),
        "submitted_by": _text(_field(raw, "submitted_by")),
    }
    item.update({key: _text(_field(raw, key)) for key in EXTRA_FIELDS})
    return item


def _competition(slug: str) -> str:
    slug = (slug or "").strip()
    if not COMPETITION_RE.fullmatch(slug):
        raise KaggleError(
            f"not a Kaggle competition slug: {slug!r} (the SLUG in kaggle.com/competitions/SLUG)"
        )
    return slug


def _list(api: Any, competition: str) -> tuple[list[dict[str, Any]], list[str]]:
    """(submissions in Kaggle's order, notes on submissions that may be missing)."""
    competition = _competition(competition)
    # Client versions with `page_number` page by number; competition_submissions drops the page token.
    paged = _accepts(api, "competition_submissions", "page_number")
    notes = (
        [] if paged else ["this kaggle client cannot page through submissions; only the first page was read"]
    )
    found: dict[str, dict[str, Any]] = {}
    for page in range(1, MAX_PAGES + 1):
        kwargs: dict[str, Any] = {"page_number": page, "page_size": PAGE_SIZE} if paged else {}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                batch = _call(api, "competition_submissions", competition, **kwargs)
        except KaggleError:
            raise
        except (Exception, SystemExit) as e:
            raise KaggleError(f"cannot list submissions of {competition}: {_reason(e, _LIST_HINTS)}") from e
        seen = fresh = 0
        for raw in batch or ():
            item = normalize_submission(raw)
            if item is None:
                continue
            seen += 1
            if item["id"] not in found:
                found[item["id"]] = item
                fresh += 1
        if not paged:
            break
        if not fresh:
            if seen:  # the server repeated a page instead of moving on
                notes.append(f"Kaggle did not page past {len(found)} submissions; older ones may be missing")
            break
    return list(found.values()), notes


def list_submissions(api: Any, competition: str) -> list[dict[str, Any]]:
    """Your team's submissions to `competition`, normalised (see `normalize_submission`), in Kaggle's order.

    Pages are followed while each one brings submissions not seen yet.
    """
    return _list(api, competition)[0]


class _Linker:
    """Finds the one registered notebook a submission description names."""

    def __init__(self, notebooks: Iterable[Notebook]) -> None:
        notebooks = list(notebooks)
        self.ids = [nb.id for nb in notebooks]
        by_name: dict[str, set[str]] = {}
        for nb in notebooks:
            if len(nb.name) >= MIN_NAME_CHARS:
                by_name.setdefault(nb.name, set()).add(nb.id)
        edge = _NAME_EDGE
        self.names = [
            (re.compile(rf"(?<![{edge}]){re.escape(name)}(?![{edge}])"), ids) for name, ids in by_name.items()
        ]

    def find(self, text: str) -> tuple[str | None, bool]:
        """(notebook id or None, ambiguous). Sha prefixes (8+ hex) decide first, then names (4+ chars).

        A name shared by several notebooks, or two different notebooks named, is ambiguous: no link.
        """
        hits = {i for tok in _SHA_PREFIX_RE.findall(text) for i in self.ids if i.startswith(tok.lower())}
        if not hits:
            hits = self._named(text)
        if len(hits) == 1:
            return next(iter(hits)), False
        return None, len(hits) > 1

    def _named(self, text: str) -> set[str]:
        spans = [(m.start(), m.end(), ids) for rx, ids in self.names for m in rx.finditer(text)]
        # a name found inside a longer matched name ("agent" in "agent.ipynb") does not count
        kept = [
            s
            for s in spans
            if not any(o[0] <= s[0] and s[1] <= o[1] and o[1] - o[0] > s[1] - s[0] for o in spans)
        ]
        return set().union(*(ids for _, _, ids in kept))


@dataclass
class SyncReport:
    """What `sync_submissions` wrote (or would write, with dry_run), as submission ids."""

    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    linked: list[str] = field(default_factory=list)  # linked to a notebook by this sync
    ambiguous: list[str] = field(default_factory=list)  # unlinked: the description names several notebooks
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)  # new and updated: Kaggle fields + "nb"
    changed: dict[str, list[str]] = field(default_factory=dict)  # updated: the fields that changed
    notes: list[str] = field(default_factory=list)
    competition: str = ""
    dry_run: bool = False


def _wanted(item: dict[str, Any], competition: str, extra: Any) -> dict[str, Any]:
    """The sub.upsert fields a Kaggle submission should have; `extra` keeps keys it does not set."""
    old_extra = dict(extra) if isinstance(extra, dict) else {}
    return {
        "competition": competition,
        "description": item["description"],
        "date": item["date"],
        "status": item["status"],
        "public_score": item["public_score"],
        "private_score": item["private_score"],
        "submitted_by": item["submitted_by"],
        "source": SOURCE,
        "extra": {**old_extra, **{k: item[k] for k in EXTRA_FIELDS if item.get(k)}},
    }


def sync_submissions(store: Store, competition: str, *, api: Any = None, dry_run: bool = False) -> SyncReport:
    """Record your team's Kaggle submissions to `competition`, writing a sub.upsert only for what changed.

    A submission keeps any notebook link it has (from `sub link` or an earlier sync). An unlinked one is
    linked when its description names exactly one notebook, by a sha prefix of 8+ hex characters or else
    by a registered name of 4+ characters.
    """
    competition = _competition(competition)
    if api is None:
        api = get_api()
    items, notes = _list(api, competition)
    report = SyncReport(notes=notes, competition=competition, dry_run=dry_run)
    state = store.state()
    linker = _Linker(state.notebooks_in_order())
    for item in sorted(items, key=lambda i: (i["date"], len(i["id"]), i["id"])):
        sid = item["id"]
        old = state.subs.get(sid)
        wanted = _wanted(item, competition, old.extra if old else None)
        changes = wanted if old is None else {k: v for k, v in wanted.items() if getattr(old, k) != v}
        nb = old.nb if old else None
        if not nb:
            target, ambiguous = linker.find(item["description"])
            if target:
                changes["nb"] = nb = target
                report.linked.append(sid)
            elif ambiguous:
                report.ambiguous.append(sid)
        if old is not None and not changes:
            report.unchanged.append(sid)
            continue
        if old is None:
            report.new.append(sid)
        else:
            report.updated.append(sid)
            report.changed[sid] = list(changes)
        report.rows[sid] = {**item, "nb": nb}
        if not dry_run:
            store.append("sub.upsert", {"id": sid, **changes})
    return report


# ----- kernels ---------------------------------------------------------------------------------------


def _code_file(folder: Path) -> Path:
    """The pulled notebook or script: the metadata's code_file, else the only .ipynb or .py file."""
    meta_path = folder / METADATA_FILE
    code_name = None
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise KaggleError(f"cannot read the pulled {METADATA_FILE}: {e}") from e
        value = meta.get("code_file") if isinstance(meta, dict) else None
        if isinstance(value, str) and value.strip():
            code_name = value.strip()
    if code_name is not None:
        path = folder / code_name
        if Path(code_name).name != code_name or not path.is_file() or path.is_symlink():
            raise KaggleError(
                f"the pulled {METADATA_FILE} names {code_name!r}, which is not a downloaded file"
            )
        return path
    found = sorted(
        p for p in folder.iterdir() if p.suffix in CODE_SUFFIXES and p.is_file() and not p.is_symlink()
    )
    if len(found) == 1:
        return found[0]
    if not found:
        raise KaggleError("the pull brought no .ipynb or .py file")
    names = ", ".join(p.name for p in found)
    raise KaggleError(f"the pull brought several code files ({names}) and no {METADATA_FILE} naming one")


def pull_kernel(
    store: Store,
    kernel_ref: str,
    *,
    api: Any = None,
    name: str | None = None,
    tags: Iterable[str] = (),
    parent: str | None = None,
    applies: Iterable[str] = (),
) -> tuple[Notebook, bool]:
    """Download the latest source of Kaggle kernel OWNER/SLUG into cache/kaggle/ and register it.

    `parent` is any notebook reference. Returns (notebook, created). The source is read, never run.
    """
    ref = (kernel_ref or "").strip()
    m = KERNEL_REF_RE.fullmatch(ref)
    if not m:
        raise KaggleError(
            f"not a Kaggle kernel: {kernel_ref!r} (expected OWNER/SLUG, as in kaggle.com/code/OWNER/SLUG)"
        )
    owner, slug = m.groups()
    state = store.state()
    parent_id = state.resolve_nb(parent).id if parent else None
    applies = list(applies)
    unknown = [a for a in applies if a not in state.ideas]
    if unknown:  # checked before downloading anything
        raise LookupError(f"unknown idea(s): {', '.join(unknown)} (add them with `pseudorca idea add`)")
    if api is None:
        api = get_api()
    folder = store.cache_dir("kaggle", f"{owner}__{slug}", util.new_ulid())
    try:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                _call(api, "kernels_pull", ref, path=str(folder), metadata=True)
        except KaggleError:
            raise
        except (Exception, SystemExit) as e:
            raise KaggleError(f"cannot pull {ref}: {_reason(e, _PULL_HINTS)}") from e
        code = _code_file(folder)
    except KaggleError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    # The cache sits inside the repository but is local-only, so record just the file name as the path.
    return register_bytes(
        store,
        code.read_bytes(),
        name=name or code.name,
        path=code.name,
        kernel=ref,
        parent=parent_id,
        applies=applies,
        tags=tags,
    )
