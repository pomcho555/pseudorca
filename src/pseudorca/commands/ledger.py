"""Ledger commands: `init`, `idea`, `nb`, `status`, `sub` and `log`.

Every change is one new event in the store; the reading commands show the state rebuilt from the events.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shlex
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import IO, Any, NoReturn

import click

from pseudorca import util
from pseudorca.commands._common import (
    LEGEND,
    Ctx,
    fail,
    pass_ctx,
    show_warnings,
    user_errors,
    verdict_symbol,
    warn,
)
from pseudorca.model import IDEA_STATUSES, NOTEBOOK_FIELDS, Effective, Idea, Notebook, State, Submission
from pseudorca.notebooks import register_file
from pseudorca.store import EVENT_TYPES, STORE_DIR, Event, Store, StoreError

SHOW_LATEST = 10  # notebooks listed by `idea show`
MAX_GREP_HITS = 500
EVIDENCE_WIDTH = 50  # display columns of the EVIDENCE cells in `idea show` and `nb show`
_KERNEL_RE = re.compile(r"[^/\s]+/[^/\s]+")
_HUNK_RE = re.compile(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_ID_HELP = "1-64 letters, digits, '.', '_' or '-', starting with a letter or digit"
_FP_HELP = "text (may span lines), re:REGEX, !text or !re:REGEX (the last two must be absent)"
_EARLIEST = datetime.min.replace(tzinfo=UTC)

# ----- small helpers -------------------------------------------------------------------------------


def _echo_json(obj: Any) -> None:
    click.echo(json.dumps(obj, ensure_ascii=False, indent=2))


def _minute(iso: str) -> str:
    """A ledger time as "YYYY-MM-DD HH:MM" in the viewer's time zone; a bare date stays a date."""
    if not iso:
        return "-"
    try:
        return util.local_time(iso)
    except (OverflowError, OSError, ValueError):  # near year 1 or 9999 the conversion leaves the range
        return str(iso)[:16].replace("T", " ")


def _day(iso: str) -> str:
    """A ledger time as the viewer's local date."""
    return _minute(iso)[:10]


def _instant(iso: object) -> datetime:
    """A ledger time as an aware datetime, so times written with different UTC offsets compare correctly.

    A time without an offset (or a bare date) is taken as the viewer's local time; what does not parse
    sorts first.
    """
    try:
        dt = datetime.fromisoformat(str(iso or "").strip())
        return dt if dt.tzinfo is not None else dt.astimezone()
    except (ValueError, OverflowError, OSError):
        return _EARLIEST


def _csv(items: Iterable[str]) -> str:
    return ", ".join(items) or "-"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _size(n: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value, i = float(n), 0
    while value >= 1024 and i < len(units) - 1:
        value, i = value / 1024, i + 1
    return f"{n} B" if i == 0 else f"{value:.1f} {units[i]}"


def _score(x: float | None) -> str:
    return "-" if x is None else str(x)


def _clean(values: Iterable[str], what: str) -> list[str]:
    """Stripped values without duplicates, in their given order; an empty one is an error."""
    out: list[str] = []
    for v in values:
        v = v.strip()
        if not v:
            fail(f"{what} cannot be empty")
        out.append(v)
    return list(dict.fromkeys(out))


def _kv(pairs: list[tuple[str, Any]]) -> str:
    """Aligned `key  value` lines. Lists and multi-line strings continue on indented lines; a list item
    that spans lines (a multi-line fingerprint) continues two columns further in."""
    width = max(len(k) for k, _ in pairs)
    lines: list[str] = []
    for key, value in pairs:
        if isinstance(value, list):
            items = []
            for v in value:
                first, *rest = str(v).splitlines() or [""]
                items += [first, *("  " + line for line in rest)]
        else:
            items = str(value).splitlines()
        items = items or ["-"]
        lines.append(f"{key:<{width}}  {items[0]}")
        lines.extend(" " * (width + 2) + item for item in items[1:])
    return "\n".join(lines)


def _edit_list(
    current: list[str], add: Iterable[str], remove: Iterable[str], what: str, owner: str
) -> list[str]:
    """`current` without `remove` and with `add` appended; warns about removals that are not there."""
    add, remove = list(dict.fromkeys(add)), list(dict.fromkeys(remove))
    if both := [v for v in add if v in remove]:
        fail(f"{what} {both[0]!r} is both added and removed")
    for v in remove:
        if v not in current:
            warn(f"{owner} has no {what} {v!r}")
    return [v for v in current if v not in remove] + [v for v in add if v not in current]


def _nothing_to_change(label: str, given: bool, command: str) -> NoReturn:
    if not given:
        fail(f"nothing to change: give at least one option (see `pseudorca {command} --help`)")
    fail(f"nothing to change: {label} already has these values")


def _nb_label(st: State, nb_id: Any) -> str:
    if not nb_id:
        return "-"
    nb = st.notebooks.get(str(nb_id))
    return nb.label if nb else util.short(str(nb_id))


def _nb_ref(nb: Notebook) -> str:
    """A notebook's label, with a sha prefix when it has a name (names repeat across versions)."""
    return f"{nb.name} {util.short(nb.id, 8)}" if nb.name else nb.label


# ----- verdict cells -------------------------------------------------------------------------------


def _verdict_text(eff: Effective | None, claimed: bool) -> str:
    symbol = verdict_symbol(eff, claimed)
    if eff is None:
        return f"{symbol} {'claimed' if claimed else 'none'}"
    return f"{symbol} {eff.status}" + (", stale" if eff.stale else "")


def _when(eff: Effective | None) -> str:
    """When the newest record behind a verdict was made (compared as instants, shown in local time)."""
    if eff is None or not eff.records:
        return "-"
    return _minute(max(eff.records, key=lambda r: _instant(r.time)).time)


def _evidence(eff: Effective | None, width: int = EVIDENCE_WIDTH) -> str:
    """The first evidence quote of a verdict (else its note) on one line, cut to `width` around the match
    when the evidence says where the match starts (`col`), else from the start."""
    if eff is None:
        return "-"
    # the records behind the verdict first (an LLM that settled a conflict, not the losing method)
    backing = [r for r in eff.records if r.status == eff.status] or eff.records
    for rec in backing:
        for ev in rec.evidence:
            quote = str(ev.get("quote") or "") if isinstance(ev, dict) else ""
            if quote.strip():
                col = ev.get("col")
                usable = isinstance(col, int) and not isinstance(col, bool) and 0 <= col < len(quote)
                return util.excerpt(quote, col if usable else None, width)
    notes = [r.note for r in backing if r.note.strip()]
    return util.clip(" ".join(notes[0].split()), width) if notes else "-"


def _verdict_json(eff: Effective | None) -> dict[str, Any] | None:
    return asdict(eff) if eff is not None else None


# ----- ideas: fingerprints and patches -------------------------------------------------------------


def _check_fingerprint(fp: str) -> str:
    """Validate a fingerprint: `text`, `re:<regex>`, `!text` or `!re:<regex>`. Each may span lines (the
    audit then matches the text as consecutive lines, or searches the regex over whole files)."""
    body = fp[1:] if fp.startswith("!") else fp
    pattern = body[3:] if body.startswith("re:") else None
    if not (body if pattern is None else pattern).strip():
        fail(f"fingerprint {fp!r} has nothing to look for")
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as e:
            fail(f"invalid regex in fingerprint {fp!r}: {e}")
    return fp


def _looks_like_diff(text: str) -> bool:
    return any(line.startswith(("@@", "+++ ")) for line in text.splitlines())


def _save_patch(store: Store, f: IO[bytes]) -> str:
    """Store a patch file as a blob and return its sha256. Warns if it does not look like a unified diff."""
    data = f.read()
    if not _looks_like_diff(data.decode("utf-8", "replace")):
        name = getattr(f, "name", "-")
        warn(f"{name} does not look like a unified diff (no '@@' or '+++ ' line); stored it anyway")
    return store.put_blob(data)


def _description(text: str | None, file: IO[str] | None) -> str | None:
    """The text of --desc or --desc-file; None when neither was given."""
    if text is not None and file is not None:
        fail("use --desc or --desc-file, not both")
    if file is not None:
        return file.read().strip()
    return None if text is None else text.strip()


@dataclass
class PatchFile:
    path: str
    hunks: int = 0
    added: int = 0
    removed: int = 0


def _diff_path(raw: str) -> str:
    p = raw.split("\t")[0].strip()
    if len(p) >= 2 and p[0] == p[-1] == '"':
        p = p[1:-1]
    return p[2:] if p.startswith(("a/", "b/")) else p


def summarize_patch(text: str) -> list[PatchFile]:
    """Files, hunks and added/removed line counts of a unified diff (git or plain); a minimal parser.

    Hunk headers with line counts are followed exactly, so a removed `-- x` line is not taken for a
    `--- file` header. Hunks without counts are read up to the next header.
    """
    files: list[PatchFile] = []
    cur: PatchFile | None = None
    old_path = ""
    old_left = new_left = 0  # lines still due in the current hunk
    loose = False  # in a hunk whose header has no line counts
    for line in text.splitlines():
        if cur is not None and (old_left > 0 or new_left > 0):
            if line.startswith("+"):
                cur.added, new_left = cur.added + 1, new_left - 1
            elif line.startswith("-"):
                cur.removed, old_left = cur.removed + 1, old_left - 1
            elif not line.startswith("\\"):  # context; "\ No newline at end of file" is not a line
                old_left, new_left = old_left - 1, new_left - 1
            continue
        if line.startswith("--- "):
            old_path, loose = _diff_path(line[4:]), False
        elif line.startswith("+++ "):
            path = _diff_path(line[4:])
            cur, loose = PatchFile(old_path if path == "/dev/null" else path), False
            files.append(cur)
        elif line.startswith("@@"):
            if cur is None:
                cur = PatchFile("?")
                files.append(cur)
            cur.hunks += 1
            m = _HUNK_RE.match(line)
            loose = m is None
            if m:
                old_left, new_left = int(m.group(1) or 1), int(m.group(2) or 1)
        elif loose and cur is not None:
            if line.startswith("+"):
                cur.added += 1
            elif line.startswith("-"):
                cur.removed += 1
    return files


def _read_patch(store: Store, sha: str | None) -> list[PatchFile] | None:
    """The files of a stored patch; None when there is no patch or its blob is not in this store."""
    if not sha:
        return None
    try:
        data = store.get_blob(sha)
    except StoreError:
        return None
    return summarize_patch(data.decode("utf-8", "replace"))


def _patch_lines(sha: str | None, files: list[PatchFile] | None) -> list[str] | str:
    if not sha:
        return "-"
    if files is None:
        return f"{util.short(sha)} (blob not in this store)"
    hunks, added, removed = (sum(getattr(f, k) for f in files) for k in ("hunks", "added", "removed"))
    head = f"{util.short(sha)}  {_plural(len(files), 'file')}, {_plural(hunks, 'hunk')}, +{added} -{removed}"
    width = max((util.display_width(f.path) for f in files), default=0)
    pad = {f.path: " " * (width - util.display_width(f.path)) for f in files}
    return [
        head,
        *(f"  {f.path}{pad[f.path]}  {_plural(f.hunks, 'hunk')}  +{f.added} -{f.removed}" for f in files),
    ]


def _patch_json(sha: str | None, files: list[PatchFile] | None) -> dict[str, Any] | None:
    if not sha:
        return None
    available, files = files is not None, files or []
    return {
        "sha": sha,
        "available": available,
        "files": [asdict(f) for f in files],
        "hunks": sum(f.hunks for f in files),
        "added": sum(f.added for f in files),
        "removed": sum(f.removed for f in files),
    }


def _checks(idea: Idea) -> str:
    """What deterministic checks an idea has: "2f+p" = two fingerprints and a patch, "-" = none."""
    parts = [f"{len(idea.fingerprints)}f"] if idea.fingerprints else []
    if idea.patch:
        parts.append("p")
    return "+".join(parts) or "-"


def _select_ideas(
    st: State, statuses: Iterable[str], tag: str | None, owner: str | None, show_all: bool
) -> tuple[list[Idea], int]:
    """Ideas in board order, filtered. Parked ones are hidden (and counted) unless --all or --status asks."""
    statuses = set(statuses)
    picked: list[Idea] = []
    hidden = 0
    for idea in st.ideas_in_order():
        if (tag and tag not in idea.tags) or (owner and idea.owner != owner):
            continue
        if statuses:
            if idea.status not in statuses:
                continue
        elif idea.status == "parked" and not show_all:
            hidden += 1
            continue
        picked.append(idea)
    return picked, hidden


# ----- event summaries (log, update messages) ------------------------------------------------------

_KEY_LABELS = {"public_score": "public", "private_score": "private"}
_SUB_KEYS = ("public_score", "private_score", "nb", "status", "description")


def _value_text(key: str, value: Any, st: State) -> str:
    if value is None:
        return "none"
    if key in ("nb", "parent"):
        return _nb_label(st, value)
    if key in ("patch", "results"):
        return util.short(str(value))
    if isinstance(value, list):  # items on one line, so a multi-line fingerprint does not break the summary
        return ",".join(" ".join(str(v).split()) for v in value) or "[]"
    if isinstance(value, dict):
        return util.clip(json.dumps(value, ensure_ascii=False), 40)
    text = " ".join(str(value).split())
    return util.clip(text, 40) if text else '""'


def _fields_text(data: dict[str, Any], st: State, keys: Iterable[str] | None = None) -> str:
    keys = [k for k in data if k != "id"] if keys is None else [k for k in keys if k in data]
    return ", ".join(f"{_KEY_LABELS.get(k, k)}={_value_text(k, data[k], st)}" for k in keys)


def _event_summary(ev: Event, st: State) -> str:
    """One readable line per event."""
    d, t = ev.data, ev.type
    if t == "idea.add":
        return f"idea {d.get('id')} added: {d.get('title') or '-'}"
    if t == "idea.update":
        return f"idea {d.get('id')}: {_fields_text(d, st) or 'no changes'}"
    if t == "nb.add":
        extra = _fields_text(d, st, [k for k in ("applies", "tags", "kernel") if d.get(k)])
        name = d.get("name") or "-"
        return f"nb {name} {util.short(str(d.get('id')))}" + (f": {extra}" if extra else "")
    if t == "nb.update":
        return f"nb {_nb_label(st, d.get('id'))} {util.short(str(d.get('id')))}: {_fields_text(d, st)}"
    if t == "sub.upsert":
        return f"sub {d.get('id')}: {_fields_text(d, st, _SUB_KEYS) or _fields_text(d, st) or 'recorded'}"
    if t == "run.add":
        where = f" on {_nb_label(st, d.get('nb'))}" if d.get("nb") else ""
        return f"run {d.get('label') or util.short(str(d.get('id')))}: {d.get('n_rows', 0)} rows{where}"
    if t == "audit.verdict":
        return (
            f"verdict {d.get('status')} ({d.get('method')}): "
            f"idea {d.get('idea')} on {_nb_label(st, d.get('nb'))}"
        )
    return _fields_text(d, st) or "-"


# ----- init ----------------------------------------------------------------------------------------


def _in_git_work_tree(directory: Path) -> bool:
    """Whether `directory` or a parent has a `.git` entry (a file for worktrees and submodules)."""
    return any((d / ".git").exists() for d in (directory, *directory.parents))


def _has_entries(directory: Path) -> bool:
    try:
        return directory.is_dir() and any(directory.iterdir())
    except OSError:
        return False


_STORE_PARTS = {  # what `init` can create, in the order it lists them
    "config.toml": "settings: commit it, and never put secrets in it",
    "events/": "the ledger itself, one JSON file per change",
    "blobs/": "stored notebooks, patches and result tables",
    "cache/": "local only, ignored by git",
    ".gitignore": "keeps cache/ out of git",
    ".gitattributes": "stops git from converting blob bytes (line endings)",
}


@click.command("init")
@click.argument(
    "directory", metavar="[DIR]", required=False, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--name", default="", metavar="NAME", help="Ledger name (default: the directory's name).")
@click.option("--competition", default="", metavar="SLUG", help="Kaggle competition slug, for `kaggle sync`.")
def init_cmd(directory: Path | None, name: str, competition: str) -> None:
    """Create a ledger (.pseudorca/), or complete one that misses some of its parts.

    It goes in DIR, by default the current directory: the top of your team's private git repository.
    Existing files, such as a committed config.toml, are never overwritten.
    """
    target = Path(directory or Path.cwd()).expanduser().resolve()
    existed = _has_entries(target / STORE_DIR)
    with user_errors():
        store = Store.init(target, name=name, competition=competition)
    if not _in_git_work_tree(target):
        warn(
            f"{target} is not inside a git repository; create one (git init) so the team can share the ledger"
        )
    if existed:
        click.echo(f"completed {store.root}, keeping its existing files; added:")
    else:
        click.echo(f"created {store.root}")
    order = list(_STORE_PARTS)
    created = sorted(store.created, key=lambda p: order.index(p) if p in order else len(order))
    width = max((len(p) for p in created), default=0)
    for path in created:
        click.echo(f"  {path:<{width}}  {_STORE_PARTS.get(path, '')}".rstrip())
    unused = [opt for opt, value in (("--name", name), ("--competition", competition)) if value]
    if unused and "config.toml" not in store.created:
        warn(
            f"{' and '.join(unused)} {'was' if len(unused) == 1 else 'were'} not applied: the existing "
            "config.toml was kept; edit it if needed"
        )
    cd = "" if target == Path.cwd().resolve() else f"cd {shlex.quote(str(target))} && "
    click.echo("\nnext steps:")
    click.echo(f'  {cd}git add .pseudorca && git commit -m "Add Pseudorca ledger"')
    click.echo("  keep the repository private: the ledger will hold your notebooks and results")
    click.echo('  pseudorca idea add ID --title "..."   then   pseudorca nb add NOTEBOOK.ipynb')


# ----- idea ----------------------------------------------------------------------------------------


@click.group("idea")
def idea_group() -> None:
    """Ideas: the improvements you want in (or out of) your notebooks."""


@idea_group.command("add")
@click.argument("idea_id", metavar="ID")
@click.option("--title", required=True, help="One-line summary.")
@click.option("--desc", "description", metavar="TEXT", help="What the idea is and how to recognise it.")
@click.option(
    "--desc-file",
    type=click.File("r", encoding="utf-8"),
    metavar="PATH",
    help="Read the description from a file (- for stdin).",
)
@click.option("--owner", metavar="NAME", help="Who drives it (default: you).")
@click.option("--status", type=click.Choice(IDEA_STATUSES), default="idea", show_default=True)
@click.option("--tag", "tags", multiple=True, metavar="TAG", help="A tag (repeatable).")
@click.option("--link", "links", multiple=True, metavar="URL", help="A related link (repeatable).")
@click.option(
    "--fingerprint",
    "fingerprints",
    multiple=True,
    metavar="FINGERPRINT",
    help=f"Code that shows the idea is in a notebook: {_FP_HELP}. Repeatable.",
)
@click.option(
    "--patch",
    "patch_file",
    type=click.File("rb"),
    metavar="FILE",
    help="A unified diff that implements the idea.",
)
@pass_ctx
def idea_add(
    ctx: Ctx,
    idea_id: str,
    title: str,
    description: str | None,
    desc_file: IO[str] | None,
    owner: str | None,
    status: str,
    tags: tuple[str, ...],
    links: tuple[str, ...],
    fingerprints: tuple[str, ...],
    patch_file: IO[bytes] | None,
) -> None:
    """Add an idea. ID is a short slug such as `fast-prompt`."""
    if not util.IDEA_ID_RE.fullmatch(idea_id):
        fail(f"invalid idea id {idea_id!r}: use {_ID_HELP}")
    store = ctx.store()
    if idea_id in store.state().ideas:
        fail(f"idea {idea_id!r} already exists (change it with `pseudorca idea update {idea_id}`)")
    if not title.strip():
        fail("--title cannot be empty")
    desc = _description(description, desc_file) or ""
    data: dict[str, Any] = {
        "id": idea_id,
        "title": title.strip(),
        "description": desc,
        "owner": owner.strip() if owner is not None else store.actor(),
        "status": status,
        "tags": _clean(tags, "a tag"),
        "links": _clean(links, "a link"),
        "fingerprints": list(dict.fromkeys(_check_fingerprint(f) for f in fingerprints)),
        "patch": None,
    }
    if patch_file is not None:
        data["patch"] = _save_patch(store, patch_file)
    store.append("idea.add", data)
    idea = store.state().ideas[idea_id]
    click.echo(f"added idea {idea.id} ({idea.status}, owner {idea.owner or '-'}, checks {_checks(idea)})")


@idea_group.command("update")
@click.argument("ref", metavar="ID")
@click.option("--title", help="New one-line summary.")
@click.option("--desc", "description", metavar="TEXT", help="New description.")
@click.option(
    "--desc-file",
    type=click.File("r", encoding="utf-8"),
    metavar="PATH",
    help="Read the description from a file (- for stdin).",
)
@click.option("--owner", metavar="NAME", help="New owner.")
@click.option("--status", type=click.Choice(IDEA_STATUSES), help="New status.")
@click.option("--add-tag", multiple=True, metavar="TAG", help="Add a tag (repeatable).")
@click.option("--remove-tag", multiple=True, metavar="TAG", help="Remove a tag (repeatable).")
@click.option("--add-link", multiple=True, metavar="URL", help="Add a link (repeatable).")
@click.option("--remove-link", multiple=True, metavar="URL", help="Remove a link (repeatable).")
@click.option(
    "--add-fingerprint",
    multiple=True,
    metavar="FINGERPRINT",
    help=f"Add a fingerprint: {_FP_HELP}. Repeatable.",
)
@click.option(
    "--remove-fingerprint",
    multiple=True,
    metavar="FINGERPRINT",
    help="Remove a fingerprint, given exactly as it was added (repeatable).",
)
@click.option(
    "--patch",
    "patch_file",
    type=click.File("rb"),
    metavar="FILE",
    help="Replace the patch with this unified diff.",
)
@click.option("--no-patch", is_flag=True, help="Remove the patch.")
@pass_ctx
def idea_update(
    ctx: Ctx,
    ref: str,
    title: str | None,
    description: str | None,
    desc_file: IO[str] | None,
    owner: str | None,
    status: str | None,
    add_tag: tuple[str, ...],
    remove_tag: tuple[str, ...],
    add_link: tuple[str, ...],
    remove_link: tuple[str, ...],
    add_fingerprint: tuple[str, ...],
    remove_fingerprint: tuple[str, ...],
    patch_file: IO[bytes] | None,
    no_patch: bool,
) -> None:
    """Change an idea. Only the fields that change are recorded."""
    store = ctx.store()
    st = store.state()
    with user_errors():
        idea = st.resolve_idea(ref)
    if patch_file is not None and no_patch:
        fail("use --patch or --no-patch, not both")
    label = f"idea {idea.id}"
    changes: dict[str, Any] = {}
    if title is not None:
        if not title.strip():
            fail("--title cannot be empty")
        changes["title"] = title.strip()
    desc = _description(description, desc_file)
    if desc is not None:
        changes["description"] = desc
    if owner is not None:
        changes["owner"] = owner.strip()
    if status is not None:
        changes["status"] = status
    tags = _edit_list(idea.tags, _clean(add_tag, "a tag"), _clean(remove_tag, "a tag"), "tag", label)
    links = _edit_list(idea.links, _clean(add_link, "a link"), _clean(remove_link, "a link"), "link", label)
    added_fps = [_check_fingerprint(f) for f in add_fingerprint]
    changes["tags"], changes["links"] = tags, links
    changes["fingerprints"] = _edit_list(
        idea.fingerprints, added_fps, remove_fingerprint, "fingerprint", label
    )
    if no_patch:
        if not idea.patch:
            warn(f"{label} has no patch")
        changes["patch"] = None
    elif patch_file is not None:
        changes["patch"] = _save_patch(store, patch_file)
    changes = {k: v for k, v in changes.items() if getattr(idea, k) != v}
    if not changes:
        lists = (add_tag, remove_tag, add_link, remove_link, add_fingerprint, remove_fingerprint)
        scalars = (title, description, desc_file, owner, status, patch_file)
        given = no_patch or any(lists) or any(v is not None for v in scalars)
        _nothing_to_change(label, given, "idea update")
    store.append("idea.update", {"id": idea.id, **changes})
    click.echo(f"updated {label}: {_fields_text(changes, st)}")
    audited = sum(1 for _, iid in st.verdicts if iid == idea.id)
    if audited and dataclasses.replace(idea, **changes).rev != idea.rev:
        click.echo(
            f"note: its checks changed, so the verdicts on {_plural(audited, 'notebook')} are now stale; "
            "audit again to refresh them"
        )


@idea_group.command("list")
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice(IDEA_STATUSES),
    help="Only ideas with this status (repeatable).",
)
@click.option("--tag", metavar="TAG", help="Only ideas with this tag.")
@click.option("--owner", metavar="NAME", help="Only ideas with this owner.")
@click.option("--all", "show_all", is_flag=True, help="Include parked ideas.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def idea_list(
    ctx: Ctx, statuses: tuple[str, ...], tag: str | None, owner: str | None, show_all: bool, as_json: bool
) -> None:
    """List ideas (parked ones only with --all or --status parked)."""
    st = ctx.store().state()
    ideas, hidden = _select_ideas(st, statuses, tag, owner, show_all)
    if as_json:
        _echo_json([asdict(i) for i in ideas])
        return
    if ideas:
        rows = [[i.id, i.status, i.owner or "-", i.title, _checks(i), _day(i.updated)] for i in ideas]
        click.echo(util.format_table(["ID", "STATUS", "OWNER", "TITLE", "CHECKS", "UPDATED"], rows))
    elif not st.ideas:
        click.echo('no ideas yet: add one with `pseudorca idea add ID --title "..."`')
    else:
        click.echo("no ideas match")
    if hidden:
        click.echo(f"({_plural(hidden, 'parked idea')} hidden; use --all)")


@idea_group.command("show")
@click.argument("ref", metavar="ID")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def idea_show(ctx: Ctx, ref: str, as_json: bool) -> None:
    """Show an idea: its fields, its patch, and its verdicts on the latest notebooks."""
    store = ctx.store()
    st = store.state()
    with user_errors():
        idea = st.resolve_idea(ref)
    files = _read_patch(store, idea.patch)
    cells = [
        (nb, st.effective_verdict(nb.id, idea.id), idea.id in nb.applies)
        for nb in st.notebooks_in_order()
        if (nb.id, idea.id) in st.verdicts or idea.id in nb.applies
    ]
    if as_json:
        _echo_json(
            {
                "idea": asdict(idea),
                "rev": idea.rev,
                "patch_summary": _patch_json(idea.patch, files),
                "notebooks": [
                    {"nb": nb.id, "label": nb.label, "claimed": claimed, "verdict": _verdict_json(eff)}
                    for nb, eff, claimed in cells
                ],
            }
        )
        return
    click.echo(
        _kv(
            [
                ("idea", idea.id),
                ("title", idea.title or "-"),
                ("status", idea.status),
                ("owner", idea.owner or "-"),
                ("tags", _csv(idea.tags)),
                ("links", idea.links),
                ("fingerprints", idea.fingerprints),
                ("patch", _patch_lines(idea.patch, files)),
                ("rev", idea.rev),
                ("created", f"{_minute(idea.created)} by {idea.created_by or '-'}"),
                ("updated", f"{_minute(idea.updated)} by {idea.updated_by or '-'}"),
                ("description", idea.description or "-"),
            ]
        )
    )
    click.echo("")
    shown = cells[-SHOW_LATEST:]
    if not shown:
        click.echo("no verdicts yet, and no notebook claims it (run `pseudorca audit`)")
        return
    if len(cells) > len(shown):
        click.echo(f"latest {len(shown)} of {len(cells)} notebooks:")
    rows = [
        [_nb_ref(nb), _verdict_text(eff, claimed), eff.method if eff else "-", _when(eff), _evidence(eff)]
        for nb, eff, claimed in shown
    ]
    click.echo(util.format_table(["NOTEBOOK", "VERDICT", "METHOD", "WHEN", "EVIDENCE"], rows))


# ----- nb ------------------------------------------------------------------------------------------


@click.group("nb")
def nb_group() -> None:
    """Notebook versions: register, list, show and inspect them."""


def _check_kernel(kernel: str | None) -> str | None:
    if kernel is None:
        return None
    kernel = kernel.strip()
    if kernel and not _KERNEL_RE.fullmatch(kernel):
        fail(f"--kernel {kernel!r} is not a Kaggle kernel ref (OWNER/SLUG)")
    return kernel


@nb_group.command("add")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", metavar="NAME", help="Name to refer to it by (default: the file name).")
@click.option("--parent", metavar="REF", help="The notebook it was derived from.")
@click.option("--applies", multiple=True, metavar="ID", help="An idea its author says is in it (repeatable).")
@click.option("--tag", "tags", multiple=True, metavar="TAG", help="A tag such as `candidate` (repeatable).")
@click.option("--kernel", metavar="OWNER/SLUG", help="Its Kaggle kernel.")
@click.option("--version", help="Its Kaggle kernel version.")
@click.option("--note", metavar="TEXT", help="Free text.")
@click.option(
    "--no-store",
    is_flag=True,
    help="Record only its hash and path, not its bytes. The file must be inside the repository: audits "
    "read it there while it keeps this content.",
)
@pass_ctx
def nb_add(
    ctx: Ctx,
    file: Path,
    name: str | None,
    parent: str | None,
    applies: tuple[str, ...],
    tags: tuple[str, ...],
    kernel: str | None,
    version: str | None,
    note: str | None,
    no_store: bool,
) -> None:
    """Register a notebook (.ipynb) or script (.py) version.

    Adding the same bytes again merges the new metadata into the existing entry (and restores its stored
    bytes if they went missing). An .ipynb must be notebook JSON, not a Git LFS pointer.
    """
    store = ctx.store()
    st = store.state()
    if file.suffix.lower() not in (".ipynb", ".py"):
        warn(f"{file.name} is neither a .ipynb nor a .py file; audits may not be able to read it")
    if name is not None and not name.strip():
        fail("--name cannot be empty")
    with user_errors():
        parent_id = st.resolve_nb(parent).id if parent else None
        idea_ids = [st.resolve_idea(a).id for a in applies]
    sha = util.sha256_file(file)
    before = st.notebooks.get(sha)
    old = asdict(before) if before else None
    had_bytes = store.has_blob(sha)
    with user_errors():
        nb, created = register_file(
            store,
            file,
            name=name.strip() if name else None,
            parent=parent_id,
            applies=idea_ids,
            tags=_clean(tags, "a tag"),
            kernel=_check_kernel(kernel) or "",
            version=(version or "").strip(),
            note=(note or "").strip(),
            keep_bytes=not no_store,
        )
    ref = f"{nb.label} {util.short(nb.id)}"
    if created:
        click.echo(f"registered {ref}")
        if no_store:  # register_file refuses a file outside the repository: audits could not find it again
            where = f"{nb.path} in the repository"
            click.echo(f"  bytes not stored: audits read {where} while it keeps this content")
        return
    if name and name.strip() != nb.name:
        warn(
            f"kept the name {nb.name!r}; rename it with `pseudorca nb update {util.short(nb.id)} --name ...`"
        )
    new = asdict(nb)
    changed = [k for k in NOTEBOOK_FIELDS if k != "stored" and old is not None and old[k] != new[k]]
    done = [f"updated {', '.join(changed)}"] if changed else []
    if not no_store and not had_bytes and store.has_blob(nb.id):  # register_file stored or restored them
        missing = old is not None and old["stored"]
        done.append("stored its bytes, which were missing" if missing else "stored its bytes")
    click.echo(f"already registered as {ref}" + (f"; {'; '.join(done)}" if done else " (no changes)"))


@nb_group.command("list")
@click.option("--tag", metavar="TAG", help="Only notebooks with this tag.")
@click.option(
    "-n",
    "count",
    type=click.IntRange(min=0),
    default=0,
    metavar="N",
    help="Show only the latest N (0 = all).",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def nb_list(ctx: Ctx, tag: str | None, count: int, as_json: bool) -> None:
    """List registered notebooks, newest last."""
    st = ctx.store().state()
    nbs = [nb for nb in st.notebooks_in_order() if not tag or tag in nb.tags]
    total = len(nbs)
    if count:
        nbs = nbs[-count:]
    if as_json:
        _echo_json([asdict(nb) for nb in nbs])
        return
    if not nbs:
        click.echo(
            "no notebooks match"
            if st.notebooks
            else "no notebooks yet: register one with `pseudorca nb add FILE`"
        )
        return
    rows = [
        [
            nb.seq,
            util.short(nb.id),
            nb.name or "-",
            _csv(nb.tags),
            _nb_label(st, nb.parent),
            len(nb.applies),
            _day(nb.added),
            nb.added_by or "-",
        ]
        for nb in nbs
    ]
    click.echo(util.format_table(["#", "SHA", "NAME", "TAGS", "PARENT", "APPLIES", "ADDED", "BY"], rows))
    if total > len(nbs):
        click.echo(f"({_plural(total - len(nbs), 'older notebook')} not shown)")


@nb_group.command("show")
@click.argument("ref", metavar="REF")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def nb_show(ctx: Ctx, ref: str, as_json: bool) -> None:
    """Show a notebook, and the ideas it claims or has verdicts for.

    REF is a sha256 (or a unique prefix of 6+ characters), `latest`, `latest:<tag>`, a name or a kernel ref.
    """
    st = ctx.store().state()
    with user_errors():
        nb = st.resolve_nb(ref)
    children = [c for c in st.notebooks_in_order() if c.parent == nb.id]
    subs = sorted((s for s in st.subs.values() if s.nb == nb.id), key=_sub_order)
    rank = {idea.id: i for i, idea in enumerate(st.ideas_in_order())}
    idea_ids = {iid for nid, iid in st.verdicts if nid == nb.id} | set(nb.applies)
    rows = []
    for iid in sorted(idea_ids, key=lambda i: (rank.get(i, len(rank)), i)):
        idea = st.ideas.get(iid)
        rows.append((iid, idea.status if idea else "-", st.effective_verdict(nb.id, iid), iid in nb.applies))
    if as_json:
        _echo_json(
            {
                "notebook": asdict(nb),
                "children": [c.id for c in children],
                "submissions": [asdict(s) for s in subs],
                "ideas": [
                    {"idea": iid, "idea_status": status, "claimed": claimed, "verdict": _verdict_json(eff)}
                    for iid, status, eff, claimed in rows
                ],
            }
        )
        return
    parent = f"{_nb_label(st, nb.parent)} {util.short(nb.parent)}" if nb.parent else "-"
    click.echo(
        _kv(
            [
                ("notebook", f"{nb.label}  (#{nb.seq})"),
                ("sha256", nb.id),
                ("path", nb.path or "-"),
                ("kernel", nb.kernel or "-"),
                ("version", nb.version or "-"),
                ("parent", parent),
                ("children", _csv(f"{c.label} {util.short(c.id)}" for c in children)),
                ("applies", _csv(nb.applies)),
                ("tags", _csv(nb.tags)),
                ("note", nb.note or "-"),
                (
                    "stored",
                    "yes" if nb.stored else "no (added with --no-store; audits read the file at its path)",
                ),
                ("size", f"{nb.size or 0:,} bytes"),
                ("added", f"{_minute(nb.added)} by {nb.added_by or '-'}"),
                ("submissions", _csv(f"{s.id} (public {_score(s.public_score)})" for s in subs)),
            ]
        )
    )
    click.echo("")
    if not rows:
        click.echo("no verdicts yet, and it claims no ideas (run `pseudorca audit`)")
        return
    table = [
        [iid, status, _verdict_text(eff, claimed), eff.method if eff else "-", _evidence(eff)]
        for iid, status, eff, claimed in rows
    ]
    click.echo(util.format_table(["IDEA", "IDEA STATUS", "VERDICT", "METHOD", "EVIDENCE"], table))


def _parent_loop(st: State, nb_id: str, parent_id: str) -> bool:
    """Whether making `parent_id` the parent of `nb_id` would close a loop in the lineage."""
    seen: set[str] = set()
    cur: str | None = parent_id
    while cur and cur not in seen:
        if cur == nb_id:
            return True
        seen.add(cur)
        p = st.notebooks.get(cur)
        cur = p.parent if p else None
    return False


@nb_group.command("update")
@click.argument("ref", metavar="REF")
@click.option("--name", metavar="NAME", help="New name.")
@click.option("--parent", metavar="REF", help="The notebook it was derived from.")
@click.option("--no-parent", is_flag=True, help="Remove the parent.")
@click.option(
    "--add-applies", multiple=True, metavar="ID", help="An idea its author says is in it (repeatable)."
)
@click.option("--remove-applies", multiple=True, metavar="ID", help="Remove a claimed idea (repeatable).")
@click.option("--add-tag", multiple=True, metavar="TAG", help="Add a tag (repeatable).")
@click.option("--remove-tag", multiple=True, metavar="TAG", help="Remove a tag (repeatable).")
@click.option("--kernel", metavar="OWNER/SLUG", help='Its Kaggle kernel ("" clears it).')
@click.option("--version", help="Its Kaggle kernel version.")
@click.option("--note", metavar="TEXT", help="Free text.")
@pass_ctx
def nb_update(
    ctx: Ctx,
    ref: str,
    name: str | None,
    parent: str | None,
    no_parent: bool,
    add_applies: tuple[str, ...],
    remove_applies: tuple[str, ...],
    add_tag: tuple[str, ...],
    remove_tag: tuple[str, ...],
    kernel: str | None,
    version: str | None,
    note: str | None,
) -> None:
    """Change a notebook's metadata. Only the fields that change are recorded."""
    store = ctx.store()
    st = store.state()
    if parent and no_parent:
        fail("use --parent or --no-parent, not both")
    with user_errors():
        nb = st.resolve_nb(ref)
        new_parent = st.resolve_nb(parent) if parent else None
        added = [st.resolve_idea(a).id for a in add_applies]
        removed = [a if a in nb.applies else st.resolve_idea(a).id for a in remove_applies]
    label = f"{nb.label} {util.short(nb.id)}"
    changes: dict[str, Any] = {}
    if name is not None:
        if not name.strip():
            fail("--name cannot be empty")
        changes["name"] = name.strip()
    if new_parent is not None:
        if new_parent.id == nb.id:
            fail("a notebook cannot be its own parent")
        if _parent_loop(st, nb.id, new_parent.id):
            fail(f"{new_parent.label} descends from {nb.label}; making it the parent would create a loop")
        changes["parent"] = new_parent.id
    if no_parent:
        changes["parent"] = None
    changes["applies"] = _edit_list(nb.applies, added, removed, "claimed idea", label)
    changes["tags"] = _edit_list(nb.tags, _clean(add_tag, "a tag"), _clean(remove_tag, "a tag"), "tag", label)
    if kernel is not None:
        changes["kernel"] = _check_kernel(kernel)
    for key, value in (("version", version), ("note", note)):
        if value is not None:
            changes[key] = value.strip()
    changes = {k: v for k, v in changes.items() if getattr(nb, k) != v}
    if not changes:
        lists = (add_applies, remove_applies, add_tag, remove_tag)
        given = no_parent or any(lists) or any(v is not None for v in (name, parent, kernel, version, note))
        _nothing_to_change(label, given, "nb update")
    store.append("nb.update", {"id": nb.id, **changes})
    click.echo(f"updated {label}: {_fields_text(changes, store.state())}")


def _excerpt(line: str, needle: str, ignore_case: bool, width: int = 200) -> str:
    """A long line cut down to a window around the first match."""
    line = line.rstrip()
    if len(line) <= width:
        return line
    hay, pat = (line.lower(), needle.lower()) if ignore_case else (line, needle)
    start = max(0, hay.find(pat) - width // 4)
    end = start + width
    return ("…" if start else "") + line[start:end] + ("…" if end < len(line) else "")


@nb_group.command("tree")
@click.argument("ref", metavar="REF")
@click.option(
    "--grep", "needle", metavar="TEXT", help="Show the lines that contain TEXT (plain text, not a regex)."
)
@click.option(
    "--path", "glob", default="*", metavar="GLOB", help="Only paths matching GLOB (`*` also matches `/`)."
)
@click.option("-i", "--ignore-case", is_flag=True, help="Case-insensitive --grep.")
@pass_ctx
def nb_tree(ctx: Ctx, ref: str, needle: str | None, glob: str, ignore_case: bool) -> None:
    """List the files extracted from a notebook, or --grep them.

    Extraction is static: nothing in the notebook is run.
    """
    from pseudorca.extract import tree_for

    store = ctx.store()
    with user_errors():
        nb = store.state().resolve_nb(ref)
        try:
            tree = tree_for(store, nb)
        except OSError as e:
            fail(f"cannot read {nb.label}: {e}")
    if needle is not None:
        if not needle:
            fail("--grep needs some text")
        hits = tree.grep(needle, ignore_case=ignore_case, path_glob=glob, max_hits=MAX_GREP_HITS)
        for h in hits:
            click.echo(f"{h.path}:{h.line}: {_excerpt(h.text, needle, ignore_case)}")
        if len(hits) >= MAX_GREP_HITS:
            warn(f"stopped after {MAX_GREP_HITS} matches; narrow the search with --path")
        if not hits:
            click.echo(f"no line contains {needle!r}", err=True)
        return
    paths = tree.paths(glob)
    if paths:
        rows = []
        for p in paths:
            f = tree.files[p]
            rows.append([p, _size(f.size) + (" (binary)" if f.binary else ""), f.origin])
        click.echo(util.format_table(["PATH", "SIZE", "ORIGIN"], rows, max_width=80))
    else:
        click.echo(f"no files match {glob!r}" if glob != "*" else "no files")
    for w in tree.warnings:
        warn(w)


# ----- status --------------------------------------------------------------------------------------


@click.command("status")
@click.option("--nb", "nb_refs", multiple=True, metavar="REF", help="A notebook to show (repeatable).")
@click.option(
    "-n",
    "count",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    metavar="N",
    help="Show the latest N notebooks (0 = all).",
)
@click.option(
    "--status",
    "statuses",
    multiple=True,
    type=click.Choice(IDEA_STATUSES),
    help="Only ideas with this status (repeatable).",
)
@click.option("--tag", metavar="TAG", help="Only ideas with this tag.")
@click.option("--all", "show_all", is_flag=True, help="Include parked ideas.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def status_cmd(
    ctx: Ctx,
    nb_refs: tuple[str, ...],
    count: int,
    statuses: tuple[str, ...],
    tag: str | None,
    show_all: bool,
    as_json: bool,
) -> None:
    """The board: which ideas are in which notebooks.

    Columns are the notebooks given with --nb, or the latest N, oldest first.
    """
    store = ctx.store()
    st = store.state()
    if nb_refs:
        with user_errors():
            cols = list({nb.id: nb for nb in (st.resolve_nb(r) for r in nb_refs)}.values())
    else:
        ordered = st.notebooks_in_order()
        cols = ordered[-count:] if count else ordered
    ideas, hidden = _select_ideas(st, statuses, tag, None, show_all)
    grid = [[(st.effective_verdict(nb.id, idea.id), idea.id in nb.applies) for nb in cols] for idea in ideas]
    if as_json:
        rows = []
        for idea, cells in zip(ideas, grid, strict=True):
            row = []
            for nb, (eff, claimed) in zip(cols, cells, strict=True):
                row.append(
                    {
                        "nb": nb.id,
                        "symbol": verdict_symbol(eff, claimed),
                        "status": eff.status if eff else None,
                        "method": eff.method if eff else None,
                        "stale": bool(eff and eff.stale),
                        "claimed": claimed,
                    }
                )
            rows.append({"idea": asdict(idea), "cells": row})
        _echo_json({"notebooks": [asdict(nb) for nb in cols], "rows": rows})
    elif not ideas:
        click.echo(
            "no ideas match"
            if st.ideas
            else 'no ideas yet: add one with `pseudorca idea add ID --title "..."`'
        )
        if hidden:
            click.echo(f"({_plural(hidden, 'parked idea')} hidden; use --all)")
    else:
        heads = ["IDEA", "STATUS", *(f"{i} {util.clip(nb.label, 12)}" for i, nb in enumerate(cols, 1))]
        table = [
            [idea.id, idea.status, *(verdict_symbol(eff, claimed) for eff, claimed in cells)]
            for idea, cells in zip(ideas, grid, strict=True)
        ]
        click.echo(util.format_table(heads, table))
        if hidden:
            click.echo(f"({_plural(hidden, 'parked idea')} hidden; use --all)")
        click.echo("")
        if cols:
            click.echo(LEGEND)
            key = [[i, nb.label, util.short(nb.id), _day(nb.added)] for i, nb in enumerate(cols, 1)]
            click.echo(util.format_table(["#", "NOTEBOOK", "SHA", "ADDED"], key))
        else:
            click.echo("no notebooks yet: register one with `pseudorca nb add FILE`")
    show_warnings(store)


# ----- sub -----------------------------------------------------------------------------------------


def _sub_order(s: Submission) -> tuple[datetime, int, str]:
    """By date (as instants: Kaggle writes UTC, `sub add` a local date), then by numeric id."""
    return (_instant(s.date), int(s.id) if s.id.isascii() and s.id.isdigit() else -1, s.id)


@click.group("sub")
def sub_group() -> None:
    """Leaderboard submissions and their scores."""


@sub_group.command("add")
@click.argument("sub_id", metavar="ID")
@click.option("--nb", "nb_ref", metavar="REF", help="The notebook it was made from.")
@click.option("--score", type=float, help="Public leaderboard score.")
@click.option("--private-score", type=float, help="Private leaderboard score.")
@click.option("--desc", "description", metavar="TEXT", help="Its description.")
@click.option("--date", "when", metavar="DATE", help="When it was submitted, YYYY-MM-DD (default: today).")
@click.option("--status", metavar="TEXT", help="Its Kaggle status (free text): complete, pending or error.")
@click.option("--competition", metavar="SLUG", help="Default: [store].competition in config.toml.")
@pass_ctx
def sub_add(
    ctx: Ctx,
    sub_id: str,
    nb_ref: str | None,
    score: float | None,
    private_score: float | None,
    description: str | None,
    when: str | None,
    status: str | None,
    competition: str | None,
) -> None:
    """Record a submission by hand, or change one already recorded."""
    if not util.IDEA_ID_RE.fullmatch(sub_id):
        fail(f"invalid submission id {sub_id!r}: use {_ID_HELP}")
    for option, value in (("--score", score), ("--private-score", private_score)):
        if value is not None and not math.isfinite(value):
            fail(f"{option} must be a finite number")
    if when is not None:
        try:
            datetime.fromisoformat(when.strip())
        except ValueError:
            fail(f"--date {when!r} is not an ISO date such as 2026-01-31")
    store = ctx.store()
    st = store.state()
    with user_errors():
        nb_id = st.resolve_nb(nb_ref).id if nb_ref else None
    given = {
        "public_score": score,
        "private_score": private_score,
        "nb": nb_id,
        "status": status.strip() if status is not None else None,
        "description": description.strip() if description is not None else None,
        "date": when.strip() if when is not None else None,
        "competition": competition.strip() if competition is not None else None,
    }
    given = {k: v for k, v in given.items() if v is not None}
    existing = st.subs.get(sub_id)
    if existing is None:
        data = {
            "id": sub_id,
            **given,
            "date": given.get("date") or date.today().isoformat(),
            "competition": given.get("competition", str(store.setting("store", "competition", "") or "")),
            "submitted_by": store.actor(),
            "source": "manual",
        }
        store.append("sub.upsert", data)
        click.echo(
            f"added submission {sub_id}: {_fields_text(data, store.state(), _SUB_KEYS) or 'no score yet'}"
        )
        return
    changes = {k: v for k, v in given.items() if getattr(existing, k) != v}
    if not changes:
        fail(f"nothing to change: submission {sub_id} already has these values")
    store.append("sub.upsert", {"id": sub_id, **changes})
    click.echo(f"updated submission {sub_id}: {_fields_text(changes, store.state())}")


@sub_group.command("list")
@click.option(
    "-n",
    "count",
    type=click.IntRange(min=0),
    default=0,
    metavar="N",
    help="Show only the latest N (0 = all).",
)
@click.option("--nb", "nb_ref", metavar="REF", help="Only submissions made from this notebook.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@pass_ctx
def sub_list(ctx: Ctx, count: int, nb_ref: str | None, as_json: bool) -> None:
    """List submissions by date, newest last."""
    st = ctx.store().state()
    subs = sorted(st.subs.values(), key=_sub_order)
    if nb_ref:
        with user_errors():
            nb_id = st.resolve_nb(nb_ref).id
        subs = [s for s in subs if s.nb == nb_id]
    total = len(subs)
    if count:
        subs = subs[-count:]
    if as_json:
        _echo_json([asdict(s) for s in subs])
        return
    if not subs:
        click.echo(
            "no submissions match" if st.subs else "no submissions yet (`pseudorca sub add` or `kaggle sync`)"
        )
        return
    rows = [
        [
            s.id,
            _minute(s.date),
            _nb_label(st, s.nb),
            _score(s.public_score),
            _score(s.private_score),
            s.status or "-",
            s.submitted_by or "-",
            s.description or "-",
        ]
        for s in subs
    ]
    heads = ["ID", "DATE", "NOTEBOOK", "PUBLIC", "PRIVATE", "STATUS", "BY", "DESCRIPTION"]
    click.echo(util.format_table(heads, rows))
    if total > len(subs):
        click.echo(f"({_plural(total - len(subs), 'older submission')} not shown)")


@sub_group.command("link")
@click.argument("sub_id", metavar="ID")
@click.argument("nb_ref", metavar="REF")
@pass_ctx
def sub_link(ctx: Ctx, sub_id: str, nb_ref: str) -> None:
    """Link submission ID to the notebook REF it was made from."""
    store = ctx.store()
    st = store.state()
    sub = st.subs.get(sub_id)
    if sub is None:
        fail(f"no submission {sub_id!r} (record it with `pseudorca sub add` or `pseudorca kaggle sync`)")
    with user_errors():
        nb = st.resolve_nb(nb_ref)
    ref = f"{nb.label} {util.short(nb.id)}"
    if sub.nb == nb.id:
        click.echo(f"submission {sub_id} is already linked to {ref}")
        return
    store.append("sub.upsert", {"id": sub_id, "nb": nb.id})
    click.echo(f"linked submission {sub_id} to {ref}")


# ----- log -----------------------------------------------------------------------------------------


@click.command("log")
@click.option(
    "-n",
    "count",
    type=click.IntRange(min=0),
    default=20,
    show_default=True,
    metavar="N",
    help="Events to show (0 = all).",
)
@click.option(
    "--type",
    "event_type",
    metavar="TYPE",
    help="Only this event type (e.g. idea.update) or family (e.g. idea).",
)
@click.option("--actor", metavar="NAME", help="Only events by this author.")
@click.option("--json", "as_json", is_flag=True, help="Print JSON (the events as stored, plus a summary).")
@pass_ctx
def log_cmd(ctx: Ctx, count: int, event_type: str | None, actor: str | None, as_json: bool) -> None:
    """Recent changes to the ledger, newest first. Times are in your time zone."""
    store = ctx.store()
    st = store.state()
    events = store.events()
    if event_type:
        families = {t.split(".")[0] for t in EVENT_TYPES}
        if event_type not in EVENT_TYPES and event_type not in families:
            fail(f"unknown event type {event_type!r}; use one of {', '.join(sorted(EVENT_TYPES | families))}")
        events = [e for e in events if e.type == event_type or e.type.startswith(event_type + ".")]
    if actor:
        events = [e for e in events if e.actor == actor]
    events = events[::-1][:count] if count else events[::-1]
    if as_json:
        _echo_json([{**e.to_json(), "summary": _event_summary(e, st)} for e in events])
        return
    if not events:
        click.echo("no events")
        return
    rows = [[_minute(e.time), e.actor, e.type, _event_summary(e, st)] for e in events]
    click.echo(util.format_table(["TIME", "ACTOR", "TYPE", "SUMMARY"], rows, max_width=90))
