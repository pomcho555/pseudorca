# Pseudorca design

Pseudorca is an experiment ledger for LLM work that does not train models: prompt engineering, agent
harness changes and inference speed-ups, as done in Kaggle competitions. It answers two questions for a team:

1. **Which improvements are in which notebook?** A ledger of ideas, notebook versions and submissions. An
   audit reads each notebook without running it and decides, idea by idea, whether the idea is in it. It
   uses deterministic checks first and an LLM only when they are inconclusive.
2. **Does a change beat the noise?** Paired, per-item comparisons of local runs with bootstrap intervals,
   and the leaderboard noise estimated from repeated submissions.

## Principles

1. **Static only.** Notebooks and scripts are never executed, imported or unpickled. Embedded payloads
   are decoded with size, depth and member limits.
2. **The ledger is team-private.** It lives in `.pseudorca/` inside the team's private git repository.
   There is no server, no hosted viewer and no telemetry. Each team has its own store.
3. **Code leaves the machine only when allowed.** An LLM endpoint that is not on localhost needs
   `allow_remote = true`. Secrets are redacted from everything sent to an LLM.
4. **Kaggle access is read-only.** Pseudorca lists submissions and pulls kernel sources. It never submits,
   pushes or creates anything.
5. **LLM claims are checked.** Every quote an LLM gives as evidence must exist in the extracted tree.
   Otherwise the verdict drops to `unknown`.
6. **Noise-aware by default.** Comparisons pair by item, average the draws within each item, and report
   intervals over items.
7. **This repository holds generic code only.** Tests and examples use synthetic data.

## Store

```
.pseudorca/
  config.toml                          settings (commit it; never put secrets in it)
  events/YYYY-MM/<ulid>-<actor>.json   one immutable event per file (actor slugged for the file name)
  blobs/<sha256[:2]>/<sha256>          notebooks, patches, result tables
  cache/                               local only (extracted trees, LLM transcripts, Kaggle pulls); gitignored
  .gitattributes                       blobs/** -text -diff, so git never rewrites blob bytes
```

Every event file has this shape:
`{"schema": 1, "id": <ULID>, "type": ..., "time": <local ISO time>, "actor": ..., "data": {...}}`.

- Events are never edited. A correction is a new event.
- Replaying events in id order rebuilds the state.
- **Order is causal first.** A new event's ULID is later than both the wall clock and every event the
  writer's store already holds (a hybrid logical clock).
  - So an edit made after pulling a teammate's change wins over it, even when the writer's clock is
    behind.
  - When the newest event is more than a minute ahead of the local clock, a warning is shown.
  - The `time` field keeps the writer's wall clock (with its offset) for display. Displays convert it to
    the viewer's time zone.
- **Conflicts:** concurrent edits resolve last-writer-wins per field, and lists are replaced whole.
  - Exception: a second `nb.add` of the same notebook merges. This covers both registering it on two clones
    before a pull and re-adding it later with more metadata.
    - Tags and applies are unioned, and empty fields are filled.
    - Another name becomes an alias that still resolves, and a different note is appended.
    - Nothing already recorded is wiped.
    - Re-registering therefore writes an `nb.add` that carries only what it adds. `nb update` is for
      deliberate changes.
- **Forward compatibility:** unknown event types, unknown fields and a newer `schema` are ignored with
  one warning each, so older versions can read newer stores.
- **Validation:**
  - Each event is checked before it changes the state. A malformed value is dropped with a warning:
    a non-sha256 `patch` or `results`, or a score that is not a finite number.
  - Blob ids must be sha256 before they become paths, and blob bytes are checked against their name on
    read. A Git LFS pointer or a line-ending conversion is an error, not a wrong verdict.
  - Adding the original file again rewrites a missing or damaged blob.
  - The local cache is untrusted too:
    - cache folders must not be symlinks or lead outside `.pseudorca/cache`;
    - a cached tree is used only when its content names are sha256 hashes of regular files.
- Because every event is a new file, git merges never conflict.

| type | data |
|---|---|
| `idea.add` | `id`, `title`, `description`, `owner`, `status`, `tags[]`, `links[]`, `fingerprints[]`, `patch` (blob sha or null) |
| `idea.update` | `id` plus any idea fields to change |
| `nb.add` | `id` (sha256 of the bytes), `name`, `path`, `kernel`, `version`, `parent`, `applies[]`, `tags[]`, `note`, `stored`, `size` |
| `nb.update` | `id` plus any of `name`, `path`, `kernel`, `version`, `parent`, `applies`, `tags`, `note`, `stored` |
| `sub.upsert` | `id` plus any of `competition`, `nb`, `description`, `date`, `status`, `public_score`, `private_score`, `submitted_by`, `source`, `extra{}` |
| `run.add` | `id` (ULID), `results` (blob sha), `format` (`csv`/`jsonl`), `nb`, `label`, `protocol`, `note`, `n_rows`, `file` (source file name) |
| `audit.verdict` | `nb`, `idea`, `idea_rev`, `status`, `method`, `evidence[{path,line,quote}]`, `note`, `model`, `detail{}` |

**Actor.** The author of new events comes from `--as`, then `$PSEUDORCA_ACTOR`, then git `user.name`, then
the OS user.

## Entities (`pseudorca.model`)

- **Idea status:** `idea` → `testing` → `adopted`, `rejected` or `parked`.
- **`Idea.rev`:** a hash of `title`, `description`, `fingerprints` and `patch`. A verdict made for an
  older rev is shown as stale (`*`).
- **Notebook references** resolve in this order:
  1. full sha256;
  2. `latest`, or `latest:<tag>`;
  3. name (the newest notebook with that name);
  4. Kaggle kernel ref;
  5. a unique sha prefix of 6 or more characters.

**Effective verdict** for a (notebook, idea) pair. Each rule applies only if the ones before it did not.
1. The latest `human` verdict.
2. The deterministic methods (`fingerprint`, `patch`) for the current rev: if they agree on `present`
   or `absent` and none says `partial`, that status.
3. A non-`unknown` `llm` verdict. This also settles a fingerprint/patch contradiction.
4. `conflict` if the deterministic methods contradict each other.
5. `partial` if a deterministic method said so.
6. `unknown`.

For each machine method, the record that counts is the one made by the newest code, and the latest
among equals:
- for fingerprint and patch, the highest (extractor version, matcher version);
- for the LLM, the highest prompt version.

A teammate on an older Pseudorca therefore cannot override a newer one's verdict. The audit engine
records these versions in each verdict's `detail`.

## Tree and extraction (`pseudorca.tree`, `pseudorca.extract`)

`Tree` is a virtual, read-only file tree. Its paths never become file names on disk, and the cache
stores contents under their sha256.

| path | content |
|---|---|
| `cells/cell_004.py` | code cell 4, with IPython magics turned into plain Python |
| `cells/cell_004.sh` | a `%%bash` or `%%sh` cell |
| `cells/cell_004.<magic>.txt` | any other cell magic |
| `markdown/cell_000.md` | a markdown cell: context only, never evidence |
| `files/<path>` | a file the notebook writes with a literal path and content |
| `payloads/cell_004/<n>.<ext>` | a decoded embedded payload |
| `payloads/cell_004/<n>/<member>` | a member of an embedded archive |
| `script.py`, `payloads/script/...` | a script kernel and its payloads |

Every file records its origin, for example `cell 4 > line 12 str > base64 > lzma > tar member src/a.py`.

**Format**
- Only a name ending in `.ipynb` (notebook) or `.py` (script) picks the format, and the registered path is
  tried before the name.
- Any other suffix is a label: `v1.5`, `qwen2.5-7b`, `solver v2.1`. In that case the bytes decide: a JSON
  object with `cells` or `worksheets` is a notebook, anything else is a script.
- nbformat 4 is read fully, and nbformat 3 on a best-effort basis. Code cells go to `cells/`, markdown
  cells to `markdown/`, and outputs are ignored.

**Magics**
- A cell magic receives its body the way IPython passes it, ending with a newline even when the cell's
  last line has none.
- `%%writefile [-a] PATH` becomes `files/PATH`, and `-a` appends.
- Lines in `files/` are numbered as in the written file. An append continues from that file's last line
  break.
- `%%bash`, `%%sh` and `%%script bash` become `.sh` files. Other cell magics become `.txt` files.
  - In both, the lines up to and including the magic line are kept, the magic line as `# %%bash`. Line N
    of the file is therefore line N of the cell.
  - Their bodies are scanned for payloads by pattern, not parsed.
- Line magics and `!cmd` are commented out so the cell still parses. Inside an indented block the line
  becomes `pass  # <line>`, so the block is not left empty. Line numbers are kept.
- Literal environment settings are collected in `tree.meta["env"]` as `[{cell, key, value}]`. They come
  from `%env K=V`, `%set_env K V` and `os.environ["K"] = "V"`.

**File writes**
- A literal path written with literal text becomes `files/<path>`.
- This covers `Path(p).write_text(s)`, `open(p, "w").write(s)` and `with open(p, "w") as f: f.write(s)`.
- What a write puts in a file is searched for payloads like a `%%writefile` body: as Python for a `.py`
  path, and by the pattern scan otherwise.
- Bytes written with `write_bytes` or in a binary mode are searched when they are text.
- A written file counts as one nesting level.

**Payloads**
- Every `str` or `bytes` constant of at least `min_payload_chars` is tried in three steps:
  1. Text decoding: base64 (standard or urlsafe, whitespace ignored), base85, ascii85 or hex.
  2. Decompression, chosen by magic bytes: zlib, gzip, xz or lzma-alone, or bz2.
  3. Archives: tar (any compression) or zip.
- Text results become files. Binary results are recorded without content.
- A pickle is recorded as binary and never loaded.
- **Limits** are enforced while decompressing. `Limits` caps:
  - bytes per payload (`max_payload_bytes`);
  - bytes per notebook (`max_total_bytes`);
  - decode steps (`max_depth`);
  - archive members (`max_members`).
- Up to 64 concatenated compressed streams are followed per payload. A stream that decodes to nothing
  ends the payload. Input is fed in 64 KiB windows, so time stays linear.
- Decoded or written Python is searched again for payloads and writes, up to `max_nesting` times. Python
  longer than `max_parse_chars` is searched by pattern instead of parsed.
- The pattern scan for quoted strings runs in linear time. It is used for code that does not parse, for
  shell and other cell magics, and for written files that are not `.py`. A one-quote string ends at a line
  end that no backslash escapes, as in Python's tokenizer.
- **Decoder memory is bounded too.**
  - An xz or lzma header that declares a dictionary of 128 MiB or more has that dictionary lowered to what
    the output cap needs (96 MiB at the default `max_payload_bytes`). A decoder never looks back further
    than it has decoded, so the output is unchanged.
  - Only headers liblzma accepts are lowered, and an xz stream's blocks are followed up to 256 per stream.
  - A stream that would still need more is kept as a binary `.xz`/`.lzma` entry with a warning.
- Origins name the xz container step `lzma`.

**Never:** `exec`, `eval`, importing notebook code, `pickle.loads`, or writing tree paths to disk.

**Line numbers** count only `\n`, `\r\n` and `\r` (`tree.split_lines`), as Python's tokenizer does.

**`tree_for(store, nb)`** caches in `cache/trees/<nb.id>/`.
- The cache is valid while `meta["extractor_version"] == EXTRACTOR_VERSION` (now 3).
- Links inside a cache entry (`manifest.json`, its temporary file, entries of `files/`) are removed, never
  followed, before the entry is read or written. A rebuild prunes only regular files directly inside
  `files/`.
- Cached text keeps its line breaks (`\n`, `\r\n` or `\r`) exactly.
- The bytes come from the blob.
- If the notebook was added with `--no-store`, the bytes come from `nb.path`, but only if that file still
  has the same sha256.

## Audit (`pseudorca.audit`)

### Fingerprints
- **Syntax:**
  - `text` must be present. Text that spans lines must occur in consecutive lines, ignoring whitespace
    differences and blank lines. Its first line may be the end of a line, and its last line the start of
    one.
  - `re:<regex>` must match a line. A regex containing a line break (typed, or written `\n` or `\r`) or
    the `(?m)` flag is searched over each whole file with `re.MULTILINE`.
  - `!text` and `!re:<regex>` must be absent.
- **Scope:** only evidence paths (no markdown, no binary files).
- **Evidence:**
  - the quote is the lines the match touches, without their outer whitespace, cut to 300 characters
    around the match;
  - `line` is the quote's first line, so a match that starts on a blank line points at the first line
    with text;
  - `col` is where the match starts in the quote, so tables can show a window around it.
- **Status with positives** (a positive is something that must be present):
  - `present` when all positives are found and no negative is;
  - `absent` when no positive is found;
  - `partial` otherwise.
- **Status with only negatives:**
  - `present` when none is found;
  - `absent` when all are found;
  - `partial` otherwise.

### Patch
- **Input:** a unified diff (git or plain), read as UTF-8. A leading byte order mark is ignored.
- **Lines:**
  - Lines are compared after whitespace normalisation, and blank lines are skipped.
  - Trivial lines (`pass`, `else:`, lone brackets, and lines under 4 characters) are never anchors, and a
    change made only of trivial lines (or only of whitespace) is ignored.
  - Next to changed lines, trivial lines are part of the change and must match like any other line. So
    when a trivial line changes later, its change becomes `unknown`.
- **Candidate files**, tried in this order:
  1. paths ending with the diff's path;
  2. the same basename;
  3. all evidence paths.
- **Matching:**
  - A hunk's non-trivial context lines split it into changes. Each change is anchored by the nearest
    non-trivial context line before it and after it.
  - **New side in place:** the added lines sit next to an anchor.
    - When the change's old side is also in place, they must sit between both anchors. A change at a hunk
      edge has only its one anchor.
    - For a change that only removes lines, the two anchors are adjacent instead.
  - **Old side in place:** the removed lines sit between both anchors. For a change that only adds lines,
    the two anchors are adjacent instead.
  - A lone anchor at a hunk edge must be the first or last line of its file.
- **Hunk state:**
  - `applied`: every change's new side is in place (and, for a pure removal, its old side is not);
  - `reverted`: every change's old side is in place and its new side is not;
  - `unknown`: anything else. This includes added lines found only away from their context, and added
    lines next to only one anchor while the change's old side is still in place.
- **Hunks without non-trivial context** are matched by their blocks alone. These are hunks for a new or
  deleted file, `-U0` diffs, and hunks whose context lines are all trivial.
  - **Where blocks are looked for:** in the diff's own file (same path, then same basename) when the
    notebook has it. Otherwise, and for a new or deleted file, in every evidence path.
  - `applied` when every added block is found;
  - `reverted` when every removed block is found and no added block is;
  - for a hunk that only adds or only removes lines, "not there" when at most 20% of its distinct
    non-trivial lines are found. For a new or deleted file this counts lines found anywhere, as long as
    no file of that name exists; otherwise it counts lines in the file itself.
- **Patch status:**
  - every hunk applied: `present`;
  - every hunk reverted: `absent`;
  - no hunk applied or reverted: `unknown`;
  - anything else: `partial`.
- **`detail`** holds the counts and the files.
- **Speed:** matching normalises and indexes the notebook's lines once per tree, and every check in an
  audit shares that index. A run of lines is looked for only where its rarest line occurs, so checks take
  near-linear time; an 8,000-line formatter diff takes well under a second.

### Orchestration
1. An audit reads the ledger once. Effective verdicts are computed at the end, from that snapshot plus
   the run's own findings.
2. For each selected idea, the deterministic methods it supports run first.
3. When the LLM is called:
   - never with `llm = "never"`;
   - always with `llm = "always"`;
   - with `llm = "auto"`, only when the deterministic result is inconclusive: there is no single agreed
     `present`/`absent`, and a fingerprint/patch conflict also counts as inconclusive. The LLM's verified
     verdict then settles it, and audit and check show its quote.
4. A method is skipped, unless `--force` is given, when what is recorded for it is at least as new as
   what this run would record:
   - **Fingerprint and patch:** skipped when the verdict that counts (`State.current_records`) has
     `(detail.extractor_version, detail.matcher_version) >= (EXTRACTOR_VERSION, MATCHER_VERSION)`.
     - Each method has its own `MATCHER_VERSION` (now 2), bumped when its matching changes.
     - So a matcher fix redoes older verdicts without rebuilding cached trees.
   - **LLM:** skipped when any verdict recorded for this rev, not only the one that counts, comes from:
     - the same model with `detail.prompt_version >= PROMPT_VERSION`; or
     - any model with a newer prompt version.

     So each model is asked once per prompt version.

   Versions compare as integers, and a missing one counts as 0. An older Pseudorca never redoes a newer
   one's verdicts, and an upgrade redoes older ones.
5. Every finding is written as one `audit.verdict` event.

**`audit NB_REF`**
- When `NB_REF` names an existing file, whatever its suffix, the registered version with that file's
  sha256 is audited, whatever its name.
- Without `NB_REF`, the latest registered notebook is audited, and a file named `latest` is not read.
- A file whose version is not registered is an error. If its name resolves to another registered
  version, it is a warning instead. Both suggest `pseudorca nb add FILE` or `pseudorca check FILE`.
- `audit` and `check` take `--json`.

### LLM judge (`pseudorca.llm`)
- **Tools:**
  - `list_files(glob)`;
  - `grep(text, path_glob, ignore_case)` (substrings only, never regex);
  - `read_file(path, start_line, end_line)`;
  - `submit_verdict(status, evidence[{path, line, quote}], reasoning)`.
- **Limits:** `max_steps` tool calls. Each tool output is cut to `max_tool_chars`.
- **Prompt:**
  - tree content is untrusted data, and instructions found in it are ignored;
  - markdown is a claim, not evidence;
  - quotes must be verbatim;
  - each prompt piece is redacted before it is cut to its length limit, and the whole message is redacted
    again.
- **Verification:**
  - each quote must be found in the cited file (never under `markdown/`) by the rule of
    `Tree.find_snippet` with partial ends. The search takes time linear in the file, and the match nearest
    the cited line wins (the earlier one on a tie);
  - only the first 10 quotes are checked, and unverified ones are dropped;
  - `present` or `partial` with no verified quote becomes `unknown`;
  - `absent` needs at least one tool call.
- **Remote guard:**
  - Local hosts are `localhost`, `*.localhost`, `127.0.0.0/8`, `::1` and `0.0.0.0`. Every other host is
    remote, and `anthropic` is always remote.
  - A remote endpoint needs `allow_remote`. `ensure_allowed` raises `LLMError` before any network use.
  - A local endpoint is reached directly. Its client ignores proxy settings from the environment and
    does not follow redirects, so its requests cannot be carried to another host.
  - A remote endpoint keeps the environment's proxy settings.
- **Redaction:** these are replaced by `[REDACTED]` before anything is sent:
  - OpenAI, Anthropic, GitHub, Hugging Face, Slack, Google and Kaggle keys;
  - AWS access key ids and secret access keys, including credentials files and `aws configure set`;
  - JWTs, Bearer and Basic credentials, and private key blocks;
  - passwords in URLs, also with an empty user name (`redis://:password@host`);
  - values assigned to secret-like names (key, token, secret, credential, password, passwd, pwd,
    passphrase):
    - quoted values with any string prefix, or in triple quotes;
    - unquoted secret-looking values on YAML, INI, `.env` or properties lines;
    - `NAME=value` in shell commands, including a notebook's `!` lines and strings handed to a shell
      (`os.system`, `subprocess`). There a value ends at whitespace, `;`, `&`, `|` or a quote, as in a
      shell.
  - **Passwords:** a name that ends in password, passwd, pwd or passphrase holds a password.
    - An unquoted value of 6 or more characters on its line is redacted, words and numbers included.
    - The exception is a value that reads as code: an environment variable's name, a dotted name, a
      CamelCase name, or a variable named for a password.
- **What is not redacted:** environment variable names, templates and code-like names, and (except on a
  password line) booleans, numbers and short words.
- Replacements keep every line break as it was, so line numbers in the redacted view match the notebook.
- **Linear time:** redaction runs in time linear in the text.
- Verification runs on the same redacted text.
- **Transcripts:** saved in `cache/llm/`.
- **Fallback:** for endpoints without tool calling, a single prompt with retrieved snippets, answered as
  JSON. The reply is read in linear time: `<think>` blocks are skipped and code fences are taken.
- **Unreadable output:** tool arguments or a reply whose JSON is nested deeper than Python's recursion
  limit count as unreadable, and the verdict becomes `unknown`.

### `check`
`pseudorca check NOTEBOOK` registers the notebook (unless `--dry-run`) and audits it. It then compares
the result with expectations:
- **Expected:** ideas whose status is in `[audit].expect_status` (default `adopted`), plus any `--expect ID`.
- **Forbidden:** ideas whose status is in `[audit].forbid_status` (default `rejected`).

Both settings must be idea statuses (`idea`, `testing`, `adopted`, `rejected`, `parked`); any other
value is an error.

| exit code | meaning |
|---|---|
| 0 | every expected idea is present, and no forbidden idea is |
| 1 | an expected idea is not fully present, a forbidden idea is present, or `--strict` is on and an expected idea is unknown |
| 2 | the check could not run: any error, including a usage error |

`check` also lists ideas the notebook claims (`applies`) that the audit found absent. `--json` carries
the sections and `passed`.

## Kaggle (`pseudorca.kaggle_api`), read-only
- **API calls:** only listing submissions and pulling kernel sources. Credentials are the `kaggle`
  package's own, and they are never printed.
- **`kaggle sync`** upserts a submission only when something changed. A submission links to a notebook
  in two ways:
  - an explicit `sub link`;
  - its description contains a registered notebook's name (4 or more characters) or a sha prefix (8 or
    more characters). The match must be unambiguous.
- **`kaggle pull OWNER/SLUG`** downloads the latest source into `cache/kaggle/` and registers it with
  `kernel = OWNER/SLUG`.

## Results and statistics (`pseudorca.results`, `pseudorca.stats`)
- **Results table:** CSV with a header, or JSONL.
  - `item` is required.
  - `draw` is optional. Without it, a repeated item gets draws 0, 1, 2… in file order.
  - Numeric columns are metrics. The default metric is `score`.
- **`compare A B`:**
  - **Inputs:**
    - Each side is a run, a run label, or all runs of a notebook.
    - The protocols must match, unless `--force` is given.
    - An arm pools each results table (blob) once. A repeated table, for example one imported twice by
      two teammates, is left out with a warning.
  - **Per item:** only items present in both arms count. Draws are averaged per item, and
    `d_i = mean_B,i − mean_A,i`.
  - **Estimate:** the mean of `d`.
  - **95% interval:** a two-stage bootstrap. Items are resampled with replacement, then draws within each
    item; 10,000 resamples with a fixed seed.
  - **Also reported:**
    - wins, losses and ties;
    - a sign-flip permutation p-value: exact up to 16 items, otherwise random flips (at least 20,000, and
      enough to resolve the confidence asked for);
    - MDE80 ≈ 2.8·SE.
  - **Verdict:**
    - `B better` or `A better` only when the interval excludes 0 and p < 1 − confidence;
    - `too few items` with fewer items than `min_items(confidence)` (6 at 95%), because then no outcome
      could be significant;
    - `within noise` otherwise.
  - **A/A check:** when an arm has 2 or more draws on 5 or more items, its draws are split into halves
    and compared the same way, as a check on the noise. It is judged by the same verdict rule.
- **LB noise:** scored submissions are grouped by notebook. The pooled SD comes from groups with at least
  2 scored submissions. Scores that are not finite numbers are skipped.

## Effects (`pseudorca.effects`)
`effects` answers two questions: which ideas each submission has, and what each idea adds to the score.
It fits an additive model, `score = baseline + Σ effect(idea in it) + noise`, by least squares.

- **Rows:**
  - **Leaderboard:** every submission, oldest first (date as an instant, then numeric id). A submission
    is used when it has a finite score (public, or private with `--private`) and its notebook is
    registered. With submissions from several competitions, `--competition` must choose one; a
    competition no submission has is an error.
  - **Local runs:** the runs of one protocol; each protocol is fitted separately (`--protocol` keeps
    one). A run is used when its notebook is registered and its table has finite values of the metric.
    A results table recorded more than once counts once, through its first run that can be used, so a
    table added again with `--nb` after a first copy without one is used. A run's value is its mean,
    over the items every used run has, of each item's mean over draws.
- **Which ideas a row has:** the effective verdict on its notebook. `present` is 1 and `absent` is 0.
  `partial`, `unknown`, `conflict` and no verdict count as missing. Stale verdicts are used, and a note
  counts them once per notebook and idea; stale human verdicts need a new `verdict`, not an audit.
- **Which ideas are modelled:**
  - Parked ideas are left out, as on the board, unless `--all` is given.
  - By default, the ideas with a present/absent verdict on every usable row.
  - An idea with gaps is left out (`excluded`), with the rows that lack a verdict and how to fill them:
    an audit for a missing verdict, `pseudorca verdict` for an inconclusive one (partial, unknown,
    conflict: another audit would say the same). When it varies among the other rows, a note says that
    leaving it out may bias the others.
  - With `--idea`, exactly those ideas are modelled (a parked one too). Rows without a present/absent
    verdict for one of them are not used, and the reason is shown. Notes name the other ideas that vary among the rows:
    one that always comes and goes with a chosen idea is counted as that idea's effect, and others go
    into the residuals.
- **The design:**
  - An idea in every used row (`always`) or in none (`never`) cannot be measured, and is listed as
    such.
  - Ideas with identical columns, that is, always in or out together, form one bundle with one effect.
  - The model is `X = [1 | bundle columns]`. It is solved through a QR decomposition and the SVD of
    `R`, and residuals are taken through an orthonormal basis of X's columns, so no `n × n` matrix is
    ever built. A contrast `c` is estimable when it is orthogonal to the
    null space of `X`. Its estimate is then `c·b̂ = w·y` with `w = (X⁺)ᵀc`, the same for every
    least-squares solution.
  - Contrasts between bundles never involve the intercept, so what they cannot see is the bundle part
    of the null space, made orthonormal (`P`). A bundle that is not estimable is grouped with the
    bundles in the support of `P·e_j`. Through the intercept, two unrelated choices (one idea of a set
    in every row) would look mixed; leaving it out keeps them apart, and `P` is block-diagonal over the
    groups.
  - Within a group, differences between two bundles are reported when they are estimable. Two bundles
    never used in the same row are a swap, "A instead of B", with the newer choice first: the one in the
    latest row that has one of them but not the other. Two that are also used together are "A vs B",
    the one that first appeared later first. Pairs are tried with the latest choices first, so that a
    choice among several is set against each alternative, and picked greedily, so that none is a
    combination of the others. Bundles that no difference covers are `tangled`, and their note names
    the bundles of their group. Nothing that is not estimable is ever given a number.
  - An estimate or bound within rounding of 0 (`1000·ε·Σ|wᵢyᵢ|`) is 0, and a residual sum of squares
    within the rows' rounding is 0, so an idea with no effect is never `helps` by rounding.
- **Leaderboard intervals:**
  - The noise is the residual SD, `sqrt(RSS / (n − rank))` with `n − rank` degrees of freedom, and the
    interval uses Student's t. Resubmissions are rows with the same columns, so they supply degrees of
    freedom.
  - `--noise SD` (or `[effects] noise` in `config.toml`) uses a known SD and the normal distribution.
  - With `n = rank` and no known noise, there is no noise estimate. Effects are shown without intervals,
    and their verdict is `unknown`. The same holds when the residuals are exactly 0 (resubmissions that
    repeat their scores, and a model that fits every submission): that says nothing about the noise
    between notebooks, so a note asks for a known noise SD instead of printing zero-width intervals.
  - The pooled resubmission SD (as in `lb`) is shown for comparison.
  - **Lack of fit:** when resubmissions give pure-error degrees of freedom and other rows give more, an
    F test compares the residuals beyond the resubmissions with the resubmissions' own spread:
    `F = (SS_lof / df_lof) / (SS_pure / df_pure)`, whatever the source of the noise used for the
    intervals. When its p-value is below 1 − confidence, a note suggests interactions, an untracked
    change, or ideas left out by `--idea`.
- **Local intervals:** a two-stage bootstrap. Items are resampled with replacement, the same items for
  every run so that the runs stay paired; then draws are resampled within each item and run. Each
  resample gives run means `y*`, and the estimate's distribution is `w·y*`. The interval is the
  percentile interval, and the SE is the SD of the resampled estimates. The bootstrap is seeded and
  drawn as in `compare`, so with two runs (given in the order they were recorded) the interval is
  exactly `compare`'s. With fewer items than `min_items(confidence)`, the verdict is `too few items`.
  - **Run-to-run noise:** the item/draw bootstrap knows nothing of noise shared by a whole run (another
    machine, a cache, a nondeterministic server). Runs with the same modelled ideas reveal it: their
    spread around their own mean, `SS_rep` with `df_rep` degrees of freedom. Its expectation without run
    noise comes from resampling the items alone, each item keeping its mean over draws (resampling the
    draws as well would count their noise twice). The part beyond it, per degree of freedom, is the
    run-to-run variance `τ²`. As `τ` is estimated from `df_rep` degrees of freedom, `N(0, (k·τ)²)` per run
    is added to every resample of the intervals, with `k = t(1 − α/2, df_rep) / z(1 − α/2)`. When
    `SS_rep` is beyond the resampled ones (p below 1 − confidence), a note says so. In simulations the
    95% intervals then cover about 95% of the time with 2 or more degrees of freedom, and about 90% with
    one. With no two runs sharing their ideas, `τ` cannot be measured, the intervals assume it is 0, and
    the output says so.
  - **Lack of fit:** the rest of the residuals (`df − df_rep` degrees of freedom) is how far runs with
    different ideas stray from the additive model. It is tested against the same item resamples plus the
    run noise, and a p-value below 1 − confidence adds a note: ideas may interact, or a change no idea
    describes matters (or, without repeated runs, runs have noise of their own).
- **Verdicts:** `helps` or `hurts` when the interval excludes 0 (by `--lower-is-better`), `within noise`
  otherwise, and `unknown` without an interval. On local runs a winner also needs the sign-flip
  p-value of the per-item contrasts (`means_i · w`) below 1 − confidence, as in `compare`; otherwise
  the verdict is `within noise` and the note says why. Measured lines are sorted by effect, best first.
- **MDE80:** the effect the test finds 80% of the time: `(t(1 − α/2, df) + t(0.8, df))·SE` with
  Student's t intervals, and `(z(1 − α/2) + z(0.8))·SE` with a known noise or the bootstrap. It is an
  approximation (the exact one inverts the noncentral t).
- **Breakdown (`--sub`, `--base`):**
  - It starts at the base's fitted score and ends at the target's.
  - There is one step for each bundle that differs whose contrast is estimable: `+` added, `−`
    removed, with a bundle in parentheses, `+ (a + b)`. The rest is one step per group of bundles that
    mix with each other. Each such step is estimable: the difference of two rows always is, so `P·d = 0`,
    and as `P` is block-diagonal over the groups, each group's part is 0 too.
  - The steps add up exactly to the difference of the fitted scores. The observed scores are shown
    beside the fitted ones.
  - Ideas outside the model whose verdicts differ between the two, or are unknown on one of them, are
    listed: the breakdown cannot account for them.
  - By default it goes from the first used submission to the best one.
- **Changes:** a row's changes are the ideas that came in or left since the latest earlier row with a
  present/absent verdict for them, so a row without verdicts in between hides nothing.
- **Configuration:** `config.toml` may hold the competition's direction and a known noise, read by
  `effects` and `report`. Flags override them; a key a flag overrides is not read, so a bad value there
  does not stop the command (the report shows a bad value as a note):

  ```toml
  [effects]
  lower_is_better = false   # true when a lower score is better
  noise = 0.012             # a known LB noise SD (default: from the residuals)
  ```
- **Assumptions:**
  - Effects add up; ideas do not interact.
  - Notebooks with the same ideas score alike. On the leaderboard, a change that no idea describes goes
    into the residuals; on local runs, into `τ` when runs with the same ideas show it.
  - The noise is independent between submissions.
  - At 95%, about one measured line in twenty excludes 0 by chance, when the noise is measured: local
    runs without repeats cannot show run-to-run noise, and their intervals leave it out.
- **Terminal output (`commands/effects.py`):**
  - The submissions table lists the latest `-n` rows (also in `--json`, where each fit has a `total`);
    the model always uses all of them. CHANGES reads `not used: …; resubmits 103; +new −old`.
  - Notebooks that would show the same label get their short sha. IDEA labels longer than 40 columns
    get a `[k]` marker and are printed in full under the table; NOTE and CHANGES are never cut.
  - The interval sketch puts 0 at `|`, the interval at `[…]` and the estimate at `*`; rounding to
    columns never moves an interval across 0.
  - The breakdown's numbers are running totals rounded once, so the printed column adds up exactly
    (the report rounds them the same way). Its target is called the best "in the model": submissions
    the model does not use may have scored higher.
    With no modelled idea that differs it says `(no modelled idea differs)`; ideas outside the model
    that differ, or may, are listed. Without any idea in the model there is no default breakdown.
  - A protocol with nothing to compare lists its runs grouped by why they are not used.
  - The local runs' noise lines say whether a run-to-run SD was measured and used.
  - When the bootstrap would draw more than 3e8 values, a line on stderr names `--resamples` and
    `--no-runs` first.
  - Parked ideas with verdicts on the shown notebooks are counted in a `--all` hint.
- **`audit --submitted`** audits every registered notebook that a submission links to, so that
  `effects` has verdicts for all of them.
  - Notebooks come in the order of their oldest submission (`effects.sub_order`), each once, with the
    same `--idea`, `--status`, LLM options, `--force` and `--dry-run` as a single audit. The LLM
    endpoint and its remote guard are set up once.
  - Skipped with one warning each, and listed under `skipped` in `--json` by reason: `no_notebook` (a
    submission without one: `sub link`), `not_registered` (`nb add`, `kaggle pull` or `sub link`),
    `not_stored` (no bytes in this store and nothing to stand in for them) and `unreadable` (a blob that
    fails its sha256 check, such as a Git LFS pointer, or bytes that are not a notebook). Readability
    follows `extract.tree_for`, so a fresh cached tree stands in for missing or damaged bytes. Bad
    bytes are found before the run; a notebook the extractor cannot read during it is skipped too, and
    the run goes on. Any other error stops the run and names the notebook; verdicts recorded before it
    stay.
  - Text output is one line per notebook and a total, each ending with `; N LLM errors` when the LLM
    failed. Each distinct LLM error is warned about once with the number of notebooks it hit (the five
    most frequent; the rest are counted, and `--json` has each one). `--json` gives `{"dry_run",
    "notebooks": [the object of `audit NB --json`], "skipped", "totals"}`, counting audited notebooks
    only.
- **Report:** `report` is one HTML file of tabs, one view each, that switch without scripts: a hidden
  radio button per tab, the labels as the tab bar, and generated CSS (`#id:checked ~ #id-panel`) that
  shows the checked one's panel. The ids are made by the report, never taken from the ledger, so the CSS
  holds no ledger string. Each panel starts with a heading that screen readers read and print shows;
  printing shows every panel. On a narrow page the tab bars wrap.
  - Tabs: Effects (when computed, and first when there are submissions or runs; else the board is
    first), Board, Submissions, Notebooks, Ideas, Activity.
  - The Effects tab holds one line on the model, then a tab per view of the same model: the ideas per
    submission (outlined cells changed since the latest earlier verdict), what each idea adds (forest
    plot), the breakdown, the score chart (fitted scores and a noise band), and the local runs of each
    protocol. Each view is drawn on its own, so one that fails on odd numbers becomes a note and the
    others are still drawn.
  - It fits public scores with `effects`' defaults, the direction and noise from `[effects]`, and
    `--metric`, `--resamples` (10,000 by default, as in `effects`) and `--all` for the runs and ideas.
    With submissions from several competitions it shows the store's competition, else a note.
  - A bad `[effects]` value, or a part that cannot be computed (overflow, recursion, a missing blob),
    becomes a note; the rest of the report is still written.
  - Notes point to `pseudorca effects` options or `config.toml`, never to flags `report` lacks.

## Display conventions
- Times are shown in the viewer's time zone. A bare date stays as written. Times are compared as instants,
  never as strings.
- `-n N` means the latest N, and `-n 0` means all.
- `--json` keeps the stored ISO times.
- `log --json` prints events as stored, plus a one-line summary.
- Strings from the ledger, and lines of notebooks, are printed through `util.clean` (`_common.echo`,
  `warn`, `format_table`, and the one-line errors): control characters (all but tab) and bidirectional
  overrides become `?`. JSON output is left as it is. The ledger comes through git and may be hand-edited, and such
  characters could otherwise move the cursor, recolour, retitle or rewrite the terminal.
- Numbers: scores to 6 significant digits; effects and intervals with decimals from the interval widths
  (about two significant digits), plus more when a bound that excludes 0 would print as 0; magnitudes
  of 1e6 and more in compact form (`2.5e+30`). Negative numbers use a Unicode minus.

## Module map

| module | owns |
|---|---|
| `util`, `store`, `model`, `tree`, `audit/types`, `cli`, `commands/_common` | core |
| `commands/ledger.py` | `init`, `idea`, `nb`, `sub`, `status`, `log` |
| `extract/` | extraction |
| `audit/fingerprint.py`, `audit/patch.py`, `audit/engine.py`, `commands/audit.py` | `audit`, `check`, `verdict` |
| `llm/` | LLM judge, backends, redaction |
| `kaggle_api.py`, `commands/kaggle_cmd.py` | `kaggle sync`, `kaggle pull` |
| `results.py`, `stats.py`, `commands/results.py` | `run`, `compare`, `lb` |
| `effects.py`, `commands/effects.py` | `effects` (the report's effects section uses `effects.py`) |
| `report.py`, `charts.py`, `commands/report.py` | `report` (single-file HTML, no external resources; inline SVG charts) |

Heavy imports (`numpy`, `openai`, `anthropic`, `kaggle`) happen inside the functions that need them.

`pseudorca init` creates a store or completes a partial one (for example a clone without `events/`).
- It never overwrites existing files such as a committed `config.toml`.
- Without `--name`, the name is left out of `config.toml`; it defaults to the repository folder's name.
  Two clones that run `init` with the same options therefore write identical files.

Expected failures become one-line errors in every command: store errors, `LookupError`, `ValueError`
and `OSError`.
- `cli.LazyGroup.invoke` does this around each command, and `commands._common.user_errors()` does it inside
  a command, for any extra exception types passed to it.
- Set `PSEUDORCA_DEBUG=1` to see tracebacks.
- After a command finishes, any ledger warning it has not shown yet is printed once to stderr. This covers
  replay warnings and the clock-skew note.
- When output is piped into a reader that closes early, such as `head`, the command ends quietly with code
  0. `check` keeps its own exit code in that case.
- `check` decides its exit code before it prints anything, so a closed stdout (`| head`) still ends with
  0 or 1 by the outcome. If output is lost before the outcome is known, `check` exits 2.
- Raise `LookupError` or `ValueError` for user mistakes, never `KeyError`, which is treated as a bug.
