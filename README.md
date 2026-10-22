# Pseudorca

**An experiment ledger for LLM work without training.**

[日本語](README.ja.md)

Training-oriented trackers record hyperparameters and loss curves. That is the wrong shape for some work:

- the experiment is a notebook assembled from small changes (a prompt tweak, a retry rule, a serving flag);
- teammates fork each other's notebooks;
- every evaluation is noisy.

Pseudorca answers the two questions that matter in Kaggle-style LLM competitions:

1. **Is this improvement actually in the notebook we are about to submit?**
   - Pseudorca reads notebooks *without running them*.
   - It unpacks the files a notebook writes or embeds, then checks every idea in your ledger.
   - It uses fingerprints and patch matching first. When those are inconclusive, it can ask an LLM, and
     every quote the LLM gives as evidence is checked against the notebook.
2. **Does a change beat the noise?** Pseudorca compares runs task by task, with bootstrap intervals, and
   estimates leaderboard noise from repeated submissions.

Everything lives in `.pseudorca/` inside your team's private git repository. There is no server, no
account and no telemetry.

> Status: alpha. Built for Kaggle code competitions first.

## Install

```bash
uv tool install "pseudorca[all] @ git+https://github.com/pomcho555/pseudorca"
# or
pip install "pseudorca[all] @ git+https://github.com/pomcho555/pseudorca"
```

- Requires Python 3.11+. The core depends only on `click` and `numpy`.
- Extras:
  - `llm`: OpenAI-compatible and Anthropic clients, for the LLM audit;
  - `kaggle`: the official Kaggle client, for `kaggle sync` and `kaggle pull`;
  - `all`: both.

## A five-minute tour

The tour uses synthetic files from `examples/make_demo_files.py`: an offline word-problem solver in two
notebook versions, a patch, and two results tables.

```bash
git clone https://github.com/pomcho555/pseudorca
python3 pseudorca/examples/make_demo_files.py ~/pseudorca-demo
cd ~/pseudorca-demo && git init
pseudorca init --competition word-problems
```

Alternatively, `bash pseudorca/examples/demo.sh <a new folder>` runs the whole tour at once.

### 1. Write down the ideas

```bash
pseudorca idea add short-cot    --title "Shorter reasoning instruction" --status adopted \
    --fingerprint "Think step by step, briefly."
pseudorca idea add retry-parse  --title "Retry when no answer parses"   --status adopted \
    --patch retry-parse.diff
pseudorca idea add prefix-cache --title "vLLM prefix caching"           --status testing \
    --fingerprint "enable_prefix_caching=True"
pseudorca idea add temp-06      --title "Sampling temperature 0.6"      --status rejected \
    --fingerprint 're:temperature\s*=\s*0\.6'
```

An idea is checked in one or both of two ways:
- **Fingerprints:** text that must appear, or with `!` must not appear. Use `re:` for a regex. Text may
  span lines.
- **A patch:** a unified diff. Pseudorca checks whether each hunk is applied or reverted.

Ideas move through the statuses `idea` → `testing` → `adopted`, `rejected` or `parked`.

### 2. Register notebooks

```bash
pseudorca nb add base.ipynb --tag baseline
pseudorca nb add next.ipynb --parent base.ipynb --applies short-cot --applies retry-parse --tag candidate
pseudorca nb tree next.ipynb
```

```text
PATH                                SIZE   ORIGIN
----------------------------------  -----  ---------------------------------------------------------------
cells/cell_001.py                   98 B   cell 1
cells/cell_002.py                   44 B   cell 2
cells/cell_003.py                   623 B  cell 3
cells/cell_004.py                   205 B  cell 4
files/solver.py                     495 B  cell 2 > %%writefile solver.py
markdown/cell_000.md                54 B   cell 0
payloads/cell_003/0/utils/io.py     109 B  cell 3 > line 3 str > base64 > lzma > tar member utils/io.py
payloads/cell_003/0/utils/timer.py  171 B  cell 3 > line 3 str > base64 > lzma > tar member utils/timer.py
```

`nb tree` shows what the audit sees:
- the code cells;
- the file written by `%%writefile`;
- the two modules inside a base64 + lzma + tar payload.

None of it was run.

### 3. Audit

```bash
pseudorca audit latest
```

```text
audit of next.ipynb (ecadc54857)

IDEA          STATUS    VERDICT    METHOD       EVIDENCE/NOTE
------------  --------  ---------  -----------  ------------------------------------------------------------
prefix-cache  testing   ✓ present  fingerprint  cells/cell_004.py:3 …eights", enable_prefix_caching=True, g…
retry-parse   adopted   ✓ present  patch        files/solver.py:11 def solve(llm, problem, sampling, retrie…
short-cot     adopted   ✓ present  fingerprint  files/solver.py:3 …d problems. Think step by step, briefly."
temp-06       rejected  ✗ absent   fingerprint  found 0/1 (missing: re:temperature\s*=\s*0\.6)

3 present, 1 absent; 4 new verdicts recorded
✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  c claimed, not audited  · none  * stale
```

### 4. Check before you submit

```bash
pseudorca check next.ipynb    # exit code 0
pseudorca check base.ipynb    # exit code 1: adopted ideas are missing (2 would mean the check could not run)
```

```text
$ pseudorca check next.ipynb
check of next.ipynb (ecadc54857), already registered

OK (2)
  retry-parse  adopted  ✓ present  patch        files/solver.py:11 def solve(llm, problem, sampling, retrie…
  short-cot    adopted  ✓ present  fingerprint  files/solver.py:3 …d problems. Think step by step, briefly."

check passed
(exit code 0)

$ pseudorca check base.ipynb
check of base.ipynb (61a5a1c1b5), already registered

MISSING (2)
  retry-parse  adopted  ✗ absent  patch        0/1 hunks applied, 1 reverted
  short-cot    adopted  ✗ absent  fingerprint  found 0/1 (missing: Think step by step, briefly.)

check failed: 2 missing
(exit code 1)
```

### 5. The board

```bash
pseudorca status
```

```text
IDEA          STATUS    1 base.ipynb  2 next.ipynb
------------  --------  ------------  ------------
prefix-cache  testing   ✗             ✓
retry-parse   adopted   ✗             ✓
short-cot     adopted   ✗             ✓
temp-06       rejected  ✗             ✗

✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  c claimed, not audited  · none  * stale
#  NOTEBOOK    SHA         ADDED
-  ----------  ----------  ----------
1  base.ipynb  61a5a1c1b5  2026-10-22
2  next.ipynb  ecadc54857  2026-10-22
```

### 6. Does it beat the noise?

```bash
pseudorca run add runs/base.csv --nb base.ipynb --label base --protocol p60x3
pseudorca run add runs/next.csv --nb next.ipynb --label next --protocol p60x3
pseudorca compare base next
```

```text
A  label base: 60 items, 3 draws per item
B  label next: 60 items, 3 draws per item
items compared: 60 (only in A: 0, only in B: 0)
mean score: A 0.51  B 0.58
B − A: +0.07  95% CI [−0.04, +0.19]
B wins/losses/ties: 20/13/27  p = 0.137 (sign-flip permutation)
SE 0.06  MDE80 0.16
A/A on A (draws split in halves): +0.03  95% CI [−0.12, +0.19]  ok
A/A on B (draws split in halves): 0.00  95% CI [−0.15, +0.16]  ok
verdict: within noise (the 95% CI includes 0)
```

A gain of 0.07 on 60 problems with 3 draws each is not distinguishable from noise. The output also gives
the smallest effect this setup can detect (MDE80). That tells you how many more draws or problems you
need.

### 7. The leaderboard

In a real competition, `pseudorca kaggle sync` fills in your team's submissions and scores from Kaggle.
It only reads. Here we add three submissions by hand:

```bash
pseudorca sub add 1001 --nb base.ipynb --score 0.412 --date 2026-01-05
pseudorca sub add 1002 --nb base.ipynb --score 0.398 --date 2026-01-06   # the same notebook again
pseudorca sub add 1003 --nb next.ipynb --score 0.437 --date 2026-01-07
pseudorca lb --diff 1001 1003   # is this LB gap bigger than the LB noise?
```

```text
A  1001  base.ipynb  public 0.412  (2026-01-05)
B  1003  next.ipynb  public 0.437  (2026-01-07)
B − A: +0.025 (public score)
z = +1.79 against the LB noise (pooled SD 0.0099 from 1 notebook and 2 submissions; z = diff / (SD·√2))
verdict: within LB noise
```

### 8. Share

```bash
pseudorca report                # writes .pseudorca/cache/report.html (local, ignored by git)
git add .pseudorca && git commit -m "Update ledger" && git push
```

## Concepts

| | |
|---|---|
| **Idea** | An improvement you want in a notebook, or out of it. It has a status, an owner, tags, links, and fingerprints and/or a patch. |
| **Notebook** | A version of a notebook or script, identified by the sha256 of its bytes. It can record a parent, the ideas its author says it applies, tags, and its Kaggle kernel. |
| **Verdict** | One method's answer to "is idea X in notebook N?": `present`, `partial`, `absent` or `unknown`. See below for the methods and how they combine. |
| **Run** | A local results table: one row per (item, draw) with numeric metrics, tied to a notebook and a protocol. |
| **Submission** | A leaderboard entry with public and private scores, linked to a notebook. |

**Verdict methods:** `fingerprint`, `patch`, `llm` and `human` (`pseudorca verdict`).

**How the methods combine:**
1. A human verdict wins.
2. Otherwise the deterministic methods win when they agree.
3. Otherwise a verified LLM verdict is used.
4. If an idea changes after an audit, its old verdicts are marked stale (`*`).

**Notebook references** accept a full sha256, a sha prefix, a name, a Kaggle kernel ref, `latest` or
`latest:<tag>`.

## How the audit reads a notebook

The audit never runs a notebook. It builds a read-only tree of text:

| path | what it holds |
|---|---|
| `cells/cell_NNN.py` | code cells, with IPython magics commented out (line numbers kept) |
| `files/<path>` | files written with `%%writefile`, `Path(...).write_text(...)` or `open(..., "w").write(...)` |
| `payloads/cell_NNN/...` | embedded payloads, decoded |
| `markdown/cell_NNN.md` | markdown cells; context only, never evidence |

- **Payload formats:** base64, base85 or hex text, compressed with zlib, gzip, xz/lzma or bz2, possibly
  packed as tar or zip.
- **Limits:** decoding is capped in size, depth and archive members.
- **Pickles** are never loaded.
- **Environment settings** such as `%env K=V` and `os.environ["K"] = "V"` are collected as well.

## The LLM judge (optional)

Configure it in `.pseudorca/config.toml`. A local model keeps your code on your machine:

```toml
[llm]
provider = "openai"                      # any OpenAI-compatible server
model = "qwen3-coder:30b"
base_url = "http://localhost:11434/v1"   # Ollama; vLLM is usually http://localhost:8000/v1
```

A hosted model has to be allowed explicitly:

```toml
[llm]
provider = "anthropic"                   # or "openai" without base_url for api.openai.com
model = "claude-sonnet-5-5"
api_key_env = "ANTHROPIC_API_KEY"        # the name of the variable, never the key
allow_remote = true                      # notebook code will be sent to this provider
```

**How it works**
- The model gets read-only tools over the extracted tree: list files, substring search and read lines.
- It must answer with quotes.
- Pseudorca checks every quote against the notebook. A quote that is not found is dropped. A `present`
  verdict with no verified quote becomes `unknown`.
- Before anything is sent, common secret formats are replaced with `[REDACTED]`.
- A local server is always contacted directly, even when your shell has a proxy configured.

**When the LLM is asked**
- By default (`--llm auto`), only for ideas the deterministic checks could not settle.
- Each answer is recorded per notebook, idea version and model, so a teammate's verdict is reused rather
  than paid for twice.

## Working as a team

- Put the store in your team's **private** repository and use one store per team.
- Every change is a new file under `.pseudorca/events/`, so `git pull` and `git push` never conflict.
- The author of each change comes from `git config user.name`, `--as` or `$PSEUDORCA_ACTOR`.
- `pseudorca log` shows who did what.
- `pseudorca check` in CI or a pre-submit script stops a submission that lost an adopted idea.
- Keep reports private. GitHub Pages sites are public even when the repository is private.

## Comparing runs

**Results tables** are CSV or JSONL files:
- `item` is required, `draw` is optional, and every other numeric column is a metric.

**`compare A B`**
- Each of `A` and `B` is a run id, a run label or a notebook (pooling its runs).
- **Method:**
  - only the items both sides share are used;
  - draws are averaged within each item;
  - the interval is a two-stage bootstrap: items are resampled, then draws within each item.
- **Output:**
  - the difference with a 95% interval;
  - wins, losses and ties with a sign-flip permutation p-value;
  - the smallest effect detectable at 80% power;
  - an A/A check that splits each arm's draws in half.
- `compare` names a winner only when the interval excludes 0 and the permutation p-value is below
  1 − confidence. With fewer than 6 items (at 95%) it says `too few items`.
- Each results table is pooled once, even if it was imported twice.
- Runs from different protocols are not compared unless you pass `--force`.

**`lb`** estimates leaderboard noise from notebooks submitted more than once, and `lb --diff` tells you
whether a gap is bigger than that noise.

## Commands

| command | what it does |
|---|---|
| `init` | create `.pseudorca/` in the current repository, or complete a partial one |
| `idea add / update / list / show` | the ideas |
| `nb add / update / list / show / tree` | notebooks, and what the audit sees in them |
| `audit [NB]` | decide which ideas are in a notebook |
| `check FILE_OR_NB` | audit, then fail if an expected idea is missing or a rejected one is present |
| `verdict NB IDEA STATUS` | record a human verdict |
| `status` | the ideas × notebooks board |
| `sub add / list / link`, `kaggle sync`, `kaggle pull` | submissions and Kaggle kernels (read-only) |
| `run add / list / show`, `compare`, `lb` | results and noise |
| `report` | a single-file HTML report |
| `log` | recent changes and who made them |

`audit`, `check`, `status`, `log`, `lb` and the `list` commands take `--json` for scripts. `-n N` limits
lists to the latest N, and `-n 0` shows all. Times are shown in your time zone.

Every command has `--help`. The design and the exact rules are in [DESIGN.md](DESIGN.md), and the safety
model is in [SECURITY.md](SECURITY.md).

## Roadmap

- Import vLLM metrics (throughput, KV-cache use, prefix-cache hits) per run.
- Export runs and transcripts to Inspect AI logs for browsing.
- Import the output files of Kaggle kernel runs.
- An optional small server with authentication, for a live team dashboard.

## License

MIT. See [LICENSE](LICENSE).
