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

The tour uses synthetic files from `examples/make_demo_files.py`: an offline word-problem solver in four
notebook versions, a patch, and four results tables.

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

### 8. What did each idea add?

Two more versions get submitted: `hot.ipynb` samples at temperature 0.6 (the rejected idea), and
`lean.ipynb` drops the retries. `audit --submitted` audits every notebook that has a submission, and
`effects` lines the submissions up against the ideas:

```bash
pseudorca nb add hot.ipynb  --parent next.ipynb --tag candidate
pseudorca nb add lean.ipynb --parent next.ipynb --tag candidate
pseudorca run add runs/hot.csv  --nb hot.ipynb  --label hot  --protocol p60x3
pseudorca run add runs/lean.csv --nb lean.ipynb --label lean --protocol p60x3
pseudorca sub add 1004 --nb hot.ipynb  --score 0.411 --date 2026-01-08
pseudorca sub add 1005 --nb lean.ipynb --score 0.428 --date 2026-01-09
pseudorca sub add 1006 --nb next.ipynb --score 0.441 --date 2026-01-10
pseudorca audit --submitted
pseudorca effects
```

```text
Submissions, oldest first (public score)
SUB   DATE        NOTEBOOK    PUBLIC  FITTED  1  2  3  4  CHANGES
----  ----------  ----------  ------  ------  -  -  -  -  -------------------------------------
1001  2026-01-05  base.ipynb  0.412   0.405   ✗  ✗  ✗  ✗
1002  2026-01-06  base.ipynb  0.398   0.405   ✗  ✗  ✗  ✗  resubmits 1001
1003  2026-01-07  next.ipynb  0.437   0.439   ✓  ✓  ✓  ✗  +prefix-cache +retry-parse +short-cot
1004  2026-01-08  hot.ipynb   0.411   0.411   ✓  ✓  ✓  ✓  +temp-06
1005  2026-01-09  lean.ipynb  0.428   0.428   ✓  ✗  ✓  ✗  −retry-parse −temp-06
1006  2026-01-10  next.ipynb  0.441   0.439   ✓  ✓  ✓  ✗  resubmits 1003; +retry-parse
ideas: 1 prefix-cache  2 retry-parse  3 short-cot  4 temp-06
✓ present  ◐ partial  ✗ absent  ? unknown  ! methods disagree  · no verdict  * stale (re-audit)

What each idea adds to the public score (additive model over 6 submissions)
IDEA                      EFFECT  95% CI            SUBS  VERDICT       −0.066    0    +0.066  NOTE
------------------------  ------  ----------------  ----  ------------  ---------------------  --------------------------------------------
prefix-cache + short-cot  +0.023  [−0.015, +0.061]  4/2   within noise          [-|--*-----]   always in or out together: measured together
retry-parse               +0.011  [−0.027, +0.049]  3/3   within noise        [---|-*----]
temp-06                   −0.028  [−0.066, +0.010]  1/5   within noise  [-----*---|-]
SUBS = submissions with/without it. VERDICT is helps or hurts when the 95% CI excludes 0.
noise: residual SD 0.00728 with 2 degrees of freedom; t intervals (few degrees of freedom: wide intervals)
resubmitting a notebook moved the score with SD 0.00728 (2 notebook(s), 4 submissions)

Local runs (protocol p60x3), score: 4 of 4 runs used, 60 items in every one
RUN                NOTEBOOK    MEAN    FITTED  1  2  3  4  NOTE
-----------------  ----------  ------  ------  -  -  -  -  ----
01M5J3ZEDT (base)  base.ipynb  0.5111  0.5111  ✗  ✗  ✗  ✗
01M5J3ZEG7 (next)  next.ipynb  0.5833  0.5833  ✓  ✓  ✓  ✗
01M5J3ZF0H (hot)   hot.ipynb   0.4556  0.4556  ✓  ✓  ✓  ✓
01M5J3ZF2R (lean)  lean.ipynb  0.5333  0.5333  ✓  ✗  ✓  ✗
ideas: 1 prefix-cache  2 retry-parse  3 short-cot  4 temp-06

What each idea adds to the score (additive model over 4 runs)
IDEA                      EFFECT  95% CI          RUNS  VERDICT       −0.24     0     +0.24  NOTE
------------------------  ------  --------------  ----  ------------  ---------------------  --------------------------------------------
retry-parse               +0.05   [−0.07, +0.17]  2/2   within noise         [--|-*----]
prefix-cache + short-cot  +0.02   [−0.10, +0.15]  3/1   within noise        [---|*----]      always in or out together: measured together
temp-06                   −0.13   [−0.24, −0.01]  1/3   hurts         [----*---]|
RUNS = runs with/without it. VERDICT is helps or hurts when the 95% CI excludes 0 and the sign-flip p < 0.05, as in `compare`.
intervals: two-stage bootstrap over items and draws (10000 resamples, seed 0), the runs paired by item
run-to-run SD: not measurable, as no two runs have the same ideas; the intervals assume none (run a notebook twice to measure it)

Breakdown of 1006 (best in the model) from 1001 (first), by the model
  1001 fitted                    0.405  observed 0.412
  + (prefix-cache + short-cot)  +0.023  [−0.015, +0.061]
  + retry-parse                 +0.011  [−0.027, +0.049]
  = 1006 fitted                  0.439  observed 0.441
```

- **The table** shows which ideas each submission has, and what changed since the one before.
- **The effects** come from an additive model, `score = baseline + the effects of its ideas + noise`. It
  is fitted to all scored submissions, and separately to the local runs of each protocol, paired by item.
- **What cannot be separated is said so.** `prefix-cache` and `short-cot` were always added together, so
  only their sum is measured. An idea that is in every submission, or in none, gets no number at all.
  Two ideas that replaced each other are measured as "A instead of B".
- **On the leaderboard nothing is beyond the noise yet.** Two resubmissions give the noise estimate only
  2 degrees of freedom, so the intervals are wide. The local runs measure the same ideas on another
  metric, and there temperature 0.6 clearly hurts.
- **The breakdown** walks from the first submission to the best one, idea by idea. `--sub` and `--base`
  choose others.

`pseudorca report` draws the same as charts in a single HTML file, one tab per view: the ideas in each
submission, what each idea adds, the breakdown, the scores over time and each protocol's local runs, next
to tabs for the board, the submissions, the notebooks, the ideas and the recent activity.

### 9. Share

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
- **Limits:** decoding is capped in size, depth, archive members and decoder memory. Whatever is not
  decoded is listed as a warning.
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
- With the OpenAI-compatible provider, a localhost server is always contacted directly, even when your shell
  has a proxy configured. Hosted providers use your normal network settings.

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

**`effects`** puts the ideas side by side instead of two notebooks:
- Each scored submission (or local run) is a row, and the verdicts on its notebook say which ideas it has.
- An additive least-squares model gives each idea's effect with an interval: Student's t from the residuals
  on the leaderboard (or a known `--noise`), and the paired two-stage bootstrap on local runs.
- Ideas that always change together are measured together, and swapped ideas as "A instead of B". What the
  rows cannot separate gets no number.
- The breakdown (`--sub`, `--base`) shows how one submission's score is built from another's, idea by idea.
- On local runs, runs with the same ideas also measure run-to-run noise, and the intervals include it.
- Set the score's direction and a known LB noise once for the team, in `.pseudorca/config.toml`:

  ```toml
  [effects]
  lower_is_better = true   # for an error metric
  noise = 0.012            # the LB noise SD, if you know it
  ```

## Commands

| command | what it does |
|---|---|
| `init` | create `.pseudorca/` in the current repository, or complete a partial one |
| `idea add / update / list / show` | the ideas |
| `nb add / update / list / show / tree` | notebooks, and what the audit sees in them |
| `audit [NB]`, `audit --submitted` | decide which ideas are in a notebook, or in every submitted one |
| `check FILE_OR_NB` | audit, then fail if an expected idea is missing or a rejected one is present |
| `verdict NB IDEA STATUS` | record a human verdict |
| `status` | the ideas × notebooks board |
| `sub add / list / link`, `kaggle sync`, `kaggle pull` | submissions and Kaggle kernels (read-only) |
| `run add / list / show`, `compare`, `lb` | results and noise |
| `effects` | which ideas each submission has, and what each idea adds to the score |
| `report` | a single-file HTML report, one tab per view (no scripts) |
| `log` | recent changes and who made them |

`audit`, `check`, `status`, `log`, `lb`, `effects` and the `list` commands take `--json` for scripts. In
`status`, `log`, `nb list`, `sub list`, `run list`, `lb`, `effects` and `report`, `-n N` shows the latest N
and `-n 0` shows all.
Times are shown in your time zone.

Every command has `--help`. The design and the exact rules are in [DESIGN.md](DESIGN.md), and the safety
model is in [SECURITY.md](SECURITY.md).

## Roadmap

- Import vLLM metrics (throughput, KV-cache use, prefix-cache hits) per run.
- Export runs and transcripts to Inspect AI logs for browsing.
- Import the output files of Kaggle kernel runs.
- An optional small server with authentication, for a live team dashboard.

## License

MIT. See [LICENSE](LICENSE).
