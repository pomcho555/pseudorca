from __future__ import annotations

import time

from pseudorca.audit import fingerprint, patch
from pseudorca.audit.types import Evidence
from pseudorca.tree import Tree


def make_tree(files: dict[str, str]) -> Tree:
    t = Tree("synthetic")
    for path, text in files.items():
        t.add_text(path, text, f"test > {path}")
    return t


# ----- fingerprints --------------------------------------------------------------------------------

CELLS = {
    "cells/cell_000.py": "import os\nMODEL = 'small-model'\ntemperature = 0.2\n",
    "cells/cell_001.py": "def build_prompt(q):\n    return SYSTEM + q\n",
    "markdown/cell_002.md": "We also set max_retries = 3 somewhere.\n",
    "files/src/agent.py": "SYSTEM = 'Be brief.'\nDEBUG = False\n",
}


def test_fingerprint_all_positives_found_is_present():
    f = fingerprint.check(["temperature = 0.2", r"re:SYSTEM\s*=\s*'Be"], make_tree(CELLS))
    assert f is not None
    assert (f.status, f.method) == ("present", "fingerprint")
    assert f.detail == {
        "found": ["temperature = 0.2", r"re:SYSTEM\s*=\s*'Be"],
        "missing": [],
        "forbidden_found": [],
    }
    assert f.evidence == [
        Evidence("cells/cell_000.py", 3, "temperature = 0.2", col=0),
        Evidence("files/src/agent.py", 1, "SYSTEM = 'Be brief.'", col=0),
    ]


def test_fingerprint_some_found_is_partial_and_none_found_is_absent():
    t = make_tree(CELLS)
    f = fingerprint.check(["temperature = 0.2", "top_p = 0.9"], t)
    assert f.status == "partial"
    assert f.detail["missing"] == ["top_p = 0.9"]
    assert "missing: top_p = 0.9" in f.note
    f = fingerprint.check(["top_p = 0.9", "re:seed\\s*="], t)
    assert f.status == "absent" and f.evidence == []


def test_fingerprint_negatives():
    t = make_tree(CELLS)
    f = fingerprint.check(["!DEBUG = True"], t)
    assert (f.status, f.note) == ("present", "0/1 forbidden found")
    f = fingerprint.check(["!DEBUG = False"], t)
    assert f.status == "absent" and f.detail["forbidden_found"] == ["!DEBUG = False"]
    assert f.evidence == [Evidence("files/src/agent.py", 2, "DEBUG = False", col=0)]
    assert fingerprint.check(["!DEBUG = False", "!re:TRACE"], t).status == "partial"
    # every positive found but a forbidden one too: partial
    f = fingerprint.check(["temperature = 0.2", "!re:DEBUG\\s*="], t)
    assert f.status == "partial" and f.detail["forbidden_found"] == ["!re:DEBUG\\s*="]
    assert f.note == "found 1/1; 1/1 forbidden found (!re:DEBUG\\s*=)"
    # no positive found: absent, whatever the negatives say
    assert fingerprint.check(["top_p", "!DEBUG = True"], t).status == "absent"


def test_fingerprint_ignores_markdown_and_binary_files():
    t = make_tree(CELLS)
    t.add_bytes("payloads/cell_003/0.bin", b"\x00\x01max_retries = 3\x00", "payload")
    f = fingerprint.check(["max_retries = 3"], t)
    assert f.status == "absent"


def test_fingerprint_regex_and_invalid_patterns():
    t = make_tree(CELLS)
    assert fingerprint.check([r"re:temperature\s*=\s*0\.[0-2]\b"], t).status == "present"
    assert fingerprint.check([r"re:^temperature = 0\.9$"], t).status == "absent"
    for bad in (["re:(unclosed"], ["!"], ["re:"]):
        f = fingerprint.check(bad, t)
        assert f.status == "unknown" and "invalid fingerprint" in f.note and f.detail["invalid"]


def test_fingerprint_without_fingerprints_or_code():
    assert fingerprint.check([], make_tree(CELLS)) is None
    assert fingerprint.check(["  "], make_tree(CELLS)) is None
    f = fingerprint.check(["x = 1"], make_tree({"markdown/cell_000.md": "x = 1"}))
    assert f.status == "unknown" and "no code" in f.note


def test_fingerprint_evidence_is_capped_and_long_lines_are_clipped_around_the_match():
    many = "".join(f"call_model(i={i})\n" for i in range(8))
    long_line = "blob = '" + "a" * 800 + "NEEDLE" + "b" * 400 + "'\n"
    t = make_tree({"cells/cell_000.py": many, "cells/cell_001.py": "x = 2\n", "cells/cell_002.py": long_line})
    f = fingerprint.check(["call_model(", "x = 2"], t)
    assert len(f.evidence) == fingerprint.MAX_EVIDENCE
    # each fingerprint's first hit comes before anyone's second
    assert [e.path for e in f.evidence[:2]] == ["cells/cell_000.py", "cells/cell_001.py"]
    quote = fingerprint.check(["NEEDLE"], t).evidence[0].quote
    assert "NEEDLE" in quote and len(quote) <= fingerprint.QUOTE_CHARS


# ----- patch parsing -------------------------------------------------------------------------------

AGENT_DIFF = """diff --git a/src/agent.py b/src/agent.py
index 1111111..2222222 100644
--- a/src/agent.py
+++ b/src/agent.py
@@ -1,4 +1,4 @@
 import os

-SYSTEM = "You are a helpful assistant."
+SYSTEM = "Be brief. Answer with one number."

@@ -6,2 +6,2 @@ def answer(q):
     prompt = SYSTEM + q
-    return client.generate(prompt, temperature=0.7)
+    return client.generate(prompt, temperature=0.2, top_p=0.9)
"""

CONFIG_DIFF = """diff --git a/src/config.py b/src/config.py
index 5555555..6666666 100644
--- a/src/config.py
+++ b/src/config.py
@@ -1,3 +1,3 @@
-BATCH = 8
+BATCH = 16
 TIMEOUT = 30
-RETRIES = 1
+RETRIES = 3
"""

NEW_FILE_DIFF = """diff --git a/src/util.py b/src/util.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/src/util.py
@@ -0,0 +1,3 @@
+def clamp(v, lo, hi):
+    return max(lo, min(hi, v))
+CLAMP_RANGE = (0, 10)
\\ No newline at end of file
"""

DELETE_DIFF = """diff --git a/src/legacy.py b/src/legacy.py
deleted file mode 100644
index 4444444..0000000
--- a/src/legacy.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def legacy_scoring(x):
-    return x * 0.5 + 1
"""

OLD_AGENT = """import os

SYSTEM = "You are a helpful assistant."

def answer(q):
    prompt = SYSTEM + q
    return client.generate(prompt, temperature=0.7)
"""
NEW_AGENT = OLD_AGENT.replace("You are a helpful assistant.", "Be brief. Answer with one number.").replace(
    "temperature=0.7)", "temperature=0.2, top_p=0.9)"
)


def test_parse_git_diff_with_several_files_and_hunks():
    files = patch.parse(AGENT_DIFF + CONFIG_DIFF)
    assert [f.path for f in files] == ["src/agent.py", "src/config.py"]
    agent, config = files
    assert (agent.old_path, agent.new_path, agent.is_new, agent.is_deleted) == (
        "src/agent.py",
        "src/agent.py",
        False,
        False,
    )
    assert len(agent.hunks) == 2
    h1, h2 = agent.hunks
    assert h1.header == "@@ -1,4 +1,4 @@"
    assert h1.context == ["import os", "", ""]
    assert h1.removed == [['SYSTEM = "You are a helpful assistant."']]
    assert h1.added == [['SYSTEM = "Be brief. Answer with one number."']]
    assert h2.header.startswith("@@ -6,2 +6,2 @@") and h2.context == ["    prompt = SYSTEM + q"]
    # interleaved changes make one block per run
    (h,) = config.hunks
    assert h.removed == [["BATCH = 8"], ["RETRIES = 1"]]
    assert h.added == [["BATCH = 16"], ["RETRIES = 3"]]
    assert h.context == ["TIMEOUT = 30"]


def test_parse_new_file_deletion_and_no_newline_markers():
    new, gone = patch.parse(NEW_FILE_DIFF + DELETE_DIFF)
    assert (new.path, new.old_path, new.is_new, new.is_deleted) == ("src/util.py", None, True, False)
    assert new.hunks[0].added == [
        ["def clamp(v, lo, hi):", "    return max(lo, min(hi, v))", "CLAMP_RANGE = (0, 10)"]
    ]
    assert new.hunks[0].context == [] and new.hunks[0].removed == []
    assert (gone.path, gone.new_path, gone.is_deleted) == ("src/legacy.py", None, True)
    assert gone.hunks[0].removed == [["def legacy_scoring(x):", "    return x * 0.5 + 1"]]

    (limits,) = patch.parse(
        "--- a/limits.py\n+++ b/limits.py\n@@ -1 +1 @@\n-MAX_TOKENS = 512\n"
        "\\ No newline at end of file\n+MAX_TOKENS = 256\n\\ No newline at end of file\n"
    )
    assert limits.hunks[0].removed == [["MAX_TOKENS = 512"]]
    assert limits.hunks[0].added == [["MAX_TOKENS = 256"]]
    assert limits.hunks[0].context == []


def test_parse_plain_diff_trusts_hunk_counts():
    text = (
        "--- queries.sql\t2026-10-01 12:00:00.000000000 +0900\n"
        "+++ queries.sql\t2026-10-02 08:30:00.000000000 +0900\n"
        "@@ -1,3 +1,3 @@\n"
        " SELECT id FROM items\n"
        "--- drop the cache before scoring\n"
        "+++ keep the cache warm\n"
        " WHERE score > 0\n"
    )
    (f,) = patch.parse(text)
    assert f.path == "queries.sql" and len(f.hunks) == 1
    assert f.hunks[0].removed == [["-- drop the cache before scoring"]]
    assert f.hunks[0].added == [["++ keep the cache warm"]]
    assert f.hunks[0].context == ["SELECT id FROM items", "WHERE score > 0"]


def test_parse_survives_miscounted_hunks():
    # hand-edited: the first hunk claims 5 lines but has 2, the second claims 1 but has 3
    text = (
        "--- a/one.py\n+++ b/one.py\n@@ -1,5 +1,5 @@\n-A = 1\n+A = 2\n"
        "--- a/two.py\n+++ b/two.py\n@@ -1 +1 @@\n-B = 1\n+B = 2\n+C = 3\n"
    )
    one, two = patch.parse(text)
    assert (one.path, one.hunks[0].added) == ("one.py", [["A = 2"]])
    assert (two.path, two.hunks[0].added) == ("two.py", [["B = 2", "C = 3"]])


def test_parse_quoted_paths_and_garbage():
    (f,) = patch.parse('--- "a/caf\\303\\251.py"\n+++ "b/caf\\303\\251.py"\n@@ -1 +1 @@\n-x = 1\n+x = 2\n')
    assert f.path == "café.py"
    assert patch.parse("") == []
    assert patch.parse("just some notes\nabout a change\n") == []


# ----- patch matching ------------------------------------------------------------------------------


def test_patch_applied_is_present_with_the_first_added_lines_as_evidence():
    f = patch.check(AGENT_DIFF, make_tree({"files/src/agent.py": NEW_AGENT}))
    assert (f.status, f.method) == ("present", "patch")
    assert f.evidence == [
        Evidence("files/src/agent.py", 3, 'SYSTEM = "Be brief. Answer with one number."'),
        Evidence("files/src/agent.py", 7, "return client.generate(prompt, temperature=0.2, top_p=0.9)"),
    ]
    assert f.detail["hunks"] == 2 and f.detail["applied"] == 2
    assert f.detail["files"] == {
        "src/agent.py": {
            "hunks": ["applied", "applied"],
            "found_in": ["files/src/agent.py"],
            "new_file": False,
            "deleted": False,
        }
    }
    assert f.note == "2/2 hunks applied"


def test_patch_reverted_is_absent_with_the_removed_lines_as_evidence():
    f = patch.check(AGENT_DIFF, make_tree({"files/src/agent.py": OLD_AGENT}))
    assert f.status == "absent" and f.detail["reverted"] == 2
    assert [(e.line, e.quote) for e in f.evidence] == [
        (3, 'SYSTEM = "You are a helpful assistant."'),
        (7, "return client.generate(prompt, temperature=0.7)"),
    ]


def test_patch_partial_and_drifted():
    half = OLD_AGENT.replace("You are a helpful assistant.", "Be brief. Answer with one number.")
    f = patch.check(AGENT_DIFF, make_tree({"files/src/agent.py": half}))
    assert f.status == "partial" and (f.detail["applied"], f.detail["reverted"]) == (1, 1)
    assert f.detail["files"]["src/agent.py"]["hunks"] == ["applied", "reverted"]

    drifted = OLD_AGENT.replace("You are a helpful assistant.", "Think step by step.").replace(
        "temperature=0.7", "temperature=0.5"
    )
    f = patch.check(AGENT_DIFF, make_tree({"files/src/agent.py": drifted}))
    assert f.status == "unknown" and f.detail["unknown"] == 2 and f.evidence == []


def test_patch_matching_ignores_whitespace_differences():
    inlined = (
        "import  os\n"
        '\tSYSTEM  =  "Be brief. Answer with one number."\n'
        "    def answer(self, q):\n"
        "\t\tprompt = SYSTEM + q\n"
        "\t\treturn client.generate(prompt,  temperature=0.2,  top_p=0.9)\n"
    )
    f = patch.check(AGENT_DIFF, make_tree({"cells/cell_003.py": inlined}))
    assert f.status == "present"
    assert [(e.path, e.line) for e in f.evidence] == [("cells/cell_003.py", 2), ("cells/cell_003.py", 5)]
    # the same lines without the context they were changed in: no longer evidence of this patch
    moved = inlined.replace("import  os\n", "class Agent:\n")
    f = patch.check(AGENT_DIFF, make_tree({"cells/cell_003.py": moved}))
    assert f.status == "partial" and f.detail["files"]["src/agent.py"]["hunks"] == ["unknown", "applied"]


def test_patch_ignores_trivial_lines():
    assert patch.is_trivial("    pass") and patch.is_trivial("else:") and patch.is_trivial("x=1")
    assert patch.is_trivial("    }));") and patch.is_trivial("")
    assert not patch.is_trivial("x = 1") and not patch.is_trivial("return x")
    only_trivial = (
        "@@ -8,2 +8,4 @@ def main():\n     for q in load():\n+        pass\n+\n         print(answer(q))\n"
    )
    f = patch.check(AGENT_DIFF + only_trivial, make_tree({"files/src/agent.py": NEW_AGENT}))
    assert f.status == "present"
    assert (f.detail["hunks"], f.detail["applied"], f.detail["trivial"]) == (3, 2, 1)

    # a trivial removed block does not count: this hunk is judged by its added line alone
    warmup = '--- a/src/agent.py\n+++ b/src/agent.py\n@@ -1,2 +1,2 @@\n def warmup():\n-    pass\n+    client.generate("ping", max_tokens=1)\n'
    applied = make_tree({"cells/cell_000.py": 'def warmup():\n    client.generate("ping", max_tokens=1)\n'})
    assert patch.check(warmup, applied).status == "present"
    assert (
        patch.check(warmup, make_tree({"cells/cell_000.py": "def warmup():\n    pass\n"})).status == "absent"
    )

    f = patch.check("--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n x = compute()\n+pass\n", applied)
    assert f.status == "unknown" and "trivial" in f.note
    # re-indenting changes nothing once whitespace is normalised
    reindent = (
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n def f():\n-  return compute(x)\n+    return compute(x)\n"
    )
    assert patch.check(reindent, applied).detail["trivial"] == 1


def test_patch_candidate_files_are_tried_by_path_then_basename_then_everywhere():
    def evidence_paths(files: dict[str, str]) -> set[str]:
        return {e.path for e in patch.check(AGENT_DIFF, make_tree(files)).evidence}

    # the same code in a cell and in the written file: the file wins, though "cells/" sorts first
    assert evidence_paths({"cells/cell_000.py": NEW_AGENT, "files/src/agent.py": NEW_AGENT}) == {
        "files/src/agent.py"
    }
    assert evidence_paths({"cells/cell_000.py": NEW_AGENT, "files/agent.py": NEW_AGENT}) == {"files/agent.py"}
    assert evidence_paths({"cells/cell_000.py": "x = 1\n", "cells/cell_007.py": NEW_AGENT}) == {
        "cells/cell_007.py"
    }
    # a file written twice keeps both versions; the second one ("~2") still matches by path
    assert evidence_paths(
        {"cells/cell_000.py": NEW_AGENT, "files/src/agent.py": OLD_AGENT, "files/src/agent.py~2": NEW_AGENT}
    ) == {"files/src/agent.py~2"}


def test_patch_new_files():
    new_util = "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))\nCLAMP_RANGE = (0, 10)\n"
    f = patch.check(NEW_FILE_DIFF, make_tree({"files/src/util.py": new_util}))
    assert f.status == "present" and f.detail["files"]["src/util.py"]["new_file"]
    # nowhere in the notebook, not even a file of that name: absent
    assert patch.check(NEW_FILE_DIFF, make_tree({"cells/cell_000.py": "import os\n"})).status == "absent"
    # a util.py with other content: the code drifted
    other = make_tree({"files/util.py": "def clamp(v):\n    return v\n"})
    assert patch.check(NEW_FILE_DIFF, other).status == "unknown"


def test_patch_deleted_files():
    f = patch.check(DELETE_DIFF, make_tree({"cells/cell_000.py": "score = new_scoring(x)\n"}))
    assert f.status == "present" and f.detail["files"]["src/legacy.py"]["deleted"]
    still_there = make_tree({"cells/cell_000.py": "def legacy_scoring(x):\n    return x * 0.5 + 1\n"})
    f = patch.check(DELETE_DIFF, still_there)
    assert f.status == "absent" and f.evidence[0] == Evidence(
        "cells/cell_000.py", 1, "def legacy_scoring(x):"
    )


def test_patch_removal_only_and_addition_only_hunks():
    removal = (
        "--- a/src/agent.py\n+++ b/src/agent.py\n@@ -6,3 +6,2 @@ def answer(q):\n"
        "     prompt = SYSTEM + q\n-    prompt = add_examples(prompt, n=8)\n"
        "     return client.generate(prompt, temperature=0.2)\n"
    )
    removed = make_tree(
        {"cells/cell_000.py": "prompt = SYSTEM + q\nreturn client.generate(prompt, temperature=0.2)\n"}
    )
    assert patch.check(removal, removed).status == "present"
    kept = make_tree(
        {
            "cells/cell_000.py": "prompt = SYSTEM + q\nprompt = add_examples(prompt, n=8)\n"
            "return client.generate(prompt, temperature=0.2)\n"
        }
    )
    assert patch.check(removal, kept).status == "absent"
    # neither the removed line nor any context line: unknown
    assert patch.check(removal, make_tree({"cells/cell_000.py": "import os\n"})).status == "unknown"

    addition = (
        "--- a/src/agent.py\n+++ b/src/agent.py\n@@ -6,2 +6,3 @@ def answer(q):\n"
        "     prompt = SYSTEM + q\n+    prompt = prompt.strip()[:4000]\n"
        "     return client.generate(prompt, temperature=0.2)\n"
    )
    assert patch.check(addition, removed).status == "absent"  # its place found, without the addition
    assert patch.check(addition, make_tree({"cells/cell_000.py": "import os\n"})).status == "unknown"
    with_it = make_tree(
        {
            "cells/cell_000.py": "prompt = SYSTEM + q\nprompt = prompt.strip()[:4000]\n"
            "return client.generate(prompt, temperature=0.2)\n"
        }
    )
    assert patch.check(addition, with_it).status == "present"
    # the added line alone, without the context it was added in, is not evidence of this hunk
    alone = make_tree({"cells/cell_000.py": "prompt = prompt.strip()[:4000]\n"})
    assert patch.check(addition, alone).status == "unknown"


def test_patch_empty_or_unparsable_diff_and_empty_tree():
    tree = make_tree({"files/src/agent.py": NEW_AGENT})
    for text in (
        "",
        "not a diff at all\n",
        "diff --git a/x.bin b/x.bin\nBinary files a/x.bin and b/x.bin differ\n",
    ):
        f = patch.check(text, tree)
        assert (f.status, f.method) == ("unknown", "patch") and "no hunks" in f.note
    f = patch.check(AGENT_DIFF, Tree())
    assert f.status == "unknown" and "no code" in f.note


def test_patch_with_a_byte_order_mark_still_has_its_first_file_header():
    plain = "﻿" + CONFIG_DIFF[CONFIG_DIFF.index("--- ") :]  # starts with the "---" line
    (f,) = patch.parse(plain)
    assert f.path == "src/config.py" and len(f.hunks) == 1
    tree = make_tree({"files/src/config.py": "BATCH = 16\nTIMEOUT = 30\nRETRIES = 3\n"})
    assert patch.check(plain, tree).detail["files"] == {
        "src/config.py": {
            "hunks": ["applied"],
            "found_in": ["files/src/config.py"],
            "new_file": False,
            "deleted": False,
        }
    }


# ----- patch matching: a change is judged at its place -----------------------------------------------

STRIP_DIFF = """--- a/agent.py
+++ b/agent.py
@@ -6,2 +6,3 @@ def solve(llm, prompt):
     out = llm.generate(prompt)
+    out = out.strip()
     return parse(out)
"""
HELPER = "def clean(out):\n    out = out.strip()\n    return out\n\n\n"
SOLVE = "def solve(llm, prompt):\n    out = llm.generate(prompt)\n    return parse(out)\n"


def test_an_added_line_found_away_from_its_context_is_not_applied():
    # the added line exists in an unrelated helper, and solve() is unpatched: the hunk is reverted
    f = patch.check(STRIP_DIFF, make_tree({"cells/cell_000.py": HELPER + SOLVE}))
    assert f.status == "absent" and f.detail["files"]["agent.py"]["hunks"] == ["reverted"]
    # solve() rewritten (its context is gone) while the helper still has the line: nothing is known
    drifted = HELPER + "def solve(llm, prompt):\n    return parse(llm.generate(prompt)[0])\n"
    assert patch.check(STRIP_DIFF, make_tree({"cells/cell_000.py": drifted})).status == "unknown"
    # patched: the evidence is the line inside solve(), not the helper's
    patched = HELPER + SOLVE.replace("(prompt)\n", "(prompt)\n    out = out.strip()\n")
    f = patch.check(STRIP_DIFF, make_tree({"cells/cell_000.py": patched}))
    assert f.status == "present" and f.evidence == [Evidence("cells/cell_000.py", 8, "out = out.strip()")]
    # the same holds after the context line: a block anchored only below is applied too
    below = "def solve(llm, prompt):\n    out = llm.generate(prompt, n=2)[0]\n    out = out.strip()\n"
    below += "    return parse(out)\n"
    assert patch.check(STRIP_DIFF, make_tree({"cells/cell_000.py": below})).status == "present"


def test_reverted_needs_the_old_side_in_place():
    # (a) an add-only hunk applied, then reformatted: unknown, not reverted
    quotes = (
        "--- a/agent.py\n+++ b/agent.py\n@@ -1,3 +1,4 @@\n def build(prompt):\n"
        "     prompt = prompt.strip()\n+    prompt = prompt.replace('\\t', ' ')\n     return prompt\n"
    )
    reformatted = (
        'def build(prompt):\n    prompt = prompt.strip()\n    prompt = prompt.replace("\\t", " ")\n'
        "    return prompt\n"
    )
    assert patch.check(quotes, make_tree({"cells/cell_000.py": reformatted})).status == "unknown"
    unpatched = "def build(prompt):\n    prompt = prompt.strip()\n    return prompt\n"
    assert patch.check(quotes, make_tree({"cells/cell_000.py": unpatched})).status == "absent"

    # (b) a removal applied, while the removed line still exists in another function: applied
    removal = (
        "--- a/agent.py\n+++ b/agent.py\n@@ -1,4 +1,3 @@\n def answer(q):\n     out = generate(q)\n"
        "-    print(out)\n     return out\n"
    )
    other = "def debug(out):\n    print(out)\n\n\ndef answer(q):\n    out = generate(q)\n    return out\n"
    f = patch.check(removal, make_tree({"cells/cell_000.py": other}))
    assert f.status == "present", f.note
    kept = "def answer(q):\n    out = generate(q)\n    print(out)\n    return out\n"
    f = patch.check(removal, make_tree({"cells/cell_000.py": kept}))
    assert f.status == "absent" and f.evidence == [Evidence("cells/cell_000.py", 3, "print(out)")]


ANSWER_PY = (
    "import re\n\n"
    'ANSWER_RE = re.compile(r"Answer:\\s*(-?\\d+)")\n\n\n'
    "def extract_answer(text):\n"
    "    match = ANSWER_RE.search(text)\n"
    "    if match is None:\n"
    "        return None\n"
    "    return int(match.group(1))\n"
)
ANSWER_DIFF = "--- /dev/null\n+++ b/src/answer.py\n@@ -0,0 +1,10 @@\n" + "".join(
    "+" + line + "\n" for line in ANSWER_PY.splitlines()
)


def test_a_new_file_is_judged_by_how_much_of_it_is_found():
    assert patch.check(ANSWER_DIFF, make_tree({"cells/cell_001.py": ANSWER_PY})).status == "present"
    # (c) inlined in a cell, then one line edited: most of it is there, so not absent
    edited = ANSWER_PY.replace("int(match", "float(match")
    f = patch.check(ANSWER_DIFF, make_tree({"cells/cell_001.py": edited}))
    assert f.status == "unknown" and f.detail["files"]["src/answer.py"]["lines_found"] == [6, 7]
    # not there at all, apart from a common import: absent
    f = patch.check(ANSWER_DIFF, make_tree({"cells/cell_000.py": "import re\nimport os\n"}))
    assert f.status == "absent" and f.detail["files"]["src/answer.py"]["lines_found"] == [1, 7]
    # a deleted file inlined elsewhere with an edit is not "deleted" either
    deleted = "--- a/src/answer.py\n+++ /dev/null\n@@ -1,10 +0,0 @@\n" + "".join(
        "-" + line + "\n" for line in ANSWER_PY.splitlines()
    )
    assert patch.check(deleted, make_tree({"cells/cell_001.py": edited})).status == "unknown"
    assert patch.check(deleted, make_tree({"cells/cell_000.py": "import re\n"})).status == "present"


def test_changes_at_the_start_or_end_of_a_file_use_the_file_boundary():
    top = "--- a/agent.py\n+++ b/agent.py\n@@ -1,2 +1,3 @@\n+import json\n import os\n import re\n"
    assert patch.check(top, make_tree({"files/agent.py": "import json\nimport os\nimport re\n"})).status == (
        "present"
    )
    assert patch.check(top, make_tree({"files/agent.py": "import os\nimport re\n"})).status == "absent"
    # "import os" further down a file is not the place this hunk changes
    assert patch.check(top, make_tree({"files/agent.py": "x = 1\nimport os\nimport re\n"})).status == (
        "unknown"
    )
    end = "--- a/agent.py\n+++ b/agent.py\n@@ -1,2 +1 @@\n x = load()\n-print(x)\n"
    assert patch.check(end, make_tree({"files/agent.py": "x = load()\n\n"})).status == "present"
    assert patch.check(end, make_tree({"files/agent.py": "x = load()\nprint(x)\n"})).status == "absent"
    assert patch.check(end, make_tree({"files/agent.py": "x = load()\nrun(x)\n"})).status == "unknown"


def test_a_hunk_with_two_changes_needs_both_in_place():
    diff = (
        "--- a/config.py\n+++ b/config.py\n@@ -1,5 +1,5 @@\n-BATCH = 8\n+BATCH = 16\n TIMEOUT = 30\n"
        "-RETRIES = 1\n+RETRIES = 3\n VERBOSE = False\n"
    )
    both = make_tree({"files/config.py": "BATCH = 16\nTIMEOUT = 30\nRETRIES = 3\nVERBOSE = False\n"})
    assert patch.check(diff, both).status == "present"
    neither = make_tree({"files/config.py": "BATCH = 8\nTIMEOUT = 30\nRETRIES = 1\nVERBOSE = False\n"})
    assert patch.check(diff, neither).status == "absent"
    half = make_tree({"files/config.py": "BATCH = 16\nTIMEOUT = 30\nRETRIES = 1\nVERBOSE = False\n"})
    f = patch.check(diff, half)
    assert f.status == "unknown" and f.detail["unknown"] == 1


TWO_SOLVERS_DIFF = (
    "--- a/agent.py\n+++ b/agent.py\n"
    "@@ -1,4 +1,5 @@\n def solve_math(llm, q):\n     out = llm.generate(MATH_PROMPT + q)\n"
    "+    out = out.strip()\n     return out\n"
    "@@ -6,4 +7,5 @@\n def solve_code(llm, q):\n     out = llm.generate(CODE_PROMPT + q)\n"
    "+    out = out.strip()\n     return out\n"
)
TWO_SOLVERS = (
    "def solve_math(llm, q):\n    out = llm.generate(MATH_PROMPT + q)\n"
    "    out = out.strip()\n    return out\n\n"
    "def solve_code(llm, q):\n    out = llm.generate(CODE_PROMPT + q)\n"
    "    out = out.strip()\n    return out\n"
)


def test_added_lines_next_to_one_anchor_do_not_count_while_the_old_side_is_in_place():
    # only solve_math is patched: solve_code's hunk finds its added line above solve_math's "return out",
    # but solve_code's own anchors are still next to each other
    half = TWO_SOLVERS.replace("CODE_PROMPT + q)\n    out = out.strip()\n", "CODE_PROMPT + q)\n")
    f = patch.check(TWO_SOLVERS_DIFF, make_tree({"files/agent.py": half}))
    assert f.status == "partial" and f.detail["files"]["agent.py"]["hunks"] == ["applied", "unknown"]
    assert f.evidence == [Evidence("files/agent.py", 3, "out = out.strip()")]
    f = patch.check(TWO_SOLVERS_DIFF, make_tree({"files/agent.py": TWO_SOLVERS}))
    assert f.status == "present" and [e.line for e in f.evidence] == [3, 8]
    none = half.replace("    out = out.strip()\n", "")
    assert patch.check(TWO_SOLVERS_DIFF, make_tree({"files/agent.py": none})).status == "absent"

    # an unpatched solve() next to a helper that has the added line before the same after-anchor
    diff = (
        "--- a/agent.py\n+++ b/agent.py\n@@ -1,3 +1,4 @@\n def solve(llm, prompt):\n"
        "     out = llm.generate(prompt)\n+    out = out.strip()\n     return out\n"
    )
    solve = "def solve(llm, prompt):\n    out = llm.generate(prompt)\n    return out\n"
    assert patch.check(diff, make_tree({"cells/cell_000.py": HELPER + solve})).status == "unknown"
    patched = HELPER + solve.replace("(prompt)\n", "(prompt)\n    out = out.strip()\n")
    f = patch.check(diff, make_tree({"cells/cell_000.py": patched}))
    assert f.status == "present" and f.evidence == [Evidence("cells/cell_000.py", 8, "out = out.strip()")]

    # a hunk that adds a copy of its own before-anchor: unpatched code has the "new side" by accident
    dup = (
        "--- a/run.py\n+++ b/run.py\n@@ -1,2 +1,3 @@\n model.warmup(batch)\n+model.warmup(batch)\n"
        " results = model.run(batch)\n"
    )
    once = "model.warmup(batch)\nresults = model.run(batch)\n"
    assert patch.check(dup, make_tree({"files/run.py": once})).status == "unknown"
    twice = "model.warmup(batch)\nmodel.warmup(batch)\nresults = model.run(batch)\n"
    assert patch.check(dup, make_tree({"files/run.py": twice})).status == "present"


def test_hunks_without_context_look_for_their_blocks_in_the_diffs_own_file():
    unpatched = "def solve(llm, prompt):\n    out = llm.generate(prompt)\n    return parse(out)\n"
    # a -U0 change: the file still has the old line, and an unrelated cell happens to have the new one
    change = (
        "--- a/src/agent.py\n+++ b/src/agent.py\n@@ -2 +2 @@\n-    out = llm.generate(prompt)\n"
        "+    out = llm.generate(prompt, temperature=0.2)\n"
    )
    other = "    out = llm.generate(prompt, temperature=0.2)\n"
    f = patch.check(change, make_tree({"files/src/agent.py": unpatched, "cells/cell_009.py": other}))
    assert f.status == "absent" and f.evidence == [
        Evidence("files/src/agent.py", 2, "out = llm.generate(prompt)")
    ]
    # without the file in the notebook, every evidence path is searched
    assert patch.check(change, make_tree({"cells/cell_009.py": other})).status == "present"
    # a -U0 addition: the file is unpatched, a helper cell has the added line
    addition = "--- a/src/agent.py\n+++ b/src/agent.py\n@@ -2,0 +3 @@\n+    out = out.strip()\n"
    files = {"files/src/agent.py": unpatched, "cells/cell_000.py": HELPER}
    assert patch.check(addition, make_tree(files)).status == "absent"
    # a -U0 removal: the file is patched, an unrelated cell still has the removed line
    removal = "--- a/src/agent.py\n+++ b/src/agent.py\n@@ -3 +2,0 @@\n-    print(out)\n"
    patched = "def answer(q):\n    out = generate(q)\n    return out\n"
    files = {"files/src/agent.py": patched, "cells/cell_000.py": "def debug(out):\n    print(out)\n"}
    assert patch.check(removal, make_tree(files)).status == "present"
    # a hunk whose only context lines are trivial ones
    trivial_context = (
        "--- a/agent.py\n+++ b/agent.py\n@@ -10,4 +10,5 @@\n     )\n \n+    retries = 3\n     return\n"
    )
    files = {"files/agent.py": "x = f(\n    a,\n    )\n\n    return\n", "cells/cell_001.py": "retries = 3\n"}
    assert patch.check(trivial_context, make_tree(files)).status == "absent"
    files["files/agent.py"] = "x = f(\n    a,\n    )\n\n    retries = 3\n    return\n"
    assert patch.check(trivial_context, make_tree(files)).status == "present"


def test_trivial_lines_inside_a_change_are_compared_like_other_lines():
    """Trivial lines are never anchors and never a change by themselves, but next to changed lines they are
    part of the change's text: they can pin its place, and when they drift the change is not recognised."""
    diff = (
        "--- a/f.py\n+++ b/f.py\n@@ -1,4 +1,5 @@\n def f(x):\n     y = g(x)\n+    y = y.strip()\n"
        "     pass\n     return y\n"
    )
    applied = "def f(x):\n    y = g(x)\n    y = y.strip()\n    pass\n    return y\n"
    assert patch.check(diff, make_tree({"files/f.py": applied})).status == "present"
    # the `pass` removed later: neither side is in place any more, so the hunk is unknown (never a guess)
    assert patch.check(diff, make_tree({"files/f.py": applied.replace("    pass\n", "")})).status == "unknown"
    # a -U1 hunk: the trivial `pass` is all that places the change, and it still does
    u1 = (
        "--- a/agent.py\n+++ b/agent.py\n@@ -11,3 +11,3 @@ class Agent:\n     def warmup(self):\n"
        '-        pass\n+        self.client.generate("ping", max_tokens=1)\n \n'
    )
    agent = "class Agent:\n    def warmup(self):\n        pass\n\n    def run(self, q):\n        return q\n"
    assert patch.check(u1, make_tree({"files/agent.py": agent})).status == "absent"
    warm = agent.replace("pass", 'self.client.generate("ping", max_tokens=1)')
    assert patch.check(u1, make_tree({"files/agent.py": warm})).status == "present"


def test_checks_of_one_tree_share_its_index_until_the_tree_changes():
    tree = make_tree({"files/src/agent.py": OLD_AGENT})
    assert patch.check(AGENT_DIFF, tree).status == "absent"
    assert patch.check(CONFIG_DIFF + AGENT_DIFF, tree).detail["files"]["src/agent.py"]["hunks"] == [
        "reverted",
        "reverted",
    ]
    tree.add_text("files/src/agent.py", NEW_AGENT, "edited", replace=True)
    f = patch.check(AGENT_DIFF, tree)
    assert f.status == "present" and f.evidence[0].quote == 'SYSTEM = "Be brief. Answer with one number."'
    tree.add_text("cells/cell_000.py", OLD_AGENT, "a cell")  # one more file: the diff's file still wins
    assert patch.check(AGENT_DIFF, tree).status == "present"


def _formatter_diff(n: int) -> tuple[str, str, str]:
    """A diff that changes the quotes on every other line of an n-line file: (diff, old file, new file)."""
    old, new = [], []
    for i in range(n):
        if i % 2:
            old.append(f"    key_{i} = 'value {i}'")
            new.append(f'    key_{i} = "value {i}"')
        else:
            old.append(f"    other_{i} = compute({i})")
            new.append(f"    other_{i} = compute({i})")
    lines = ["--- a/src/m.py", "+++ b/src/m.py", f"@@ -1,{n} +1,{n} @@"]
    for o, w in zip(old, new, strict=True):
        lines += [" " + o] if o == w else ["-" + o, "+" + w]
    return "\n".join(lines) + "\n", "\n".join(old) + "\n", "\n".join(new) + "\n"


def test_patch_matching_takes_near_linear_time():
    diff, old, new = _formatter_diff(8000)
    filler = {f"cells/cell_{i:03d}.py": "".join(f"v_{i}_{j} = {j}\n" for j in range(500)) for i in range(40)}
    for files, status in (
        ({"files/src/m.py": new}, "present"),
        ({"files/src/m.py": old}, "absent"),
        ({**filler, "cells/cell_999.py": new}, "present"),  # inlined in a cell: every path is a candidate
    ):
        tree = make_tree(files)
        start = time.perf_counter()
        f = patch.check(diff, tree)
        elapsed = time.perf_counter() - start
        assert f.status == status and elapsed < 1.0, (status, elapsed)
    # many small hunks over a long file
    hunks, body = [], []
    for i in range(1000):
        body += [f"def f_{i}(x):", f"    y_{i} = g(x)", f"    return y_{i}"]
        hunks.append(
            f"@@ -{3 * i + 1},3 +{3 * i + 1},3 @@\n def f_{i}(x):\n-    y_{i} = g(x)\n+    y_{i} = h(x)\n"
            f"     return y_{i}\n"
        )
    diff = "--- a/big.py\n+++ b/big.py\n" + "".join(hunks)
    tree = make_tree({"files/big.py": "\n".join(body) + "\n"})
    start = time.perf_counter()
    f = patch.check(diff, tree)
    elapsed = time.perf_counter() - start
    assert f.status == "absent" and f.detail["reverted"] == 1000 and elapsed < 1.0, elapsed


# ----- multi-line fingerprints and match columns ------------------------------------------------------

SOLVER = (
    "import re\n\n"
    "def solve(llm, problem, sampling, retries=2):\n"
    "    for _ in range(retries + 1):\n"
    "        out = llm.generate(problem, sampling)\n"
    "        answer = parse_answer(out)\n"
)


def test_text_fingerprints_spanning_lines_match_consecutive_lines():
    t = make_tree({"cells/cell_000.py": "x = 1\n", "files/solver.py": SOLVER})
    f = fingerprint.check(["for _ in range(retries + 1):\n    out = llm.generate"], t)
    assert f.status == "present" and f.detail["found"]
    (ev,) = f.evidence
    assert (ev.path, ev.line) == ("files/solver.py", 4)
    assert ev.quote.startswith("for _ in range(retries + 1):") and "out = llm.generate" in ev.quote
    assert ev.col == 0
    # the first line may be the end of a line, whitespace and blank lines do not matter
    f = fingerprint.check(["solve(llm, problem,  sampling, retries=2):\n\n  for _ in range("], t)
    assert f.status == "present" and f.evidence[0].line == 3
    assert f.evidence[0].quote[f.evidence[0].col :].startswith("solve(llm")
    # negative ones are forbidden when found, and pasted snippets keep working with \r\n
    f = fingerprint.check(["!def solve(llm, problem, sampling, retries=2):\r\n    for _ in"], t)
    assert f.status == "absent" and f.detail["forbidden_found"]
    assert fingerprint.check(["out = llm.generate(problem)\nanswer = parse_answer(out)"], t).status == (
        "absent"
    )


def test_regex_fingerprints_spanning_lines_search_whole_files():
    t = make_tree({"files/solver.py": SOLVER})
    f = fingerprint.check(["re:range\\(retries \\+ 1\\):\n\\s+out = llm\\.generate"], t)
    assert f.status == "present" and (f.evidence[0].path, f.evidence[0].line) == ("files/solver.py", 4)
    assert f.evidence[0].quote[f.evidence[0].col :].startswith("range(retries + 1):")
    # an escaped \n works the same way as a typed newline
    assert fingerprint.check([r"re:retries=2\):\n\s+for "], t).evidence[0].line == 3
    # (?m) searches the whole file, with ^ and $ at line boundaries
    f = fingerprint.check([r"re:(?m)^\s+answer = parse_answer\(out\)$"], t)
    assert f.status == "present" and f.evidence[0].line == 6
    assert fingerprint.check([r"!re:(?m)^import re$\n^def solve"], t).status == "present"


def test_multi_line_regex_evidence_starts_at_the_first_line_of_its_quote():
    solver = SOLVER.replace("        answer", "\n        answer")  # a blank line 6, the answer on line 7
    spec = r"re:(?m)^\s*answer = parse_answer\(out\)$"
    (ev,) = fingerprint.check([spec], make_tree({"files/solver.py": solver})).evidence
    assert (ev.line, ev.quote, ev.col) == (7, "answer = parse_answer(out)", 0)
    (ev,) = fingerprint.check([spec], make_tree({"cells/cell_000.py": solver.replace("\n", "\r\n")})).evidence
    assert (ev.line, ev.quote) == (7, "answer = parse_answer(out)")
    spaced = solver.replace("\n\n        answer", "\n   \t \n        answer")  # a whitespace-only line 6
    (ev,) = fingerprint.check([spec], make_tree({"files/solver.py": spaced})).evidence
    assert (ev.line, ev.quote) == (7, "answer = parse_answer(out)")
    # a long match is cut around where it starts: the line is the one its cut quote starts on
    text = "x = 1" + " " * 150 + "\n    call(" + "a" * 400 + ")\n    done = True\n"
    (ev,) = fingerprint.check([r"re:\n\s*call\(a+\)"], make_tree({"files/long.py": text})).evidence
    assert (ev.line, ev.col) == (2, 0) and ev.quote.startswith("call(aaa")


def test_fingerprint_evidence_records_where_the_match_starts():
    prompt = 'SYSTEM_PROMPT = "You solve math word problems. Think step by step, briefly."\n'
    t = make_tree({"files/solver.py": "    " + prompt})
    (ev,) = fingerprint.check(["Think step by step, briefly."], t).evidence
    assert ev.quote == prompt.strip() and ev.quote[ev.col :].startswith("Think step by step")
    (ev,) = fingerprint.check([r"re:^\s+SYSTEM_PROMPT"], t).evidence
    assert ev.col == 0
    long_line = "blob = '" + "a" * 800 + "NEEDLE" + "b" * 400 + "'\n"
    (ev,) = fingerprint.check(["NEEDLE"], make_tree({"cells/cell_000.py": long_line})).evidence
    assert len(ev.quote) <= fingerprint.QUOTE_CHARS and ev.quote[ev.col :].startswith("NEEDLE")
    assert ev.to_dict()["col"] == ev.col
