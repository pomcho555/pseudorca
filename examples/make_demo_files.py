"""Write synthetic demo inputs for Pseudorca: four notebook versions, a patch and four results tables.

    python examples/make_demo_files.py DEMO_DIR

Everything here is made up: a small offline word-problem solver, nothing from any real competition.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import lzma
import random
import sys
import tarfile
from pathlib import Path

SOLVER_V1 = """\
import re

SYSTEM_PROMPT = "You solve math word problems. Think step by step."


def parse_answer(text):
    m = re.search(r"Answer:\\s*(-?\\d+(?:\\.\\d+)?)", text)
    return float(m.group(1)) if m else None


def solve(llm, problem, sampling):
    out = llm.generate([SYSTEM_PROMPT + "\\n\\n" + problem], sampling)[0].outputs[0].text
    return parse_answer(out)
"""

SOLVER_V2 = """\
import re

SYSTEM_PROMPT = "You solve math word problems. Think step by step, briefly."


def parse_answer(text):
    m = re.search(r"Answer:\\s*(-?\\d+(?:\\.\\d+)?)", text)
    return float(m.group(1)) if m else None


def solve(llm, problem, sampling, retries=2):
    for _ in range(retries + 1):
        out = llm.generate([SYSTEM_PROMPT + "\\n\\n" + problem], sampling)[0].outputs[0].text
        answer = parse_answer(out)
        if answer is not None:
            return answer
    return None
"""

SOLVER_LEAN = SOLVER_V1.replace("Think step by step.", "Think step by step, briefly.")  # no retries

RETRY_PATCH = """\
--- a/solver.py
+++ b/solver.py
@@ -10,6 +10,10 @@ def parse_answer(text):
     return float(m.group(1)) if m else None


-def solve(llm, problem, sampling):
-    out = llm.generate([SYSTEM_PROMPT + "\\n\\n" + problem], sampling)[0].outputs[0].text
-    return parse_answer(out)
+def solve(llm, problem, sampling, retries=2):
+    for _ in range(retries + 1):
+        out = llm.generate([SYSTEM_PROMPT + "\\n\\n" + problem], sampling)[0].outputs[0].text
+        answer = parse_answer(out)
+        if answer is not None:
+            return answer
+    return None
"""

UTILS = {
    "utils/timer.py": "import time\n\n\nclass Budget:\n    def __init__(self, seconds):\n        self.end = time.monotonic() + seconds\n\n    def left(self):\n        return self.end - time.monotonic()\n",
    "utils/io.py": "import json\n\n\ndef write_answers(path, answers):\n    with open(path, 'w') as f:\n        json.dump(answers, f)\n",
}


def _payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, text in UTILS.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(lzma.compress(buf.getvalue())).decode()


def _notebook(solver: str, llm_kwargs: str, temperature: str) -> dict:
    def code(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "# Word-problem solver\nOffline vLLM and a small solver.",
            },
            code(
                "!pip install --no-index --find-links /kaggle/input/wheels vllm\n%env VLLM_LOGGING_LEVEL=WARNING"
            ),
            code("%%writefile solver.py\n" + solver),
            code(
                "import base64, io, lzma, tarfile\n\n"
                f'UTILS = "{_payload()}"\n'
                "with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode(UTILS)))) as tf:\n"
                '    tf.extractall("utils_pkg")\n'
            ),
            code(
                "from vllm import LLM, SamplingParams\n\n"
                f'llm = LLM(model="/kaggle/input/my-model/weights"{llm_kwargs})\n'
                f"sampling = SamplingParams(temperature={temperature}, max_tokens=1024)\n"
            ),
        ],
    }


def _results(path: Path, lift: float, seed: int) -> None:
    rng = random.Random(seed)
    base = random.Random(7)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "draw", "score", "seconds"])
        for i in range(60):
            p = min(0.97, base.uniform(0.2, 0.9) + lift)
            for d in range(3):
                w.writerow([f"problem-{i:03d}", d, int(rng.random() < p), round(rng.uniform(20, 90), 1)])


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "base.ipynb").write_text(json.dumps(_notebook(SOLVER_V1, "", "0.7"), indent=1))
    (out / "next.ipynb").write_text(
        json.dumps(
            _notebook(SOLVER_V2, ", enable_prefix_caching=True, gpu_memory_utilization=0.92", "0.7"), indent=1
        )
    )
    cache = ", enable_prefix_caching=True, gpu_memory_utilization=0.92"
    (out / "hot.ipynb").write_text(json.dumps(_notebook(SOLVER_V2, cache, "0.6"), indent=1))  # next, hotter
    (out / "lean.ipynb").write_text(json.dumps(_notebook(SOLVER_LEAN, cache, "0.7"), indent=1))  # next, no retries
    (out / "retry-parse.diff").write_text(RETRY_PATCH)
    _results(out / "runs" / "base.csv", 0.0, seed=1)
    _results(out / "runs" / "next.csv", 0.08, seed=2)
    _results(out / "runs" / "hot.csv", -0.04, seed=3)
    _results(out / "runs" / "lean.csv", 0.06, seed=4)
    print(f"wrote demo files to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pseudorca-demo"))
