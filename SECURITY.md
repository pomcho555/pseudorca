# Security

Pseudorca reads notebooks written by other people, sometimes downloaded from the internet. It is built so
that reading them is safe and that a team's code and results stay private.

## What Pseudorca does and does not do

- **It never runs notebook content.** Notebooks and scripts are parsed with `ast.parse` and searched as
  text. Nothing from a notebook, an embedded payload or an archive is executed, imported, evaluated or
  unpickled. Pickled payloads are recorded as opaque binary entries.
- **Decoding is bounded.** Embedded payloads (base64, zlib, gzip, xz/lzma, bz2, tar, zip) are decoded with
  caps on bytes per payload, bytes per notebook, decode depth and archive members. Archive member names are
  normalised into virtual paths. The extracted tree is never written to disk under those names; the cache
  stores contents under their sha256.
- **Code leaves your machine only when you allow it.** The LLM audit is off until you configure a model.
  Any endpoint that is not on localhost needs `allow_remote = true`, including the Anthropic API.
  - Before any text goes to an LLM, common secret formats are replaced with `[REDACTED]`: API keys and
    tokens for OpenAI, Anthropic, GitHub, Hugging Face, AWS, Google, Slack and Kaggle, private keys, and
    `password = "..."`-style assignments.
  - Redaction is best effort. Do not rely on it as your only protection.
- **LLM answers are not trusted.** Notebook text is treated as untrusted data in the prompt. Any quote the
  LLM gives as evidence must be found in the notebook; otherwise the verdict becomes `unknown`. The LLM
  has read-only tools (list, substring search, read) over the extracted text and cannot act on anything.
- **Kaggle access is read-only.** Pseudorca only lists submissions and pulls kernel sources. It never
  submits, pushes or creates anything, and it never prints credentials.
- **No telemetry and no server.** The ledger is plain files in your repository.

## Keeping your data private

- The ledger (`.pseudorca/`) holds your notebooks, results and verdicts. Keep it in a **private**
  repository.
- Do not publish `pseudorca report` output. GitHub Pages sites are public even when the repository is
  private.
- In shared competitions, use one store per team. Do not move one team's store into another team's
  repository.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting ("Report a vulnerability" under the repository's
Security tab) rather than a public issue. If that button is not there, open an issue that asks for a private
contact and leave out every detail of the problem.
