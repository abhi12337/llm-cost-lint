# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-12

### Fixed

- A Python file with a syntax error was skipped silently, so a scan of broken code printed "No
  findings" and exited 0. Every file that cannot be parsed is now reported: a
  `warning: <path>: syntax error line N; file not scanned` line on stderr, a count in the text
  header, and a `"skipped": [{"path": ..., "reason": ...}]` list in the JSON report. The exit
  code still depends on findings only, unless `--strict` is given.

### Added

- `--strict`: exit `2` when any file could not be parsed.

## [0.1.0] - 2026-11-11

First release.

### Added

- `llm-cost-lint scan [PATH ...]`: static scan of folders or single files (Python via `ast`,
  JavaScript / TypeScript via a lexer). Never imports or runs the scanned code, never calls a
  model; no network access.
- `llm-cost-lint rules`: every rule with why it matters and its fix for each SDK.
- SDK coverage: OpenAI (Python and Node; chat.completions, responses, embeddings), Anthropic
  (Python and TypeScript; messages), Google GenAI (`google-genai`, `@google/genai`), Vercel AI
  SDK (`generateText`, `streamText`, `generateObject`, `streamObject`, `embed`, `embedMany`),
  LangChain chat model constructors and `.invoke()`, LiteLLM.
- Rules:
  - LCL001 no output token cap (medium)
  - LCL002 very large `max_tokens` on a non-streaming Anthropic call (low)
  - LCL003 no request timeout on the client, chat model or call (low)
  - LCL004 LLM call per item in a loop: unbounded parallel fan-out (medium) or sequential (low)
  - LCL005 loop of LLM calls with retries disabled (low)
  - LCL006 large static prompt sent to Anthropic without `cache_control` (low)
  - LCL007 dynamic value near the start of a large prompt, which defeats prefix caching (info)
  - LCL008 top-tier model for a simple or repeated task (info / low, heuristic)
  - LCL009 embeddings recomputed without a cache (low, heuristic)
  - LCL010 conversation history that grows without a limit (medium, heuristic)
  - LCL011 retired or deprecated model id (medium)
  - LCL012 sampling / `n` / streaming settings worth a second look (info; low for Claude models
    that reject sampling parameters)
  - LCL013 LLM call in a request handler without error handling (low)
- Dated model data file (`data/models.json`, as of 2026-11-11): model tiers, retired /
  deprecated ids, models that reject sampling parameters. No prices.
- Text and JSON reports, an SDK-specific fix for every finding, `--fail-on` (default `medium`),
  `--min-severity`, `--disable`, `--exclude`, `--include-tests`, `--large-prompt-chars`,
  `--brief`, `--no-color`, `llm-cost-lint: ignore [RULE ...]` comments, exit codes `0` / `1` / `2`.
- Demo project generator (`examples/make_demo.py`) and a GitHub Actions example.
