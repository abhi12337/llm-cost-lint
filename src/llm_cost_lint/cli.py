"""Command line interface.

  llm-cost-lint scan [PATH ...]
  llm-cost-lint rules
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from . import __version__, models
from .checks import LARGE_PROMPT_DEFAULT, Options
from .findings import RULES, SEVERITIES, SEVERITY_RANK, Finding
from .report import render_json, render_text, use_color
from .scanner import MAX_BYTES_DEFAULT, ScanResult, scan_path

EXIT_OK, EXIT_FINDINGS, EXIT_ERROR = 0, 1, 2


def _exit_code(findings: list[Finding], fail_on: str) -> int:
    if fail_on == "never":
        return EXIT_OK
    lim = SEVERITY_RANK[fail_on]
    return EXIT_FINDINGS if any(SEVERITY_RANK[f.severity] <= lim for f in findings) else EXIT_OK


def _filter_min(findings: list[Finding], min_sev: str) -> list[Finding]:
    lim = SEVERITY_RANK[min_sev]
    return [f for f in findings if SEVERITY_RANK[f.severity] <= lim]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cost-lint",
        description="Static lint for LLM API calls that waste money or break in production (OpenAI, Anthropic, "
                    "Google GenAI, Vercel AI SDK, LangChain, LiteLLM). Reads files only: never runs your code, "
                    "never calls a model, no network.",
        epilog="A clean report is not a guarantee. See 'Known limitations' in the README.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="scan folders / files (Python, JavaScript, TypeScript)")
    s.add_argument("paths", nargs="*", default=None, help="folders or files to scan (default: .)")
    s.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="glob of paths to skip (repeatable), e.g. --exclude 'scripts/*'")
    s.add_argument("--include-tests", action="store_true", help="also scan test / fixture files")
    s.add_argument("--disable", action="append", default=[], metavar="RULE",
                   help="ignore a rule id (repeatable), e.g. --disable LCL008")
    s.add_argument("--large-prompt-chars", type=int, default=LARGE_PROMPT_DEFAULT, metavar="N",
                   help=f"prompt length in characters that counts as large for LCL006 / LCL007 "
                        f"(default: {LARGE_PROMPT_DEFAULT})")
    s.add_argument("--max-file-mb", type=float, default=MAX_BYTES_DEFAULT / 1024 / 1024,
                   help="skip files larger than this (default: 2)")
    s.add_argument("--json", action="store_true", help="print a JSON report instead of text")
    s.add_argument("--fail-on", choices=[*SEVERITIES, "never"], default="medium",
                   help="exit 1 if any finding is at or above this severity (default: medium)")
    s.add_argument("--min-severity", choices=SEVERITIES, default="info",
                   help="hide findings below this severity (default: info = show all)")
    s.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    s.add_argument("--brief", action="store_true", help="omit the why/fix text")

    sub.add_parser("rules", help="list every rule with why it matters and how to fix it")
    return parser


def cmd_scan(args) -> int:
    paths = args.paths or ["."]
    for p in paths:
        if not os.path.exists(p):
            print(f"error: {p} does not exist", file=sys.stderr)
            return EXIT_ERROR
    unknown = [r for r in args.disable if r.upper() not in RULES]
    if unknown:
        print(f"error: unknown rule id(s): {', '.join(unknown)} (see `llm-cost-lint rules`)", file=sys.stderr)
        return EXIT_ERROR
    if args.large_prompt_chars < 1:
        print("error: --large-prompt-chars must be a positive number", file=sys.stderr)
        return EXIT_ERROR
    max_bytes = int(args.max_file_mb * 1024 * 1024)
    opts = Options(large_prompt_chars=args.large_prompt_chars)
    results: list[ScanResult] = [scan_path(p, excludes=args.exclude, include_tests=args.include_tests,
                                           max_bytes=max_bytes, opts=opts) for p in paths]
    if len(results) > 1:   # make locations unambiguous
        for r, label in zip(results, paths):
            if os.path.isfile(label) or label == ".":
                continue
            prefix = label.replace("\\", "/").rstrip("/")
            for f in r.findings:
                f.file = f"{prefix}/{f.file}"
            r.llm_files = [f"{prefix}/{x}" for x in r.llm_files]
    disabled = {r.upper() for r in args.disable}
    findings = [f for r in results for f in r.findings if f.rule_id not in disabled]
    exit_code = _exit_code(findings, args.fail_on)
    shown = _filter_min(findings, args.min_severity)
    files = sum(r.files_scanned for r in results)
    py = sum(r.py_files for r in results)
    js = sum(r.js_files for r in results)
    llm_files = [x for r in results for x in r.llm_files]
    calls = sum(r.llm_calls for r in results)
    if args.json:
        sys.stdout.write(render_json(shown, meta={
            "targets": [r.root for r in results], "files_scanned": files, "python_files": py, "js_files": js,
            "llm_files": llm_files, "llm_calls": calls, "model_data_as_of": models.as_of(), "exit_code": exit_code,
        }))
    else:
        header = [f"Scanned {', '.join(paths)}",
                  f"{files} file(s) ({py} Python, {js} JS/TS); {calls} LLM call(s) in {len(llm_files)} file(s); "
                  f"model list as of {models.as_of()}"]
        if not llm_files:
            header.append("No LLM SDK calls found (OpenAI, Anthropic, Google GenAI, Vercel AI SDK, LangChain, "
                          "LiteLLM).")
        sys.stdout.write(render_text(shown, header=header, color=use_color(force=False if args.no_color else None),
                                     verbose_fix=not args.brief, footer=[f"Exit code: {exit_code}"]))
    return exit_code


def cmd_rules(_args) -> int:
    print(f"Model data (tiers, retired / deprecated ids) as of {models.as_of()}.\n")
    for r in RULES.values():
        print(f"{r.id}  {r.title}")
        print(textwrap.fill(r.why, 92, initial_indent="    ", subsequent_indent="    "))
        print("    Fix:")
        print(textwrap.indent(r.fix.replace("<as_of>", models.as_of()), "      "))
        print()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    return {"scan": cmd_scan, "rules": cmd_rules}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
