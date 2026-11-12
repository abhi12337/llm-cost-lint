"""Walks a folder (or takes a single file) and runs the Python / JS extractors and the rules.
Reads local files only. Never imports, runs or connects to anything it scans."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from .checks import Options, check
from .findings import Finding
from .js_scan import extract_js
from .py_scan import extract_python

MAX_BYTES_DEFAULT = 2 * 1024 * 1024
SKIP_DIRS = {"node_modules", ".git", ".hg", ".svn", "dist", "build", "out", ".next", ".nuxt", ".output", ".svelte-kit",
             ".venv", "venv", "env", "__pycache__", "site-packages", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
             ".ruff_cache", "coverage", ".turbo", ".cache", ".idea", "target", "vendor", ".gradle", ".vercel"}
TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "fixtures", "__fixtures__", "testdata", "e2e", "__mocks__"}
PY_EXT = {".py"}
JS_EXT = {".ts", ".mts", ".cts", ".js", ".mjs", ".cjs", ".tsx", ".jsx"}


@dataclass
class ScanResult:
    root: str
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    py_files: int = 0
    js_files: int = 0
    llm_files: list[str] = field(default_factory=list)
    llm_calls: int = 0


def _is_test_file(name: str) -> bool:
    n = name.lower()
    return n.startswith("test_") or n == "conftest.py" or n.endswith((
        "_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", ".test.mjs", ".spec.mjs", ".test.tsx",
        ".spec.tsx", ".test.jsx", ".spec.jsx"))


def _read(p: Path, max_bytes: int) -> str | None:
    try:
        if p.stat().st_size > max_bytes:
            return None
        data = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def scan_file(path: Path, rel: str, res: ScanResult, max_bytes: int, opts: Options) -> None:
    suffix = path.suffix.lower()
    name = path.name
    if suffix in PY_EXT:
        lang = "py"
    elif suffix in JS_EXT and not name.endswith((".d.ts", ".min.js")):
        lang = "js"
    else:
        return
    text = _read(path, max_bytes)
    if text is None:
        return
    res.files_scanned += 1
    if lang == "py":
        res.py_files += 1
        ff = extract_python(rel, text)
    else:
        res.js_files += 1
        ff = extract_js(rel, text)
    if ff is None:
        return
    if any(c.kind in ("gen", "embed") for c in ff.calls):
        res.llm_files.append(rel)
        res.llm_calls += sum(1 for c in ff.calls if c.kind in ("gen", "embed"))
    res.findings += check(ff, opts)


def scan_path(target: str, *, excludes: list[str] | None = None, include_tests: bool = False,
              max_bytes: int = MAX_BYTES_DEFAULT, opts: Options | None = None) -> ScanResult:
    excludes = excludes or []
    opts = opts or Options()
    root = Path(target)
    if root.is_file():
        res = ScanResult(str(root.resolve().parent))
        scan_file(root, target.replace("\\", "/"), res, max_bytes, opts)
        return res
    res = ScanResult(str(root.resolve()))
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
        rel_dir = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIRS or d.startswith(".") or (not include_tests and d.lower() in TEST_DIRS):
                continue
            if any(fnmatch.fnmatch(rel_dir + d, pat) or fnmatch.fnmatch(rel_dir + d + "/", pat) for pat in excludes):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            rel = rel_dir + fn
            if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                continue
            if not include_tests and _is_test_file(fn):
                continue
            scan_file(Path(dirpath) / fn, rel, res, max_bytes, opts)
    return res
