"""Shared helpers for llm-cost-lint's tests."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest

from llm_cost_lint.scanner import scan_path

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def make_repo(tmp_path):
    """make_repo({"rel/path": "content", ...}) -> Path. Content is dedented."""

    def _make(files: dict) -> Path:
        root = tmp_path / "repo"
        root.mkdir(exist_ok=True)
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
        return root

    return _make


@pytest.fixture
def lint(make_repo):
    """lint(code, name="app.py") -> list of (rule_id, line, severity)."""

    def _lint(code: str, name: str = "app.py", **kw):
        root = make_repo({name: code})
        res = scan_path(str(root), **kw)
        return sorted((f.rule_id, f.line, f.severity) for f in res.findings)

    return _lint


@pytest.fixture
def lint_findings(make_repo):
    def _lint(code: str, name: str = "app.py", **kw):
        root = make_repo({name: code})
        return scan_path(str(root), **kw).findings

    return _lint


@pytest.fixture
def copy_fixture(tmp_path):
    def _copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copytree(FIXTURES / name, dst)
        return dst

    return _copy


def ids(findings) -> set[str]:
    return {f[0] for f in findings}
