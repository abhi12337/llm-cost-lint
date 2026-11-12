import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from llm_cost_lint import models
from llm_cost_lint.cli import main
from llm_cost_lint.findings import RULES, SEVERITIES

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_scan_vulnerable_text_report(capsys):
    code, out, _ = run(capsys, "scan", str(FIX / "py_vulnerable"), "--no-color")
    assert code == 1
    assert "MEDIUM   LCL001" in out and "app.py:102" in out
    assert "why:" in out and "fix:" in out and "max_completion_tokens=800" in out
    assert "Exit code: 1" in out


def test_scan_clean_exits_0(capsys):
    code, out, _ = run(capsys, "scan", str(FIX / "py_clean"), str(FIX / "ts_clean"), "--no-color")
    assert code == 0
    assert "No findings" in out and "14 LLM call(s) in 2 file(s)" in out


def test_json_report_shape(capsys):
    code, out, _ = run(capsys, "scan", str(FIX / "ts_vulnerable"), "--json")
    doc = json.loads(out)
    assert code == 1 and doc["exit_code"] == 1 and doc["tool"] == "llm-cost-lint"
    assert doc["llm_files"] == ["src/server.ts"] and doc["model_data_as_of"] == models.as_of()
    assert set(doc["summary"]) == set(SEVERITIES)
    first = doc["findings"][0]
    assert first["severity"] == "medium"
    assert {"rule_id", "severity", "message", "file", "line", "evidence", "title", "why", "fix", "heuristic"} <= set(first)
    assert "fixkeys" not in json.dumps(doc)


def test_default_fail_on_is_medium(capsys, tmp_path):
    (tmp_path / "a.py").write_text("from openai import OpenAI\nclient = OpenAI()\n")      # LCL003 is low
    assert run(capsys, "scan", str(tmp_path))[0] == 0
    assert run(capsys, "scan", str(tmp_path), "--fail-on", "low")[0] == 1
    (tmp_path / "b.py").write_text("from openai import OpenAI\nc = OpenAI(timeout=5)\n"
                                   "c.chat.completions.create(model='m', messages=[])\n")       # LCL001 is medium
    assert run(capsys, "scan", str(tmp_path))[0] == 1
    assert run(capsys, "scan", str(tmp_path), "--fail-on", "high")[0] == 0
    assert run(capsys, "scan", str(tmp_path), "--fail-on", "never")[0] == 0


def test_min_severity_and_disable(capsys):
    code, out, _ = run(capsys, "scan", str(FIX / "py_vulnerable"), "--json", "--min-severity", "medium")
    assert {f["severity"] for f in json.loads(out)["findings"]} == {"medium"}
    code, out, _ = run(capsys, "scan", str(FIX / "py_vulnerable"), "--json", "--disable", "LCL001",
                       "--disable", "lcl004", "--disable", "LCL010", "--disable", "LCL011")
    doc = json.loads(out)
    assert not {f["rule_id"] for f in doc["findings"]} & {"LCL001", "LCL004", "LCL010", "LCL011"}
    assert code == 0                       # everything left is low / info


def test_brief_omits_why_and_fix(capsys):
    _, out, _ = run(capsys, "scan", str(FIX / "py_vulnerable"), "--brief", "--no-color")
    assert "why:" not in out and "fix:" not in out and "LCL001" in out


def test_unknown_rule_missing_path_bad_option(capsys, tmp_path):
    code, _, err = run(capsys, "scan", str(tmp_path / "nope"))
    assert code == 2 and "does not exist" in err
    code, _, err = run(capsys, "scan", str(tmp_path), "--disable", "LCL999")
    assert code == 2 and "unknown rule" in err
    code, _, err = run(capsys, "scan", str(tmp_path), "--large-prompt-chars", "0")
    assert code == 2
    with pytest.raises(SystemExit) as e:
        main(["scan", "--fail-on", "sometimes"])
    assert e.value.code == 2


def test_scan_single_file(capsys):
    code, out, _ = run(capsys, "scan", str(FIX / "ts_vulnerable" / "src" / "server.ts"), "--json")
    doc = json.loads(out)
    assert code == 1 and doc["files_scanned"] == 1 and len(doc["findings"]) == 14


def test_tests_folders_skipped_unless_included(capsys, tmp_path):
    vuln = (FIX / "py_vulnerable" / "app.py").read_text()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "app.py").write_text(vuln)
    (tmp_path / "test_app.py").write_text(vuln)
    code, out, _ = run(capsys, "scan", str(tmp_path), "--json")
    assert code == 0 and json.loads(out)["findings"] == []
    code, out, _ = run(capsys, "scan", str(tmp_path), "--json", "--include-tests")
    assert code == 1 and len(json.loads(out)["findings"]) == 50


def test_exclude_glob(capsys, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "a.py").write_text((FIX / "py_vulnerable" / "app.py").read_text())
    code, out, _ = run(capsys, "scan", str(tmp_path), "--exclude", "scripts/*", "--json")
    assert code == 0 and json.loads(out)["files_scanned"] == 0


def test_node_modules_and_venv_skipped(capsys, tmp_path):
    for d in ("node_modules/openai", ".venv/lib", "dist"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "x.py").write_text("from openai import OpenAI\nOpenAI()\n")
    code, out, _ = run(capsys, "scan", str(tmp_path), "--json")
    assert json.loads(out)["files_scanned"] == 0


def test_rules_lists_every_rule(capsys):
    code, out, _ = run(capsys, "rules")
    assert code == 0
    for rid in RULES:
        assert rid in out
    assert models.as_of() in out
    # every placeholder is filled in or a documented generic one
    leftovers = set(re.findall(r"<[a-z_]+>", out))
    assert leftovers <= {"<client>", "<model>", "<ctor>", "<capkw>", "<fn>", "<call>", "<var>", "<smaller>",
                         "<replacement>", "<max_tokens>", "<suggest>"}


def test_every_rule_has_title_why_and_fix_per_sdk():
    for r in RULES.values():
        assert r.title and len(r.why) > 80 and r.fixes
        for text in r.fixes.values():
            assert len(text) > 40


@pytest.mark.parametrize("name", ["py_vulnerable", "ts_vulnerable"])
def test_every_finding_has_location_evidence_and_specific_fix(capsys, name):
    _, out, _ = run(capsys, "scan", str(FIX / name), "--json")
    for f in json.loads(out)["findings"]:
        assert f["file"] and f["line"] and f["evidence"] and f["fix"] and f["why"]
        assert not re.search(r"<(client|model|ctor|capkw|fn|call|var|smaller|replacement)>", f["fix"]), f


def test_module_entry_point():
    env = {"PYTHONPATH": str(ROOT / "src")}
    if sys.platform == "win32":
        env.update({"SYSTEMROOT": "C:\\Windows", "PATH": ""})
    res = subprocess.run([sys.executable, "-m", "llm_cost_lint", "scan", str(FIX / "ts_clean"), "--brief"],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert "No findings" in res.stdout


def test_no_network_imports():
    """The scanner must never be able to contact anything."""
    for p in (ROOT / "src" / "llm_cost_lint").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        for mod in ("urllib.request", "http.client", "socket", "requests", "httpx", "subprocess"):
            assert f"import {mod}" not in text and f"from {mod} import" not in text, (p.name, mod)


# ---------------------------------------------------------------- model data file
def test_model_data_is_dated_and_has_no_prices():
    d = models.data()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["as_of"])
    raw = models.DATA_FILE.read_text(encoding="utf-8")
    assert not re.search(r"\$\s*\d|usd|per 1m|per million", raw.lower())
    assert "price" not in raw.replace("No prices", "").lower()
    seen = set()
    for r in d["retired"]:
        assert r["id"] not in seen
        seen.add(r["id"])
        assert r["provider"] in ("openai", "anthropic", "google") and r["status"] in ("retired", "deprecated")
        assert r["replacement"]
        assert r["date"] is None or re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"])


@pytest.mark.parametrize("model,retired", [
    ("gpt-4-32k", True), ("openai/gpt-4-32k", True), ("models/gemini-1.5-pro", True),
    ("anthropic/claude-2.1", True), ("anthropic:claude-3-opus-20240229", True),
    ("gpt-4.1-mini", False), ("claude-sonnet-5-5", False), ("gemini-2.5-flash", False), ("gpt-4", False),
])
def test_retired_lookup(model, retired):
    assert (models.retired(model) is not None) is retired


@pytest.mark.parametrize("model,top", [
    ("claude-opus-5-5", True), ("claude-fable-5-2", True), ("gpt-5", True), ("o3", True), ("gemini-2.5-pro", True),
    ("gpt-4o", True), ("claude-haiku-5-5", False), ("claude-sonnet-5-5", False), ("gpt-5-mini", False),
    ("gpt-4o-mini", False), ("gpt-4.1-nano", False), ("gemini-2.5-flash", False), ("o4-mini", False),
])
def test_top_tier_lookup(model, top):
    assert (models.top_tier(model) is not None) is top
