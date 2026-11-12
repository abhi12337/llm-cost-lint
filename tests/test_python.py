"""Python: every rule has code that must be reported and well-configured code that must not."""

import pytest

from llm_cost_lint.scanner import scan_path

from conftest import ids

OPENAI = "from openai import OpenAI\nclient = OpenAI(timeout=30)\n"
ANTHROPIC = "import anthropic\nclient = anthropic.Anthropic(timeout=30)\n"
BIG = "x" * 4200


def rules(findings):
    return sorted((f.rule_id, f.file, f.line, f.severity) for f in findings)


# ---------------------------------------------------------------- fixtures
def test_vulnerable_fixture(copy_fixture):
    res = scan_path(str(copy_fixture("py_vulnerable")))
    assert rules(res.findings) == [
        ("LCL001", "app.py", 102, "medium"),
        ("LCL001", "app.py", 134, "medium"),
        ("LCL001", "app.py", 241, "medium"),
        ("LCL001", "app.py", 245, "medium"),
        ("LCL002", "app.py", 109, "low"),
        ("LCL003", "app.py", 11, "low"),
        ("LCL003", "app.py", 12, "low"),
        ("LCL003", "app.py", 13, "low"),
        ("LCL003", "app.py", 14, "low"),
        ("LCL003", "app.py", 241, "low"),
        ("LCL004", "app.py", 122, "low"),
        ("LCL004", "app.py", 134, "medium"),
        ("LCL005", "app.py", 134, "low"),
        ("LCL006", "app.py", 109, "low"),
        ("LCL007", "app.py", 226, "info"),
        ("LCL008", "app.py", 122, "low"),
        ("LCL008", "app.py", 134, "low"),
        ("LCL009", "app.py", 142, "low"),
        ("LCL010", "app.py", 102, "medium"),
        ("LCL011", "app.py", 241, "medium"),
        ("LCL012", "app.py", 109, "low"),
        ("LCL012", "app.py", 234, "info"),
        ("LCL013", "app.py", 102, "low"),
        ("LCL013", "app.py", 109, "low"),
        ("LCL013", "app.py", 142, "low"),
    ]
    assert res.llm_calls == 9


def test_clean_fixture(copy_fixture):
    res = scan_path(str(copy_fixture("py_clean")))
    assert res.findings == []
    assert res.llm_calls == 9


# ---------------------------------------------------------------- LCL001
@pytest.mark.parametrize("code", [
    OPENAI + "client.chat.completions.create(model='gpt-4.1-mini', messages=[])\n",
    OPENAI + "client.beta.chat.completions.parse(model='gpt-4.1-mini', messages=[], response_format=X)\n",
    OPENAI + "client.responses.create(model='gpt-4.1-mini', input='hi')\n",
    "from google import genai\nfrom google.genai import types\nc = genai.Client(http_options=types.HttpOptions(timeout=1))\n"
    "c.models.generate_content(model='gemini-2.5-flash', contents='hi')\n",
    "from google import genai\nfrom google.genai import types\nc = genai.Client(http_options=types.HttpOptions(timeout=1))\n"
    "c.models.generate_content(model='gemini-2.5-flash', contents='hi', config=types.GenerateContentConfig(temperature=0))\n",
    "import litellm\nlitellm.completion(model='gpt-4.1-mini', messages=[], timeout=5)\n",
    "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model='gpt-4.1-mini', timeout=5)\n",
])
def test_lcl001_no_cap(lint, code):
    assert "LCL001" in ids(lint(code))


@pytest.mark.parametrize("code", [
    OPENAI + "client.chat.completions.create(model='gpt-4.1-mini', messages=[], max_completion_tokens=500)\n",
    OPENAI + "client.chat.completions.create(model='gpt-4.1-mini', messages=[], max_tokens=500)\n",
    OPENAI + "client.responses.create(model='gpt-4.1-mini', input='hi', max_output_tokens=500)\n",
    OPENAI + "client.chat.completions.create(**params)\n",                       # unknown: stay silent
    "from google import genai\nfrom google.genai import types\nc = genai.Client(http_options=types.HttpOptions(timeout=1))\n"
    "c.models.generate_content(model='gemini-2.5-flash', contents='hi', config=types.GenerateContentConfig(max_output_tokens=100))\n",
    "from google import genai\nfrom google.genai import types\nc = genai.Client(http_options=types.HttpOptions(timeout=1))\n"
    "c.models.generate_content(model='gemini-2.5-flash', contents='hi', config=CONFIG)\n",   # unknown config
    "import litellm\nlitellm.completion(model='gpt-4.1-mini', messages=[], max_tokens=300, timeout=5)\n",
    "import litellm\nlitellm.completion(model='anthropic/claude-sonnet-5-5', messages=[], timeout=5)\n",
    ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=1024, messages=[])\n",
    "from langchain_anthropic import ChatAnthropic\nllm = ChatAnthropic(model='claude-sonnet-5-5', timeout=5)\n",
    "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model='gpt-4.1-mini', max_tokens=300, timeout=5)\n",
])
def test_lcl001_capped_or_unknown(lint, code):
    assert "LCL001" not in ids(lint(code))


def test_twilio_messages_create_is_not_anthropic(lint):
    code = ("from twilio.rest import Client\nc = Client('a', 'b')\n"
            "def view(request):\n    c.messages.create(to='+1', from_='+2', body='hi')\n")
    assert lint(code) == []


# ---------------------------------------------------------------- LCL002
def test_lcl002_large_non_streaming(lint):
    code = ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=64000, messages=[])\n"
    assert ("LCL002", 3, "low") in lint(code)


@pytest.mark.parametrize("code", [
    ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=16000, messages=[])\n",
    ANTHROPIC + "with client.messages.stream(model='claude-sonnet-5-5', max_tokens=64000, messages=[]) as s:\n    pass\n",
    ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=64000, stream=True, messages=[])\n",
])
def test_lcl002_ok(lint, code):
    assert "LCL002" not in ids(lint(code))


# ---------------------------------------------------------------- LCL003
@pytest.mark.parametrize("code", [
    "from openai import OpenAI\nclient = OpenAI()\n",
    "from openai import AsyncAzureOpenAI\nclient = AsyncAzureOpenAI(api_version='x')\n",
    "import anthropic\nclient = anthropic.AsyncAnthropic(max_retries=3)\n",
    "from google import genai\nclient = genai.Client()\n",
    "from langchain_anthropic import ChatAnthropic\nllm = ChatAnthropic(model='claude-sonnet-5-5')\n",
    "import litellm\nlitellm.completion(model='gpt-4.1-mini', messages=[], max_tokens=5)\n",
])
def test_lcl003_no_timeout(lint, code):
    assert "LCL003" in ids(lint(code))


@pytest.mark.parametrize("code", [
    "from openai import OpenAI\nclient = OpenAI(timeout=20.0)\n",
    "from openai import OpenAI\nimport httpx\nclient = OpenAI(http_client=httpx.Client(timeout=5))\n",
    "from openai import OpenAI\nclient = OpenAI()\nclient.with_options(timeout=5).chat.completions.create("
    "model='gpt-4.1-mini', max_tokens=5, messages=[])\n",
    "from openai import OpenAI\nclient = OpenAI(**settings)\n",
    "from google import genai\nfrom google.genai import types\nclient = genai.Client(http_options=types.HttpOptions(timeout=9))\n",
    "from langchain_anthropic import ChatAnthropic\nllm = ChatAnthropic(model='claude-sonnet-5-5', default_request_timeout=9)\n",
    "import litellm\nlitellm.request_timeout = 30\nlitellm.completion(model='gpt-4.1-mini', messages=[], max_tokens=5)\n",
])
def test_lcl003_ok(lint, code):
    assert "LCL003" not in ids(lint(code))


# ---------------------------------------------------------------- LCL004 / LCL005
def test_lcl004_sequential_loop(lint):
    code = OPENAI + ("def tag(items):\n    for it in items:\n"
                     "        client.chat.completions.create(model='gpt-4.1-mini', max_tokens=5, messages=[])\n")
    assert ("LCL004", 5, "low") in lint(code)


def test_lcl004_gather_comprehension_is_parallel(lint):
    code = ("import asyncio\nfrom openai import AsyncOpenAI\nclient = AsyncOpenAI(timeout=5)\n"
            "async def run(items):\n    return await asyncio.gather(*[client.chat.completions.create("
            "model='gpt-4.1-mini', max_tokens=5, messages=[]) for i in items])\n")
    assert ("LCL004", 5, "medium") in lint(code)


def test_lcl004_create_task_in_loop_is_parallel(lint):
    code = ("import asyncio\nfrom openai import AsyncOpenAI\nclient = AsyncOpenAI(timeout=5)\n"
            "async def one(x):\n    return await client.responses.create(model='gpt-4.1-mini', input=x, max_output_tokens=9)\n"
            "async def run(items):\n    tasks = []\n    for x in items:\n        tasks.append(asyncio.create_task(one(x)))\n"
            "    return await asyncio.gather(*tasks)\n")
    assert ("LCL004", 5, "medium") in lint(code)


@pytest.mark.parametrize("code", [
    # retry loop over a literal range
    OPENAI + "for attempt in range(3):\n    client.chat.completions.create(model='m', max_tokens=5, messages=[])\n",
    # agent / tool loop
    OPENAI + "def agent(msgs):\n    while True:\n        r = client.chat.completions.create(model='m', max_tokens=5, "
             "messages=msgs)\n        if r: break\n",
    # bounded with a semaphore
    "import asyncio\nfrom openai import AsyncOpenAI\nclient = AsyncOpenAI(timeout=5)\nsem = asyncio.Semaphore(4)\n"
    "async def one(x):\n    async with sem:\n        return await client.responses.create(model='m', input=x, "
    "max_output_tokens=5)\nasync def run(items):\n    return await asyncio.gather(*[one(x) for x in items])\n",
    # LangChain batch with max_concurrency
    "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(model='m', max_tokens=5, timeout=5)\n"
    "llm.batch(items, config={'max_concurrency': 4})\n",
])
def test_lcl004_ok(lint, code):
    assert "LCL004" not in ids(lint(code))


def test_lcl005_retries_disabled_on_that_client_only(lint):
    code = ("from openai import OpenAI\nfast = OpenAI(timeout=5, max_retries=0)\nsafe = OpenAI(timeout=5)\n"
            "def run(items):\n    for x in items:\n"
            "        fast.chat.completions.create(model='m', max_tokens=5, messages=[])\n"
            "        safe.chat.completions.create(model='m', max_tokens=5, messages=[])\n")
    found = lint(code)
    assert ("LCL005", 6, "low") in found
    assert not any(f[0] == "LCL005" and f[1] == 7 for f in found)


def test_lcl005_litellm_without_num_retries(lint):
    code = ("import litellm\ndef run(items):\n    for x in items:\n"
            "        litellm.completion(model='gpt-4.1-mini', messages=[], max_tokens=5, timeout=5)\n")
    assert "LCL005" in ids(lint(code))
    ok = code.replace("timeout=5)", "timeout=5, num_retries=3)")
    assert "LCL005" not in ids(lint(ok))


def test_lcl005_own_retry_counts(lint):
    code = ("import tenacity\nfrom openai import OpenAI\nc = OpenAI(timeout=5, max_retries=0)\n"
            "def run(items):\n    for x in items:\n        c.chat.completions.create(model='m', max_tokens=5, messages=[])\n")
    assert "LCL005" not in ids(lint(code))


# ---------------------------------------------------------------- LCL006 / LCL007
def test_lcl006_anthropic_static_prompt(lint):
    code = ANTHROPIC + f"SYSTEM = '{BIG}'\n" + \
        "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, system=SYSTEM, messages=[])\n"
    assert ("LCL006", 4, "low") in lint(code)


@pytest.mark.parametrize("call", [
    "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, system=[{'type': 'text', 'text': SYSTEM, "
    "'cache_control': {'type': 'ephemeral'}}], messages=[])",
    "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, cache_control={'type': 'ephemeral'}, "
    "system=SYSTEM, messages=[])",
])
def test_lcl006_cached_ok(lint, call):
    code = ANTHROPIC + f"SYSTEM = '{BIG}'\n" + call + "\n"
    assert "LCL006" not in ids(lint(code))


def test_lcl006_short_prompt_and_openai_ok(lint):
    code = ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, system='be brief', messages=[])\n"
    assert "LCL006" not in ids(lint(code))
    # OpenAI caches long prefixes automatically: a static prompt is not reported
    code = OPENAI + f"SYSTEM = '{BIG}'\n" + ("client.chat.completions.create(model='m', max_tokens=5, "
                                            "messages=[{'role': 'system', 'content': SYSTEM}])\n")
    assert lint(code) == []


def test_lcl007_dynamic_prefix(lint):
    code = OPENAI + ("def f(user):\n    client.chat.completions.create(model='m', max_tokens=5, messages=["
                     f"{{'role': 'system', 'content': f'Hello {{user}}. {BIG}'}}])\n")
    assert ("LCL007", 4, "info") in lint(code)
    tail = OPENAI + ("def f(user):\n    client.chat.completions.create(model='m', max_tokens=5, messages=["
                     f"{{'role': 'system', 'content': f'{BIG} The user is {{user}}.'}}])\n")
    assert "LCL007" not in ids(lint(tail))


def test_large_prompt_threshold_option(lint):
    code = ANTHROPIC + "SYSTEM = '" + "y" * 600 + "'\n" + \
        "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, system=SYSTEM, messages=[])\n"
    assert "LCL006" not in ids(lint(code))
    from llm_cost_lint.checks import Options
    assert "LCL006" in ids(lint(code, opts=Options(large_prompt_chars=500)))


# ---------------------------------------------------------------- LCL008
def test_lcl008_top_tier_for_classification(lint):
    code = ANTHROPIC + ("client.messages.create(model='claude-opus-5-5', max_tokens=500, messages=["
                        "{'role': 'user', 'content': 'Classify this email as spam or not spam'}])\n")
    assert ("LCL008", 3, "info") in lint(code)


def test_lcl008_short_cap(lint):
    code = OPENAI + "client.chat.completions.create(model='gpt-5', max_completion_tokens=10, messages=[])\n"
    assert "LCL008" in ids(lint(code))


@pytest.mark.parametrize("code", [
    ANTHROPIC + "client.messages.create(model='claude-haiku-5-5', max_tokens=10, messages=["
                "{'role': 'user', 'content': 'Classify this email as spam'}])\n",
    ANTHROPIC + "client.messages.create(model='claude-opus-5-5', max_tokens=8000, messages=["
                "{'role': 'user', 'content': 'Review this architecture document'}])\n",
])
def test_lcl008_ok(lint, code):
    assert "LCL008" not in ids(lint(code))


# ---------------------------------------------------------------- LCL009
def test_lcl009_embed_in_loop(lint):
    code = OPENAI + ("def ingest(texts):\n    for t in texts:\n"
                     "        client.embeddings.create(model='text-embedding-3-small', input=t)\n")
    assert ("LCL009", 5, "low") in lint(code)


@pytest.mark.parametrize("code", [
    # the user's query, once per request: fine
    "from fastapi import FastAPI\n" + OPENAI + "app = FastAPI()\n@app.get('/s')\ndef s(q: str):\n    try:\n"
    "        return client.embeddings.create(model='text-embedding-3-small', input=q)\n    except Exception:\n        pass\n",
    # cached
    OPENAI + "from functools import lru_cache\n@lru_cache\ndef emb(t):\n    return client.embeddings.create(model='m', input=t)\n"
    "def ingest(texts):\n    for t in texts:\n        emb(t)\n",
])
def test_lcl009_ok(lint, code):
    assert "LCL009" not in ids(lint(code))


# ---------------------------------------------------------------- LCL010
@pytest.mark.parametrize("code,line", [
    (OPENAI + "history = []\ndef chat(m):\n    history.append({'role': 'user', 'content': m})\n"
              "    return client.chat.completions.create(model='m', max_tokens=5, messages=history)\n", 6),
    (OPENAI + "class Bot:\n    def __init__(self):\n        self.messages = []\n    def send(self, m):\n"
              "        self.messages.append({'role': 'user', 'content': m})\n"
              "        return client.chat.completions.create(model='m', max_tokens=5, messages=self.messages)\n", 8),
    (OPENAI + "SESSIONS = {}\ndef chat(sid, m):\n    h = SESSIONS.setdefault(sid, [])\n    h.append(m)\n"
              "    return client.chat.completions.create(model='m', max_tokens=5, messages=h)\n", 7),
    (OPENAI + "def repl():\n    msgs = []\n    while True:\n        msgs.append({'role': 'user', 'content': input()})\n"
              "        client.chat.completions.create(model='m', max_tokens=5, messages=msgs)\n", 7),
    (ANTHROPIC + "history = []\ndef chat(m):\n    history.append({'role': 'user', 'content': m})\n"
                 "    return client.messages.create(model='claude-sonnet-5-5', max_tokens=500, messages=[*history])\n", 6),
])
def test_lcl010_unbounded_history(lint, code, line):
    assert ("LCL010", line, "medium") in lint(code)


@pytest.mark.parametrize("code", [
    OPENAI + "history = []\ndef chat(m):\n    history.append(m)\n"
             "    return client.chat.completions.create(model='m', max_tokens=5, messages=history[-20:])\n",
    OPENAI + "history = []\ndef chat(m):\n    history.append(m)\n    if len(history) > 40:\n        del history[:2]\n"
             "    return client.chat.completions.create(model='m', max_tokens=5, messages=history)\n",
    OPENAI + "history = []\ndef chat(m):\n    history.append(m)\n"
             "    return client.chat.completions.create(model='m', max_tokens=5, messages=trim_history(history))\n",
    # agent loop: local list, one task
    OPENAI + "def agent(task):\n    msgs = [task]\n    while True:\n"
             "        r = client.chat.completions.create(model='m', max_tokens=5, messages=msgs)\n        msgs.append(r)\n",
])
def test_lcl010_ok(lint, code):
    assert "LCL010" not in ids(lint(code))


# ---------------------------------------------------------------- LCL011
@pytest.mark.parametrize("code", [
    OPENAI + "client.chat.completions.create(model='gpt-4-32k', max_tokens=5, messages=[])\n",
    ANTHROPIC + "MODEL = 'claude-3-opus-20240229'\n",
    "import os\nMODEL = os.getenv('MODEL', 'gemini-1.5-flash')\n",                 # no SDK import: model context
    "settings = {'model': 'anthropic/claude-3-5-sonnet-20241022'}\n",
])
def test_lcl011_retired(lint, code):
    assert "LCL011" in ids(lint(code))


@pytest.mark.parametrize("code", [
    OPENAI + "client.chat.completions.create(model='gpt-4.1-mini', max_tokens=5, messages=[])\n",
    "NOTES = 'gpt-4-32k was retired'\n",       # not an exact id, not a model setting
    "names = ['claude-2.1']\n",                # no SDK and not a model setting
])
def test_lcl011_ok(lint, code):
    assert "LCL011" not in ids(lint(code))


def test_lcl011_deprecated_mentions_date(lint_findings):
    fs = lint_findings(ANTHROPIC + "MODEL = 'claude-sonnet-4-5'\n")
    f = next(f for f in fs if f.rule_id == "LCL011")
    assert "deprecated" in f.message and "2026-11-30" in f.message and "claude-sonnet-5-5" in f.fix


# ---------------------------------------------------------------- LCL012
@pytest.mark.parametrize("code,sev", [
    (OPENAI + "client.chat.completions.create(model='m', max_tokens=5, temperature=0.2, top_p=0.9, messages=[])\n",
     "info"),
    (OPENAI + "client.chat.completions.create(model='m', max_tokens=5, n=4, messages=[])\n", "info"),
    (ANTHROPIC + "client.messages.create(model='claude-sonnet-5-5', max_tokens=500, temperature=0, messages=[])\n",
     "low"),
])
def test_lcl012(lint, code, sev):
    assert ("LCL012", 3, sev) in lint(code)


@pytest.mark.parametrize("code", [
    OPENAI + "client.chat.completions.create(model='m', max_tokens=5, temperature=0.2, messages=[])\n",
    ANTHROPIC + "client.messages.create(model='claude-haiku-4-5', max_tokens=500, temperature=0, messages=[])\n",
    OPENAI + "client.chat.completions.create(model='m', max_tokens=5, n=1, messages=[])\n",
])
def test_lcl012_ok(lint, code):
    assert "LCL012" not in ids(lint(code))


# ---------------------------------------------------------------- LCL013
@pytest.mark.parametrize("code", [
    "from fastapi import FastAPI\n" + OPENAI + "app = FastAPI()\n@app.post('/a')\ndef a(q: str):\n"
    "    return client.chat.completions.create(model='m', max_tokens=5, messages=[])\n",
    "from flask import Flask\n" + OPENAI + "app = Flask(__name__)\n@app.route('/a', methods=['POST'])\ndef a():\n"
    "    return client.chat.completions.create(model='m', max_tokens=5, messages=[])\n",
    OPENAI + "def view(request):\n    return client.chat.completions.create(model='m', max_tokens=5, messages=[])\n",
])
def test_lcl013_unhandled_in_handler(lint, code):
    assert "LCL013" in ids(lint(code))


@pytest.mark.parametrize("code", [
    "from fastapi import FastAPI\n" + OPENAI + "app = FastAPI()\n@app.post('/a')\ndef a(q: str):\n    try:\n"
    "        return client.chat.completions.create(model='m', max_tokens=5, messages=[])\n"
    "    except Exception:\n        return None\n",
    OPENAI + "def job():\n    return client.chat.completions.create(model='m', max_tokens=5, messages=[])\n",
])
def test_lcl013_ok(lint, code):
    assert "LCL013" not in ids(lint(code))


# ---------------------------------------------------------------- ignore comments
def test_inline_ignore(lint):
    base = OPENAI + "client.chat.completions.create(model='gpt-4-32k', messages=[])"
    assert {"LCL001", "LCL011"} <= ids(lint(base + "\n"))
    assert lint(base + "  # llm-cost-lint: ignore\n") == []
    only = lint(base + "  # llm-cost-lint: ignore LCL011\n")
    assert ids(only) == {"LCL001"}
    above = OPENAI + "# llm-cost-lint: ignore\nclient.chat.completions.create(model='gpt-4-32k', messages=[])\n"
    assert lint(above) == []


def test_syntax_error_file_is_skipped(lint):
    assert lint("from openai import OpenAI\ndef broken(:\n") == []


def test_fix_is_sdk_specific(lint_findings):
    fs = lint_findings(OPENAI + "client.responses.create(model='gpt-4.1-mini', input='hi')\n")
    f = next(f for f in fs if f.rule_id == "LCL001")
    assert "max_output_tokens=800" in f.fix and "client.responses.create" in f.fix
    fs = lint_findings("from google import genai\nc = genai.Client()\n")
    f = next(f for f in fs if f.rule_id == "LCL003")
    assert "HttpOptions(timeout=30_000)" in f.fix
