"""JavaScript / TypeScript: every rule has code that must be reported and code that must not."""

import pytest

from llm_cost_lint.jslex import lex
from llm_cost_lint.scanner import scan_path

from conftest import ids

OPENAI = 'import OpenAI from "openai";\nconst client = new OpenAI({ timeout: 30_000 });\n'
ANTHROPIC = 'import Anthropic from "@anthropic-ai/sdk";\nconst client = new Anthropic({ timeout: 30_000 });\n'
AI = 'import { generateText, streamText, embed, embedMany } from "ai";\nimport { openai } from "@ai-sdk/openai";\n'
BIG = "x" * 4200


def rules(findings):
    return sorted((f.rule_id, f.file, f.line, f.severity) for f in findings)


def js(lint, code, name="src/app.ts"):
    return lint(code, name=name)


# ---------------------------------------------------------------- fixtures
def test_vulnerable_fixture(copy_fixture):
    res = scan_path(str(copy_fixture("ts_vulnerable")))
    assert rules(res.findings) == [
        ("LCL001", "src/server.ts", 95, "medium"),
        ("LCL001", "src/server.ts", 127, "medium"),
        ("LCL003", "src/server.ts", 7, "low"),
        ("LCL003", "src/server.ts", 8, "low"),
        ("LCL003", "src/server.ts", 127, "low"),
        ("LCL004", "src/server.ts", 116, "medium"),
        ("LCL005", "src/server.ts", 116, "low"),
        ("LCL006", "src/server.ts", 104, "low"),
        ("LCL008", "src/server.ts", 116, "low"),
        ("LCL009", "src/server.ts", 137, "low"),
        ("LCL010", "src/server.ts", 95, "medium"),
        ("LCL011", "src/server.ts", 105, "medium"),
        ("LCL013", "src/server.ts", 95, "low"),
        ("LCL013", "src/server.ts", 104, "low"),
    ]
    assert res.llm_calls == 5


def test_clean_fixture(copy_fixture):
    res = scan_path(str(copy_fixture("ts_clean")))
    assert res.findings == []
    assert res.llm_calls == 5


# ---------------------------------------------------------------- lexer
def test_lexer_keeps_offsets_and_reads_templates():
    src = 'const a = `hi ${name} there`; // comment "x"\nconst b = "s\\"q"; /* c */'
    clean, code, toks = lex(src)
    assert len(clean) == len(code) == len(src)
    assert "comment" not in clean and "there" in clean and "there" not in code
    t = toks[0]
    assert t.template and t.has_interp and t.text == "hi  there" and t.after_first_interp == len(" there")


def test_strings_and_comments_do_not_match(lint):
    code = OPENAI + '// client.chat.completions.create({ model: "gpt-4-32k" })\nconst s = "client.chat.completions.create(";\n'
    assert js(lint, code) == []


# ---------------------------------------------------------------- LCL001
@pytest.mark.parametrize("code", [
    OPENAI + 'await client.chat.completions.create({ model: "gpt-4.1-mini", messages: [] });\n',
    OPENAI + 'await client.responses.create({ model: "gpt-4.1-mini", input: "hi" });\n',
    'import { GoogleGenAI } from "@google/genai";\nconst ai = new GoogleGenAI({ httpOptions: { timeout: 1 } });\n'
    'await ai.models.generateContent({ model: "gemini-2.5-flash", contents: "hi" });\n',
    AI + 'await generateText({ model: openai("gpt-4.1-mini"), prompt: "hi", abortSignal: s });\n',
    AI + 'streamText({ model: openai("gpt-4.1-mini"), prompt: "hi" });\n',
    'import { ChatOpenAI } from "@langchain/openai";\nconst llm = new ChatOpenAI({ model: "gpt-4.1-mini", timeout: 5 });\n',
])
def test_lcl001_no_cap(lint, code):
    assert "LCL001" in ids(js(lint, code))


@pytest.mark.parametrize("code", [
    OPENAI + 'await client.chat.completions.create({ model: "m", messages: [], max_completion_tokens: 500 });\n',
    OPENAI + 'await client.responses.create({ model: "m", input: "hi", max_output_tokens: 500 });\n',
    OPENAI + 'await client.chat.completions.create(params);\n',
    OPENAI + 'await client.chat.completions.create({ ...base, messages: [] });\n',
    'import { GoogleGenAI } from "@google/genai";\nconst ai = new GoogleGenAI({ httpOptions: { timeout: 1 } });\n'
    'await ai.models.generateContent({ model: "m", contents: "hi", config: { maxOutputTokens: 100 } });\n',
    AI + 'await generateText({ model: openai("m"), prompt: "hi", maxOutputTokens: 300, abortSignal: s });\n',
    AI + 'await generateText({ model: openai("m"), prompt: "hi", maxTokens: 300, abortSignal: s });\n',
    'import { generateText } from "ai";\nimport { anthropic } from "@ai-sdk/anthropic";\n'
    'await generateText({ model: anthropic("claude-sonnet-5-5"), prompt: "hi", abortSignal: s });\n',
    ANTHROPIC + 'await client.messages.create({ model: "claude-sonnet-5-5", max_tokens: 1024, messages: [] });\n',
])
def test_lcl001_ok(lint, code):
    assert "LCL001" not in ids(js(lint, code))


def test_method_definition_named_like_sdk_is_ignored(lint):
    code = AI + "class X {\n  async generateText(p) { return p; }\n}\n"
    assert js(lint, code) == []


def test_openai_assistants_threads_messages_not_anthropic(lint):
    code = OPENAI + 'await client.beta.threads.messages.create(threadId, { role: "user", content: "hi" });\n'
    assert js(lint, code) == []


# ---------------------------------------------------------------- LCL002
def test_lcl002(lint):
    code = ANTHROPIC + 'await client.messages.create({ model: "claude-sonnet-5-5", max_tokens: 64_000, messages: [] });\n'
    assert ("LCL002", 3, "low") in js(lint, code)
    ok = ANTHROPIC + 'const s = client.messages.stream({ model: "claude-sonnet-5-5", max_tokens: 64_000, messages: [] });\n'
    assert "LCL002" not in ids(js(lint, ok))


# ---------------------------------------------------------------- LCL003
@pytest.mark.parametrize("code", [
    'import OpenAI from "openai";\nconst client = new OpenAI();\n',
    'import Anthropic from "@anthropic-ai/sdk";\nconst a = new Anthropic({ apiKey: k });\n',
    'import { GoogleGenAI } from "@google/genai";\nconst ai = new GoogleGenAI({ apiKey: k });\n',
    AI + 'await generateText({ model: openai("m"), prompt: "hi", maxOutputTokens: 9 });\n',
    'import { ChatOpenAI } from "@langchain/openai";\nconst llm = new ChatOpenAI({ model: "m", maxTokens: 9 });\n',
])
def test_lcl003(lint, code):
    assert "LCL003" in ids(js(lint, code))


@pytest.mark.parametrize("code", [
    OPENAI,
    'import OpenAI from "openai";\nconst client = new OpenAI(config);\n',
    'import OpenAI from "openai";\nconst client = new OpenAI();\n'
    'await client.chat.completions.create({ model: "m", messages: [], max_tokens: 5 }, { timeout: 10_000 });\n',
    'import { GoogleGenAI } from "@google/genai";\nconst ai = new GoogleGenAI({ httpOptions: { timeout: 9 } });\n',
    AI + 'streamText({ model: openai("m"), prompt: "hi", maxOutputTokens: 9 });\n',
])
def test_lcl003_ok(lint, code):
    assert "LCL003" not in ids(js(lint, code))


# ---------------------------------------------------------------- LCL004 / LCL005
@pytest.mark.parametrize("code,sev", [
    (OPENAI + 'export async function f(items: string[]) {\n  for (const it of items) {\n'
              '    await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n  }\n}\n', "low"),
    (OPENAI + 'export async function f(items: string[]) {\n  return Promise.all(items.map(async (it) =>\n'
              '    client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] })));\n}\n', "medium"),
    (OPENAI + 'export function f(items: string[]) {\n  items.forEach(async (it) => {\n'
              '    await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n  });\n}\n', "medium"),
    (OPENAI + 'async function one(x) {\n  return client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n}\n'
              'export const all = (xs) => Promise.all(xs.map(one));\n', "medium"),
])
def test_lcl004(lint, code, sev):
    found = js(lint, code)
    assert any(f[0] == "LCL004" and f[2] == sev for f in found), found


@pytest.mark.parametrize("code", [
    OPENAI + 'for (let attempt = 0; attempt < 3; attempt++) {\n'
             '  await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n}\n',
    OPENAI + 'while (true) {\n  const r = await client.chat.completions.create({ model: "m", max_tokens: 5, messages });\n'
             '  if (r) break;\n}\n',
    OPENAI + 'import pLimit from "p-limit";\nconst limit = pLimit(4);\nexport const all = (xs) => Promise.all(xs.map((x) =>'
             ' limit(() => client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] }))));\n',
])
def test_lcl004_ok(lint, code):
    assert "LCL004" not in ids(js(lint, code))


def test_lcl005(lint):
    code = ('import OpenAI from "openai";\nconst c = new OpenAI({ timeout: 5, maxRetries: 0 });\n'
            'export async function f(xs) {\n  for (const x of xs) {\n'
            '    await c.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n  }\n}\n')
    assert ("LCL005", 5, "low") in js(lint, code)
    assert "LCL005" not in ids(js(lint, code.replace("maxRetries: 0", "maxRetries: 3")))
    assert "LCL005" not in ids(js(lint, 'import pRetry from "p-retry";\n' + code))


# ---------------------------------------------------------------- LCL006 / LCL007
def test_lcl006(lint):
    code = ANTHROPIC + f'const SYSTEM = "{BIG}";\n' + \
        'await client.messages.create({ model: "claude-sonnet-5-5", max_tokens: 500, system: SYSTEM, messages: [] });\n'
    assert ("LCL006", 4, "low") in js(lint, code)
    cached = ANTHROPIC + f'const SYSTEM = "{BIG}";\n' + \
        ('await client.messages.create({ model: "claude-sonnet-5-5", max_tokens: 500, '
         'system: [{ type: "text", text: SYSTEM, cache_control: { type: "ephemeral" } }], messages: [] });\n')
    assert "LCL006" not in ids(js(lint, cached))


def test_lcl006_array_join_prompt(lint):
    lines = ",\n".join(f'  "{"rule " * 20}{i}"' for i in range(50))
    code = ANTHROPIC + f"const SYSTEM = [\n{lines}\n].join(\"\\n\");\n" + \
        'await client.messages.create({ model: "claude-sonnet-5-5", max_tokens: 500, system: SYSTEM, messages: [] });\n'
    assert "LCL006" in ids(js(lint, code))


def test_lcl007(lint):
    code = OPENAI + ('export async function f(user) {\n  await client.responses.create({ model: "m", max_output_tokens: 5, '
                     f'input: "q", instructions: `Hello ${{user}}. {BIG}` }});\n}}\n')
    assert ("LCL007", 4, "info") in js(lint, code)


# ---------------------------------------------------------------- LCL008
def test_lcl008(lint):
    code = ANTHROPIC + ('await client.messages.create({ model: "claude-opus-5-5", max_tokens: 500, messages: '
                        '[{ role: "user", content: "Is this spam? Answer with only yes or no." }] });\n')
    assert ("LCL008", 3, "info") in js(lint, code)
    ok = code.replace("claude-opus-5-5", "claude-haiku-5-5")
    assert "LCL008" not in ids(js(lint, ok))


# ---------------------------------------------------------------- LCL009
def test_lcl009(lint):
    code = AI + ('export async function idx(docs) {\n  for (const d of docs) {\n'
                 '    await embed({ model: openai.embedding("m"), value: d });\n  }\n}\n')
    assert ("LCL009", 5, "low") in js(lint, code)
    ok = AI + 'export async function idx(docs) {\n  return embedMany({ model: openai.embedding("m"), values: docs });\n}\n'
    assert "LCL009" not in ids(js(lint, ok))


def test_lcl009_handler_collection(lint):
    code = OPENAI + ('export async function POST(req) {\n  try {\n'
                     '    return await client.embeddings.create({ model: "m", input: documents });\n  } catch (e) { throw e; }\n}\n')
    assert "LCL009" in ids(js(lint, code))
    query = code.replace("input: documents", "input: query")
    assert "LCL009" not in ids(js(lint, query))


# ---------------------------------------------------------------- LCL010
def test_lcl010(lint):
    code = OPENAI + ('const history = [];\nexport async function chat(m) {\n  history.push({ role: "user", content: m });\n'
                     '  return client.chat.completions.create({ model: "m", max_tokens: 5, messages: history });\n}\n')
    assert ("LCL010", 6, "medium") in js(lint, code)
    spread = code.replace("messages: history", 'messages: [{ role: "system", content: "s" }, ...history]')
    assert "LCL010" in ids(js(lint, spread))
    cls = OPENAI + ('class Bot {\n  messages = [];\n  async send(m) {\n    this.messages.push(m);\n'
                    '    return client.chat.completions.create({ model: "m", max_tokens: 5, messages: this.messages });\n  }\n}\n')
    assert "LCL010" in ids(js(lint, cls))


@pytest.mark.parametrize("code", [
    OPENAI + 'const history = [];\nexport async function chat(m) {\n  history.push(m);\n'
             '  return client.chat.completions.create({ model: "m", max_tokens: 5, messages: history.slice(-20) });\n}\n',
    OPENAI + 'let history = [];\nexport async function chat(m) {\n  history.push(m);\n  history = history.slice(-20);\n'
             '  return client.chat.completions.create({ model: "m", max_tokens: 5, messages: history });\n}\n',
    OPENAI + 'export async function agent(task) {\n  const messages = [task];\n  while (true) {\n'
             '    const r = await client.chat.completions.create({ model: "m", max_tokens: 5, messages });\n'
             '    messages.push(r.choices[0].message);\n    if (!r) break;\n  }\n}\n',
])
def test_lcl010_ok(lint, code):
    assert "LCL010" not in ids(js(lint, code))


# ---------------------------------------------------------------- LCL011 / LCL012
def test_lcl011(lint):
    assert "LCL011" in ids(js(lint, OPENAI + 'const MODEL = process.env.MODEL ?? "gpt-4-32k";\n', name="config.ts"))
    assert "LCL011" in ids(js(lint, 'export const config = { model: "claude-2.1" };\n', name="config.ts"))
    assert "LCL011" not in ids(js(lint, 'export const names = ["claude-2.1"];\n', name="config.ts"))


def test_lcl012(lint):
    code = OPENAI + 'await client.chat.completions.create({ model: "m", max_tokens: 5, temperature: 1, top_p: 0.5, messages: [] });\n'
    assert ("LCL012", 3, "info") in js(lint, code)
    code = ANTHROPIC + 'await client.messages.create({ model: "claude-opus-5-5", max_tokens: 500, temperature: 0, messages: [] });\n'
    assert ("LCL012", 3, "low") in js(lint, code)


# ---------------------------------------------------------------- LCL013
@pytest.mark.parametrize("code", [
    OPENAI + 'export async function POST(req: Request) {\n'
             '  const r = await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n'
             '  return Response.json(r);\n}\n',
    OPENAI + 'export const POST = async (req: Request) => {\n'
             '  return Response.json(await client.responses.create({ model: "m", input: "x", max_output_tokens: 5 }));\n};\n',
    OPENAI + 'import express from "express";\nconst app = express();\napp.post("/x", async (req, res) => {\n'
             '  res.json(await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] }));\n});\n',
    OPENAI + 'export default async function handler(req, res) {\n'
             '  res.json(await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] }));\n}\n',
])
def test_lcl013(lint, code):
    assert "LCL013" in ids(js(lint, code))


@pytest.mark.parametrize("code", [
    OPENAI + 'export async function POST(req: Request) {\n  try {\n'
             '    return Response.json(await client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] }));\n'
             '  } catch (err) {\n    return Response.json({ error: "busy" }, { status: 429 });\n  }\n}\n',
    AI + 'export async function POST(req: Request) {\n'
         '  return streamText({ model: openai("m"), prompt: "x", maxOutputTokens: 9 }).toTextStreamResponse();\n}\n',
    OPENAI + 'export async function job() {\n  return client.chat.completions.create({ model: "m", max_tokens: 5, messages: [] });\n}\n',
])
def test_lcl013_ok(lint, code):
    assert "LCL013" not in ids(js(lint, code))


def test_inline_ignore_js(lint):
    code = OPENAI + 'await client.chat.completions.create({ model: "gpt-4-32k", messages: [] }); // llm-cost-lint: ignore\n'
    assert js(lint, code) == []
    code = OPENAI + '// llm-cost-lint: ignore LCL011\nawait client.chat.completions.create({ model: "gpt-4-32k", messages: [] });\n'
    assert ids(js(lint, code)) == {"LCL001"}
