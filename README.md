# llm-cost-lint

A static lint for LLM API calls that waste money or break in production. It reads your
Python and JavaScript/TypeScript source and finds calls to:

- **OpenAI** (Python and Node): `chat.completions.create / parse / stream`, `responses.create`,
  `embeddings.create`, sync and async clients, Azure clients
- **Anthropic** (Python and TypeScript): `messages.create / stream / parse`, Bedrock / Vertex /
  Foundry clients
- **Google GenAI** (`google-genai` for Python, `@google/genai` for JS): `models.generate_content`,
  `generateContent`, `embed_content`
- **Vercel AI SDK** (`ai`): `generateText`, `streamText`, `generateObject`, `streamObject`, `embed`,
  `embedMany`
- **LangChain** chat model constructors (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`,
  `ChatVertexAI`, `ChatLiteLLM`, `init_chat_model`, ...) and their `.invoke()` calls
- **LiteLLM**: `completion`, `acompletion`, `embedding`

and reports the patterns that quietly multiply the bill or fail under load: no output cap,
no timeout, one call per item fired all at once, a conversation history that is never trimmed,
a large system prompt sent uncached on every call, a top-tier model doing a classification job,
embeddings recomputed on every request, retired model names, and more (13 rules).

Every finding has a rule id, a severity, `file:line`, the offending line, why it matters, and
an exact fix written for the SDK and language it was found in, with your variable names.

Zero dependencies (Python 3.10+ standard library only). **Static only:** it never imports or
runs your code, never calls a model and makes no network requests. Model facts (which models
are retired, which are top tier) come from a small dated data file in the package, not from
the internet. CI-friendly exit codes.

---

## Quick start

```bash
pipx install git+https://github.com/abhi12337/llm-cost-lint   # straight from GitHub
pipx install .                          # from a clone of this repo (or: pip install .)

llm-cost-lint scan                      # current folder
llm-cost-lint scan path/to/api path/to/web
llm-cost-lint scan src/agent.py         # a single file
llm-cost-lint scan . --json > report.json
llm-cost-lint rules                     # every rule, why it matters, how to fix it per SDK
```

Try it on the demo project (a FastAPI backend and an Express + Vercel AI SDK server, each with
one wasteful file and one fixed copy, all taken from `tests/fixtures`):

```bash
python examples/make_demo.py demo-llm-app
llm-cost-lint scan demo-llm-app
```

Exit codes: `0` = nothing at or above `--fail-on` (default `medium`), `1` = findings at or above
it, `2` = usage error (or, with `--strict`, a file that could not be parsed). Use
`--fail-on critical|high|medium|low|info|never` to tune CI.

A Python file with a syntax error cannot be checked. It is never treated as clean: each one is
reported on stderr (`warning: api/app.py: syntax error line 12; file not scanned`), counted in
the text header and listed under `"skipped"` in the JSON report. Add `--strict` to make such
files fail the run with exit code `2`.

**Why the default is `medium`:** this tool has no
critical or high rules, because nothing it finds leaks data. Its `medium` rules are the ones
with no upper bound on cost or that cause an outage: a call with no output cap, an unbounded
parallel fan-out, a chat history that grows forever, and a retired model id. Those should block
a merge. `low` findings are gaps where the SDK default usually works (timeouts, error handling,
missing prompt caching) or heuristics, and `info` findings are advice; they are reported but do
not fail the build unless you ask with `--fail-on low` or `--fail-on info`.

## Example output

All output below was produced by running this version of the tool on the demo project (nothing
is hand-edited).

`llm-cost-lint scan demo-llm-app/web --brief` (the TypeScript server only):

```text
llm-cost-lint 0.1.1
Scanned demo-llm-app/web
2 file(s) (0 Python, 2 JS/TS); 10 LLM call(s) in 2 file(s); model list as of 2026-11-11

MEDIUM   LCL001  No output token cap
      at: src/server.ts:95
      client.chat.completions.create() sets no max_completion_tokens / max_tokens: the answer can
      run to the model's maximum output length and every token is billed.
      code: const completion = await client.chat.completions.create({

MEDIUM   LCL001  No output token cap
      at: src/server.ts:127
      generateText() sets no maxOutputTokens (maxTokens in AI SDK 4): the answer can run to the
      model's maximum output length and every token is billed.
      code: const { text } = await generateText({

MEDIUM   LCL004  LLM call per item in a loop
      at: src/server.ts:116
      anthropic.messages.create() is fired for every item at once (loop at line 115) with no
      concurrency limit: a large list means a burst of parallel requests, rate-limit errors and
      retries of billed work.
      code: const r = await anthropic.messages.create({

MEDIUM   LCL010  Conversation history grows without a limit [heuristic]
      at: src/server.ts:95
      'messages' is kept in a module-level array (it lives as long as the server process), grows
      every turn and is sent whole to client.chat.completions.create() with no trimming or
      summarising: every turn re-bills the entire conversation.
      code: const completion = await client.chat.completions.create({

MEDIUM   LCL011  Retired or deprecated model name
      at: src/server.ts:105
      'claude-3-5-sonnet-20241022' is listed as retired by Anthropic since 2025-10-28 (llm-cost-
      lint list as of 2026-11-11): calls with it fail.
      code: model: "claude-3-5-sonnet-20241022",

LOW      LCL003  No request timeout configured
      at: src/server.ts:7
      OpenAI() is created without a timeout option: the SDK default is 10 minutes per attempt,
      with 2 retries by default, so one stuck call can hold a worker far longer.
      code: const client = new OpenAI();

LOW      LCL003  No request timeout configured
      at: src/server.ts:8
      Anthropic() is created without a timeout option: the SDK default is 10 minutes per attempt,
      with 2 retries by default, so one stuck call can hold a worker far longer.
      code: const anthropic = new Anthropic({ maxRetries: 0 });

LOW      LCL003  No request timeout configured
      at: src/server.ts:127
      generateText() has no abortSignal: the AI SDK sets no request timeout of its own, so a slow
      provider holds the request until the platform kills it.
      code: const { text } = await generateText({

LOW      LCL005  Loop of LLM calls with retries disabled
      at: src/server.ts:116
      anthropic.messages.create() runs in a loop with retries disabled (anthropic is created with
      maxRetries: 0) and no retry/backoff of its own: one 429 stops the job.
      code: const r = await anthropic.messages.create({

LOW      LCL006  Large static prompt sent without prompt caching (Anthropic)
      at: src/server.ts:104
      The system prompt (SUPPORT_PROMPT) is 6,175 characters, identical on every call, and the
      request has no cache_control: it is billed as fresh input every time.
      code: const msg = await anthropic.messages.create({

LOW      LCL008  Top-tier model used for a simple or repeated task [heuristic]
      at: src/server.ts:116
      'claude-opus-5-5' is a top-tier model and it runs once per item in a loop: a smaller model
      (claude-haiku-5-5 or claude-sonnet-5-5) is usually enough and costs much less per token.
      code: const r = await anthropic.messages.create({

LOW      LCL009  Embeddings recomputed without a cache [heuristic]
      at: src/server.ts:137
      embed() is called once per item in a loop (line 136) with no cache in this file: the same
      texts are re-embedded on every run, one request each.
      code: const { embedding } = await embed({ model: openai.embedding("text-embedding-3-small"), value: doc });

LOW      LCL013  LLM call in a request handler without error handling
      at: src/server.ts:95
      client.chat.completions.create() runs in a request handler outside any try/catch: a rate
      limit or timeout becomes an unhandled 500.
      code: const completion = await client.chat.completions.create({

LOW      LCL013  LLM call in a request handler without error handling
      at: src/server.ts:104
      anthropic.messages.create() runs in a request handler outside any try/catch: a rate limit or
      timeout becomes an unhandled 500.
      code: const msg = await anthropic.messages.create({

Summary: 0 critical, 0 high, 5 medium, 9 low, 0 info
Exit code: 1
```

A full finding (without `--brief`) carries the reason once per rule and a fix with the names
from your code:

```text
MEDIUM   LCL001  No output token cap
      at: api/app.py:102
      client.chat.completions.create() sets no max_completion_tokens / max_tokens: the answer can
      run to the model's maximum output length and every token is billed.
      code: resp = client.chat.completions.create(model="gpt-4o-mini", messages=history)
      why: Without an output cap the model may write until it reaches the model's maximum
           output length, and you pay for every output token. A prompt bug, a jailbreak or a
           model that loops ("repeat forever") turns one request into the most expensive
           request the model can produce. An explicit cap also makes cost per request
           predictable and lets you detect truncation instead of silently paying for it.
      fix:
        Set a cap sized to the longest answer you need, and detect truncation:

          resp = client.chat.completions.create(
              model="gpt-4o-mini", messages=messages,
              max_completion_tokens=800,   # o-series / gpt-5: includes reasoning tokens
          )                                # (older chat models also accept max_tokens=800)
          if resp.choices[0].finish_reason == "length": ...   # hit the cap

LOW      LCL006  Large static prompt sent without prompt caching (Anthropic)
      at: api/app.py:109
      The system prompt (SUPPORT_PROMPT) is 6,175 characters, identical on every call, and the
      request has no cache_control: it is billed as fresh input every time.
      code: msg = claude.messages.create(
      why: Anthropic only caches a prompt prefix when the request asks for it (cache_control
           on a block, or top-level cache_control for automatic caching). A large system
           prompt that is identical on every call is otherwise billed as fresh input every
           time. Cached reads are billed at a small fraction of the normal input price and
           are faster. Prefixes shorter than the model's minimum cacheable length (between
           512 and 4,096 tokens depending on the model, per Anthropic's docs) are not cached
           even when marked.
      fix:
        Mark the static prefix as cacheable:

          resp = claude.messages.create(
              model="claude-opus-4-8", max_tokens=...,
              system=[{"type": "text", "text": SUPPORT_PROMPT, "cache_control": {"type": "ephemeral"}}],
              messages=messages,
          )
          # or let the API place the breakpoint: messages.create(..., cache_control={"type": "ephemeral"})

        Verify with resp.usage.cache_read_input_tokens > 0 on the second call.
```

<details>
<summary>The whole demo project: <code>llm-cost-lint scan demo-llm-app --brief</code> (Python + TypeScript)</summary>

```text
llm-cost-lint 0.1.1
Scanned demo-llm-app
4 file(s) (2 Python, 2 JS/TS); 28 LLM call(s) in 4 file(s); model list as of 2026-11-11

MEDIUM   LCL001  No output token cap
      at: api/app.py:102
      client.chat.completions.create() sets no max_completion_tokens / max_tokens: the answer can
      run to the model's maximum output length and every token is billed.
      code: resp = client.chat.completions.create(model="gpt-4o-mini", messages=history)

MEDIUM   LCL001  No output token cap
      at: api/app.py:134
      aclient.responses.create() sets no max_output_tokens: the answer can run to the model's
      maximum output length and every token is billed.
      code: r = await aclient.responses.create(model="gpt-5", input=doc)

MEDIUM   LCL001  No output token cap
      at: api/app.py:241
      litellm.completion() sets no max_tokens: the answer can run to the model's maximum output
      length and every token is billed.
      code: return litellm.completion(model="gpt-4-32k", messages=[{"role": "user", "content": text}])

MEDIUM   LCL001  No output token cap
      at: api/app.py:245
      gemini.models.generate_content() sets no config.max_output_tokens: the answer can run to the
      model's maximum output length and every token is billed.
      code: return gemini.models.generate_content(model="gemini-2.5-flash", contents=question)

MEDIUM   LCL001  No output token cap
      at: web/src/server.ts:95
      client.chat.completions.create() sets no max_completion_tokens / max_tokens: the answer can
      run to the model's maximum output length and every token is billed.
      code: const completion = await client.chat.completions.create({

MEDIUM   LCL001  No output token cap
      at: web/src/server.ts:127
      generateText() sets no maxOutputTokens (maxTokens in AI SDK 4): the answer can run to the
      model's maximum output length and every token is billed.
      code: const { text } = await generateText({

MEDIUM   LCL004  LLM call per item in a loop
      at: api/app.py:134
      aclient.responses.create() is fired for every item at once (through one(), called per item
      at line 137) with no concurrency limit: a large list means a burst of parallel requests,
      rate-limit errors and retries of billed work.
      code: r = await aclient.responses.create(model="gpt-5", input=doc)

MEDIUM   LCL004  LLM call per item in a loop
      at: web/src/server.ts:116
      anthropic.messages.create() is fired for every item at once (loop at line 115) with no
      concurrency limit: a large list means a burst of parallel requests, rate-limit errors and
      retries of billed work.
      code: const r = await anthropic.messages.create({

MEDIUM   LCL010  Conversation history grows without a limit [heuristic]
      at: api/app.py:102
      'history' is kept in a module-level list (it lives as long as the process), grows every turn
      and is sent whole to client.chat.completions.create() with no trimming or summarising: every
      turn re-bills the entire conversation.
      code: resp = client.chat.completions.create(model="gpt-4o-mini", messages=history)

MEDIUM   LCL010  Conversation history grows without a limit [heuristic]
      at: web/src/server.ts:95
      'messages' is kept in a module-level array (it lives as long as the server process), grows
      every turn and is sent whole to client.chat.completions.create() with no trimming or
      summarising: every turn re-bills the entire conversation.
      code: const completion = await client.chat.completions.create({

MEDIUM   LCL011  Retired or deprecated model name
      at: api/app.py:241
      'gpt-4-32k' is listed as retired by OpenAI since 2025-06-06 (llm-cost-lint list as of
      2026-11-11): calls with it fail.
      code: return litellm.completion(model="gpt-4-32k", messages=[{"role": "user", "content": text}])

MEDIUM   LCL011  Retired or deprecated model name
      at: web/src/server.ts:105
      'claude-3-5-sonnet-20241022' is listed as retired by Anthropic since 2025-10-28 (llm-cost-
      lint list as of 2026-11-11): calls with it fail.
      code: model: "claude-3-5-sonnet-20241022",

LOW      LCL002  Very large max_tokens on a non-streaming Anthropic call
      at: api/app.py:109
      claude.messages.create() asks for max_tokens=64,000 without streaming: the request can run
      past HTTP timeouts and the cap no longer bounds the cost of one call.
      code: msg = claude.messages.create(

LOW      LCL003  No request timeout configured
      at: api/app.py:11
      OpenAI() is created without timeout=: the SDK default is 10 minutes per attempt, with 2
      retries by default, so one stuck call can hold a worker far longer.
      code: client = OpenAI()

LOW      LCL003  No request timeout configured
      at: api/app.py:12
      AsyncOpenAI() is created without timeout=: the SDK default is 10 minutes per attempt, with 2
      retries by default, so one stuck call can hold a worker far longer.
      code: aclient = AsyncOpenAI(max_retries=0)

LOW      LCL003  No request timeout configured
      at: api/app.py:13
      Anthropic() is created without timeout=: the SDK default is 10 minutes per attempt, with 2
      retries by default, so one stuck call can hold a worker far longer.
      code: claude = anthropic.Anthropic()

LOW      LCL003  No request timeout configured
      at: api/app.py:14
      Client() is created without a request timeout (http_options.timeout): a slow or stuck call
      can hold the caller for as long as the HTTP layer allows.
      code: gemini = genai.Client()

LOW      LCL003  No request timeout configured
      at: api/app.py:241
      litellm.completion() has no timeout=: LiteLLM's built-in default request timeout is long
      (check litellm.request_timeout for your version).
      code: return litellm.completion(model="gpt-4-32k", messages=[{"role": "user", "content": text}])

LOW      LCL003  No request timeout configured
      at: web/src/server.ts:7
      OpenAI() is created without a timeout option: the SDK default is 10 minutes per attempt,
      with 2 retries by default, so one stuck call can hold a worker far longer.
      code: const client = new OpenAI();

LOW      LCL003  No request timeout configured
      at: web/src/server.ts:8
      Anthropic() is created without a timeout option: the SDK default is 10 minutes per attempt,
      with 2 retries by default, so one stuck call can hold a worker far longer.
      code: const anthropic = new Anthropic({ maxRetries: 0 });

LOW      LCL003  No request timeout configured
      at: web/src/server.ts:127
      generateText() has no abortSignal: the AI SDK sets no request timeout of its own, so a slow
      provider holds the request until the platform kills it.
      code: const { text } = await generateText({

LOW      LCL004  LLM call per item in a loop
      at: api/app.py:122
      client.chat.completions.create() runs once per item (loop at line 121): cost grows with the
      list; batch items into one prompt or use the provider's batch API for offline work.
      code: r = client.chat.completions.create(

LOW      LCL005  Loop of LLM calls with retries disabled
      at: api/app.py:134
      aclient.responses.create() runs in a loop with retries disabled (aclient is created with
      max_retries=0) and no retry/backoff of its own: one 429 stops the job.
      code: r = await aclient.responses.create(model="gpt-5", input=doc)

LOW      LCL005  Loop of LLM calls with retries disabled
      at: web/src/server.ts:116
      anthropic.messages.create() runs in a loop with retries disabled (anthropic is created with
      maxRetries: 0) and no retry/backoff of its own: one 429 stops the job.
      code: const r = await anthropic.messages.create({

LOW      LCL006  Large static prompt sent without prompt caching (Anthropic)
      at: api/app.py:109
      The system prompt (SUPPORT_PROMPT) is 6,175 characters, identical on every call, and the
      request has no cache_control: it is billed as fresh input every time.
      code: msg = claude.messages.create(

LOW      LCL006  Large static prompt sent without prompt caching (Anthropic)
      at: web/src/server.ts:104
      The system prompt (SUPPORT_PROMPT) is 6,175 characters, identical on every call, and the
      request has no cache_control: it is billed as fresh input every time.
      code: const msg = await anthropic.messages.create({

LOW      LCL008  Top-tier model used for a simple or repeated task [heuristic]
      at: api/app.py:122
      'gpt-4o' is a top-tier model and it runs once per item in a loop: a smaller model
      (gpt-4o-mini or gpt-5-mini) is usually enough and costs much less per token.
      code: r = client.chat.completions.create(

LOW      LCL008  Top-tier model used for a simple or repeated task [heuristic]
      at: api/app.py:134
      'gpt-5' is a top-tier model and it runs once per item in a loop: a smaller model (gpt-5-mini
      or gpt-5-nano) is usually enough and costs much less per token.
      code: r = await aclient.responses.create(model="gpt-5", input=doc)

LOW      LCL008  Top-tier model used for a simple or repeated task [heuristic]
      at: web/src/server.ts:116
      'claude-opus-5-5' is a top-tier model and it runs once per item in a loop: a smaller model
      (claude-haiku-5-5 or claude-sonnet-5-5) is usually enough and costs much less per token.
      code: const r = await anthropic.messages.create({

LOW      LCL009  Embeddings recomputed without a cache [heuristic]
      at: api/app.py:142
      client.embeddings.create() embeds a collection inside a request handler with no cache in
      this file: document embeddings are recomputed and billed on every request.
      code: vectors = client.embeddings.create(model="text-embedding-3-small", input=documents)

LOW      LCL009  Embeddings recomputed without a cache [heuristic]
      at: web/src/server.ts:137
      embed() is called once per item in a loop (line 136) with no cache in this file: the same
      texts are re-embedded on every run, one request each.
      code: const { embedding } = await embed({ model: openai.embedding("text-embedding-3-small"), value: doc });

LOW      LCL012  Sampling, n or streaming settings worth a second look
      at: api/app.py:109
      claude.messages.create(): temperature is rejected with HTTP 400 by 'claude-opus-4-8' (per
      Anthropic's docs); up to 64,000 output tokens are generated without streaming in a request
      handler (the user sees nothing until the end).
      code: msg = claude.messages.create(

LOW      LCL013  LLM call in a request handler without error handling
      at: api/app.py:102
      client.chat.completions.create() runs in a request handler outside any try/except: a rate
      limit or timeout becomes an unhandled 500.
      code: resp = client.chat.completions.create(model="gpt-4o-mini", messages=history)

LOW      LCL013  LLM call in a request handler without error handling
      at: api/app.py:109
      claude.messages.create() runs in a request handler outside any try/except: a rate limit or
      timeout becomes an unhandled 500.
      code: msg = claude.messages.create(

LOW      LCL013  LLM call in a request handler without error handling
      at: api/app.py:142
      client.embeddings.create() runs in a request handler outside any try/except: a rate limit or
      timeout becomes an unhandled 500.
      code: vectors = client.embeddings.create(model="text-embedding-3-small", input=documents)

LOW      LCL013  LLM call in a request handler without error handling
      at: web/src/server.ts:95
      client.chat.completions.create() runs in a request handler outside any try/catch: a rate
      limit or timeout becomes an unhandled 500.
      code: const completion = await client.chat.completions.create({

LOW      LCL013  LLM call in a request handler without error handling
      at: web/src/server.ts:104
      anthropic.messages.create() runs in a request handler outside any try/catch: a rate limit or
      timeout becomes an unhandled 500.
      code: const msg = await anthropic.messages.create({

INFO     LCL007  Dynamic value near the start of a large prompt
      at: api/app.py:226
      The system message (system) is 6,154 characters with a per-request value inserted before
      6,145 of them: that part can never be served from the prompt cache.
      code: return client.chat.completions.create(

INFO     LCL012  Sampling, n or streaming settings worth a second look
      at: api/app.py:234
      client.chat.completions.create(): temperature and top_p are both set (tune one, not both); 3
      completions are requested and billed per call.
      code: return client.chat.completions.create(

Summary: 0 critical, 0 high, 12 medium, 25 low, 2 info
Exit code: 1
```

</details>

The same two servers written carefully (`app_fixed.py`, `server_fixed.ts`, which do the same
work with caps, timeouts, a concurrency limit, a trimmed history, prompt caching, a cache for
embeddings, current models and error handling) produce no findings:

```text
llm-cost-lint 0.1.1
Scanned demo-llm-app/api/app_fixed.py, demo-llm-app/web/src/server_fixed.ts
2 file(s) (1 Python, 1 JS/TS); 14 LLM call(s) in 2 file(s); model list as of 2026-11-11

No findings. (This is not a guarantee: see 'Known limitations' in the README.)
Summary: 0 critical, 0 high, 0 medium, 0 low, 0 info
Exit code: 0
```

## What it checks

| Rule | Severity | What it finds |
|---|---|---|
| LCL001 | Medium | A call with no output cap where the SDK makes it optional: OpenAI `chat.completions` without `max_completion_tokens` / `max_tokens`, `responses` without `max_output_tokens`, Google GenAI without `config.max_output_tokens`, Vercel AI SDK without `maxOutputTokens` (`maxTokens` in AI SDK 4), LiteLLM without `max_tokens`, LangChain `ChatOpenAI` / `ChatGoogleGenerativeAI` / `ChatVertexAI` / `ChatLiteLLM` without a cap. Not reported for Anthropic (where `max_tokens` is required), for the Vercel Anthropic provider and LiteLLM with Claude models (both send a default), or when the options come from a variable or `**kwargs` (unknown). |
| LCL002 | Low | An Anthropic `messages.create` with `max_tokens` above 16,384 and no streaming: long non-streaming requests run into HTTP timeouts and the cap stops bounding the cost of a call. Streaming calls are not reported. |
| LCL003 | Low | No request timeout: `OpenAI()`, `AsyncOpenAI()`, `Anthropic()`, `new OpenAI({...})`, `new Anthropic({...})` without `timeout` (the OpenAI and Anthropic SDKs default to 10 minutes per attempt, with 2 retries), `genai.Client()` / `new GoogleGenAI()` without `http_options.timeout` / `httpOptions.timeout`, LangChain chat models without `timeout`, LiteLLM calls without `timeout=` (and no `litellm.request_timeout`), Vercel `generateText` / `generateObject` without `abortSignal`. A custom `http_client`, a `with_options(timeout=...)` or a per-request timeout anywhere in the file counts. |
| LCL004 | Medium / Low | An LLM call per item of a list. Medium when all items are fired at once with no limiter: `asyncio.gather` over a comprehension, `create_task` in a loop, `Promise.all(items.map(...))`, `forEach(async ...)`, including through one level of same-file helper function. Low for a sequential `for` loop (cost grows with the list; batch or use the batch API). Retry loops (`for attempt in range(3)`, `for (let i = 0; i < 3; i++)`), `while` loops (agent / tool loops) and files that use a limiter (`asyncio.Semaphore`, `p-limit`, `p-queue`, `Bottleneck`, thread pools, `max_concurrency`) are not reported. |
| LCL005 | Low | A loop of LLM calls whose client was created with `max_retries=0` / `maxRetries: 0`, or LiteLLM without `num_retries`, and no retry of its own (`tenacity`, `backoff`, `p-retry`, ...): one 429 stops the job. |
| LCL006 | Low | Anthropic only: a static system prompt of 4,000+ characters (`--large-prompt-chars`) sent with no `cache_control` (block-level or top-level automatic caching). The prompt is resolved through module-level constants, string concatenation, `textwrap.dedent`, and JS `const`s, template literals and `[...].join()`. |
| LCL007 | Info | Any provider: a per-request value (f-string / template `${}` / `.format`) inserted before at least half of a large static prompt. Prompt caching is a prefix match everywhere (OpenAI caches long prefixes automatically, Gemini 2.5+ caches implicitly, Anthropic caches what you mark), so the static text after the dynamic value is never cached. A large static OpenAI or Gemini prompt alone is **not** reported, because those providers cache it without any code change. |
| LCL008 | Info / Low | Heuristic: a top-tier model (`claude-opus-*`, `claude-fable-*`, `gpt-5`, `o3`, `gpt-4o`, `gemini-*-pro`, ... from the data file) used where a smaller model usually suffices: the prompt looks like classification / labelling / yes-no / extraction, or the output cap is 64 tokens or less (info), or the call runs once per item in a loop (low). |
| LCL009 | Low | Heuristic: embeddings called once per item in a loop, or a collection (`documents`, `chunks`, a list literal, a comprehension, `embedMany`, `embed_documents`) embedded inside a request handler, in a file with no cache (`cache`, `lru_cache`, `redis`, `memo`, a vector store `upsert`, ...). Embedding the user's single query per request is not reported. |
| LCL010 | Medium | Heuristic: a message list that lives across turns (module-level, `self.messages` / `this.messages`, a per-session store such as `sessions.setdefault(id, [])`, or a `while ... input()` chat loop), is appended to, and is sent whole as `messages` with no slicing (`[-20:]`, `.slice(-20)`), `del` / `pop` / `splice` / `shift`, length check, or trim/summarise/truncate/compact helper. Agent loops over a local list inside one function are not reported. |
| LCL011 | Medium | A string that exactly equals a retired or deprecated model id from the dated data file (`gpt-4-32k`, `o1-preview`, `claude-3-5-sonnet-20241022`, `claude-2.1`, `gemini-1.5-pro`, ...), with provider prefixes (`openai/`, `anthropic/`, `models/`, `anthropic:`) stripped. In files that import an LLM SDK every string literal is checked; elsewhere only values assigned to a `model`-named variable, key or argument (including `os.getenv("MODEL", "...")` / `process.env.MODEL ?? "..."`). |
| LCL012 | Info / Low | `temperature` and `top_p` both set; `n` / `candidate_count` above 1; a non-streaming call in a request handler allowed to generate 4,000+ tokens. Low when an Anthropic call sends `temperature` / `top_p` / `top_k` to a model that Anthropic documents as rejecting them with HTTP 400 (Opus 4.7 and later, Fable, Mythos, Sonnet 5 / 5.5, Haiku 5.5, per the data file). |
| LCL013 | Low | An LLM call inside a request handler (FastAPI / Flask / Django-style decorators or `request` parameter, Next.js `GET`/`POST` route handlers, Pages API `handler(req, res)`, Express / Hono / Fastify routes, `"use server"` files) that is not inside a `try` with an `except` / `catch`, or wrapped in a retry decorator. `streamText` / `streamObject` (errors arrive through the stream) are not reported. |

Findings marked `[heuristic]` (LCL008, LCL009, LCL010) are pattern-based guesses about intent;
treat them as review prompts.

### The model data file

`src/llm_cost_lint/data/models.json` holds the model tiers for LCL008, the retired / deprecated
ids for LCL011 and the "rejects sampling parameters" patterns for LCL012, with an `as_of` date
(currently **2026-11-11**) that is printed in every report. It contains **no prices**: the tool never
states a price, only that one tier costs much less than another within one provider. The
retired list is deliberately conservative (exact ids that the providers have shut down or
announced a shutdown date for); dates the author could not confirm are left empty. Check the
providers' deprecation pages before acting, and send a pull request when they change.

## Known limitations

Be clear about the limits before relying on it:

- **It is static and per file.** It sees the code, not the traffic. A finding says a pattern is
  present, not how much it costs you; a clean report is no guarantee. Calls whose client or
  options come from another module are only partly understood: a call with options built in a
  variable or passed through `**kwargs` / `...spread` is treated as unknown and the rules that
  need those options stay silent (fewer false positives, some misses).
- **JavaScript/TypeScript is analysed without a full parser.** Comments and strings are lexed
  out and patterns are matched on what is left. Only object-literal arguments are inspected.
  Unusual shapes (calls built dynamically, clients behind wrappers, handlers registered far
  from their definition) may be missed. Python uses the real `ast` module and is more precise.
- **Loop, handler and history detection follow one level of same-file helpers.** A call made in
  a helper that is looped over from another file is not linked to the loop.
- **Recognition of mitigations is by pattern.** Any limiter, retry library or cache name in the
  file counts as handling the problem for the whole file, even if it guards a different call.
  A `try` with any `except` counts as error handling.
- **Model facts age.** Tiers and retirements are as of the data file's date; newer models are
  unknown to it (unknown models are never reported by LCL008 or LCL011). It does not read model
  names from `.env`, YAML or JSON config files.
- **Provider behaviour is summarised, not guaranteed.** Default timeouts, automatic caching and
  minimum cacheable lengths are described as the providers document them on the data file's
  date; SDK versions differ. Messages are phrased conservatively where behaviour varies.
- **Not covered yet:** other SDKs (Cohere, Mistral, Bedrock `converse`, Ollama, raw `fetch` /
  `requests` to provider URLs), LangChain chains and agents beyond chat model constructors and
  `.invoke()`, LlamaIndex, prompt caching via the Vercel AI SDK `providerOptions`, Go / Java /
  other languages, and anything about prompt quality.

## How to fix each finding

`llm-cost-lint rules` prints every rule with its fix for each SDK. In a report, each finding
shows only the fix for the SDK and language it was found in, with your client, model and
variable names filled in. The short version:

**LCL001 - no output cap.** OpenAI: `max_completion_tokens=800` (chat) or `max_output_tokens=800`
(responses); Google: `config=types.GenerateContentConfig(max_output_tokens=800)` /
`config: { maxOutputTokens: 800 }`; Vercel: `maxOutputTokens: 800`; LangChain:
`ChatOpenAI(max_tokens=800)` / `new ChatOpenAI({ maxTokens: 800 })`; LiteLLM: `max_tokens=800`.
Check the finish reason (`"length"`, `incomplete`, `MAX_TOKENS`) to detect truncation.

**LCL002 - huge non-streaming max_tokens.** Lower it, or `client.messages.stream(...)` +
`get_final_message()` / `finalMessage()`.

**LCL003 - no timeout.** `OpenAI(timeout=30.0)`, `new Anthropic({ timeout: 30_000 })` (Python takes
seconds, JS milliseconds), `genai.Client(http_options=types.HttpOptions(timeout=30_000))`
(milliseconds), `abortSignal: AbortSignal.timeout(30_000)` for the AI SDK, `timeout=30` for LiteLLM
and LangChain.

**LCL004 - per-item calls.** Bound concurrency (`asyncio.Semaphore(8)`, `p-limit`), put several items
in one prompt, or use the provider's batch API for offline work.

**LCL005 - retries disabled in a loop.** Keep the SDK retries (`max_retries=3`), `num_retries=3`
for LiteLLM, or `tenacity` / `p-retry` with exponential backoff.

**LCL006 - uncached Anthropic prompt.**
`system=[{"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}]` or top-level
`cache_control={"type": "ephemeral"}`; verify `usage.cache_read_input_tokens > 0`.

**LCL007 - dynamic value early in a large prompt.** Keep the static text first and identical;
move dates, names and ids to the end or into the user message.

**LCL008 - top-tier model for a simple task.** Try the smaller model of the same provider on real
examples, keep the big one only where it is measurably better.

**LCL009 - embeddings without a cache.** Cache by a hash of the text, embed lists in one request,
embed documents at ingest time and only the query per request.

**LCL010 - unbounded history.** Send `history[-N:]` / `history.slice(-N)`, summarise older turns,
or use server-side compaction where the provider offers it.

**LCL011 - retired model.** Replace the id (the finding suggests a current one), read it from
configuration, re-run your evals.

**LCL012 - sampling settings.** One of `temperature` / `top_p`; `n` only when you compare
candidates; drop sampling parameters on Claude models that reject them; stream long
user-facing answers.

**LCL013 - no error handling in a handler.** Catch the SDK's rate-limit / timeout / connection
errors (`openai.RateLimitError`, `anthropic.RateLimitError`, `google.genai.errors.APIError`,
`litellm.RateLimitError`, `APICallError` in the AI SDK) and return a 429 / 503.

## Suppressing a finding

Add `llm-cost-lint: ignore` as a comment on the reported line or on the line above it. Add rule
ids to silence only those:

```python
resp = client.chat.completions.create(model=MODEL, messages=msgs)  # llm-cost-lint: ignore LCL001
```

```ts
// llm-cost-lint: ignore
const r = await generateText({ model, prompt });
```

To turn a rule off for a whole run: `--disable LCL008` (repeatable). To change what counts as a
large prompt: `--large-prompt-chars 8000`.

## Use in CI

See [`examples/github-actions/llm-cost-lint.yml`](examples/github-actions/llm-cost-lint.yml). It
runs on every pull request, fails the build on medium findings, and uploads the JSON report as
an artifact.

## JSON output

`--json` prints (one finding shown, from the demo project; `targets` shortened):

```json
{
  "tool": "llm-cost-lint",
  "version": "0.1.1",
  "targets": [
    "/abs/path/demo-llm-app"
  ],
  "files_scanned": 4,
  "python_files": 2,
  "js_files": 2,
  "llm_files": [
    "api/app.py",
    "api/app_fixed.py",
    "web/src/server.ts",
    "web/src/server_fixed.ts"
  ],
  "llm_calls": 28,
  "model_data_as_of": "2026-11-11",
  "skipped": [],
  "exit_code": 1,
  "summary": {
    "critical": 0,
    "high": 0,
    "medium": 12,
    "low": 25,
    "info": 2
  },
  "findings": [
    {
      "rule_id": "LCL010",
      "severity": "medium",
      "message": "'history' is kept in a module-level list (it lives as long as the process), grows every turn and is sent whole to client.chat.completions.create() with no trimming or summarising: every turn re-bills the entire conversation.",
      "file": "api/app.py",
      "line": 102,
      "evidence": "resp = client.chat.completions.create(model=\"gpt-4o-mini\", messages=history)",
      "heuristic": true,
      "title": "Conversation history grows without a limit",
      "why": "Chat APIs are stateless: the whole message list is sent and billed as input on every turn. A history list that is appended to on every turn and never trimmed makes each turn more expensive than the last (total cost grows roughly with the square of the number of turns), and long sessions eventually fail when they exceed the model's context window.",
      "fix": "Bound what you send. Keep the system prompt plus the last N turns, and summarise or drop\nolder ones:\n\n  MAX_TURNS = 20\n  history.append({\"role\": \"user\", \"content\": text})\n  window = history[-MAX_TURNS * 2:]\n  resp = client.chat.completions.create(..., messages=window)\n\nCount tokens (or characters) rather than turns if messages vary a lot in size. Anthropic\nalso offers server-side compaction (beta) for long conversations.",
      "extra": {
        "var": "history",
        "call": "client.chat.completions.create",
        "sdk": "openai-chat",
        "lang": "py"
      }
    }
  ]
}
```

## Options

```text
llm-cost-lint scan [PATH ...]
  --exclude GLOB              skip paths (repeatable)
  --include-tests             also scan test / fixture files (skipped by default)
  --disable RULE              ignore a rule id (repeatable)
  --large-prompt-chars N      prompt size that counts as large for LCL006 / LCL007 (default 4000)
  --max-file-mb N             skip files larger than this (default 2)
  --json                      JSON report
  --fail-on SEVERITY          exit 1 at or above this severity (default medium; or never)
  --min-severity SEVERITY     hide findings below this severity
  --brief                     omit the why / fix text
  --strict                    exit 2 if any file could not be parsed (default: warn on stderr only)
  --no-color                  disable ANSI colours (also honours NO_COLOR)
llm-cost-lint rules
```

`node_modules`, `.git`, virtualenvs, build output (`dist`, `build`, `.next`, ...) and hidden
folders are always skipped.

## Development

```bash
pip install -e ".[test]"
pytest
```

Every rule has tests for code that must be reported and for well-configured code that must not
be, in Python and in TypeScript (`tests/test_python.py`, `tests/test_js.py`), plus the two
fixture pairs used by the demo.

## License

MIT, Copyright (c) 2026 Abheesht Tripathi. See [LICENSE](LICENSE).

---

Questions, or want a review of what your LLM features cost before the bill arrives? abheeshttripathi890@gmail.com
