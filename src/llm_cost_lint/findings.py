"""Finding model and the rule catalogue (ids, titles, why it matters, SDK-specific fixes).

Each rule has one fix text per SDK/language ("openai-chat-py", "anthropic-js", ...). A finding
shows only the fix for the SDK and language it was found in, with its own names filled in."""

from __future__ import annotations

import textwrap
from dataclasses import asdict, dataclass, field

SEVERITIES = ("critical", "high", "medium", "low", "info")
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}  # lower = worse

SDK_LABELS = {
    "openai-chat-py": "OpenAI Python (chat.completions)",
    "openai-chat-js": "OpenAI Node (chat.completions)",
    "openai-responses-py": "OpenAI Python (responses)",
    "openai-responses-js": "OpenAI Node (responses)",
    "openai-py": "OpenAI Python",
    "openai-js": "OpenAI Node",
    "anthropic-py": "Anthropic Python",
    "anthropic-js": "Anthropic TypeScript",
    "google-py": "Google GenAI Python (google-genai)",
    "google-js": "Google GenAI JS (@google/genai)",
    "vercel-js": "Vercel AI SDK (ai)",
    "langchain-py": "LangChain Python",
    "langchain-js": "LangChain JS",
    "litellm-py": "LiteLLM",
    "py": "Python",
    "js": "JavaScript / TypeScript",
}


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    why: str
    fixes: dict

    @property
    def fix(self) -> str:
        """Every variant, for `llm-cost-lint rules`."""
        if list(self.fixes) == ["default"]:
            return self.fixes["default"]
        parts = []
        for key, text in self.fixes.items():
            label = SDK_LABELS.get(key, "Any SDK" if key == "default" else key)
            parts.append(f"[{label}]\n" + textwrap.indent(text, "  "))
        return "\n".join(parts)


RULES: dict[str, Rule] = {}


def _rule(id: str, title: str, why: str, fixes: dict[str, str]) -> None:
    RULES[id] = Rule(id, title, " ".join(why.split()),
                     {k: textwrap.dedent(v).strip("\n") for k, v in fixes.items()})


# ---------------------------------------------------------------- LCL001
_rule(
    "LCL001",
    "No output token cap",
    """
Without an output cap the model may write until it reaches the model's maximum output
length, and you pay for every output token. A prompt bug, a jailbreak or a
model that loops ("repeat forever") turns one request into the most expensive request the
model can produce. An explicit cap also makes cost per request predictable and lets you
detect truncation instead of silently paying for it.
""",
    {
        "openai-chat-py": """
            Set a cap sized to the longest answer you need, and detect truncation:

              resp = <client>.chat.completions.create(
                  model=<model>, messages=messages,
                  max_completion_tokens=800,   # o-series / gpt-5: includes reasoning tokens
              )                                # (older chat models also accept max_tokens=800)
              if resp.choices[0].finish_reason == "length": ...   # hit the cap
        """,
        "openai-chat-js": """
            Set a cap sized to the longest answer you need, and detect truncation:

              const resp = await <client>.chat.completions.create({
                model: <model>, messages,
                max_completion_tokens: 800,   // o-series / gpt-5: includes reasoning tokens
              });                             // (older chat models also accept max_tokens: 800)
              if (resp.choices[0].finish_reason === "length") { /* hit the cap */ }
        """,
        "openai-responses-py": """
            Set max_output_tokens (it includes reasoning tokens) and detect truncation:

              resp = <client>.responses.create(model=<model>, input=..., max_output_tokens=800)
              if resp.status == "incomplete" and resp.incomplete_details.reason == "max_output_tokens": ...
        """,
        "openai-responses-js": """
            Set max_output_tokens (it includes reasoning tokens) and detect truncation:

              const resp = await <client>.responses.create({ model: <model>, input, max_output_tokens: 800 });
              if (resp.status === "incomplete") { /* resp.incomplete_details?.reason === "max_output_tokens" */ }
        """,
        "google-py": """
            Pass max_output_tokens in the request config:

              from google.genai import types
              resp = <client>.models.generate_content(
                  model=<model>, contents=...,
                  config=types.GenerateContentConfig(max_output_tokens=800),
              )
              # resp.candidates[0].finish_reason == types.FinishReason.MAX_TOKENS when truncated
        """,
        "google-js": """
            Pass maxOutputTokens in the request config:

              const resp = await <client>.models.generateContent({
                model: <model>, contents,
                config: { maxOutputTokens: 800 },
              });
              // resp.candidates?.[0]?.finishReason === "MAX_TOKENS" when truncated
        """,
        "vercel-js": """
            Pass an output cap to <fn>():

              const result = await <fn>({
                model: <model>, prompt,
                maxOutputTokens: 800,   // AI SDK 5+; AI SDK 4 calls it maxTokens
              });
              // result.finishReason === "length" when truncated
        """,
        "langchain-py": """
            Set the cap on the chat model constructor:

              llm = <ctor>(model=<model>, <capkw>=800)
        """,
        "langchain-js": """
            Set the cap on the chat model constructor:

              const llm = new <ctor>({ model: <model>, <capkw>: 800 });
        """,
        "litellm-py": """
            Pass max_tokens (LiteLLM maps it to each provider's parameter):

              resp = litellm.completion(model=<model>, messages=messages, max_tokens=800)
              if resp.choices[0].finish_reason == "length": ...
        """,
    },
)

# ---------------------------------------------------------------- LCL002
_rule(
    "LCL002",
    "Very large max_tokens on a non-streaming Anthropic call",
    """
Anthropic requires max_tokens, so the risk is not a missing cap but a cap far above what the
answer needs on a request that waits for the whole response. A non-streaming request that
is allowed to generate tens of thousands of tokens can run past HTTP timeouts (the SDKs'
default request timeout is 10 minutes, and the SDKs may refuse or warn on non-streaming
requests whose max_tokens implies a longer run), and the cap no longer bounds cost per
request in any useful way.
""",
    {
        "anthropic-py": """
            Either lower max_tokens to what the task needs (max_tokens=<suggest>), or stream
            long generations and collect the final message:

              with <client>.messages.stream(model=<model>, max_tokens=<max_tokens>, messages=messages) as stream:
                  message = stream.get_final_message()
        """,
        "anthropic-js": """
            Either lower max_tokens to what the task needs (max_tokens: <suggest>), or stream
            long generations and collect the final message:

              const stream = <client>.messages.stream({ model: <model>, max_tokens: <max_tokens>, messages });
              const message = await stream.finalMessage();
        """,
    },
)

# ---------------------------------------------------------------- LCL003
_rule(
    "LCL003",
    "No request timeout configured",
    """
The OpenAI and Anthropic SDKs default to a 10-minute request timeout (with 2 automatic
retries, so one call can block for much longer). Other clients may wait with no client-side
timeout at all. In a web request or a worker, a hung or very slow model call ties up a
worker, a connection and the user for minutes, and the retry of a slow call can bill a
second full generation. Set a timeout that matches how long the caller can actually wait.
""",
    {
        "openai-py": """
            Set a client-wide timeout in seconds (and override per call where needed):

              client = <ctor>(timeout=30.0, max_retries=2)
              # or: httpx.Timeout(30.0, connect=5.0)
              # one slow endpoint: client.with_options(timeout=120.0).chat.completions.create(...)
        """,
        "openai-js": """
            Set a client-wide timeout in milliseconds (and override per request where needed):

              const client = new <ctor>({ timeout: 30_000, maxRetries: 2 });
              // per request: await client.chat.completions.create({...}, { timeout: 120_000 });
        """,
        "anthropic-py": """
            Set a client-wide timeout in seconds (and override per call where needed):

              client = <ctor>(timeout=30.0, max_retries=2)
              # one slow call: client.with_options(timeout=120.0).messages.create(...)
        """,
        "anthropic-js": """
            Set a client-wide timeout in milliseconds (and override per request where needed):

              const client = new <ctor>({ timeout: 30_000, maxRetries: 2 });
              // per request: await client.messages.create({...}, { timeout: 120_000 });
        """,
        "google-py": """
            google-genai takes the timeout in milliseconds via HttpOptions:

              from google.genai import types
              client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
        """,
        "google-js": """
            @google/genai takes the timeout in milliseconds via httpOptions:

              const ai = new GoogleGenAI({ apiKey, httpOptions: { timeout: 30_000 } });
        """,
        "vercel-js": """
            Abort the request after a deadline:

              const result = await <fn>({
                model: <model>, prompt,
                abortSignal: AbortSignal.timeout(30_000),
              });
        """,
        "langchain-py": """
            Set timeout (seconds) and max_retries on the chat model:

              llm = <ctor>(model=<model>, timeout=30, max_retries=2)
        """,
        "langchain-js": """
            Set timeout (milliseconds) and maxRetries on the chat model:

              const llm = new <ctor>({ model: <model>, timeout: 30_000, maxRetries: 2 });
        """,
        "litellm-py": """
            Pass a timeout (seconds) per call, or set litellm.request_timeout once:

              resp = litellm.completion(model=<model>, messages=messages, timeout=30, num_retries=2)
        """,
    },
)

# ---------------------------------------------------------------- LCL004
_rule(
    "LCL004",
    "LLM call per item in a loop",
    """
One request per item multiplies cost and rate-limit pressure by the size of the list, which
usually grows over time. Firing them all at once (asyncio.gather over a comprehension,
Promise.all(items.map(...)), forEach(async ...)) sends hundreds of concurrent requests,
hits the provider's rate limits, and the automatic retries then re-send work that was
already billed. Offline jobs are usually cheaper through the providers' batch APIs, and
many small classification/extraction items fit in one prompt.
""",
    {
        "py-parallel": """
            Bound the concurrency (and keep the SDK's retries on):

              sem = asyncio.Semaphore(8)

              async def one(item):
                  async with sem:
                      return await <call>(...)

              results = await asyncio.gather(*(one(x) for x in items))

            For offline work, use the provider's batch API instead (OpenAI: client.batches.create
            with a JSONL file; Anthropic: client.messages.batches.create(requests=[...])), which is
            billed at a discount and has its own rate limits.
        """,
        "py-sequential": """
            Options, cheapest first:
              1. Put several items in one prompt and ask for a JSON list back.
              2. For offline work, use the provider's batch API (OpenAI: client.batches.create;
                 Anthropic: client.messages.batches.create(requests=[...])), billed at a discount.
              3. If you must call per item, cache results by item hash so reruns do not re-bill.
        """,
        "js-parallel": """
            Bound the concurrency, e.g. with p-limit:

              import pLimit from "p-limit";
              const limit = pLimit(8);
              const results = await Promise.all(items.map((x) => limit(() => <call>(...))));

            For offline work, use the provider's batch API instead (OpenAI: client.batches.create;
            Anthropic: client.messages.batches.create({ requests })), billed at a discount.
        """,
        "js-sequential": """
            Options, cheapest first:
              1. Put several items in one prompt and ask for a JSON array back.
              2. For offline work, use the provider's batch API (OpenAI: client.batches.create;
                 Anthropic: client.messages.batches.create({ requests })), billed at a discount.
              3. If you must call per item, cache results by item hash so reruns do not re-bill.
        """,
    },
)

# ---------------------------------------------------------------- LCL005
_rule(
    "LCL005",
    "Loop of LLM calls with retries disabled",
    """
A loop over many items will meet rate-limit (429) and transient 5xx errors. With retries
turned off (max_retries=0 / maxRetries: 0, or LiteLLM without num_retries) and no retry or
backoff of your own, one 429 aborts the job, and rerunning it re-bills every item that
already succeeded.
""",
    {
        "openai-py": "Keep the SDK's retry with backoff on: <ctor>(max_retries=3), or wrap the call with tenacity\n(@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))).",
        "anthropic-py": "Keep the SDK's retry with backoff on: <ctor>(max_retries=3), or wrap the call with tenacity\n(@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))).",
        "openai-js": "Keep the SDK's retry with backoff on: new <ctor>({ maxRetries: 3 }), or wrap the call with p-retry.",
        "anthropic-js": "Keep the SDK's retry with backoff on: new <ctor>({ maxRetries: 3 }), or wrap the call with p-retry.",
        "vercel-js": "Remove maxRetries: 0 (the AI SDK retries with backoff by default) or set maxRetries: 3.",
        "litellm-py": "Pass num_retries (LiteLLM retries with backoff): litellm.completion(..., num_retries=3),\nor set litellm.num_retries = 3 once at startup.",
        "langchain-py": "Set max_retries on the chat model: <ctor>(..., max_retries=3).",
        "langchain-js": "Set maxRetries on the chat model: new <ctor>({ ..., maxRetries: 3 }).",
    },
)

# ---------------------------------------------------------------- LCL006
_rule(
    "LCL006",
    "Large static prompt sent without prompt caching (Anthropic)",
    """
Anthropic only caches a prompt prefix when the request asks for it (cache_control on a
block, or top-level cache_control for automatic caching). A large system prompt that is
identical on every call is otherwise billed as fresh input every time. Cached reads are
billed at a small fraction of the normal input price and are faster. Prefixes shorter than
the model's minimum cacheable length (between 512 and 4,096 tokens depending on the model,
per Anthropic's docs) are not cached even when marked.
""",
    {
        "anthropic-py": """
            Mark the static prefix as cacheable:

              resp = <client>.messages.create(
                  model=<model>, max_tokens=...,
                  system=[{"type": "text", "text": <var>, "cache_control": {"type": "ephemeral"}}],
                  messages=messages,
              )
              # or let the API place the breakpoint: messages.create(..., cache_control={"type": "ephemeral"})

            Verify with resp.usage.cache_read_input_tokens > 0 on the second call.
        """,
        "anthropic-js": """
            Mark the static prefix as cacheable:

              const resp = await <client>.messages.create({
                model: <model>, max_tokens: ...,
                system: [{ type: "text", text: <var>, cache_control: { type: "ephemeral" } }],
                messages,
              });
              // or let the API place the breakpoint: cache_control: { type: "ephemeral" } at the top level

            Verify with resp.usage.cache_read_input_tokens > 0 on the second call.
        """,
    },
)

# ---------------------------------------------------------------- LCL007
_rule(
    "LCL007",
    "Dynamic value near the start of a large prompt",
    """
Prompt caching is a prefix match on every provider: OpenAI caches long prompt prefixes
automatically, Gemini 2.5+ models cache implicitly, and Anthropic caches what you mark.
A value that changes per request (a date, the user's name, an id) interpolated near the
start of a long static prompt makes every request a different prefix, so the large
static part after it is never served from cache.
""",
    {
        "default": """
            Keep the static text first and byte-identical, and move per-request values to the end
            of the prompt or into the user message:

              system = STATIC_INSTRUCTIONS                       # no interpolation
              messages = [..., {"role": "user", "content": f"Today is {today}. User: {name}\\n\\n{question}"}]
        """,
    },
)

# ---------------------------------------------------------------- LCL008
_rule(
    "LCL008",
    "Top-tier model used for a simple or repeated task",
    """
Classification, labelling, yes/no answers and short extractions rarely need the largest
model in a family, and the price difference between tiers of the same provider is large.
When the call also runs once per item in a loop, the difference is multiplied by the
number of items. This is a heuristic: confirm with an eval on real examples before switching.
""",
    {
        "default": """
            Try a smaller model of the same provider on a sample of real inputs and compare:

              model=<smaller>

            Keep the top-tier model only for the inputs where the smaller one is measurably worse.
            (Model tiers come from llm-cost-lint's data file, dated <as_of>.)
        """,
    },
)

# ---------------------------------------------------------------- LCL009
_rule(
    "LCL009",
    "Embeddings recomputed without a cache",
    """
Embedding the same documents again on every request (or one item at a time in a loop) pays
for the same vectors repeatedly and adds a network round trip per item. Document
embeddings only change when the text changes, so they belong in a cache or vector store
keyed by a hash of the text; embedding endpoints also accept a list of inputs per request.
""",
    {
        "py": """
            Embed once, store, and reuse; batch what is new:

              key = hashlib.sha256(text.encode()).hexdigest()
              vec = cache.get(key)
              if vec is None:
                  vec = <call>(...)            # send a list of texts in one request when you can
                  cache.set(key, vec)

            For document collections, embed at ingest time into a vector store and only embed the
            user's query per request.
        """,
        "js": """
            Embed once, store, and reuse; batch what is new:

              const key = createHash("sha256").update(text).digest("hex");
              let vec = await cache.get(key);
              if (!vec) { vec = await <call>(...); await cache.set(key, vec); }   // embedMany / input: [...] for lists

            For document collections, embed at ingest time into a vector store and only embed the
            user's query per request.
        """,
    },
)

# ---------------------------------------------------------------- LCL010
_rule(
    "LCL010",
    "Conversation history grows without a limit",
    """
Chat APIs are stateless: the whole message list is sent and billed as input on every turn.
A history list that is appended to on every turn and never trimmed makes each turn more
expensive than the last (total cost grows roughly with the square of the number of turns),
and long sessions eventually fail when they exceed the model's context window.
""",
    {
        "py": """
            Bound what you send. Keep the system prompt plus the last N turns, and summarise or drop
            older ones:

              MAX_TURNS = 20
              <var>.append({"role": "user", "content": text})
              window = <var>[-MAX_TURNS * 2:]
              resp = <call>(..., messages=window)

            Count tokens (or characters) rather than turns if messages vary a lot in size. Anthropic
            also offers server-side compaction (beta) for long conversations.
        """,
        "js": """
            Bound what you send. Keep the system prompt plus the last N turns, and summarise or drop
            older ones:

              const MAX_TURNS = 20;
              <var>.push({ role: "user", content: text });
              const window = <var>.slice(-MAX_TURNS * 2);
              const resp = await <call>({ ..., messages: window });

            Count tokens (or characters) rather than turns if messages vary a lot in size.
        """,
    },
)

# ---------------------------------------------------------------- LCL011
_rule(
    "LCL011",
    "Retired or deprecated model name",
    """
Providers shut models down on announced dates. After that, every call with the old model
id fails (typically a 404 / model-not-found error), which is a production outage that no
test with a mocked client will catch. Deprecated models still work until their shutdown
date, so the fix can be planned instead of rushed.
""",
    {
        "default": """
            Replace '<model>' with a current model and re-run your evals; suggested: <replacement>.
            Read the model id from configuration (an env var with a current default) so the next
            migration is a config change. Check the provider's deprecations page: this list is
            dated <as_of> and llm-cost-lint never fetches it.
        """,
    },
)

# ---------------------------------------------------------------- LCL012
_rule(
    "LCL012",
    "Sampling, n or streaming settings worth a second look",
    """
temperature and top_p both change sampling; OpenAI and Anthropic recommend adjusting one,
not both. n > 1 (or candidate_count) bills every extra completion. Newer Claude models
reject sampling parameters outright (HTTP 400). And long answers in a user-facing request
handler without streaming make the user wait for the whole generation before anything
appears, which also invites retries that bill twice.
""",
    {
        "default": """
            - Set temperature or top_p, not both.
            - Use n / candidate_count > 1 only when you really compare candidates.
            - For Claude models listed as rejecting sampling parameters: remove temperature / top_p /
              top_k and steer with the prompt (and output_config.effort on models that support it).
            - Stream long user-facing answers (stream=True / client.messages.stream(...) /
              streamText(...)) and forward the chunks to the client.
        """,
    },
)

# ---------------------------------------------------------------- LCL013
_rule(
    "LCL013",
    "LLM call in a request handler without error handling",
    """
Rate limits (429), overloads (5xx/529) and timeouts are normal for LLM APIs. Unhandled in a
route handler, they surface as a generic 500 with a stack trace, the client cannot tell
"try again in a minute" from a bug, and naive client-side retries multiply the load (and
the bill). Catch the provider's errors and return a clear 429/503.
""",
    {
        "openai-py": """
            try:
                resp = <call>(...)
            except openai.RateLimitError:
                raise HTTPException(429, "busy, retry shortly")          # or your framework's equivalent
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                raise HTTPException(503, "model unavailable")
        """,
        "anthropic-py": """
            try:
                resp = <call>(...)
            except anthropic.RateLimitError:
                raise HTTPException(429, "busy, retry shortly")          # or your framework's equivalent
            except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError):
                raise HTTPException(503, "model unavailable")
        """,
        "google-py": """
            from google.genai import errors
            try:
                resp = <call>(...)
            except errors.APIError as e:
                raise HTTPException(429 if e.code == 429 else 503, "model unavailable")
        """,
        "litellm-py": """
            try:
                resp = <call>(...)
            except litellm.RateLimitError:
                raise HTTPException(429, "busy, retry shortly")
            except (litellm.Timeout, litellm.APIConnectionError, litellm.ServiceUnavailableError):
                raise HTTPException(503, "model unavailable")
        """,
        "langchain-py": """
            LangChain re-raises the provider SDK's errors:

            try:
                resp = <call>(...)
            except openai.RateLimitError:          # anthropic.RateLimitError for ChatAnthropic
                raise HTTPException(429, "busy, retry shortly")
        """,
        "openai-js": """
            try {
              const resp = await <call>(...);
            } catch (err) {
              if (err instanceof OpenAI.RateLimitError) return Response.json({ error: "busy" }, { status: 429 });
              if (err instanceof OpenAI.APIError) return Response.json({ error: "model unavailable" }, { status: 503 });
              throw err;
            }
        """,
        "anthropic-js": """
            try {
              const resp = await <call>(...);
            } catch (err) {
              if (err instanceof Anthropic.RateLimitError) return Response.json({ error: "busy" }, { status: 429 });
              if (err instanceof Anthropic.APIError) return Response.json({ error: "model unavailable" }, { status: 503 });
              throw err;
            }
        """,
        "vercel-js": """
            import { APICallError } from "ai";
            try {
              const result = await <call>(...);
            } catch (err) {
              if (APICallError.isInstance(err) && err.statusCode === 429)
                return Response.json({ error: "busy" }, { status: 429 });
              throw err;
            }
            (streamText reports errors through onError / the stream instead of throwing.)
        """,
        "google-js": """
            try {
              const resp = await <call>(...);
            } catch (err) {
              if (err instanceof ApiError && err.status === 429) return Response.json({ error: "busy" }, { status: 429 });
              throw err;
            }
        """,
        "langchain-js": """
            LangChain re-raises the provider SDK's errors; catch them and return 429/503:

            try { const resp = await <call>(...); }
            catch (err) { if (err?.status === 429) return Response.json({ error: "busy" }, { status: 429 }); throw err; }
        """,
    },
)


@dataclass
class Finding:
    rule_id: str
    severity: str
    message: str
    file: str | None = None
    line: int | None = None
    evidence: str | None = None
    heuristic: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def rule(self) -> Rule:
        return RULES[self.rule_id]

    @property
    def location(self) -> str:
        if self.file is None:
            return ""
        return f"{self.file}:{self.line}" if self.line else self.file

    @property
    def fix(self) -> str:
        """The fix for this finding's SDK and language, with its names filled in."""
        fixes = self.rule.fixes
        text = None
        for key in self.extra.get("fixkeys", []):
            if key in fixes:
                text = fixes[key]
                break
        if text is None:
            text = fixes.get("default") or self.rule.fix
        e = self.extra
        subs = {
            "<client>": e.get("client") or "client",
            "<model>": e.get("model_expr") or '"..."',
            "<ctor>": e.get("ctor") or "OpenAI",
            "<capkw>": e.get("capkw") or "max_tokens",
            "<fn>": e.get("fn") or "generateText",
            "<call>": e.get("call") or "client.chat.completions.create",
            "<var>": e.get("var") or "SYSTEM_PROMPT",
            "<smaller>": e.get("smaller") or "a smaller model",
            "<replacement>": e.get("replacement") or "a current model",
            "<as_of>": e.get("as_of") or "",
            "<max_tokens>": str(e.get("max_tokens") or ""),
            "<suggest>": str(e.get("suggest") or "4096"),
        }
        for k, v in subs.items():
            text = text.replace(k, v)
        return text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["title"] = self.rule.title
        d["why"] = self.rule.why
        d["fix"] = self.fix
        extra = {k: v for k, v in d.pop("extra").items() if k not in ("fixkeys",)}
        if extra:
            d["extra"] = extra
        return d


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_RANK[f.severity], f.rule_id, f.file or "", f.line or 0))


def clip_line(line: str, limit: int = 160) -> str:
    s = line.strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."
