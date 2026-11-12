"""Language-neutral description of the LLM calls found in one file.

py_scan.py (ast) and js_scan.py (lexer) both produce a FileFacts; checks.py turns it into
findings, so every rule behaves the same for Python and JavaScript/TypeScript."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Text:
    """A prompt string as far as it can be known statically."""
    static_len: int                 # characters of literal text
    has_interp: bool = False        # f-string / template ${} / .format / concatenation with a variable
    after_first_interp: int = 0     # literal characters that come after the first dynamic value
    text: str = ""                  # the literal text (for keyword heuristics)
    var: str | None = None          # constant name, when passed by name


@dataclass
class Val:
    kind: str                       # num | str | bool | none | name | obj | list | other
    value: object = None
    src: str = ""                   # source text of the expression
    text: Text | None = None        # for strings
    obj: dict | None = None         # nested keyword/object values (config objects)
    spread: bool = False            # obj contains **x / ...x
    items: int = 0                  # list literal length
    comp: bool = False              # list comprehension / .map(...)


@dataclass
class Call:
    lang: str                       # py | js
    sdk: str                        # openai-chat | openai-responses | anthropic | google | vercel | langchain | litellm
    kind: str                       # gen | embed | client | chatmodel
    line: int
    call_expr: str                  # e.g. client.chat.completions.create
    args: dict | None               # keyword / object-literal args; None when unknown (spread / variable)
    client: str | None = None
    ctor: str | None = None         # constructor name for client / chatmodel
    stream: bool = False
    model: str | None = None        # literal model id when known
    model_expr: str | None = None   # source of the model argument
    loop: str | None = None         # None | parallel | sequential
    loop_line: int | None = None
    via: str | None = None          # same-file helper that is called in the loop
    handler: bool = False           # inside a web request handler
    in_try: bool = False
    prompt: Text | None = None      # system prompt / instructions
    prompt_label: str = "system prompt"
    cached: bool = False            # cache_control present (Anthropic)
    literals: str = ""              # all literal text in the call (heuristics)
    embed_many: bool = False        # embedding a list / collection
    per_call_timeout: bool = False
    provider: str | None = None     # openai | anthropic | google (for vercel / langchain / litellm models)
    extra_retry: bool = False       # retry decorator / wrapper around this call


@dataclass
class History:
    line: int
    var: str
    call_expr: str
    sdk: str
    lang: str
    why: str                        # where the list lives (module-level, self.x, session store, REPL loop)


@dataclass
class FileFacts:
    rel: str
    lang: str
    lines: list[str]
    calls: list[Call] = field(default_factory=list)
    histories: list[History] = field(default_factory=list)
    literals: list[tuple[int, str]] = field(default_factory=list)   # (line, string literal)
    limiter: bool = False           # semaphore / p-limit / Bottleneck / executor with max_workers
    own_retry: bool = False         # tenacity / backoff / p-retry / retry loop with sleep
    cache_words: bool = False       # identifiers mentioning cache / memo / redis / lru_cache
    timeout_elsewhere: bool = False # with_options(timeout=...), per-request timeout, litellm.request_timeout
    retries_disabled: bool = False  # max_retries=0 / maxRetries: 0 on a client
    litellm_retries: bool = False   # litellm.num_retries set globally
    sdk_detected: bool = False
