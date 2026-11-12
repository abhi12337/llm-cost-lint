"""The rules. Language-neutral: they read the Calls / FileFacts produced by py_scan / js_scan."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import models
from .calls import Call, FileFacts
from .findings import Finding, clip_line

LARGE_PROMPT_DEFAULT = 4000
ANTHROPIC_BIG_MAX_TOKENS = 16384
SIMPLE_TASK_RE = re.compile(
    r"(?i)\b(classify|classification|categori[sz]e|categori[sz]ation|label (?:this|the|each)|sentiment|"
    r"yes or no|true or false|answer (?:only )?(?:with )?(?:yes|no)|one word|single word|respond (?:with )?only|"
    r"reply (?:with )?only|answer (?:with )?only|return only|output only|extract (?:the|all)|is this (?:spam|toxic))\b")
SHORT_CAP = 64
LONG_ANSWER = 4000

CAP_KEYS = {
    "openai-chat": ("max_tokens", "max_completion_tokens"),
    "openai-responses": ("max_output_tokens",),
    "litellm": ("max_tokens", "max_completion_tokens", "max_output_tokens"),
    "vercel": ("maxTokens", "maxOutputTokens"),
}
LC_CAP = {
    ("py", "ChatOpenAI"): ("max_tokens", "max_completion_tokens"),
    ("py", "AzureChatOpenAI"): ("max_tokens", "max_completion_tokens"),
    ("py", "ChatGoogleGenerativeAI"): ("max_output_tokens", "max_tokens"),
    ("py", "ChatVertexAI"): ("max_output_tokens", "max_tokens"),
    ("py", "ChatLiteLLM"): ("max_tokens",),
    ("js", "ChatOpenAI"): ("maxTokens", "max_tokens", "maxCompletionTokens"),
    ("js", "AzureChatOpenAI"): ("maxTokens", "max_tokens", "maxCompletionTokens"),
    ("js", "ChatGoogleGenerativeAI"): ("maxOutputTokens", "maxTokens"),
    ("js", "ChatVertexAI"): ("maxOutputTokens", "maxTokens"),
}
TIMEOUT_KEYS = ("timeout", "request_timeout", "default_request_timeout", "http_client", "httpOptions", "fetch",
                "abortSignal", "signal", "clientOptions", "configuration")


@dataclass
class Options:
    large_prompt_chars: int = LARGE_PROMPT_DEFAULT


def family(c: Call) -> str:
    """Fix-text family key for a call."""
    s = c.sdk
    if s.startswith("openai"):
        return "openai"
    if s.startswith("google"):
        return "google"
    if s.startswith("vercel"):
        return "vercel"
    if s.startswith("litellm"):
        return "litellm"
    if s.startswith("langchain"):
        return "langchain"
    return s


def fixkeys(c: Call, *extra: str) -> list[str]:
    keys = [f"{e}-{c.lang}" for e in extra]
    keys += [f"{c.sdk}-{c.lang}", f"{family(c)}-{c.lang}", c.lang]
    return keys


def _num(v) -> float | None:
    if v is not None and v.kind == "num":
        return v.value
    return None


def _cap(c: Call):
    """(known, value) for the output cap of a gen call."""
    if c.args is None:
        return False, None
    if c.sdk == "anthropic":
        return True, _num(c.args.get("max_tokens"))
    if c.sdk == "google":
        cfg = c.args.get("config")
        if cfg is None:
            return True, None
        if cfg.kind != "obj" or cfg.spread:
            return False, None
        v = cfg.obj.get("max_output_tokens") or cfg.obj.get("maxOutputTokens")
        return True, _num(v) if v is not None else None
    for k in CAP_KEYS.get(c.sdk, ()):
        if k in c.args:
            return True, _num(c.args[k]) if _num(c.args[k]) is not None else -1
    return True, None


def _has_cap(c: Call) -> bool | None:
    """True / False / None (unknown)."""
    if c.args is None:
        return None
    if c.sdk == "google":
        cfg = c.args.get("config")
        if cfg is None:
            return False
        if cfg.kind != "obj" or cfg.spread:
            return None
        return "max_output_tokens" in cfg.obj or "maxOutputTokens" in cfg.obj
    keys = CAP_KEYS.get(c.sdk)
    if keys is None:
        return None
    return any(k in c.args for k in keys)


class Checker:
    def __init__(self, ff: FileFacts, opts: Options):
        self.ff = ff
        self.opts = opts
        self.findings: list[Finding] = []
        self._seen: set[tuple[str, int]] = set()

    def add(self, rule: str, sev: str, msg: str, c: Call | None, line: int, heuristic: bool = False,
            keys: list[str] | None = None, **extra) -> None:
        if (rule, line) in self._seen:
            return
        raw = self.ff.lines[line - 1] if 0 < line <= len(self.ff.lines) else ""
        prev = self.ff.lines[line - 2] if 1 < line <= len(self.ff.lines) + 1 else ""
        for text in (raw, prev if prev.strip().startswith(("#", "//")) else ""):
            m = re.search(r"llm-cost-lint:\s*ignore\b([\w ,]*)", text)
            if m:
                ids = re.findall(r"LCL\d{3}", m.group(1).upper())
                if not ids or rule in ids:
                    return
        self._seen.add((rule, line))
        if c is not None:
            extra.setdefault("client", c.client)
            extra.setdefault("model_expr", c.model_expr)
            extra.setdefault("call", c.call_expr)
            extra.setdefault("ctor", c.ctor)
            extra.setdefault("sdk", c.sdk)
            if c.model:
                extra.setdefault("model", c.model)
        extra["fixkeys"] = keys or (fixkeys(c) if c else [])
        extra["lang"] = self.ff.lang
        self.findings.append(Finding(rule, sev, msg, self.ff.rel, line, clip_line(raw), heuristic,
                                     extra={k: v for k, v in extra.items() if v is not None}))

    # ------------------------------------------------------------ run
    def run(self) -> list[Finding]:
        ff = self.ff
        for c in ff.calls:
            if c.kind == "gen":
                self.cap(c)
                self.timeout_call(c)
                self.loop(c)
                self.retries(c)
                self.caching(c)
                self.tier(c)
                self.sampling(c)
                self.errors(c)
            elif c.kind == "embed":
                self.embeddings(c)
                self.errors(c)
            elif c.kind == "client":
                self.timeout_client(c)
            elif c.kind == "chatmodel":
                self.cap_chatmodel(c)
                self.timeout_client(c)
                self.sampling(c)
        self.history()
        self.retired()
        return self.findings

    # ------------------------------------------------------------ LCL001 / LCL002
    def cap(self, c: Call) -> None:
        if c.sdk == "anthropic":
            known, v = _cap(c)
            if known and v is not None and v > ANTHROPIC_BIG_MAX_TOKENS and not c.stream:
                self.add("LCL002", "low",
                         f"{c.call_expr}() asks for max_tokens={int(v):,} without streaming: the request can run past "
                         f"HTTP timeouts and the cap no longer bounds the cost of one call.",
                         c, c.line, max_tokens=int(v), suggest=4096)
            return
        if c.sdk == "vercel" and c.provider == "anthropic":
            return     # the Anthropic provider sends a default max_tokens
        if c.sdk == "litellm" and c.provider == "anthropic":
            return     # LiteLLM fills Anthropic's required max_tokens
        has = _has_cap(c)
        if has is False:
            keys = {"openai-chat": "max_completion_tokens / max_tokens", "openai-responses": "max_output_tokens",
                    "google": "config.max_output_tokens" if c.lang == "py" else "config.maxOutputTokens",
                    "vercel": "maxOutputTokens (maxTokens in AI SDK 4)",
                    "litellm": "max_tokens"}.get(c.sdk, "an output cap")
            what = f"{c.call_expr}()" if c.sdk != "vercel" else f"{c.call_expr}()"
            self.add("LCL001", "medium",
                     f"{what} sets no {keys}: the answer can run to the model's maximum output length and every "
                     f"token is billed.", c, c.line, fn=c.call_expr)

    def cap_chatmodel(self, c: Call) -> None:
        keys = LC_CAP.get((c.lang, c.ctor or ""))
        if keys is None or c.args is None:
            return
        if c.ctor == "ChatLiteLLM" and c.provider == "anthropic":
            return
        if not any(k in c.args for k in keys):
            self.add("LCL001", "medium",
                     f"{c.ctor} is created without {keys[0]}: every call through it can run to the model's maximum "
                     f"output length.", c, c.line, capkw=keys[0])

    # ------------------------------------------------------------ LCL003
    def timeout_client(self, c: Call) -> None:
        if c.args is None or self.ff.timeout_elsewhere:
            return
        if c.kind == "client":
            if any(k in c.args for k in TIMEOUT_KEYS):
                if c.sdk == "google":
                    ho = c.args.get("http_options") or c.args.get("httpOptions")
                    if ho is not None and ho.kind == "obj" and not ho.spread and "timeout" not in ho.obj:
                        pass
                    else:
                        return
                else:
                    return
            if c.sdk == "google" and ("http_options" in c.args or "httpOptions" in c.args):
                ho = c.args.get("http_options") or c.args.get("httpOptions")
                if ho.kind != "obj" or ho.spread or "timeout" in ho.obj:
                    return
            if c.sdk in ("openai", "anthropic"):
                opt = "timeout=" if c.lang == "py" else "a timeout option"
                msg = (f"{c.ctor}() is created without {opt}: the SDK default is 10 minutes per attempt, with 2 "
                       f"retries by default, so one stuck call can hold a worker far longer.")
            else:
                opt = "http_options.timeout" if c.lang == "py" else "httpOptions.timeout"
                msg = (f"{c.ctor}() is created without a request timeout ({opt}): a slow or stuck "
                       f"call can hold the caller for as long as the HTTP layer allows.")
            self.add("LCL003", "low", msg, c, c.line, keys=[f"{c.sdk}-{c.lang}"])
        elif c.kind == "chatmodel":
            if any(k in c.args for k in TIMEOUT_KEYS):
                return
            self.add("LCL003", "low",
                     f"{c.ctor} is created without a timeout: it falls back to the provider SDK's default "
                     f"(10 minutes for the OpenAI and Anthropic SDKs).", c, c.line)

    def timeout_call(self, c: Call) -> None:
        if c.args is None:
            return
        if c.sdk == "vercel" and c.call_expr in ("generateText", "generateObject"):
            if not any(k in c.args for k in ("abortSignal", "timeout")):
                self.add("LCL003", "low",
                         f"{c.call_expr}() has no abortSignal: the AI SDK sets no request timeout of its own, so a "
                         f"slow provider holds the request until the platform kills it.", c, c.line, fn=c.call_expr)
        elif c.sdk == "litellm" and not self.ff.timeout_elsewhere:
            if not any(k in c.args for k in ("timeout", "request_timeout")):
                self.add("LCL003", "low",
                         f"{c.call_expr}() has no timeout=: LiteLLM's built-in default request timeout is long "
                         f"(check litellm.request_timeout for your version).", c, c.line)

    # ------------------------------------------------------------ LCL004
    def loop(self, c: Call) -> None:
        if c.loop is None or self.ff.limiter:
            return
        via = f" (through {c.via}(), called per item at line {c.loop_line})" if c.via else \
            f" (loop at line {c.loop_line})"
        if c.loop == "parallel":
            self.add("LCL004", "medium",
                     f"{c.call_expr}() is fired for every item at once{via} with no concurrency limit: a large list "
                     f"means a burst of parallel requests, rate-limit errors and retries of billed work.",
                     c, c.line, keys=[f"{c.lang}-parallel"])
        else:
            self.add("LCL004", "low",
                     f"{c.call_expr}() runs once per item{via}: cost grows with the list; batch items into one "
                     f"prompt or use the provider's batch API for offline work.",
                     c, c.line, keys=[f"{c.lang}-sequential"])

    # ------------------------------------------------------------ LCL005
    def retries(self, c: Call) -> None:
        if c.loop is None or self.ff.own_retry or c.extra_retry:
            return
        disabled = None
        a = c.args or {}
        for k in ("max_retries", "maxRetries", "num_retries"):
            if k in a and _num(a[k]) == 0:
                disabled = f"{k}=0"
        if disabled is None and c.client:
            for k in self.ff.calls:
                if k.kind in ("client", "chatmodel") and k.client == c.client and k.args:
                    for key in ("max_retries", "maxRetries"):
                        if key in k.args and _num(k.args[key]) == 0:
                            disabled = f"{c.client} is created with {key}=0" if c.lang == "py" else f"{c.client} is created with {key}: 0"
        if disabled is None and c.sdk == "litellm" and c.args is not None and "num_retries" not in a \
                and not self.ff.litellm_retries:
            disabled = "no num_retries (LiteLLM does not retry by default)"
        if disabled:
            self.add("LCL005", "low",
                     f"{c.call_expr}() runs in a loop with retries disabled ({disabled}) and no retry/backoff of "
                     f"its own: one 429 stops the job.", c, c.line)

    # ------------------------------------------------------------ LCL006 / LCL007
    def caching(self, c: Call) -> None:
        p = c.prompt
        n = self.opts.large_prompt_chars
        if p is None or p.static_len < n:
            return
        name = f" ({p.var})" if p.var else ""
        early_dynamic = p.has_interp and p.after_first_interp >= n // 2
        if early_dynamic:
            self.add("LCL007", "info",
                     f"The {c.prompt_label}{name} is {p.static_len:,} characters with a per-request value inserted "
                     f"before {p.after_first_interp:,} of them: that part can never be served from the prompt cache.",
                     c, c.line, keys=["default"], var=p.var)
            return
        if c.sdk == "anthropic" and not c.cached and not p.has_interp:
            self.add("LCL006", "low",
                     f"The {c.prompt_label}{name} is {p.static_len:,} characters, identical on every call, and the "
                     f"request has no cache_control: it is billed as fresh input every time.",
                     c, c.line, var=p.var or "SYSTEM_PROMPT")

    # ------------------------------------------------------------ LCL008
    def tier(self, c: Call) -> None:
        if not c.model:
            return
        t = models.top_tier(c.model)
        if t is None:
            return
        if models.retired(c.model):
            return
        reason = None
        sev = "info"
        if c.loop:
            reason = "it runs once per item in a loop"
            sev = "low"
        else:
            m = SIMPLE_TASK_RE.search(c.literals or "")
            known, cap = _cap(c)
            if m:
                reason = f"the prompt looks like a simple task (\"{m.group(0)}\")"
            elif known and cap is not None and 0 < cap <= SHORT_CAP:
                reason = f"the output is capped at {int(cap)} tokens (a short label or answer)"
        if reason:
            self.add("LCL008", sev,
                     f"'{c.model}' is a top-tier model and {reason}: a smaller model ({t['smaller']}) is usually "
                     f"enough and costs much less per token.",
                     c, c.line, heuristic=True, keys=["default"], smaller=t["smaller"], as_of=models.as_of())

    # ------------------------------------------------------------ LCL009
    def embeddings(self, c: Call) -> None:
        if self.ff.cache_words:
            return
        if c.loop:
            msg = (f"{c.call_expr}() is called once per item in a loop (line {c.loop_line}) with no cache in this "
                   f"file: the same texts are re-embedded on every run, one request each.")
        elif c.handler and c.embed_many:
            msg = (f"{c.call_expr}() embeds a collection inside a request handler with no cache in this file: "
                   f"document embeddings are recomputed and billed on every request.")
        else:
            return
        self.add("LCL009", "low", msg, c, c.line, heuristic=True, keys=[c.lang])

    # ------------------------------------------------------------ LCL010
    def history(self) -> None:
        for h in self.ff.histories:
            self.add("LCL010", "medium",
                     f"'{h.var}' is kept in {h.why}, grows every turn and is sent whole to {h.call_expr}() with "
                     f"no trimming or summarising: every turn re-bills the entire conversation.",
                     None, h.line, heuristic=True, keys=[h.lang], var=h.var, call=h.call_expr, sdk=h.sdk)

    # ------------------------------------------------------------ LCL011
    def retired(self) -> None:
        for line, lit in self.ff.literals:
            r = models.retired(lit)
            if r is None:
                continue
            when = f"since {r['date']}" if r["date"] and r["status"] == "retired" else \
                f"with shutdown on {r['date']}" if r["date"] else f"before {models.as_of()}"
            what = "retired" if r["status"] == "retired" else "deprecated"
            prov = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google (Gemini API)"}[r["provider"]]
            self.add("LCL011", "medium",
                     f"'{lit}' is listed as {what} by {prov} {when} (llm-cost-lint list as of {models.as_of()}): "
                     f"calls with it {'fail' if what == 'retired' else 'will fail after that date'}.",
                     None, line, keys=["default"], model=lit, model_expr=lit,
                     replacement=r["replacement"], as_of=models.as_of(), provider=r["provider"])

    # ------------------------------------------------------------ LCL012
    def sampling(self, c: Call) -> None:
        a = c.args
        if a is None:
            return
        cfg = a.get("config")
        pool = dict(a)
        if cfg is not None and cfg.kind == "obj":
            pool.update(cfg.obj)
        issues = []
        sev = "info"
        has_t = "temperature" in pool
        has_p = "top_p" in pool or "topP" in pool
        if has_t and has_p:
            issues.append("temperature and top_p are both set (tune one, not both)")
        n = _num(pool.get("n")) or _num(pool.get("candidate_count")) or _num(pool.get("candidateCount"))
        if n is not None and n > 1:
            issues.append(f"{int(n)} completions are requested and billed per call")
        anth = c.sdk == "anthropic" or c.provider == "anthropic"
        samp = [k for k in ("temperature", "top_p", "top_k", "topP", "topK") if k in pool]
        if anth and samp and c.model and models.rejects_sampling(c.model):
            issues.insert(0, f"{', '.join(samp)} is rejected with HTTP 400 by '{c.model}' (per Anthropic's docs)")
            sev = "low"
        if c.kind == "gen" and c.handler and not c.stream and c.sdk not in ("langchain",):
            known, cap = _cap(c)
            if known and cap is not None and cap >= LONG_ANSWER:
                issues.append(f"up to {int(cap):,} output tokens are generated without streaming in a request "
                              f"handler (the user sees nothing until the end)")
        if issues:
            self.add("LCL012", sev, f"{c.call_expr}(): " + "; ".join(issues) + ".", c, c.line, keys=["default"])

    # ------------------------------------------------------------ LCL013
    def errors(self, c: Call) -> None:
        if not c.handler or c.in_try or c.extra_retry:
            return
        if c.sdk == "vercel" and c.call_expr in ("streamText", "streamObject"):
            return     # errors arrive through onError / the stream, not as a throw
        self.add("LCL013", "low",
                 f"{c.call_expr}() runs in a request handler outside any try/{'except' if c.lang == 'py' else 'catch'}"
                 f": a rate limit or timeout becomes an unhandled 500.", c, c.line)


def check(ff: FileFacts, opts: Options | None = None) -> list[Finding]:
    return Checker(ff, opts or Options()).run()
