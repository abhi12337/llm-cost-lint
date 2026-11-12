"""JavaScript / TypeScript extractor, built on the lexer in jslex.py.

Recognised: OpenAI Node (chat.completions / responses / embeddings), Anthropic TypeScript
(messages.create / stream / parse), Google GenAI JS (@google/genai: models.generateContent,
embedContent), the Vercel AI SDK (generateText / streamText / generateObject / streamObject,
embed / embedMany), LangChain JS chat models (new ChatOpenAI({...}) ...) and their .invoke().
Only object-literal arguments are inspected; a call whose options come from a variable is
treated as unknown and the rules that need its options stay silent."""

from __future__ import annotations

import bisect
import re

from .calls import Call, FileFacts, History, Text, Unparseable, Val
from .jslex import StrTok, lex, match_close, split_args, strip_span

IMPORT_RE = re.compile(r"""(?:from\s+|require\(\s*|import\(\s*)["']([^"']+)["']""")
SDK_PKG_RE = re.compile(r"^(openai|@anthropic-ai/[\w.-]+|@google/genai|ai|@ai-sdk/[\w.-]+|@langchain/[\w.-]+|"
                        r"langchain(?:/.*)?)$")
IDENT = r"[A-Za-z_$][\w$]*"
GEN_PATTERNS = [
    (re.compile(r"\.\s*chat\s*\.\s*completions\s*\.\s*(create|parse|stream)\s*\("), "openai-chat", "gen"),
    (re.compile(r"\.\s*responses\s*\.\s*(create|parse|stream)\s*\("), "openai-responses", "gen"),
    (re.compile(r"\.\s*messages\s*\.\s*(create|stream|parse)\s*\("), "anthropic", "gen"),
    (re.compile(r"\.\s*models\s*\.\s*(generateContent|generateContentStream)\s*\("), "google", "gen"),
    (re.compile(r"(?<![\w$.])(generateText|streamText|generateObject|streamObject)\s*\("), "vercel", "gen"),
    (re.compile(r"\.\s*embeddings\s*\.\s*(create)\s*\("), "openai-embed", "embed"),
    (re.compile(r"\.\s*models\s*\.\s*(embedContent)\s*\("), "google-embed", "embed"),
    (re.compile(r"(?<![\w$.])(embedMany|embed)\s*\("), "vercel-embed", "embed"),
    (re.compile(r"\.\s*(embedDocuments)\s*\("), "langchain-embed", "embed"),
    (re.compile(r"\bnew\s+(OpenAI|AzureOpenAI)\s*\("), "openai", "client"),
    (re.compile(r"\bnew\s+(Anthropic|AnthropicBedrock|AnthropicVertex|AnthropicFoundry|AnthropicBedrockMantle)\s*\("),
     "anthropic", "client"),
    (re.compile(r"\bnew\s+(GoogleGenAI)\s*\("), "google", "client"),
    (re.compile(r"\bnew\s+(ChatOpenAI|AzureChatOpenAI|ChatAnthropic|ChatGoogleGenerativeAI|ChatVertexAI)\s*\("),
     "langchain", "chatmodel"),
    (re.compile(r"(?<![\w$.])(initChatModel)\s*\("), "langchain", "chatmodel"),
]
LC_PROVIDER = {"ChatOpenAI": "openai", "AzureChatOpenAI": "openai", "ChatAnthropic": "anthropic",
               "ChatGoogleGenerativeAI": "google", "ChatVertexAI": "google"}
LIMITER_RE = re.compile(r"p-limit|pLimit|p-queue|PQueue|[Bb]ottleneck|p-map|pMap|mapLimit|eachLimit|asyncPool|"
                        r"\bconcurrency\s*:|Semaphore|RateLimiter|rateLimit")
RETRY_RE = re.compile(r"p-retry|pRetry|async-retry|\bretry\(|backoff|exponential-backoff|withRetry|retryWithBackoff")
TRIM_WORDS = re.compile(r"(?i)trim|truncat|summari[sz]|compact|maxHistory|max_history|maxTurns|max_turns|maxMessages|"
                        r"historyLimit|tokenLimit|countTokens|sliding")
CACHE_WORDS = re.compile(r"(?i)cache|memo|redis|\blru|upstash|\bkv\.|vectorStore|vector_store|upsert|unstable_cache")
PLURAL_NAMES = re.compile(r"(?i)^(docs|documents|chunks|texts|items|passages|corpus|products|faqs|articles|pages|"
                          r"records|rows|sentences|paragraphs|all\w+|\w+List|\w+s)$")


def js_imports(clean: str) -> set[str]:
    return {m.group(1) for m in IMPORT_RE.finditer(clean)}


class JsExtractor:
    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text
        self.lines = text.splitlines()
        self.clean, self.code, self.toks = lex(text)
        self.tok_at = {t.start: t for t in self.toks}
        self.tok_starts = [t.start for t in self.toks]
        self.line_starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
        self.imports = js_imports(self.clean)
        self.facts = FileFacts(rel, "js", self.lines)
        self.consts: dict[str, tuple[int, int]] = {}
        self.const_count: dict[str, int] = {}
        self.lc_vars: dict[str, Call] = {}
        self._depth_cache: list[int] | None = None

    # ------------------------------------------------------------ positions
    def line_of(self, off: int) -> int:
        return bisect.bisect_right(self.line_starts, off)

    def depth_at(self, off: int) -> int:
        if self._depth_cache is None:
            d, arr = 0, []
            for ch in self.code:
                arr.append(d)
                if ch in "{([":
                    d += 1
                elif ch in "})]":
                    d -= 1
            self._depth_cache = arr
        return self._depth_cache[off] if off < len(self._depth_cache) else 0

    def tok_in(self, a: int, b: int) -> list[StrTok]:
        i = bisect.bisect_left(self.tok_starts, a)
        out = []
        while i < len(self.toks) and self.toks[i].start < b:
            out.append(self.toks[i])
            i += 1
        return out

    def receiver_before(self, idx: int) -> str:
        j = idx
        while j > 0 and (self.code[j - 1].isalnum() or self.code[j - 1] in "_$.? \t" or
                         (self.code[j - 1] == "\n" and self.code[j - 2:j - 1] in (".", " "))):
            j -= 1
        chunk = re.sub(r"\s*\.\s*", ".", self.code[j:idx].strip())
        chunk = re.sub(r"\?\.", ".", chunk)
        return chunk.split()[-1].lstrip(".") if chunk.split() else ""

    # ------------------------------------------------------------ values
    def text_of_span(self, a: int, b: int) -> Text | None:
        """Text for an expression span: a string, a template, "a" + "b", [..].join(), or a const name."""
        a, b = strip_span(self.code, a, b)
        if a >= b:
            return None
        expr = self.code[a:b]
        tok = self.tok_at.get(a)
        if tok is not None and tok.end >= b:
            return Text(len(tok.text), tok.has_interp, tok.after_first_interp, tok.text)
        m = re.fullmatch(r"\s*(" + IDENT + r")\s*(?:\.\s*trim\s*\(\s*\))?\s*", expr)
        if m and m.group(1) in self.consts and self.const_count.get(m.group(1)) == 1:
            ca, cb = self.consts[m.group(1)]
            if (ca, cb) != (a, b):
                t = self.text_of_span(ca, cb)
                if t is not None:
                    t.var = m.group(1)
                return t
            return None
        if expr.startswith("[") and re.search(r"\]\s*\.\s*join\s*\(", expr):
            close = match_close(self.code, a)
            return self._concat(self.tok_in(a, close), a, close, sep_ok=True)
        if "+" in expr and tok is not None:
            return self._concat(self.tok_in(a, b), a, b)
        return None

    def _concat(self, toks: list[StrTok], a: int, b: int, sep_ok: bool = False) -> Text | None:
        if not toks:
            return None
        t = Text(0)
        pos = a
        for tk in toks:
            between = self.code[pos:tk.start]
            if re.search(r"[A-Za-z_$]", between.replace("join", "")):
                if not t.has_interp:
                    t.has_interp = True
            if tk.has_interp and not t.has_interp:
                t.has_interp = True
                t.after_first_interp += tk.after_first_interp
            elif t.has_interp:
                t.after_first_interp += len(tk.text)
            t.static_len += len(tk.text)
            t.text += tk.text + ("\n" if sep_ok else "")
            pos = tk.end
        tail = self.code[pos:b]
        if re.search(r"[A-Za-z_$]", re.sub(r"\.\s*join\s*\(.*", "", tail, flags=re.S)):
            t.has_interp = True
        return t

    def val(self, a: int, b: int, depth: int = 0) -> Val:
        a, b = strip_span(self.code, a, b)
        src = self.clean[a:b]
        expr = self.code[a:b]
        if not expr:
            return Val("other", None, src)
        if re.fullmatch(r"-?[\d_]+(?:\.\d+)?", expr):
            return Val("num", float(expr.replace("_", "")) if "." in expr else int(expr.replace("_", "")), src)
        if expr in ("true", "false"):
            return Val("bool", expr == "true", src)
        if expr in ("null", "undefined"):
            return Val("none", None, src)
        tok = self.tok_at.get(a)
        if tok is not None and tok.end >= b:
            return Val("str", None if tok.has_interp else tok.text, src, text=self.text_of_span(a, b))
        if expr.startswith("{") and match_close(self.code, a) == b - 1:
            return self.obj(a + 1, b - 1, depth + 1)
        if expr.startswith("["):
            close = match_close(self.code, a)
            if close == b - 1:
                return Val("list", None, src, items=len(split_args(self.code, a + 1, b - 1)))
            t = self.text_of_span(a, b)
            if t is not None:
                return Val("str", None, src, text=t)
        if re.fullmatch(IDENT, expr):
            if expr in self.consts and self.const_count.get(expr) == 1 and depth < 3:
                ca, cb = self.consts[expr]
                v = self.val(ca, cb, depth + 1)
                if v.kind in ("str", "num", "bool", "obj"):
                    v.src = src
                    if v.text is not None:
                        v.text.var = expr
                    return v
            return Val("name", expr, src)
        if re.search(r"\.\s*(?:map|flatMap)\s*\(", expr):
            return Val("list", None, src, comp=True)
        m = re.fullmatch(r"(?:process\.env\.[\w$]+|[\w$.]+)\s*(?:\?\?|\|\|)\s*(.+)", expr, re.S)
        if m:
            off = a + m.start(1)
            return self.val(off, b, depth + 1)
        t = self.text_of_span(a, b)
        if t is not None:
            return Val("str", None, src, text=t)
        return Val("other", None, src)

    def obj(self, a: int, b: int, depth: int = 0) -> Val:
        out: dict[str, Val] = {}
        spread = False
        for pa, pb in split_args(self.code, a, b):
            pa, pb = strip_span(self.code, pa, pb)
            part = self.code[pa:pb]
            if part.startswith("..."):
                spread = True
                continue
            m = re.match(r"(?:([\"'])\s*\1|(" + IDENT + r"))\s*:", part)
            if m:
                if m.group(2):
                    key = m.group(2)
                else:
                    tk = self.tok_at.get(pa)
                    key = tk.text if tk else ""
                out[key] = self.val(pa + m.end(), pb, depth)
                continue
            m = re.fullmatch(IDENT, part)
            if m:
                out[part] = self.val(pa, pb, depth) if part in self.consts else Val("name", part, part)
                continue
            m = re.match(r"(?:async\s+)?(" + IDENT + r")\s*\(", part)
            if m:
                out[m.group(1)] = Val("other", None, part)
        return Val("obj", None, self.clean[a - 1:b + 1], obj=out, spread=spread)

    # ------------------------------------------------------------ spans
    def body_after(self, idx: int) -> tuple[int, int] | None:
        """From idx, find the next `{` (skipping one paren group) and return its span."""
        m = re.compile(r"\s*(?:async\s*)?(?:\([^)]*\)\s*)?(?::\s*[^={]+)?(?:=>\s*)?\{").match(self.code, idx)
        if not m:
            return None
        o = m.end() - 1
        return o, match_close(self.code, o)

    def collect_spans(self) -> None:
        code = self.code
        self.loops: list[tuple[int, int, str, int]] = []      # (a, b, kind, line)
        for m in re.finditer(r"\bfor\s*(?:await\s*)?\(", code):
            o = m.end() - 1
            c = match_close(code, o)
            header = code[o + 1:c]
            if re.search(r";\s*\w+\s*<=?\s*\d+\s*;", header) or re.search(r"(?i)\b(attempt|retr\w*|tries)\b", header):
                continue
            k = c + 1
            while k < len(code) and code[k].isspace():
                k += 1
            end = match_close(code, k) if k < len(code) and code[k] == "{" else code.find(";", k)
            self.loops.append((o, end if end > 0 else len(code), "sequential", self.line_of(m.start())))
        for m in re.finditer(r"\.\s*(forEach|map|flatMap)\s*\(", code):
            o = m.end() - 1
            c = match_close(code, o)
            inner = code[o + 1:c]
            if m.group(1) == "forEach" and not re.match(r"\s*async\b", inner):
                kind = "forEach"
            else:
                kind = "parallel"
            self.loops.append((o, c, kind, self.line_of(m.start())))
        self.whiles = [(m.start(), self._block_end(m.end() - 1)) for m in re.finditer(r"\bwhile\s*\(", code)]
        self.tries = []
        for m in re.finditer(r"\btry\s*\{", code):
            o = m.end() - 1
            c = match_close(code, o)
            if re.match(r"\s*catch\b", code[c + 1:c + 40]):
                self.tries.append((o, c))
        self.funcs: dict[str, tuple[int, int]] = {}
        for m in re.finditer(r"\b(?:async\s+)?function\s*\*?\s*(" + IDENT + r")\s*\(", code):
            span = self.body_after(m.end() - 1)
            if span:
                self.funcs.setdefault(m.group(1), span)
        for m in re.finditer(r"\b(?:const|let|var)\s+(" + IDENT + r")\s*(?::[^=]+)?=\s*(?:async\s+)?"
                             r"(?:\([^)]*\)|" + IDENT + r")\s*(?::\s*[^=]+)?=>\s*\{", code):
            o = m.end() - 1
            self.funcs.setdefault(m.group(1), (o, match_close(code, o)))
        for m in re.finditer(r"^\s*(?:(?:public|private|protected|static|async)\s+)*(" + IDENT + r")\s*\([^)]*\)\s*"
                             r"(?::\s*[^{;]+)?\{", code, re.M):
            if m.group(1) not in ("if", "for", "while", "switch", "catch", "function", "return"):
                o = m.end() - 1
                self.funcs.setdefault(m.group(1), (o, match_close(code, o)))
        self.handlers: list[tuple[int, int]] = []
        for m in re.finditer(r"\bexport\s+(?:async\s+)?function\s+(?:GET|POST|PUT|PATCH|DELETE)\b", code):
            span = self.body_after(m.end())
            if span:
                self.handlers.append(span)
        for m in re.finditer(r"\bexport\s+const\s+(?:GET|POST|PUT|PATCH|DELETE)\b[^=]*=", code):
            span = self.body_after(m.end())
            if span:
                self.handlers.append(span)
        for m in re.finditer(r"\bexport\s+default\s+(?:async\s+)?function\s*(?:" + IDENT + r")?\s*(?=\(\s*req\b)",
                             code):
            span = self.body_after(m.end())
            if span:
                self.handlers.append(span)
        for m in re.finditer(r"\b(?:app|router|server|api|fastify|hono|route|routes|r)\s*\.\s*"
                             r"(?:get|post|put|patch|delete|all)\s*\(\s*['\"`]", code):
            o = code.index("(", m.start())
            self.handlers.append((o, match_close(code, o)))
        if re.match(r"\s*['\"]use server['\"]", self.clean):
            self.handlers.append((0, len(code)))

    def _block_end(self, paren_open: int) -> int:
        c = match_close(self.code, paren_open)
        k = c + 1
        while k < len(self.code) and self.code[k].isspace():
            k += 1
        return match_close(self.code, k) if k < len(self.code) and self.code[k] == "{" else c

    def enclosing_func(self, off: int) -> tuple[str, int, int] | None:
        best = None
        for name, (a, b) in self.funcs.items():
            if a <= off <= b and (best is None or a > best[1]):
                best = (name, a, b)
        return best

    def loop_at(self, off: int, awaited: bool) -> tuple[str | None, int | None]:
        func = self.enclosing_func(off)
        fa = func[1] if func else -1
        inner = None
        for a, b, kind, line in self.loops:
            if a < off <= b and (inner is None or a > inner[0]):
                inner = (a, b, kind, line)
        if inner is None:
            return None, None
        a, b, kind, line = inner
        if func is not None and a < fa and not (a < fa <= b):
            return None, None
        # an inner while (agent / retry loop) between the for and the call is not a per-item loop
        for wa, wb in self.whiles:
            if a < wa < off <= wb:
                return None, None
        if kind == "forEach":
            return ("sequential" if awaited else "parallel"), line
        if kind == "sequential":
            seg = self.code[a:off]
            if not awaited and re.search(r"\.\s*push\s*\(\s*$", seg):
                return "parallel", line
            return "sequential", line
        return "parallel", line

    def helper_loop(self, off: int) -> tuple[str | None, int | None, str | None]:
        func = self.enclosing_func(off)
        if func is None:
            return None, None, None
        name, fa, fb = func
        for m in re.finditer(r"(?<![\w$])" + re.escape(name) + r"\b", self.code):
            if fa - 200 <= m.start() <= fb and self._is_def(m.start()):
                continue
            nxt = self.code[m.end():m.end() + 2].lstrip()
            if nxt.startswith("("):
                kind, line = self.loop_at(m.start(), self._awaited(m.start()))
                if kind:
                    return kind, line, name
            prev = self.code[max(0, m.start() - 12):m.start()]
            if re.search(r"\.\s*(?:map|forEach)\s*\(\s*$", prev):
                return "parallel", self.line_of(m.start()), name
        return None, None, None

    def _is_def(self, idx: int) -> bool:
        before = self.code[max(0, idx - 30):idx]
        return bool(re.search(r"(function\s*\*?\s*|(?:const|let|var)\s+)$", before)) or \
            bool(re.match(r"\s*\([^)]*\)\s*(?::[^{]+)?\{", self.code[idx + 1:idx + 200]) and
                 re.search(r"(^|\n)\s*(?:async\s+)?$", before))

    def _awaited(self, idx: int) -> bool:
        before = self.code[max(0, idx - 80):idx]
        return bool(re.search(r"\bawait\s+(?:[\w$.]*\s*)?$", before)) or bool(re.search(r"\bawait\s*$", before))

    # ------------------------------------------------------------ main
    def run(self) -> FileFacts:
        ff = self.facts
        pkgs = {p for p in self.imports if SDK_PKG_RE.match(p)}
        ff.sdk_detected = bool(pkgs)
        self._consts()
        self._literals()
        if not ff.sdk_detected:
            return ff
        ff.limiter = bool(LIMITER_RE.search(self.clean))
        ff.own_retry = bool(RETRY_RE.search(self.clean))
        ff.cache_words = bool(CACHE_WORDS.search(self.code))
        ff.retries_disabled = bool(re.search(r"\bmaxRetries\s*:\s*0\b", self.code))
        self.collect_spans()
        self.has_ai = "ai" in self.imports
        self.has_anthropic = any(p.startswith("@anthropic-ai/") for p in self.imports)
        self.has_openai = "openai" in self.imports
        self.has_google = "@google/genai" in self.imports
        self.has_lc = any(p.startswith("@langchain/") or p.startswith("langchain") for p in self.imports)
        found: list[tuple[int, Call]] = []
        for pat, sdk, kind in GEN_PATTERNS:
            for m in pat.finditer(self.code):
                c = self.build(m, sdk, kind)
                if c is not None:
                    found.append((m.start(), c))
        # LangChain .invoke() on chat model variables
        for var, ctor in list(self.lc_vars.items()):
            for m in re.finditer(r"(?<![\w$.])" + re.escape(var) + r"\s*\.\s*(invoke|stream)\s*\(", self.code):
                c = self.build_lc_invoke(m, ctor)
                found.append((m.start(), c))
        found.sort(key=lambda x: x[0])
        ff.calls = [c for _, c in found]
        if any(c.per_call_timeout for c in ff.calls if c.kind == "gen" and c.sdk != "vercel"):
            ff.timeout_elsewhere = True
        self._histories(found)
        return ff

    def _consts(self) -> None:
        for m in re.finditer(r"\b(const|let|var)\s+(" + IDENT + r")\s*(?::\s*[\w$<>\[\]|. ]+)?\s*=(?![=>])\s*",
                             self.code):
            name = m.group(2)
            a = m.end()
            depth = 0
            j = a
            while j < len(self.code):
                ch = self.code[j]
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    if depth == 0:
                        break
                    depth -= 1
                elif (ch == ";" or ch == "\n") and depth == 0:
                    rest = self.code[j:j + 40].lstrip()
                    if ch == "\n" and rest[:1] in (".", "+", "?", ":", "|", "&"):
                        j += 1
                        continue
                    if ch == "\n" and self.code[a:j].rstrip().endswith(("+", "=", "(", ",")):
                        j += 1
                        continue
                    break
                elif ch == "," and depth == 0:
                    break
                j += 1
            self.consts[name] = (a, j)
            self.const_count[name] = self.const_count.get(name, 0) + 1 + (1 if m.group(1) != "const" and
                                                                           re.search(r"(?<![\w$.])" + re.escape(name) +
                                                                                     r"\s*=(?!=)", self.code[j:]) else 0)

    def _literals(self) -> None:
        sdk = self.facts.sdk_detected
        for t in self.toks:
            if t.has_interp or not (3 <= len(t.text) <= 60):
                continue
            if sdk or re.search(r"(?i)model\w*[\"']?\s*[:=]\s*(?:[\w$.]+\s*(?:\?\?|\|\|)\s*)?$",
                                self.code[max(0, t.start - 60):t.start]):
                self.facts.literals.append((self.line_of(t.start), t.text))

    def _ok(self, sdk: str, kind: str, args: dict | None, recv: str) -> bool:
        keys = set(args or {})
        if sdk == "openai-chat":
            return self.has_openai or "model" in keys or args is None
        if sdk == "openai-responses":
            return self.has_openai and ("model" in keys or args is None)
        if sdk == "anthropic":
            if recv.endswith("threads") or recv.endswith("beta.threads"):
                return False
            return self.has_anthropic or {"model", "messages", "max_tokens"} <= keys
        if sdk in ("google", "google-embed", "google-client"):
            return self.has_google
        if sdk in ("vercel", "vercel-embed"):
            return self.has_ai
        if sdk == "openai-embed":
            return self.has_openai or "input" in keys
        if sdk == "langchain-embed":
            return self.has_lc
        if kind == "client":
            return {"openai": self.has_openai, "anthropic": self.has_anthropic, "google": self.has_google}[sdk]
        if kind == "chatmodel":
            return self.has_lc
        return True

    def call_args(self, open_idx: int) -> tuple[dict | None, list[tuple[int, int]], int]:
        close = match_close(self.code, open_idx)
        parts = split_args(self.code, open_idx + 1, close)
        if not parts:
            return {}, parts, close
        a, b = strip_span(self.code, *parts[0])
        if self.code[a:a + 1] == "{" and match_close(self.code, a) == b - 1:
            v = self.obj(a + 1, b - 1)
            return (None if v.spread else v.obj), parts, close
        return None, parts, close

    def build(self, m: re.Match, sdk: str, kind: str) -> Call | None:
        open_idx = m.end() - 1
        if sdk in ("vercel", "vercel-embed"):
            before = self.code[max(0, m.start() - 20):m.start()]
            if re.search(r"(function\s*|\bdef\s*|\.\s*)$", before):
                return None
        recv = self.receiver_before(m.start()) if kind in ("gen", "embed") and sdk not in ("vercel", "vercel-embed") \
            else ""
        args, parts, close = self.call_args(open_idx)
        if not self._ok(sdk, kind, args, recv):
            return None
        method = m.group(1)
        if sdk in ("vercel", "vercel-embed"):
            expr = method
        elif kind in ("client", "chatmodel"):
            expr = f"new {method}" if method != "initChatModel" else method
        else:
            expr = re.sub(r"\s+", "", self.code[m.start():open_idx]).lstrip(".")
            expr = f"{recv}.{expr}" if recv else expr
        line = self.line_of(m.start())
        c = Call("js", sdk, kind, line, expr, args)
        c.client = recv or None
        if kind in ("client", "chatmodel"):
            c.ctor = method
            before = self.code[max(0, m.start() - 120):m.start()]
            am = re.search(r"(?:(?:const|let|var)\s+(" + IDENT + r")|(this\.[\w$]+)|(" + IDENT + r"))\s*(?::[^=]+)?=\s*$",
                           before)
            if am:
                c.client = am.group(1) or am.group(2) or am.group(3)
                if kind == "chatmodel":
                    self.lc_vars[c.client] = c
            if kind == "chatmodel":
                c.provider = LC_PROVIDER.get(method)
        if len(parts) > 1:
            c.per_call_timeout = "timeout" in self.code[parts[1][0]:parts[1][1]]
        if args is not None:
            mv = args.get("model") or args.get("modelName")
            if mv is not None:
                c.model_expr = mv.src
                if mv.kind == "str" and isinstance(mv.value, str):
                    c.model = mv.value
                else:
                    pm = re.match(r"\s*(?:(openai|anthropic|google|vertex|azure)(?:\.\w+)?)\s*\(\s*[\"']([^\"']+)[\"']",
                                  mv.src)
                    if pm:
                        c.provider = {"vertex": "google", "azure": "openai"}.get(pm.group(1), pm.group(1))
                        c.model = pm.group(2)
                    else:
                        sm = re.fullmatch(r"\s*[\"']([^\"']+)[\"']\s*", mv.src)
                        if sm:
                            c.model = sm.group(1)
            if method == "initChatModel" and parts:
                tk = self.tok_at.get(strip_span(self.code, *parts[0])[0])
                if tk:
                    c.model = tk.text
            sv = args.get("stream")
            if sv is not None and sv.kind == "bool" and sv.value:
                c.stream = True
        elif method == "initChatModel" and parts:
            tk = self.tok_at.get(strip_span(self.code, *parts[0])[0])
            if tk:
                c.model, c.model_expr = tk.text, f'"{tk.text}"'
        if method in ("stream", "streamText", "streamObject", "generateContentStream"):
            c.stream = True
        if c.model and not c.provider:
            from .py_scan import _provider_of
            c.provider = _provider_of(c.model)
        span_text = self.clean[m.start():close + 1]
        c.cached = "cache_control" in span_text or "cacheControl" in span_text
        if kind in ("gen", "embed"):
            c.literals = "\n".join(t.text for t in self.tok_in(open_idx, close))
            self._context(c, m.start())
            self._prompt(c, args)
        if kind == "embed":
            c.embed_many = self._embed_many(c, args, method)
        return c

    def build_lc_invoke(self, m: re.Match, ctor: Call) -> Call:
        open_idx = m.end() - 1
        close = match_close(self.code, open_idx)
        var = self.code[m.start():m.start() + len(m.group(0))].split(".")[0].strip()
        c = Call("js", "langchain", "gen", self.line_of(m.start()), f"{var}.{m.group(1)}", None)
        c.model, c.model_expr, c.provider, c.ctor = ctor.model, ctor.model_expr, ctor.provider, ctor.ctor
        c.client = var
        c.stream = m.group(1) == "stream"
        c.literals = "\n".join(t.text for t in self.tok_in(open_idx, close))
        self._context(c, m.start())
        parts = split_args(self.code, open_idx + 1, close)
        self._lc_messages = getattr(self, "_lc_messages", {})
        if parts:
            a, b = strip_span(self.code, *parts[0])
            self._lc_messages[id(c)] = (a, b)
        return c

    def _context(self, c: Call, off: int) -> None:
        awaited = self._awaited(off)
        c.loop, c.loop_line = self.loop_at(off, awaited)
        if c.loop is None:
            c.loop, c.loop_line, c.via = self.helper_loop(off)
        c.handler = any(a <= off <= b for a, b in self.handlers)
        c.in_try = any(a <= off <= b for a, b in self.tries)
        func = self.enclosing_func(off)
        if func is not None and not c.in_try:
            # wrapped by a retry helper: retry(() => client...)
            seg = self.code[max(func[1], off - 200):off]
            if re.search(r"\b(?:pRetry|retry|withRetry|backoff)\s*\(\s*(?:async\s*)?\(\s*\)\s*=>\s*(?:await\s*)?$", seg):
                c.extra_retry = True

    def _prompt(self, c: Call, args: dict | None) -> None:
        if args is None:
            return
        if c.sdk == "anthropic" or c.sdk == "vercel":
            s = args.get("system")
            if s is not None and s.kind == "str":
                c.prompt = s.text
        elif c.sdk == "openai-responses":
            s = args.get("instructions")
            if s is not None and s.kind == "str":
                c.prompt, c.prompt_label = s.text, "instructions"
        elif c.sdk == "openai-chat":
            mv = args.get("messages")
            if mv is not None and mv.src.lstrip().startswith("["):
                # first element { role: "system", content: ... }
                start = self.clean.find(mv.src.lstrip()[:20])
                if start >= 0:
                    o = self.code.find("[", start)
                    close = match_close(self.code, o)
                    parts = split_args(self.code, o + 1, close)
                    if parts:
                        a, b = strip_span(self.code, *parts[0])
                        if self.code[a:a + 1] == "{":
                            first = self.obj(a + 1, match_close(self.code, a))
                            role = first.obj.get("role") if first.obj else None
                            content = first.obj.get("content") if first.obj else None
                            if role is not None and role.value in ("system", "developer") and content is not None \
                                    and content.kind == "str":
                                c.prompt, c.prompt_label = content.text, f"{role.value} message"
        elif c.sdk == "google":
            cfg = args.get("config")
            if cfg is not None and cfg.kind == "obj":
                s = cfg.obj.get("systemInstruction")
                if s is not None and s.kind == "str":
                    c.prompt, c.prompt_label = s.text, "systemInstruction"

    def _embed_many(self, c: Call, args: dict | None, method: str) -> bool:
        if method in ("embedMany", "embedDocuments"):
            return True
        if args is None:
            return False
        v = args.get("input") or args.get("contents") or args.get("value")
        if v is None:
            return False
        if v.kind == "list":
            return v.items > 1 or v.comp
        if v.kind == "name":
            return bool(PLURAL_NAMES.match(str(v.value))) and not str(v.value).lower().endswith(("ss", "us", "query"))
        return False

    # ------------------------------------------------------------ history
    def _histories(self, found: list[tuple[int, Call]]) -> None:
        seen = set()
        lc_msgs = getattr(self, "_lc_messages", {})
        for off, c in found:
            if c.kind != "gen":
                continue
            var = None
            if c.args is not None and "messages" in c.args:
                v = c.args["messages"]
                src = v.src.strip()
                mm = re.fullmatch(r"(this\.[\w$]+|" + IDENT + r")", src)
                if mm:
                    var = mm.group(1)
                else:
                    mm = re.match(r"\[\s*(?:[^\]]*?,\s*)?\.\.\.\s*(this\.[\w$]+|" + IDENT + r")\s*[,\]]", src, re.S)
                    if mm:
                        var = mm.group(1)
            elif id(c) in lc_msgs:
                a, b = lc_msgs[id(c)]
                src = self.code[a:b].strip()
                if re.fullmatch(r"this\.[\w$]+|" + IDENT, src):
                    var = src
            if not var or var in seen:
                continue
            where = self._persistent(var, off)
            if where is None:
                continue
            v = re.escape(var)
            if not re.search(v + r"\s*\.\s*(?:push|unshift|splice)\s*\(", self.code):
                continue
            trimmed = [v + r"\s*\.\s*slice\s*\(\s*-", v + r"\s*\.\s*splice\s*\(", v + r"\s*\.\s*shift\s*\(",
                       v + r"\s*=\s*" + v + r"\s*\.\s*slice", v + r"\s*\.\s*length\s*[>=]", v + r"\.length\s*=\s*\d"]
            trim_call = r"(?<![\w$])[\w$]*(?:" + TRIM_WORDS.pattern[4:] + r")[\w$]*\s*\(\s*" + v + r"(?![\w$])"
            if any(re.search(p, self.code) for p in trimmed) or re.search(r"(?i)" + trim_call, self.code):
                continue
            seen.add(var)
            self.facts.histories.append(History(c.line, var, c.call_expr, c.sdk, "js", where))

    def _persistent(self, var: str, call_off: int) -> str | None:
        if var.startswith("this."):
            return f"an instance field ({var}) that lives across calls"
        decls = list(re.finditer(r"\b(?:const|let|var)\s+" + re.escape(var) + r"\b\s*(?::[^=]+)?=\s*", self.code))
        if not decls:
            return None
        d = decls[0]
        rhs = self.code[d.end():d.end() + 80]
        if self.depth_at(d.start()) == 0 and rhs.startswith("["):
            return "a module-level array (it lives as long as the server process)"
        if re.match(r"[\w$.]+\s*(?:\.\s*get\s*\(|\[)", rhs) and self.depth_at(d.start()) > 0:
            return f"a per-session store ({self.clean[d.end():d.end() + 40].split(';')[0].strip()})"
        # REPL: while (true) { const q = await rl.question(...) ... } with the array declared outside
        for wa, wb in self.whiles:
            if wa < call_off <= wb and d.start() < wa and re.search(r"\b(question|prompt|readline|rl\.)", self.code[wa:wb]):
                return "a chat loop (while ... readline)"
        return None


def extract_js(rel: str, text: str) -> FileFacts | None:
    try:
        return JsExtractor(rel, text).run()
    except RecursionError:
        raise Unparseable("too deeply nested to parse") from None
