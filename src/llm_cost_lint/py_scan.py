"""Python extractor: finds LLM SDK calls with the real `ast` module and describes them as Calls.

Recognised: OpenAI (chat.completions / responses / embeddings, sync and async clients),
Anthropic (messages.create / stream / parse), Google GenAI (google-genai: models.generate_content,
embed_content), LiteLLM (completion / acompletion / embedding), LangChain chat model constructors
(ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ...) and their .invoke() calls."""

from __future__ import annotations

import ast
import re

from .calls import Call, FileFacts, History, Text, Unparseable, Val

SDK_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(openai|anthropic|google\.genai|google\s+import\s+genai|litellm|"
                           r"langchain[\w.]*)", re.M)
CLIENT_CTORS = {
    "openai": {"OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"},
    "anthropic": {"Anthropic", "AsyncAnthropic", "AnthropicBedrock", "AsyncAnthropicBedrock", "AnthropicVertex",
                  "AsyncAnthropicVertex", "AnthropicFoundry", "AsyncAnthropicFoundry", "AnthropicBedrockMantle",
                  "AsyncAnthropicBedrockMantle"},
}
LC_CHAT = {"ChatOpenAI": "openai", "AzureChatOpenAI": "openai", "ChatAnthropic": "anthropic",
           "ChatGoogleGenerativeAI": "google", "ChatVertexAI": "google", "ChatLiteLLM": None, "init_chat_model": None}
LC_INVOKE = {"invoke", "ainvoke", "stream", "astream"}
HANDLER_DECO_LAST = {"get", "post", "put", "patch", "delete", "route", "api_route", "websocket", "api_view",
                     "require_POST", "require_GET", "require_http_methods", "csrf_exempt"}
LIMITER_RE = re.compile(r"\b(Semaphore|BoundedSemaphore|AsyncLimiter|aiolimiter|ThreadPoolExecutor|ProcessPoolExecutor|"
                        r"CapacityLimiter|max_concurrency|RateLimiter|ratelimit|limits\.)\b")
RETRY_RE = re.compile(r"\b(tenacity|backoff|retrying|stamina|@retry\b|retry\()")
TRIM_WORDS = re.compile(r"(?i)trim|truncat|summari[sz]|compact|max_history|max_turns|max_messages|history_limit|"
                        r"token_limit|context_management|count_tokens|sliding")
SIMPLE_CONTAINERS = (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
PLURAL_NAMES = re.compile(r"(?i)^(docs|documents|chunks|texts|items|passages|corpus|products|faqs|articles|pages|"
                          r"records|rows|sentences|paragraphs|all_\w+|\w+_list|\w+s)$")


def is_llm_python(text: str) -> bool:
    return bool(SDK_IMPORT_RE.search(text)) or ".chat.completions." in text


def _chain(node: ast.AST) -> list[str] | None:
    """a.b.c(...) -> ['a','b','c'];  OpenAI().chat -> ['OpenAI()', 'chat']."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        inner = _chain(node.func)
        parts.append((inner[-1] if inner else "?") + "()")
    elif isinstance(node, ast.Subscript):
        parts.append(ast.unparse(node))
    else:
        return None
    return list(reversed(parts))


class PyExtractor:
    def __init__(self, rel: str, text: str):
        self.rel = rel
        self.text = text
        self.lines = text.splitlines()
        self.tree = ast.parse(text)
        self.parent: dict[ast.AST, ast.AST] = {}
        for n in ast.walk(self.tree):
            for ch in ast.iter_child_nodes(n):
                self.parent[ch] = n
        self.alias: dict[str, str] = {}
        self.assign_count: dict[str, int] = {}
        self.assign_val: dict[str, ast.AST] = {}
        self.module_names: set[str] = set()
        self.funcs: dict[str, ast.AST] = {}
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    self.alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    self.alias[a.asname or a.name] = f"{n.module}.{a.name}"
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if isinstance(t, ast.Name) and n.value is not None:
                        self.assign_count[t.id] = self.assign_count.get(t.id, 0) + 1
                        self.assign_val[t.id] = n.value
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs.setdefault(n.name, n)
        for n in self.tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        self.module_names.add(t.id)
        self.async_code = bool(re.search(r"AsyncOpenAI|AsyncAnthropic|\.aio\.|acompletion|ainvoke|async\s+def", text))
        self.lc_vars: dict[str, Call] = {}
        self.facts = FileFacts(rel, "py", self.lines)

    # ------------------------------------------------------------ helpers
    def resolve(self, chain: list[str]) -> str:
        head = chain[0]
        if head in self.alias:
            return ".".join([self.alias[head], *chain[1:]])
        return ".".join(chain)

    def seg(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.text, node) or ast.unparse(node)

    def const_node(self, name: str) -> ast.AST | None:
        if self.assign_count.get(name) == 1:
            return self.assign_val[name]
        return None

    def text_of(self, node: ast.AST, depth: int = 0) -> Text | None:
        """Static knowledge about a string expression."""
        if depth > 4:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Text(len(node.value), text=node.value)
        if isinstance(node, ast.JoinedStr):
            t = Text(0)
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    t.static_len += len(v.value)
                    t.text += v.value
                    if t.has_interp:
                        t.after_first_interp += len(v.value)
                else:
                    t.has_interp = True
            return t
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a = self.text_of(node.left, depth + 1)
            b = self.text_of(node.right, depth + 1)
            if a is None and b is None:
                return None
            a = a or Text(0, has_interp=True)
            b = b or Text(0, has_interp=True)
            after = a.after_first_interp + b.static_len if a.has_interp else b.after_first_interp
            return Text(a.static_len + b.static_len, a.has_interp or b.has_interp, after, a.text + b.text)
        if isinstance(node, ast.Name):
            c = self.const_node(node.id)
            if c is not None and not isinstance(c, ast.Name):
                t = self.text_of(c, depth + 1)
                if t is not None:
                    t.var = node.id
                return t
            return None
        if isinstance(node, ast.Call):
            fn = _chain(node.func) or []
            if fn and fn[-1] in ("dedent", "cleandoc", "strip", "lstrip", "rstrip") :
                inner = node.args[0] if node.args and fn[-1] in ("dedent", "cleandoc") else \
                    (node.func.value if isinstance(node.func, ast.Attribute) else None)
                return self.text_of(inner, depth + 1) if inner is not None else None
            if fn and fn[-1] == "format" and isinstance(node.func, ast.Attribute):
                base = self.text_of(node.func.value, depth + 1)
                if base is not None and (node.args or node.keywords):
                    i = base.text.find("{")
                    if i >= 0:
                        base.has_interp = True
                        base.after_first_interp = base.static_len - i
                return base
        return None

    def val(self, node: ast.AST, depth: int = 0) -> Val:
        src = self.seg(node)
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                return Val("bool", v, src)
            if isinstance(v, (int, float)):
                return Val("num", v, src)
            if isinstance(v, str):
                return Val("str", v, src, text=self.text_of(node))
            if v is None:
                return Val("none", None, src)
        if isinstance(node, (ast.JoinedStr, ast.BinOp)):
            t = self.text_of(node)
            if t is not None:
                return Val("str", None, src, text=t)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return Val("num", -node.operand.value, src)
        if isinstance(node, ast.Name):
            c = self.const_node(node.id)
            if c is not None and depth < 3 and not isinstance(c, ast.Name):
                v = self.val(c, depth + 1)
                if v.kind in ("str", "num", "bool", "obj"):
                    v.src = src
                    if v.text is not None:
                        v.text.var = node.id
                    return v
            return Val("name", node.id, src)
        if isinstance(node, ast.Dict):
            obj, spread = {}, False
            for k, v in zip(node.keys, node.values):
                if k is None:
                    spread = True
                elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                    obj[k.value] = self.val(v, depth + 1)
            return Val("obj", None, src, obj=obj, spread=spread)
        if isinstance(node, ast.Call):
            fn = _chain(node.func) or []
            last = fn[-1] if fn else ""
            if last in ("getenv", "get") and len(node.args) == 2 and ".".join(fn[-3:-1]) in ("os", "os.environ", "environ"):
                return self.val(node.args[1], depth + 1)
            if last in ("dedent", "cleandoc", "strip", "format"):
                t = self.text_of(node)
                if t is not None:
                    return Val("str", None, src, text=t)
            if last.endswith(("Config", "Options")) or last == "dict":
                obj = {kw.arg: self.val(kw.value, depth + 1) for kw in node.keywords if kw.arg}
                return Val("obj", None, src, obj=obj, spread=any(kw.arg is None for kw in node.keywords))
        if isinstance(node, ast.List):
            return Val("list", None, src, items=len(node.elts))
        if isinstance(node, SIMPLE_CONTAINERS):
            return Val("list", None, src, comp=True)
        return Val("other", None, src)

    def kwargs(self, node: ast.Call) -> dict | None:
        if any(kw.arg is None for kw in node.keywords):
            return None
        return {kw.arg: self.val(kw.value) for kw in node.keywords}

    def kw_node(self, node: ast.Call, name: str) -> ast.AST | None:
        for kw in node.keywords:
            if kw.arg == name:
                return kw.value
        return None

    def ancestors(self, node: ast.AST):
        n = self.parent.get(node)
        while n is not None:
            yield n
            n = self.parent.get(n)

    def enclosing_func(self, node: ast.AST):
        for a in self.ancestors(node):
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return a
        return None

    def literal_text(self, node: ast.AST) -> str:
        out = []
        for n in ast.walk(node):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
            elif isinstance(n, ast.Name):
                c = self.const_node(n.id)
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out.append(c.value)
                elif c is not None:
                    t = self.text_of(c)
                    if t:
                        out.append(t.text)
        return "\n".join(out)

    # ------------------------------------------------------------ context: loops, handlers, try
    def _is_awaited(self, node: ast.AST) -> bool:
        return isinstance(self.parent.get(node), ast.Await)

    def loop_ctx(self, node: ast.AST) -> tuple[str | None, int | None]:
        """Loop kind around node inside its own function: (parallel|sequential|None, line)."""
        child = node
        for a in self.ancestors(node):
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.Module)):
                return None, None
            if isinstance(a, ast.While):
                return None, None
            if isinstance(a, SIMPLE_CONTAINERS) and child is not None and child not in a.generators:
                gen = a.generators[0]
                if self._range_literal(gen.iter):
                    return None, None
                parallel = self._in_gather(a) or (not self._is_awaited(node) and self.async_code and
                                                  not isinstance(a, ast.GeneratorExp) and self._returns_coroutine(node))
                return ("parallel" if parallel else "sequential"), a.lineno
            if isinstance(a, (ast.For, ast.AsyncFor)) and child is not a.iter and child is not a.target:
                if self._range_literal(a.iter) or self._retry_target(a.target):
                    return None, None
                if self._spawned(node):
                    return "parallel", a.lineno
                return "sequential", a.lineno
            child = a
        return None, None

    def _returns_coroutine(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            fn = _chain(node.func) or []
            if fn and fn[-1] in self.funcs:
                return isinstance(self.funcs[fn[-1]], ast.AsyncFunctionDef)
            return any(p in ("aio", "acompletion", "ainvoke", "aembedding") for p in fn) or \
                bool(re.search(r"AsyncOpenAI|AsyncAnthropic", self.text))
        return False

    def _range_literal(self, it: ast.AST) -> bool:
        return (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range"
                and all(isinstance(x, ast.Constant) for x in it.args))

    @staticmethod
    def _retry_target(t: ast.AST) -> bool:
        return isinstance(t, ast.Name) and bool(re.search(r"(?i)^_?(attempt|retr|tries$|try$|n_?tries)", t.id))

    def _in_gather(self, node: ast.AST) -> bool:
        for a in self.ancestors(node):
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            if isinstance(a, ast.Call):
                fn = _chain(a.func) or []
                if fn and fn[-1] in ("gather", "as_completed", "wait", "map", "imap", "imap_unordered", "starmap",
                                     "create_task", "ensure_future", "submit"):
                    return True
        return False

    def _spawned(self, node: ast.AST) -> bool:
        """call wrapped in create_task / ensure_future / executor.submit, or an unawaited coroutine appended."""
        p = self.parent.get(node)
        if isinstance(p, ast.Call):
            fn = _chain(p.func) or []
            if fn and fn[-1] in ("create_task", "ensure_future", "submit", "start_soon"):
                return True
            if fn and fn[-1] == "append" and not self._is_awaited(node) and self._returns_coroutine(node):
                return True
        return False

    def helper_loop(self, func: ast.AST) -> tuple[str | None, int | None, str | None]:
        """Is the helper `func` invoked per item somewhere in this file?"""
        name = func.name
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                fn = _chain(n.func) or []
                if fn and fn[-1] == name and (len(fn) == 1 or fn[0] in ("self", "cls")):
                    kind, line = self.loop_ctx(n)
                    if kind:
                        return kind, line, name
                fn_last = fn[-1] if fn else ""
                if fn_last in ("map", "imap", "imap_unordered", "starmap", "submit") and n.args and \
                        isinstance(n.args[0], (ast.Name, ast.Attribute)) and \
                        (_chain(n.args[0]) or [""])[-1] == name:
                    return "parallel", n.lineno, name
        return None, None, None

    def is_handler(self, func: ast.AST | None) -> bool:
        while func is not None:
            for d in func.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                ch = _chain(target) or []
                if ch and ch[-1] in HANDLER_DECO_LAST and (len(ch) > 1 or ch[-1] in (
                        "api_view", "require_POST", "require_GET", "require_http_methods", "csrf_exempt")):
                    return True
            args = [a.arg for a in func.args.args]
            if args[:1] == ["request"] or (args[:1] in (["self"], ["cls"]) and args[1:2] == ["request"]):
                return True
            func = self.enclosing_func(func)
        return False

    def in_try(self, node: ast.AST) -> bool:
        child = node
        for a in self.ancestors(node):
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in a.decorator_list:
                    if "retry" in ast.unparse(d).lower():
                        return True
                return False
            if isinstance(a, ast.Try) and a.handlers and child in a.body:
                return True
            child = a
        return False

    # ------------------------------------------------------------ classification
    def classify(self, node: ast.Call) -> tuple[str, str] | None:
        ch = _chain(node.func)
        if not ch:
            return None
        full = self.resolve(ch)
        last = ch[-1]
        kw = {k.arg for k in node.keywords}
        if len(ch) >= 3 and ch[-3:-1] == ["chat", "completions"] and last in ("create", "parse", "stream"):
            return "openai-chat", "gen"
        if len(ch) >= 2 and ch[-2] == "responses" and last in ("create", "parse", "stream") and "model" in kw:
            return "openai-responses", "gen"
        if len(ch) >= 2 and ch[-2] == "messages" and last in ("create", "stream", "parse") and "threads" not in ch:
            if "anthropic" in self.alias.values() or any(v.startswith("anthropic") for v in self.alias.values()) or \
                    {"model", "messages", "max_tokens"} <= kw:
                return "anthropic", "gen"
            return None
        if len(ch) >= 2 and ch[-2] == "models" and last in ("generate_content", "generate_content_stream"):
            return "google", "gen"
        if len(ch) >= 2 and ch[-2] == "embeddings" and last == "create" and "input" in kw:
            return "openai-embed", "embed"
        if len(ch) >= 2 and ch[-2] == "models" and last == "embed_content":
            return "google-embed", "embed"
        if full in ("litellm.completion", "litellm.acompletion", "litellm.text_completion"):
            return "litellm", "gen"
        if full in ("litellm.embedding", "litellm.aembedding"):
            return "litellm-embed", "embed"
        mod = full.rsplit(".", 1)[0] if "." in full else ""
        if last in CLIENT_CTORS["openai"] and mod.startswith("openai"):
            return "openai", "client"
        if last in CLIENT_CTORS["anthropic"] and mod.startswith("anthropic"):
            return "anthropic", "client"
        if last == "Client" and (mod in ("google.genai", "genai") and ("google.genai" in full or
                                                                       self.alias.get("genai", "").endswith("genai"))):
            return "google", "client"
        if last in LC_CHAT and mod.startswith("langchain"):
            return "langchain", "chatmodel"
        if last == "embed_documents":
            return "langchain-embed", "embed"
        if last in LC_INVOKE and len(ch) >= 2:
            base = ".".join(ch[:-1])
            if base in self.lc_vars:
                return "langchain", "gen"
        return None

    # ------------------------------------------------------------ main
    def run(self) -> FileFacts:
        ff = self.facts
        ff.sdk_detected = is_llm_python(self.text)
        ff.limiter = bool(LIMITER_RE.search(self.text))
        ff.own_retry = bool(RETRY_RE.search(self.text))
        ff.cache_words = self._has_cache_words()
        ff.litellm_retries = bool(re.search(r"litellm\.num_retries\s*=", self.text))
        if re.search(r"with_options\([^)]*timeout\s*=", self.text) or re.search(r"litellm\.request_timeout\s*=", self.text):
            ff.timeout_elsewhere = True
        if re.search(r"max_retries\s*=\s*0\b", self.text):
            ff.retries_disabled = True
        self._literals()
        if not ff.sdk_detected:
            return ff
        # pass 1: LangChain chat model variables
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                ch = _chain(n.func) or []
                if ch and ch[-1] in LC_CHAT and self.resolve(ch).startswith("langchain"):
                    p = self.parent.get(n)
                    if isinstance(p, (ast.Assign, ast.AnnAssign)):
                        tg = p.targets[0] if isinstance(p, ast.Assign) else p.target
                        self.lc_vars[ast.unparse(tg)] = None  # filled below
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                c = self.build(n)
                if c is not None:
                    ff.calls.append(c)
        for c in ff.calls:
            if c.kind == "gen" and "timeout" in (c.args or {}):
                ff.timeout_elsewhere = True
        self._histories()
        return ff

    def _has_cache_words(self) -> bool:
        for n in ast.walk(self.tree):
            name = None
            if isinstance(n, ast.Name):
                name = n.id
            elif isinstance(n, ast.Attribute):
                name = n.attr
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = n.name
            elif isinstance(n, ast.alias):
                name = n.name
            if name and re.search(r"(?i)cache|memo|redis|lru|diskcache|vector_?store|upsert", name):
                return True
        return False

    def _literals(self) -> None:
        sdk = self.facts.sdk_detected
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and 3 <= len(n.value) <= 60:
                if sdk or self._model_context(n):
                    self.facts.literals.append((n.lineno, n.value))

    def _model_context(self, n: ast.AST) -> bool:
        p = self.parent.get(n)
        if isinstance(p, ast.keyword) and p.arg and "model" in p.arg.lower():
            return True
        if isinstance(p, (ast.Assign, ast.AnnAssign)):
            tg = p.targets if isinstance(p, ast.Assign) else [p.target]
            return any("model" in ast.unparse(t).lower() for t in tg)
        if isinstance(p, ast.Dict):
            for k, v in zip(p.keys, p.values):
                if v is n and isinstance(k, ast.Constant) and "model" in str(k.value).lower():
                    return True
        if isinstance(p, ast.Call) and n in p.args[1:2]:      # os.getenv("MODEL", "gpt-4-32k")
            return "model" in ast.unparse(p).lower()
        return False

    def build(self, node: ast.Call) -> Call | None:
        cls = self.classify(node)
        if cls is None:
            return None
        sdk, kind = cls
        ch = _chain(node.func) or []
        expr = ".".join(ch)
        args = self.kwargs(node)
        c = Call("py", sdk, kind, node.lineno, expr, args)
        c.client = ".".join(ch[:-3]) if sdk == "openai-chat" else ".".join(ch[:-2]) if len(ch) > 2 else None
        if kind in ("client", "chatmodel"):
            c.ctor = ch[-1]
            p = self.parent.get(node)
            if isinstance(p, (ast.Assign, ast.AnnAssign)):
                tg = p.targets[0] if isinstance(p, ast.Assign) else p.target
                c.client = ast.unparse(tg)
                if kind == "chatmodel":
                    self.lc_vars[c.client] = c
        # model
        mnode = self.kw_node(node, "model") or self.kw_node(node, "model_name")
        if mnode is None and sdk in ("litellm", "litellm-embed") and node.args:
            mnode = node.args[0]
        if mnode is None and kind == "chatmodel" and node.args:
            mnode = node.args[0]
        if mnode is not None:
            c.model_expr = self.seg(mnode)
            mv = self.val(mnode)
            if mv.kind == "str" and isinstance(mv.value, str):
                c.model = mv.value
        if sdk == "langchain" and kind == "gen":
            base = ".".join(ch[:-1])
            ctor = self.lc_vars.get(base)
            if ctor is not None:
                c.model, c.model_expr, c.provider = ctor.model, ctor.model_expr, LC_CHAT.get(ctor.ctor or "")
                c.ctor = ctor.ctor
                c.args = None
            c.stream = ch[-1] in ("stream", "astream")
        if kind == "chatmodel":
            c.provider = LC_CHAT.get(c.ctor or "")
            if c.provider is None and c.model:
                c.provider = _provider_of(c.model)
        if sdk == "litellm" and c.model:
            c.provider = _provider_of(c.model)
        # stream
        sv = self.kw_node(node, "stream")
        if ch[-1] in ("stream", "generate_content_stream") or (isinstance(sv, ast.Constant) and sv.value is True):
            c.stream = True
        # prompt
        self._prompt(node, c)
        c.cached = "cache_control" in self.seg(node) or (args is not None and "cache_control" in args)
        if kind in ("gen", "embed"):
            c.literals = self.literal_text(node)
            c.loop, c.loop_line = self.loop_ctx(node)
            func = self.enclosing_func(node)
            if c.loop is None and func is not None:
                c.loop, c.loop_line, c.via = self.helper_loop(func)
            c.handler = self.is_handler(func)
            c.in_try = self.in_try(node)
            c.per_call_timeout = "timeout" in (args or {}) or "request_timeout" in (args or {})
        if kind == "embed":
            c.embed_many = self._embed_many(node, sdk)
        return c

    def _embed_many(self, node: ast.Call, sdk: str) -> bool:
        if sdk == "langchain-embed":
            return True
        target = self.kw_node(node, "input") or self.kw_node(node, "contents")
        if target is None and node.args:
            target = node.args[-1]
        if target is None:
            return False
        if isinstance(target, ast.List):
            return len(target.elts) > 1
        if isinstance(target, SIMPLE_CONTAINERS):
            return True
        if isinstance(target, ast.Name):
            return bool(PLURAL_NAMES.match(target.id)) and not target.id.lower().endswith(("ss", "us", "query"))
        return False

    def _prompt(self, node: ast.Call, c: Call) -> None:
        if c.sdk == "anthropic":
            s = self.kw_node(node, "system")
            if s is not None:
                c.prompt = self.text_of(s)
                if isinstance(s, (ast.List, ast.Name)) and c.prompt is None:
                    c.cached = c.cached or "cache_control" in self.seg(s)
        elif c.sdk == "openai-responses":
            s = self.kw_node(node, "instructions")
            if s is not None:
                c.prompt, c.prompt_label = self.text_of(s), "instructions"
        elif c.sdk in ("openai-chat", "litellm"):
            m = self.kw_node(node, "messages")
            if m is None and c.sdk == "litellm" and len(node.args) > 1:
                m = node.args[1]
            if isinstance(m, ast.Name):
                cn = self.const_node(m.id)
                m = cn if isinstance(cn, ast.List) else m
            if isinstance(m, ast.List) and m.elts and isinstance(m.elts[0], ast.Dict):
                d = m.elts[0]
                role = content = None
                for k, v in zip(d.keys, d.values):
                    if isinstance(k, ast.Constant) and k.value == "role" and isinstance(v, ast.Constant):
                        role = v.value
                    elif isinstance(k, ast.Constant) and k.value == "content":
                        content = v
                if role in ("system", "developer") and content is not None:
                    c.prompt, c.prompt_label = self.text_of(content), f"{role} message"
        elif c.sdk == "google":
            cfg = self.kw_node(node, "config")
            if isinstance(cfg, (ast.Call, ast.Dict)):
                s = None
                if isinstance(cfg, ast.Call):
                    s = self.kw_node(cfg, "system_instruction")
                else:
                    for k, v in zip(cfg.keys, cfg.values):
                        if isinstance(k, ast.Constant) and k.value == "system_instruction":
                            s = v
                if s is not None:
                    c.prompt, c.prompt_label = self.text_of(s), "system_instruction"

    # ------------------------------------------------------------ history
    def _histories(self) -> None:
        seen = set()
        for c in self.facts.calls:
            if c.kind != "gen":
                continue
            node = self._node_at(c)
            if node is None:
                continue
            m = self.kw_node(node, "messages")
            if m is None and c.sdk == "langchain" and node.args:
                m = node.args[0]
            if m is None and c.sdk == "litellm" and len(node.args) > 1:
                m = node.args[1]
            base = self._history_base(m)
            if base is None:
                continue
            var = ast.unparse(base)
            if var in seen:
                continue
            where = self._persistent(base, node)
            if where is None:
                continue
            if not re.search(re.escape(var) + r"\s*(?:\.(?:append|extend|insert)\(|\+=)", self.text):
                continue
            if self._trimmed(var):
                continue
            seen.add(var)
            self.facts.histories.append(History(c.line, var, c.call_expr, c.sdk, "py", where))

    def _node_at(self, c: Call) -> ast.Call | None:
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call) and n.lineno == c.line and ".".join(_chain(n.func) or []) == c.call_expr:
                return n
        return None

    @staticmethod
    def _history_base(m: ast.AST | None) -> ast.AST | None:
        if m is None:
            return None
        if isinstance(m, (ast.Name, ast.Attribute, ast.Subscript)):
            if isinstance(m, ast.Subscript) and isinstance(m.slice, ast.Slice):
                return None      # history[-20:] -> already windowed
            return m
        if isinstance(m, ast.List):
            for e in m.elts:
                if isinstance(e, ast.Starred) and isinstance(e.value, (ast.Name, ast.Attribute, ast.Subscript)):
                    if isinstance(e.value, ast.Subscript) and isinstance(e.value.slice, ast.Slice):
                        return None
                    return e.value
        if isinstance(m, ast.BinOp) and isinstance(m.op, ast.Add):
            for side in (m.left, m.right):
                if isinstance(side, (ast.Name, ast.Attribute)):
                    return side
        return None

    def _persistent(self, base: ast.AST, call: ast.Call) -> str | None:
        func = self.enclosing_func(call)
        if isinstance(base, ast.Attribute):
            if isinstance(base.value, ast.Name) and base.value.id in ("self", "cls"):
                return f"an instance attribute ({ast.unparse(base)}) that lives across calls"
            return f"an object attribute ({ast.unparse(base)}) that lives across calls"
        if isinstance(base, ast.Subscript):
            return f"a per-session store ({ast.unparse(base)})"
        name = base.id
        local_assign = None
        if func is not None:
            for n in ast.walk(func):
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                    local_assign = n
                    break
            globals_ = {g for n in ast.walk(func) if isinstance(n, ast.Global) for g in n.names}
        else:
            globals_ = set()
        if local_assign is None and (name in self.module_names or name in globals_):
            return "a module-level list (it lives as long as the process)"
        if local_assign is not None:
            v = local_assign.value
            if isinstance(v, ast.Subscript) or (isinstance(v, ast.Call) and (_chain(v.func) or [""])[-1] in (
                    "setdefault", "get", "load_history", "get_history")) or isinstance(v, ast.Attribute):
                return f"a per-session store ({ast.unparse(v)})"
        # REPL: while loop with input() around the call, list created outside that loop
        for a in self.ancestors(call):
            if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
            if isinstance(a, ast.While):
                has_input = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "input"
                                for n in ast.walk(a))
                created_inside = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                                        for t in n.targets) for n in ast.walk(a))
                if has_input and not created_inside:
                    return "a chat loop (while ... input())"
        return None

    def _trimmed(self, var: str) -> bool:
        v = re.escape(var)
        pats = [v + r"\s*\[\s*-", v + r"\s*=\s*" + v + r"\s*\[", r"del\s+" + v + r"\s*\[", v + r"\.pop\(",
                v + r"\.clear\(", v + r"\s*\[\s*:\s*\]\s*=", v + r"\s*\[\s*\d*\s*:\s*-?\s*\w+\s*\]"]
        if any(re.search(p, self.text) for p in pats):
            return True
        if re.search(r"len\(\s*" + v + r"\s*\)\s*[<>=]", self.text):
            return True
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                fn = (_chain(n.func) or [""])[-1]
                if TRIM_WORDS.search(fn) and any(var == ast.unparse(a) for a in [*n.args, *[k.value for k in n.keywords]]):
                    return True
        return False


def _provider_of(model: str) -> str | None:
    m = model.lower()
    if "claude" in m or m.startswith("anthropic"):
        return "anthropic"
    if "gemini" in m or m.startswith(("google", "vertex")):
        return "google"
    if re.match(r"^(openai/|azure/)?(gpt|o\d|text-|chatgpt)", m):
        return "openai"
    return None


def parse_failure(exc: BaseException) -> str:
    if isinstance(exc, SyntaxError):
        return f"syntax error line {exc.lineno}" if exc.lineno else "syntax error"
    if isinstance(exc, RecursionError):
        return "too deeply nested to parse"
    return f"could not be parsed ({exc})"


def extract_python(rel: str, text: str) -> FileFacts | None:
    """Facts for one file. Raises Unparseable when the file is not valid Python."""
    try:
        return PyExtractor(rel, text).run()
    except (SyntaxError, ValueError, RecursionError) as e:
        raise Unparseable(parse_failure(e)) from None
