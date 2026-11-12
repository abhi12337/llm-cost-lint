"""A small JavaScript / TypeScript lexer.

There is no JS parser in the standard library, so the JS checks work on two copies of the
source with the same length as the original (offsets and line numbers stay valid):

  clean: comments blanked, strings intact
  code:  comments blanked and string / template contents blanked (template ${...} kept)

It also returns every string literal as a StrTok with its static text and where the first
${...} interpolation is, which the prompt rules need."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class StrTok:
    start: int                      # index of the opening quote / backtick
    end: int                        # index after the closing quote / backtick
    text: str = ""                  # static text (escapes kept roughly as written)
    template: bool = False
    has_interp: bool = False
    after_first_interp: int = 0
    interps: list = field(default_factory=list)


def lex(text: str) -> tuple[str, str, list[StrTok]]:
    n = len(text)
    clean = list(text)
    code = list(text)
    toks: list[StrTok] = []
    i = 0
    tmpl_stack: list[tuple[int, StrTok]] = []     # (brace depth to return at, template token)
    brace = 0
    prev_sig = ""

    def blank(lst, a, b):
        for k in range(a, b):
            if lst[k] != "\n":
                lst[k] = " "

    def template_from(start: int, tok: StrTok) -> int:
        """start is just after ` or the } closing a ${}; returns end index, or -(index) to resume code."""
        nonlocal brace
        j = start
        while j < n:
            c = text[j]
            if c == "\\":
                blank(code, j, min(j + 2, n))
                tok.text += text[j:j + 2]
                if tok.has_interp:
                    tok.after_first_interp += 1
                j += 2
                continue
            if c == "`":
                tok.end = j + 1
                return j + 1
            if c == "$" and j + 1 < n and text[j + 1] == "{":
                tok.has_interp = True
                tok.interps.append(len(tok.text))
                tmpl_stack.append((brace, tok))
                brace += 1
                return -(j + 2)
            blank(code, j, j + 1)
            tok.text += c
            if tok.has_interp:
                tok.after_first_interp += 1
            j += 1
        tok.end = n
        return n

    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            blank(clean, i, j)
            blank(code, i, j)
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(clean, i, j)
            blank(code, i, j)
            i = j
            continue
        if c in "'\"":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            tok = StrTok(i, min(j + 1, n), text[i + 1:min(j, n)])
            toks.append(tok)
            blank(code, i + 1, min(j, n))
            i = j + 1
            prev_sig = c
            continue
        if c == "`":
            tok = StrTok(i, n, template=True)
            toks.append(tok)
            r = template_from(i + 1, tok)
            if r < 0:
                i = -r
                prev_sig = "{"
            else:
                i = r
                prev_sig = "`"
            continue
        if c == "/" and (prev_sig == "" or prev_sig in "(,=:[!&|?{};+-*%<>~^" or
                         re.search(r"\b(return|typeof|case|in|of)\s*$", text[max(0, i - 8):i])):
            j = i + 1
            in_class = False
            while j < n and text[j] != "\n":
                ch = text[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "[":
                    in_class = True
                elif ch == "]":
                    in_class = False
                elif ch == "/" and not in_class:
                    break
                j += 1
            blank(code, i + 1, min(j, n))
            i = j + 1
            prev_sig = "/"
            continue
        if c == "{":
            brace += 1
        elif c == "}":
            brace -= 1
            if tmpl_stack and brace == tmpl_stack[-1][0]:
                _, tok = tmpl_stack.pop()
                r = template_from(i + 1, tok)
                if r < 0:
                    i = -r
                    prev_sig = "{"
                else:
                    i = r
                    prev_sig = "`"
                continue
        prev_sig = c
        i += 1
    return "".join(clean), "".join(code), toks


def match_close(code: str, open_idx: int) -> int:
    """Index of the bracket closing code[open_idx]; len(code) if unbalanced."""
    depth = 0
    for j in range(open_idx, len(code)):
        c = code[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return j
    return len(code)


def split_args(code: str, start: int, end: int) -> list[tuple[int, int]]:
    """Top-level comma-separated spans of code[start:end]."""
    out, depth, a = [], 0, start
    for j in range(start, end):
        c = code[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append((a, j))
            a = j + 1
    if code[a:end].strip():
        out.append((a, end))
    return out


def strip_span(code: str, a: int, b: int) -> tuple[int, int]:
    while a < b and code[a].isspace():
        a += 1
    while b > a and code[b - 1].isspace():
        b -= 1
    return a, b
