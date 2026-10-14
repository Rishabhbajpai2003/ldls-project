"""HTML -> LDLS tree JSON (see schema/tree.schema.json).

Tree: document (prompt preamble) -> DOM elements. An element is a *container*
(its own node, children below it) when it has child elements, at least one of
them block-level, and no text of its own; otherwise it is a *leaf* holding the
whole element (start tag, content, end tag). A container's own text is its start
tag; closing tags of containers are not stored, the tree carries the nesting.
Small leaves (li, meta, td, ...) are merged by the shared rules in parsers/common.py.

Markup: tags including attribute values, <script>/<style> contents, comments,
the doctype and whitespace. Content: visible text.

Titles (used to address nodes): tag#id, else "tag: <heading text>" (for headings and
elements with a heading as a direct child), else tag.class, else tag.

Usage: python parsers/html/parser.py "parsers/html/docs/*.html" -o parsers/html/json
"""

import html
import os
import sys

from bs4 import BeautifulSoup, Comment, Doctype, NavigableString, Tag

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PREAMBLE, Node, run_cli  # noqa: E402

INDENT = "  "
INLINE = {"a", "abbr", "b", "br", "button", "cite", "code", "em", "i", "img", "input", "kbd", "label",
          "mark", "q", "s", "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr"}
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
RAW_TEXT = {"script", "style"}
HEADINGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


def start_tag(el):
    attrs = ""
    for k, v in el.attrs.items():
        if isinstance(v, list):
            v = " ".join(v)
        attrs += f" {k}" if v == "" else f' {k}="{html.escape(v, quote=True)}"'
    return f"<{el.name}{attrs}>"


def serialize(el, segs):
    """Append (text, is_markup) segments for a whole element or string."""
    if isinstance(el, Comment):
        segs.append((f"<!--{el}-->", True))
    elif isinstance(el, NavigableString):
        if el.parent is not None and el.parent.name in RAW_TEXT:
            segs.append((str(el), True))
        else:
            segs.append((html.escape(str(el), quote=False), False))
    else:
        segs.append((start_tag(el), True))
        for c in el.children:
            serialize(c, segs)
        if el.name not in VOID:
            segs.append((f"</{el.name}>", True))


def is_container(el):
    if el.name in RAW_TEXT:
        return False
    kids = [c for c in el.children if isinstance(c, Tag)]
    own_text = any(isinstance(c, NavigableString) and not isinstance(c, Comment) and c.strip() for c in el.children)
    return bool(kids) and not own_text and any(k.name not in INLINE for k in kids)


class Titles:
    def __init__(self):
        self.seen = {}

    def __call__(self, el):
        """-> (title, named): named = the title comes from an id or a heading"""
        t = self.base(el)
        return self.unique(t), "#" in t or ": " in t

    def base(self, el):
        if el.get("id"):
            t = f"{el.name}#{el['id']}"
        elif el.name in HEADINGS:
            t = f"{el.name}: {el.get_text(' ', strip=True)[:40]}"
        elif (h := el.find(HEADINGS[:3], recursive=False)) is not None:
            t = f"{el.name}: {h.get_text(' ', strip=True)[:40]}"
        elif el.get("class"):
            t = f"{el.name}.{el['class'][0]}"
        else:
            t = el.name
        return t

    def unique(self, t):
        self.seen[t] = self.seen.get(t, 0) + 1
        return t if self.seen[t] == 1 else f"{t} ({self.seen[t]})"


def make_node(node_type, segs, **kw):
    merged = []  # join neighbouring segments of the same kind, so tokenization stays natural
    for text, flag in segs:
        if not text:
            continue
        if merged and merged[-1][1] == flag:
            merged[-1] = (merged[-1][0] + text, flag)
        else:
            merged.append((text, flag))
    return Node(node_type, "".join(t for t, _ in merged), segments=merged, **kw)


def build(el, depth, titles, prefix=()):
    ind = INDENT * depth
    if isinstance(el, Comment):
        return make_node("comment", [(ind, True), (f"<!--{el}-->", True), ("\n", True)], title=titles.unique("comment"))
    title, named = titles(el)
    if is_container(el):
        node = make_node(el.name, [*prefix, (ind, True), (start_tag(el), True), ("\n", True)],
                         title=title, container=True)
        for c in el.children:
            if isinstance(c, (Tag, Comment)):
                node.children.append(build(c, depth + 1, titles))
        return node
    segs = [*prefix, (ind, True)]
    serialize(el, segs)
    segs.append(("\n", True))
    return make_node(el.name, segs, title=title, keep_title=named)


def build_raw_tree(src):
    soup = BeautifulSoup(src, "lxml")
    prefix = [(f"<!DOCTYPE {d}>\n", True) for d in soup.contents if isinstance(d, Doctype)]
    root = Node("document", PREAMBLE, title="(document)")
    root.children.append(build(soup.html, 0, Titles(), prefix))
    return root


if __name__ == "__main__":
    run_cli("html", build_raw_tree, lambda piece: piece.strip() == "", __doc__)
