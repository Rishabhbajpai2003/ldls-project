"""Markdown -> LDLS tree JSON (see schema/tree.schema.json).

Tree: document (prompt preamble) -> sections (one per heading, nested by level)
-> leaf blocks (paragraph, list, code, table, blockquote, html, rule).
Small blocks are merged by the shared rules in parsers/common.py; in Markdown a
section's blocks always precede its subsections, so merging never reorders text.

Markup: a token counts as markup if it is only whitespace or only Markdown
syntax characters (# * _ ` > | ~ - = [ ] :).

Usage: python parsers/markdown/parser.py "parsers/markdown/docs/*.md" -o parsers/markdown/json
"""

import os
import sys

from markdown_it import MarkdownIt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PREAMBLE, Node, run_cli  # noqa: E402

BLOCK_TYPES = {
    "paragraph_open": "paragraph",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "fence": "code",
    "code_block": "code",
    "blockquote_open": "blockquote",
    "table_open": "table",
    "html_block": "html",
    "hr": "rule",
}

MARKUP_CHARS = set("#*_`>|~-=[]:")


def is_markup(piece):
    s = piece.strip()
    return s == "" or all(ch in MARKUP_CHARS for ch in s)


def block_text(lines, span):
    return "\n".join(lines[span[0]:span[1]]).rstrip("\n") + "\n\n"


def build_raw_tree(src):
    lines = src.split("\n")
    toks = MarkdownIt("commonmark").enable("table").parse(src)

    root = Node("document", PREAMBLE, title="(document)", level=0)
    stack = [root]
    for i, t in enumerate(toks):
        if t.level != 0 or t.nesting == -1:
            continue
        if t.type == "heading_open":
            lvl = int(t.tag[1])
            while stack[-1].level >= lvl:
                stack.pop()
            node = Node("section", block_text(lines, t.map), title=toks[i + 1].content, level=lvl, container=True)
            stack[-1].children.append(node)
            stack.append(node)
        elif t.type in BLOCK_TYPES:
            stack[-1].children.append(Node(BLOCK_TYPES[t.type], block_text(lines, t.map)))
    return root


if __name__ == "__main__":
    run_cli("markdown", build_raw_tree, is_markup, __doc__)
