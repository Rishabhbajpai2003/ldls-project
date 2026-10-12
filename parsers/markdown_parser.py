"""Markdown -> LDLS tree JSON (see schema/tree.schema.json).

Tree: document (prompt preamble) -> sections (one per heading, nested by level)
-> leaf blocks (paragraph, list, code, table, blockquote, html, rule).

Merge rules (llama.cpp only allows a limited number of sequences, so tiny
subtrees must not get their own sequence ID):
  1. Consecutive leaf blocks of a section are packed greedily into chunks of at
     least `min_leaf` tokens; a short tail joins the previous chunk.
  2. A section whose leaf content is a single chunk with no subsections, or
     whose leaf content is shorter than `min_leaf`, is folded into the heading
     node itself (heading + body become one node).
  3. If the tree still has more than `max_nodes` nodes, `min_leaf` is doubled
     and the merge is redone; as a last resort the smallest bottom-level
     section is collapsed into a single node until the budget is met.

Tokens come from the Llama tokenizer through `ldls_driver tokenize`, so the
token IDs are exactly the ones the driver will decode.

Usage: python parsers/markdown_parser.py data/markdown/docs/*.md -o data/markdown/json
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from markdown_it import MarkdownIt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import ldls_paths  # noqa: E402

DRIVER = ldls_paths.driver()
MODEL = ldls_paths.model()

MAX_DOC_TOKENS = 2000

# root node: opens a Llama 3 chat; the driver appends the question and closes the user turn
PREAMBLE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Answer the question using only the document. Answer in one short sentence."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nDocument:\n\n"
)

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

# a token counts as markup if it is only whitespace or only Markdown syntax characters
MARKUP_CHARS = set("#*_`>|~-=[]:")


def is_markup(piece):
    s = piece.strip()
    return s == "" or all(ch in MARKUP_CHARS for ch in s)


@dataclass
class Node:
    node_type: str
    text: str
    title: str = ""
    level: int = 7  # heading level; 0 = document root, 7 = leaf block
    children: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    pieces: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.ids)

    def is_section(self):
        return self.node_type == "section"

    def absorb(self, others):
        for o in others:
            self.text += o.text
            self.ids += o.ids
            self.pieces += o.pieces


def tokenize(items):
    """items: list of (text, special) -> list of (ids, pieces)"""
    payload = json.dumps([{"text": t, "special": s} for t, s in items])
    res = subprocess.run(
        [DRIVER, "tokenize", "-m", MODEL],
        input=payload.encode("utf-8"),
        capture_output=True,
        check=True,
    )
    out = json.loads(res.stdout.decode("utf-8"))
    return [(o["ids"], o["pieces"]) for o in out]


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
            node = Node("section", block_text(lines, t.map), title=toks[i + 1].content, level=lvl)
            stack[-1].children.append(node)
            stack.append(node)
        elif t.type in BLOCK_TYPES:
            stack[-1].children.append(Node(BLOCK_TYPES[t.type], block_text(lines, t.map)))

    nodes = list(walk(root))
    for node, (ids, pieces) in zip(nodes, tokenize([(n.text, n is root) for n in nodes])):
        node.ids, node.pieces = ids, pieces
    return root


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


def pack(run, min_leaf, section_title):
    chunks, cur = [], []
    for b in run:
        cur.append(b)
        if sum(x.n for x in cur) >= min_leaf:
            chunks.append(cur)
            cur = []
    if cur:
        if chunks:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)

    out = []
    for k, chunk in enumerate(chunks):
        if len(chunk) == 1:
            node = chunk[0]
        else:
            types = {b.node_type for b in chunk}
            node = Node(types.pop() if len(types) == 1 else "merged", "")
            node.absorb(chunk)
        node.title = f"{section_title} #{k + 1}"
        out.append(node)
    return out


def merge(node, min_leaf):
    leaves = [c for c in node.children if not c.is_section()]
    sections = [c for c in node.children if c.is_section()]
    for s in sections:
        merge(s, min_leaf)

    # leaves always precede subsections in Markdown, so order is preserved
    chunks = pack(leaves, min_leaf, node.title) if leaves else []
    fold = node.is_section() and chunks and (
        (len(chunks) == 1 and not sections) or sum(c.n for c in chunks) < min_leaf
    )
    if fold:
        node.absorb(chunks)
        chunks = []
    node.children = chunks + sections


def count(node):
    return sum(1 for _ in walk(node))


def subtree_tokens(node):
    return sum(n.n for n in walk(node))


def collapse_smallest(root):
    bottom = [n for n in walk(root) if n.is_section() and n.children and not any(c.is_section() for c in n.children)]
    if not bottom:
        return False
    victim = min(bottom, key=subtree_tokens)
    victim.absorb(list(walk(victim))[1:])
    victim.children = []
    return True


def build_tree(raw, min_leaf, max_nodes):
    while True:
        tree = copy.deepcopy(raw)
        merge(tree, min_leaf)
        if count(tree) <= max_nodes or min_leaf >= 4096:
            break
        min_leaf *= 2
    while count(tree) > max_nodes and collapse_smallest(tree):
        pass
    return tree, min_leaf


def to_json(tree, source, min_leaf, max_nodes, n_raw):
    order = list(walk(tree))
    index = {id(n): i for i, n in enumerate(order)}
    parent = {}
    for n in order:
        for c in n.children:
            parent[id(c)] = index[id(n)]

    out_nodes = []
    for i, n in enumerate(order):
        p = parent.get(id(n))
        start = 0 if p is None else out_nodes[p]["positions"][-1] + 1
        out_nodes.append({
            "id": i,
            "parent": p,
            "node_type": n.node_type,
            "title": n.title,
            "seq_ids": [index[id(d)] for d in walk(n)],  # own seq first, then descendants
            "positions": list(range(start, start + n.n)),
            "tokens": n.ids,
            "n_markup": sum(is_markup(pc) for pc in n.pieces),
            "text": n.text,
        })

    return {
        "schema_version": 1,
        "domain": "markdown",
        "source": os.path.basename(source),
        "model": os.path.basename(MODEL),
        "merge": {"min_leaf_tokens": min_leaf, "max_nodes": max_nodes, "nodes_before_merge": n_raw},
        "n_tokens": sum(n.n for n in order),
        "nodes": out_nodes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--min-leaf", type=int, default=48, help="minimum tokens per leaf chunk")
    ap.add_argument("--max-nodes", type=int, default=48, help="sequence budget per document")
    args = ap.parse_args()

    # expand patterns ourselves: PowerShell and cmd pass "*.md" through unexpanded
    inputs = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            sys.exit(f"no file matches {pattern}")
        inputs += matches

    os.makedirs(args.out_dir, exist_ok=True)
    for path in inputs:
        with open(path, encoding="utf-8") as f:
            raw = build_raw_tree(f.read())
        tree, min_leaf = build_tree(raw, args.min_leaf, args.max_nodes)
        doc = to_json(tree, path, min_leaf, args.max_nodes, count(raw))
        if doc["n_tokens"] > MAX_DOC_TOKENS:
            sys.exit(f"{path}: {doc['n_tokens']} tokens exceeds the {MAX_DOC_TOKENS}-token limit")

        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"{os.path.basename(path)}: {len(doc['nodes'])} nodes ({count(raw)} before merge), "
              f"{doc['n_tokens']} tokens, min_leaf={min_leaf} -> {out}")


if __name__ == "__main__":
    main()
