"""Domain-independent parts of the LDLS parsers (see schema/SCHEMA.md).

A domain parser only builds the raw tree of `Node`s and calls `run_cli()`; this
module tokenizes it with the Llama tokenizer (through `ldls_driver tokenize`),
merges small subtrees, assigns sequence IDs and tree positions, and writes the
JSON of schema/tree.schema.json.

Merge rules (llama.cpp only allows a limited number of sequences, so tiny
subtrees must not get their own sequence ID):
  1. Each run of consecutive leaf children (between container children) is
     packed greedily into chunks of at least `min_leaf` tokens; a short tail
     joins the previous chunk. Sibling order is kept.
  2. A container whose leading leaf run is a single chunk and which has no
     container children, or whose leading leaf run is shorter than `min_leaf`,
     absorbs that run into its own text (e.g. heading + body become one node).
  3. If the tree still has more than `max_nodes` nodes, `min_leaf` is doubled
     and the merge is redone; as a last resort the smallest bottom-level
     container is collapsed into a single node until the budget is met.

Markup flags: every token is flagged as markup/syntax or content. A node either
gives `segments` [(text, is_markup)] -- each segment is tokenized on its own and
the node's tokens are their concatenation -- or is tokenized as one text and a
per-piece `classify` function decides.
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
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


@dataclass
class Node:
    node_type: str
    text: str = ""
    title: str = ""
    container: bool = False  # structural node whose children are merged by the rules above
    keep_title: bool = False # a leaf packed alone keeps its title (e.g. "server: admin") instead of "<parent> #k"
    level: int = 0           # free for the domain parser (Markdown: heading level)
    segments: list = None    # optional [(text, is_markup)]; must join to `text`
    children: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    pieces: list = field(default_factory=list)
    markup: list = field(default_factory=list)  # one flag per token

    @property
    def n(self):
        return len(self.ids)

    def absorb(self, others):
        for o in others:
            self.text += o.text
            self.ids += o.ids
            self.pieces += o.pieces
            self.markup += o.markup


def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)


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


def tokenize_tree(root, classify):
    """Tokenize every node in one driver call. The root holds the prompt preamble (special tokens)."""
    nodes = list(walk(root))
    items, spans = [], []
    for n in nodes:
        segs = n.segments if n.segments is not None else [(n.text, None)]
        if "".join(s for s, _ in segs) != n.text:
            raise ValueError(f"segments of node '{n.title}' do not join to its text")
        spans.append((len(items), segs))
        items += [(s, n is root) for s, _ in segs]

    out = tokenize(items)
    for n, (start, segs) in zip(nodes, spans):
        n.ids, n.pieces, n.markup = [], [], []
        for (_, flag), (ids, pieces) in zip(segs, out[start:start + len(segs)]):
            n.ids += ids
            n.pieces += pieces
            n.markup += [classify(p) if flag is None else (flag or p.strip() == "") for p in pieces]


def pack(run, min_leaf, parent_title, first):
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
            node = Node(types.pop() if len(types) == 1 else "merged")
            node.absorb(chunk)
        if not node.keep_title:
            node.title = f"{parent_title} #{first + k}"
        out.append(node)
    return out


def merge(node, min_leaf):
    for c in node.children:
        if c.container:
            merge(c, min_leaf)

    # split the children into runs of consecutive leaves and single containers
    groups = []
    for c in node.children:
        if c.container:
            groups.append(c)
        elif groups and isinstance(groups[-1], list):
            groups[-1].append(c)
        else:
            groups.append([c])

    packed, k = [], 1
    for g in groups:
        if isinstance(g, list):
            g = pack(g, min_leaf, node.title, k)
            k += len(g)
        packed.append(g)

    has_containers = any(c.container for c in node.children)
    leading = packed[0] if packed and isinstance(packed[0], list) else []
    fold = node.container and leading and (
        (len(leading) == 1 and not has_containers) or sum(c.n for c in leading) < min_leaf
    )
    if fold:
        node.absorb(leading)
        packed = packed[1:]

    node.children = []
    for g in packed:
        node.children += g if isinstance(g, list) else [g]


def count(node):
    return sum(1 for _ in walk(node))


def subtree_tokens(node):
    return sum(n.n for n in walk(node))


def collapse_smallest(root):
    bottom = [n for n in walk(root) if n.container and n.children and not any(c.container for c in n.children)]
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


def to_json(tree, domain, source, min_leaf, max_nodes, n_raw):
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
            "n_markup": sum(n.markup),
            "text": n.text,
        })

    return {
        "schema_version": 1,
        "domain": domain,
        "source": os.path.basename(source),
        "model": os.path.basename(MODEL),
        "merge": {"min_leaf_tokens": min_leaf, "max_nodes": max_nodes, "nodes_before_merge": n_raw},
        "n_tokens": sum(n.n for n in order),
        "nodes": out_nodes,
    }


def run_cli(domain, build_raw_tree, classify, description):
    """Command line shared by all domain parsers: parse files (globs allowed) into JSON."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="input files or quoted glob patterns")
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
        tokenize_tree(raw, classify)
        tree, min_leaf = build_tree(raw, args.min_leaf, args.max_nodes)
        doc = to_json(tree, domain, path, min_leaf, args.max_nodes, count(raw))
        if doc["n_tokens"] > MAX_DOC_TOKENS:
            sys.exit(f"{path}: {doc['n_tokens']} tokens exceeds the {MAX_DOC_TOKENS}-token limit")

        out = os.path.join(args.out_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"{os.path.basename(path)}: {len(doc['nodes'])} nodes ({count(raw)} before merge), "
              f"{doc['n_tokens']} tokens, min_leaf={min_leaf} -> {out}")
