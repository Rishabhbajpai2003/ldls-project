"""XML -> LDLS tree JSON (see schema/tree.schema.json).

Tree: document (prompt preamble) -> XML elements. An element is a *container*
(its own node, children below it) when it has child elements and no text of its
own (no mixed content); otherwise it is a *leaf* holding the whole element. A
container's own text is its start tag; closing tags of containers are not
stored, the tree carries the nesting. The XML declaration, doctype and anything
before the root element go into the root element's own text. Small leaves are
merged by the shared rules in parsers/common.py.

Markup: tag and attribute names with their syntax, namespace declarations,
comments, processing instructions, the declaration/doctype and whitespace.
Content: text and attribute values (in XML they carry data, e.g. version="5.10.2").

Titles (used to address nodes): tag#id, else "tag: <name attribute>", else
"tag: <text of a title/article-title/name/artifactId/label child>", else tag.

Usage: python parsers/xml/parser.py "parsers/xml/docs/*.xml" -o parsers/xml/json
"""

import os
import sys
from xml.sax.saxutils import escape

from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import PREAMBLE, Node, run_cli  # noqa: E402

INDENT = "  "
XML_NS = "http://www.w3.org/XML/1998/namespace"
TITLE_CHILDREN = {"title", "article-title", "name", "artifactId", "label"}


def is_element(el):
    return isinstance(el.tag, str)


def local(el):
    return etree.QName(el).localname


def tag_name(el):
    return f"{el.prefix}:{local(el)}" if el.prefix else local(el)


def attr_name(key, el):
    q = etree.QName(key)
    if not q.namespace:
        return q.localname
    if q.namespace == XML_NS:
        return f"xml:{q.localname}"
    prefix = next(p for p, uri in el.nsmap.items() if uri == q.namespace and p)
    return f"{prefix}:{q.localname}"


def start_tag(el, empty):
    segs = [("<" + tag_name(el), True)]
    parent = el.getparent()
    inherited = parent.nsmap if parent is not None else {}
    for p, uri in el.nsmap.items():
        if inherited.get(p) != uri:
            segs.append((f' xmlns{":" + p if p else ""}="{uri}"', True))
    for k, v in el.attrib.items():
        segs += [(f' {attr_name(k, el)}="', True), (escape(v, {'"': "&quot;"}), False), ('"', True)]
    segs.append(("/>" if empty else ">", True))
    return segs


def serialize(el, segs):
    """Append (text, is_markup) segments for a whole element, comment or PI."""
    if el.tag is etree.Comment:
        segs.append((f"<!--{el.text}-->", True))
    elif el.tag is etree.PI:
        segs.append((f"<?{el.target}{' ' + el.text if el.text else ''}?>", True))
    elif el.tag is etree.Entity:
        segs.append((f"&{el.name};", True))
    else:
        empty = not el.text and len(el) == 0
        segs += start_tag(el, empty)
        if el.text:
            segs.append((escape(el.text), False))
        for c in el:
            serialize(c, segs)
            if c.tail:
                segs.append((escape(c.tail), False))
        if not empty:
            segs.append((f"</{tag_name(el)}>", True))


def is_container(el):
    kids = [c for c in el if is_element(c)]
    mixed = (el.text and el.text.strip()) or any(c.tail and c.tail.strip() for c in el)
    return bool(kids) and not mixed


class Titles:
    def __init__(self):
        self.seen = {}

    def __call__(self, el):
        """-> (title, named): named = the title comes from an id, name or title child"""
        t = tag_name(el)
        if el.get("id"):
            t = f"{t}#{el.get('id')}"
        elif el.get("name"):
            t = f"{t}: {el.get('name')}"
        else:
            for c in el:
                if is_element(c) and local(c) in TITLE_CHILDREN and (c.text or "").strip():
                    t = f"{t}: {c.text.strip()[:40]}"
                    break
        return self.unique(t), t != tag_name(el)

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
    if not is_element(el):
        segs = [(ind, True)]
        serialize(el, segs)
        kind = "comment" if el.tag is etree.Comment else "pi"
        return make_node(kind, segs + [("\n", True)], title=titles.unique(kind))
    title, named = titles(el)
    if is_container(el):
        node = make_node(local(el), [*prefix, (ind, True), *start_tag(el, False), ("\n", True)],
                         title=title, container=True)
        for c in el:
            node.children.append(build(c, depth + 1, titles))
        return node
    segs = [*prefix, (ind, True)]
    serialize(el, segs)
    return make_node(local(el), segs + [("\n", True)], title=title, keep_title=named)


def prolog(src, root_el):
    """XML declaration, doctype, and comments/PIs before the root element."""
    segs = []
    s = src.lstrip()
    if s.startswith("<?xml"):
        segs.append((s[:s.index("?>") + 2] + "\n", True))
    doctype = root_el.getroottree().docinfo.doctype
    if doctype:
        segs.append((doctype + "\n", True))
    for sib in reversed(list(root_el.itersiblings(preceding=True))):
        serialize(sib, segs)
        segs.append(("\n", True))
    return segs


def build_raw_tree(src):
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
    root_el = etree.fromstring(src.encode("utf-8"), parser)
    root = Node("document", PREAMBLE, title="(document)")
    root.children.append(build(root_el, 0, Titles(), prolog(src, root_el)))
    return root


if __name__ == "__main__":
    run_cli("xml", build_raw_tree, lambda piece: piece.strip() == "", __doc__)
