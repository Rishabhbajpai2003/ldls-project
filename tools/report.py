"""Tree statistics + CPU timings + test results -> one report per domain.

Usage: python tools/report.py [--domain markdown|html|xml] [--no-bench]
       (--docs <dir> overrides parsers/<domain>/json)
Writes results/report_<domain>.json and results/report_<domain>.md.

Statistics (the root node is the prompt preamble and is excluded from document stats):
  depth         longest root-to-leaf path in edges (root = 0)
  subtree size  tokens in a node's subtree, over all non-root nodes
  leaf size     tokens in a leaf node
  markup share  tokens flagged as markup/syntax by the domain parser (n_markup; see each parser's docstring)
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

import ldls_paths

DRIVER = ldls_paths.driver()
MODEL = ldls_paths.model()
ROOT = ldls_paths.ROOT


def tree_stats(doc):
    nodes = doc["nodes"]
    children = {n["id"]: [] for n in nodes}
    for n in nodes:
        if n["parent"] is not None:
            children[n["parent"]].append(n["id"])

    depth = {0: 0}
    for n in nodes[1:]:
        depth[n["id"]] = depth[n["parent"]] + 1

    def sub_tokens(i):
        return len(nodes[i]["tokens"]) + sum(sub_tokens(c) for c in children[i])

    body = nodes[1:]
    sub = [sub_tokens(n["id"]) for n in body]
    leaves = [len(n["tokens"]) for n in body if not children[n["id"]]]
    doc_tokens = sum(len(n["tokens"]) for n in body)
    markup = sum(n["n_markup"] for n in body)

    def dist(v):
        return {"min": min(v), "median": statistics.median(v), "max": max(v)}

    return {
        "doc": doc["source"],
        "nodes": len(nodes),
        "nodes_before_merge": doc["merge"].get("nodes_before_merge"),
        "tokens_total": doc["n_tokens"],
        "tokens_document": doc_tokens,
        "depth": max(depth.values()),
        "subtree_tokens": dist(sub),
        "leaf_tokens": dist(leaves),
        "n_leaves": len(leaves),
        "markup_tokens": markup,
        "markup_share": markup / doc_tokens,
        "node_types": {t: sum(1 for n in body if n["node_type"] == t) for t in sorted({n["node_type"] for n in body})},
    }


def bench(path):
    res = subprocess.run([DRIVER, "bench", "-m", MODEL, "-d", path, "-q", "Summarize the document in one sentence.",
                          "-n", "32", "-r", "3"], capture_output=True, check=True)
    return json.loads(res.stdout.decode("utf-8"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="markdown", help="reads parsers/<domain>/json")
    ap.add_argument("--docs", default=None)
    ap.add_argument("--no-bench", action="store_true", help="reuse timings from the previous report")
    args = ap.parse_args()
    args.docs = args.docs or os.path.join(ROOT, "parsers", args.domain, "json")

    paths = sorted(glob.glob(os.path.join(args.docs, "*.json")))
    domain = json.load(open(paths[0], encoding="utf-8"))["domain"]
    out_json = os.path.join(ROOT, "results", f"report_{domain}.json")

    previous = {}
    if args.no_bench and os.path.exists(out_json):
        previous = {d["doc"]: d.get("bench") for d in json.load(open(out_json, encoding="utf-8"))["docs"]}

    docs = []
    for p in paths:
        doc = json.load(open(p, encoding="utf-8"))
        s = tree_stats(doc)
        s["bench"] = previous.get(s["doc"]) if args.no_bench else bench(p)
        docs.append(s)
        print(f"{s['doc']}: done", file=sys.stderr)

    tests_path = os.path.join(ROOT, "results", f"test_results_{domain}.json")
    tests = json.load(open(tests_path, encoding="utf-8")) if os.path.exists(tests_path) else None

    scoped_share = None
    if tests:
        shares = [r["baseline_scoped"]["n_context"] / r["n_tokens"] for r in tests["results"]]
        scoped_share = statistics.mean(shares)

    total = {
        "docs": len(docs),
        "nodes": sum(d["nodes"] for d in docs),
        "nodes_before_merge": sum(d["nodes_before_merge"] for d in docs),
        "tokens": sum(d["tokens_total"] for d in docs),
        "max_depth": max(d["depth"] for d in docs),
        "markup_share": sum(d["markup_tokens"] for d in docs) / sum(d["tokens_document"] for d in docs),
        "scoped_question_context_share": scoped_share,
    }
    if all(d["bench"] for d in docs):
        total["prefill_flat_tok_s"] = statistics.mean(d["bench"]["prefill_flat_tok_s"] for d in docs)
        total["prefill_tree_tok_s"] = statistics.mean(d["bench"]["prefill_tree_tok_s"] for d in docs)
        total["generate_tok_s"] = statistics.mean(d["bench"]["generate_tok_s"] for d in docs)
        total["kv_bytes_per_token"] = statistics.mean(d["bench"]["kv_bytes"] / d["tokens_total"] for d in docs)

    report = {"domain": domain, "total": total, "tests": tests["summary"] if tests else None, "docs": docs}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)

    lines = [
        f"# Report: {domain}",
        "",
        "| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup | Prefill flat / tree (ms) | Gen tok/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in docs:
        st, b = d["subtree_tokens"], d["bench"] or {}
        timing = (f"{b['prefill_flat_ms']:.0f} / {b['prefill_tree_one_batch_ms']:.0f}" if b else "-")
        gen = f"{b['generate_tok_s']:.1f}" if b else "-"
        lines.append(f"| {d['doc']} | {d['nodes']} ({d['nodes_before_merge']}) | {d['tokens_total']} | {d['depth']} | "
                     f"{st['min']}/{st['median']:g}/{st['max']} | {d['markup_share']:.0%} | {timing} | {gen} |")
    lines += ["", "```json", json.dumps({"total": total, "tests": report["tests"]}, indent=1), "```", ""]
    with open(out_json.replace(".json", ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
