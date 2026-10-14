"""Run test 1 (edit B, ask about A) and test 2 (evict B, ask about A) for every case.

Usage: python tests/run_tests.py [--domain markdown|html|xml]
       (= --cases tests/cases_<domain>.json --docs parsers/<domain>/json)
Writes results/test_results_<domain>.json and prints a summary table.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ldls_paths  # noqa: E402

DRIVER = ldls_paths.driver()
MODEL = ldls_paths.model()


def run_case(case, docs_dir, n_predict):
    cmd = [
        DRIVER, "test", "-m", MODEL,
        "-d", os.path.join(docs_dir, case["doc"]),
        "-q", case["question"],
        "--a", case["A"],
        "--b", case["B"],
        "--edit-text", case["edit_text"],
        "-n", str(n_predict),
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode not in (0, 1):
        sys.exit(f"driver failed on {case['doc']} / {case['A']}:\n{res.stderr.decode('utf-8', 'replace')}")
    r = json.loads(res.stdout.decode("utf-8"))
    expect = case["expect"].lower()
    r["expect"] = case["expect"]
    r["baseline_correct"] = expect in r["baseline_scoped"]["text"].lower()
    for variant in ("same_length", "free_length"):
        t = r["test1_edit_B"][variant]
        t["correct"] = expect in t["scoped_answer"]["text"].lower()
    r["test2_evict_B"]["correct"] = expect in r["test2_evict_B"]["scoped_answer"]["text"].lower()
    r["test2_evict_B"]["full_correct"] = expect in r["test2_evict_B"]["full_answer"]["text"].lower()
    return r


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="markdown", help="picks tests/cases_<domain>.json and parsers/<domain>/json")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--docs", default=None)
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    args.cases = args.cases or os.path.join(ROOT, "tests", f"cases_{args.domain}.json")
    args.docs = args.docs or os.path.join(ROOT, "parsers", args.domain, "json")

    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)
    domain = os.path.splitext(os.path.basename(args.cases))[0].replace("cases_", "")
    out_path = args.out or os.path.join(ROOT, "results", f"test_results_{domain}.json")

    results = []
    print(f"{'doc':<20} {'A':<20} {'base':<5} {'T1 tree/flat logit diff':<24} {'T1 ans':<7} {'T2 ans':<7} {'KV MB':<12} answer")
    for case in cases:
        r = run_case(case, args.docs, args.n)
        results.append(r)
        sl, fl, t2 = r["test1_edit_B"]["same_length"], r["test1_edit_B"]["free_length"], r["test2_evict_B"]
        diffs = f"{sl['scoped_first_logits_diff']:.3f} / {sl['flat_control_logits_diff']:.3f}"
        kv = f"{t2['kv_bytes_before'] / 1e6:.1f}->{t2['kv_bytes_after'] / 1e6:.1f}"
        print(f"{case['doc'][:20]:<20} {case['A'][:20]:<20} {'ok' if r['baseline_correct'] else 'MISS':<5} "
              f"{diffs:<24} {'same' if fl['scoped_unchanged'] else 'DIFF':<7} {'same' if t2['scoped_unchanged'] else 'DIFF':<7} "
              f"{kv:<12} {r['baseline_scoped']['text'][:50]!r}")

    def count(pred):
        return sum(1 for r in results if pred(r))

    summary = {
        "cases": len(results),
        "baseline_correct": count(lambda r: r["baseline_correct"]),
        "test1_pass": count(lambda r: r["test1_edit_B"]["pass"]),
        "test1_same_length_bit_identical": count(lambda r: r["test1_edit_B"]["same_length"]["pass"]),
        "test1_same_length_flat_control_changed": count(lambda r: r["test1_edit_B"]["same_length"]["flat_control_logits_diff"] > 0),
        "test1_free_length_answer_unchanged": count(lambda r: r["test1_edit_B"]["free_length"]["scoped_unchanged"]),
        "test1_free_length_correct": count(lambda r: r["test1_edit_B"]["free_length"]["correct"]),
        "test2_pass": count(lambda r: r["test2_evict_B"]["pass"]),
        "test2_correct": count(lambda r: r["test2_evict_B"]["correct"]),
        "test2_kv_dropped": count(lambda r: r["test2_evict_B"]["kv_bytes_after"] < r["test2_evict_B"]["kv_bytes_before"]),
        "test2_full_scope_unchanged": count(lambda r: r["test2_evict_B"]["full_unchanged"]),
        "test2_full_scope_correct": count(lambda r: r["test2_evict_B"]["full_correct"]),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)
    print("\n" + json.dumps(summary, indent=1))
    print(f"results -> {out_path}")
    ok = summary["test1_pass"] == summary["test2_pass"] == len(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
