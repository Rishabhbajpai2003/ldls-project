# Tree schema (v1)

Machine-readable definition: [`tree.schema.json`](tree.schema.json). This page explains the rules behind it. Every domain parser must produce this format, and the driver checks all of it when loading (`validate()` in `driver/ldls_driver.cpp`).

## Fields per node

| Field | Meaning |
|---|---|
| `id` | Index in `nodes`. Nodes are in **pre-order**: a parent always comes before its children. |
| `parent` | Parent id, or `null` for the root. |
| `node_type` | Structural type (`document`, `section`, `paragraph`, `list`, `code`, `table`, `merged`, ...). |
| `tokens` | Llama token IDs of the node's **own** text. Children's text is not repeated. |
| `positions` | One position per token (see below). |
| `seq_ids` | Sequence IDs every token of the node is tagged with: **own seq first**, then all descendants' seqs. |
| `title`, `n_markup`, `text` | Optional: label for addressing nodes, markup-token count, and source text. |

## Invariants

1. **One sequence per node.** Node *i* owns one sequence ID, unique in the document. The parsers use `seq = id`. IDs must stay below the driver's query sequence (63).
2. **Visibility via tags.** `seq_ids(N) = {seq(N)} ∪ seq_ids(children of N)`. llama.cpp masks a query token against the cell tags using the query's **first** sequence ID only. So a token of N sees exactly N's ancestors plus earlier tokens of N. Siblings and cousins are masked out.
3. **Tree positions.** A node's positions are consecutive and start at `parent.last + 1` (the root starts at 0). Siblings therefore **overlap** in position. Two things force this. llama.cpp requires each sequence's positions to be continuous. Overlap also makes a subtree's positions independent of the length of its siblings, which is what lets Test 1 compare a subtree before and after an edit to a sibling.
4. **Pre-order decoding.** Decoding the nodes in pre-order keeps every sequence continuous, so the whole document can be prefilled in one `llama_decode` call.

## Merge rules for small subtrees

llama.cpp allows only a limited number of sequences (the driver uses `n_seq_max = 64`; the hard cap is `LLAMA_MAX_SEQ = 256`). Tiny nodes would also waste them. The Markdown parser therefore applies three rules:

1. **Pack leaves.** Consecutive leaf blocks of one section (paragraphs, lists, code, tables) are packed greedily into chunks of at least `min_leaf` tokens (default 48). A short final chunk joins the previous one. A chunk mixing block types gets `node_type = "merged"`.
2. **Fold small sections.** If a section's body becomes a single chunk and the section has no subsections, or if the whole body is shorter than `min_leaf`, the body is folded into the heading node. The heading and its text then form one node.
3. **Respect the budget.** If the tree still has more than `max_nodes` nodes (default 48), `min_leaf` is doubled and the merge is redone. As a last resort, the smallest bottom-level section is collapsed into a single node until the budget is met.

In Markdown, a section's leaf blocks always come before its subsections, so merging never reorders text.

Domain notes from the project brief: grids (CSV) and cross-references (math documents) are not trees. Approximate them by grouping per row, or per cited block, with the same rules.
