# Report: markdown

| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup | Prefill flat / tree (ms) | Gen tok/s |
|---|---|---|---|---|---|---|---|
| lab_safety.md | 13 (24) | 514 | 3 | 35/47/488 | 13% | 5249 / 5849 | 11.2 |
| orbit_cli.md | 16 (29) | 589 | 3 | 21/59/563 | 27% | 6231 / 4785 | 19.0 |
| port_avelin_guide.md | 13 (24) | 525 | 3 | 38/49.5/499 | 18% | 4070 / 4000 | 18.8 |
| rye_sourdough.md | 12 (21) | 524 | 3 | 33/58/498 | 23% | 4201 / 4105 | 19.5 |
| team_handbook.md | 14 (25) | 499 | 3 | 22/47/473 | 12% | 4212 / 4050 | 17.4 |

```json
{
 "total": {
  "docs": 5,
  "nodes": 68,
  "nodes_before_merge": 123,
  "tokens": 2651,
  "max_depth": 3,
  "markup_share": 0.18921063070210234,
  "scoped_question_context_share": 0.24246645726464228,
  "prefill_flat_tok_s": 112.93399417672398,
  "prefill_tree_tok_s": 118.61800544949224,
  "generate_tok_s": 17.17132223381907,
  "kv_bytes_per_token": 32786.93943357322
 },
 "tests": {
  "cases": 10,
  "baseline_correct": 10,
  "test1_pass": 10,
  "test1_same_length_bit_identical": 10,
  "test1_same_length_flat_control_changed": 10,
  "test1_free_length_answer_unchanged": 10,
  "test1_free_length_correct": 10,
  "test2_pass": 10,
  "test2_correct": 10,
  "test2_kv_dropped": 10,
  "test2_full_scope_unchanged": 9,
  "test2_full_scope_correct": 10
 }
}
```
