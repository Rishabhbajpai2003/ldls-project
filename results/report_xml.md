# Report: xml

| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup | Prefill flat / tree (ms) | Gen tok/s |
|---|---|---|---|---|---|---|---|
| harbor_gateway_config.xml | 10 (27) | 408 | 3 | 26/54/382 | 72% | 3572 / 3409 | 19.6 |
| jats_dunlin_foraging.xml | 28 (53) | 951 | 6 | 18/56/925 | 56% | 7566 / 7593 | 17.7 |
| jats_school_sleep.xml | 25 (54) | 690 | 6 | 14/46.5/664 | 67% | 5289 / 5252 | 19.3 |
| jats_subtree_eviction.xml | 24 (52) | 773 | 6 | 19/50/747 | 59% | 6258 / 6238 | 18.6 |
| orbit_server_pom.xml | 21 (57) | 614 | 6 | 5/48/588 | 73% | 4782 / 4680 | 18.5 |

```json
{
 "total": {
  "docs": 5,
  "nodes": 108,
  "nodes_before_merge": 243,
  "tokens": 3436,
  "max_depth": 6,
  "markup_share": 0.6361161524500908,
  "scoped_question_context_share": 0.2661549732275732,
  "prefill_flat_tok_s": 124.45780398762825,
  "prefill_tree_tok_s": 126.28679631307106,
  "generate_tok_s": 18.748264080719792,
  "kv_bytes_per_token": 32794.775073258446
 },
 "tests": {
  "cases": 10,
  "baseline_correct": 10,
  "test1_pass": 8,
  "test1_same_length_bit_identical": 10,
  "test1_same_length_flat_control_changed": 10,
  "test1_free_length_answer_unchanged": 8,
  "test1_free_length_correct": 10,
  "test2_pass": 9,
  "test2_correct": 10,
  "test2_kv_dropped": 10,
  "test2_full_scope_unchanged": 10,
  "test2_full_scope_correct": 10
 }
}
```
