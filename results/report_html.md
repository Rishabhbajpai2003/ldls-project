# Report: html

| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup | Prefill flat / tree (ms) | Gen tok/s |
|---|---|---|---|---|---|---|---|
| avelin_gazette.html | 20 (40) | 822 | 5 | 16/82/796 | 73% | 6154 / 6081 | 18.5 |
| brightloom_landing.html | 35 (77) | 1242 | 6 | 23/54.5/1216 | 79% | 10094 / 9673 | 17.8 |
| cs214_course.html | 28 (73) | 901 | 7 | 24/49/875 | 74% | 7454 / 7358 | 18.3 |
| hotel_greyhook.html | 23 (52) | 865 | 6 | 30/80/839 | 76% | 6583 / 6477 | 18.4 |
| saltmarsh_recipe.html | 29 (66) | 1206 | 6 | 10/81/1180 | 75% | 9691 / 9367 | 17.2 |

```json
{
 "total": {
  "docs": 5,
  "nodes": 135,
  "nodes_before_merge": 308,
  "tokens": 5036,
  "max_depth": 7,
  "markup_share": 0.7574398695474929,
  "scoped_question_context_share": 0.19497959932374254,
  "prefill_flat_tok_s": 126.66841511068405,
  "prefill_tree_tok_s": 129.66445815628538,
  "generate_tok_s": 18.026250497620673,
  "kv_bytes_per_token": 32786.40650463837
 },
 "tests": {
  "cases": 10,
  "baseline_correct": 10,
  "test1_pass": 9,
  "test1_same_length_bit_identical": 10,
  "test1_same_length_flat_control_changed": 10,
  "test1_free_length_answer_unchanged": 9,
  "test1_free_length_correct": 10,
  "test2_pass": 9,
  "test2_correct": 10,
  "test2_kv_dropped": 10,
  "test2_full_scope_unchanged": 9,
  "test2_full_scope_correct": 10
 }
}
```
