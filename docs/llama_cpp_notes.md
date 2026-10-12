# llama.cpp notes: batched example and mask construction

Traced against llama.cpp commit `08ffba2` (12 Oct 2026). Line numbers refer to that commit.

## 1. The API surface we use (`include/llama.h`)

| Concept | API | Notes |
|---|---|---|
| Batch | `llama_batch` (l. 264): `token`, `pos`, `n_seq_id`, `seq_id`, `logits` | Each token carries its own position and a **list** of sequence IDs. `logits[i] != 0` asks for output at token *i*. |
| Allocate | `llama_batch_init(n_tokens, 0, n_seq_max)` | `n_seq_max` = the maximum number of seq IDs **per token**. |
| Decode | `llama_decode(ctx, batch)` (l. 1006) | 0 = ok, 1 = no KV slot, -1 = invalid batch. |
| Memory | `llama_get_memory(ctx)` | Handle for the KV cache. |
| Evict | `llama_memory_seq_rm(mem, seq, p0, p1)` (l. 764) | Removes seq's tags from cells with positions in `[p0, p1)`. A cell with no tags left is freed. |
| Share | `llama_memory_seq_cp(mem, src, dst, p0, p1)` (l. 773) | Adds tag `dst` to every cell tagged `src` in the range. With a unified cache this is metadata only. |
| Inspect | `llama_memory_seq_pos_min/max` | Lowest/highest position stored for a sequence. |
| Size | `llama_state_get_size(ctx)` | Serialized state size. It is dominated by the K/V data of the cells in use, so we use it to measure KV memory. |
| Context | `llama_context_params.n_seq_max`, `.kv_unified` | `kv_unified = true` gives one KV buffer shared by all sequences. Without it, every sequence has its own stream and a token cannot belong to two of them. |

## 2. The `batched` example (`examples/batched/batched.cpp`)

1. Tokenize one prompt (l. 60). Create a context with `n_ctx = prompt + (n_predict - prompt) * n_parallel` (l. 62–68).
2. Put each prompt token into the batch **once**, tagged with all `n_parallel` sequence IDs (l. 122–130). All streams share the prompt's KV cells.
3. Request logits only for the last prompt token (l. 149), then decode.
4. Loop: each stream samples from its own output row (`i_batch[i]`, l. 191) and adds its token with its **own** sequence ID at position `n_cur` (l. 214). One `llama_decode` advances all streams together.

Our driver applies the same idea to a tree: a token shared by several sequences is stored once and tagged with all of them.

## 3. How the attention mask is built from sequence IDs

`src/llama-kv-cache.cpp`, `set_input_kq_mask_impl` (l. 1569–1718). For each query token *i* in the micro-batch and each KV cell *j*:

```
seq_id = ubatch->seq_id[i][0]             // l. 1605: only the query's FIRST seq id is used
if cell j is empty                -> -inf  // l. 1659
if !cells.seq_has(j, seq_id)      -> -inf  // l. 1664: cell must carry the query's sequence
if causal && pos(j) > pos(i)      -> -inf  // l. 1681: no attending to the future (by position, not slot)
otherwise                         -> 0
```

Consequences we rely on:

* **Visibility is decided by the cell's tags.** A cell tagged `{3, 4, 7}` is visible to queries whose primary sequence is 3, 4 or 7.
* **The query's extra seq IDs do not widen what it sees.** They only decide which sequences the new cell will belong to.
* **Causality uses positions.** Overlapping positions in different sequences are therefore harmless.
* Rows for tokens of the same primary sequence reuse the mask of the first row and patch only the cells near the batch positions (l. 1618–1657). This is an optimisation and does not change the result.

`LLAMA_MAX_SEQ = 256` (`src/llama-cparams.h:8`) caps sequence IDs. With a unified cache, every cell stores a `std::bitset<256>` of tags.

## 4. Batch validation rules (`src/llama-batch.cpp`)

* l. 186–191: every seq ID must be `< n_seq_max`.
* l. 395–424: for every sequence, the batch must continue exactly where memory ends (`Y = X + 1`), and positions inside the batch must be **continuous**.
* l. 428–440: sequences that appear together on one token ("coupled") must have identical position ranges in memory.
* l. 459–491: positions of a sequence must not decrease within the batch.

These rules are why the schema uses **tree positions** and **pre-order** (see `schema/SCHEMA.md`). With linear positions, the second child of a node would have a position gap in its sequence and the batch would be rejected.

## 5. How the batch is split into micro-batches

With `kv_unified = true` there is one stream, so `llama-kv-cache.cpp:731` uses `split_simple(n_ubatch)`: consecutive chunks of up to `n_ubatch` tokens. Within a micro-batch, all K/V rows are written to the cache before attention runs, so a child node in the same micro-batch as its parent sees the parent correctly.

## 6. Two-sequence check

`ldls_driver twoseq` decodes two prompts in sequences 0 and 1 of one batch, then decodes each alone. Greedy output is identical. The first-step logits differ by about 0.5 at most: a different batch shape changes accumulation order, and the 8-bit quantization of activations (Q4_K × Q8_K) amplifies these tiny differences. Repeating a run in the same state gives bit-identical logits, independent of thread count (`ldls_driver noise`).
