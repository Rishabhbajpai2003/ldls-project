# LDLS: Structured Sequence IDs for Efficient Tree Attention in llama.cpp
## Project 0 Week 1 — Individual Report & Repository

**Student Name:** Aditya Sai Teja B  
**GitHub Profile & Submission Repository:** [aditya25058/ldls-project](https://github.com/aditya25058/ldls-project)  
**Assigned Individual Domains:** **Domain 7 (JSON / API Responses)** and **Domain 12 (Agent / Chat Logs)**  
**Platform & Environment:** CPU-only execution with Llama-3.2-1B-Instruct (`Q4_K_M`) using pinned `llama.cpp` (commit `08ffba2`)  

---

## Executive Summary

Large Language Models (LLMs) traditionally process documents as flat token streams with quadratic causal attention masks. This project implements and evaluates **Structured Sequence IDs** within `llama.cpp` to enforce hierarchical tree-structured attention on structured documents. 

In this system:
1. Documents are parsed into hierarchical trees adhering to [`schema/tree.schema.json`](schema/tree.schema.json).
2. Each node $N$ is assigned a sequence ID such that **each subtree attends strictly to its ancestors**, preventing cross-attention between sibling subtrees.
3. Sibling subtrees share overlapping token position indices, maintaining position continuity while keeping subtrees independent.
4. Independent subtrees can be dynamically **evicted** from the KV cache (`llama_memory_seq_rm`), reducing memory footprint without degrading the generation quality of unrelated subtrees.

### Key Individual Contributions:
- **Domain 7 (JSON / API Responses):** Designed and implemented [`parsers/json/parser.py`](parsers/json/parser.py) to parse arbitrary nested JSON objects, arrays, and primitive fields into hierarchical trees with semantic leaf boundaries and token merge rules. Built 5 representative API response documents (`flight_search`, `github_repo`, `movie_db`, `product_catalog`, `weather_api`).
- **Domain 12 (Agent / Chat Logs):** Designed and implemented [`parsers/chatlogs/parser.py`](parsers/chatlogs/parser.py) to model conversational interactions and multi-step tool invocations as `turn` $\to$ `tool_call` $\to$ `tool_result` trees. Built 5 multi-turn tool interaction datasets (`flight_booking`, `repo_issue`, `restaurant_reserve`, `shopping_return`, `weather_lookup`).
- **Verification & Test Harness:** Designed 10 evaluation test cases per domain ([`tests/cases_json.json`](tests/cases_json.json) and [`tests/cases_chatlogs.json`](tests/cases_chatlogs.json)). Achieved **100% pass rate (10/10 on Test 1 and 10/10 on Test 2 across both domains)**, demonstrating bit-identical logit invariance under edits and guaranteed KV cache reduction upon subtree eviction.

---

## Table of Contents

| Section | Description |
|---|---|
| [1. Invariant Architecture](#1-invariant-architecture) | Formal sequence ID tagging, position allocation, and KV cache operations |
| [2. Domain 7: JSON / API Responses](#2-domain-7-json--api-responses) | Parser design, document trees, and structural statistics |
| [3. Domain 12: Agent / Chat Logs](#3-domain-12-agent--chat-logs) | Conversational & tool-trace hierarchy, parser implementation, and statistics |
| [4. Test Evaluation & Empirical Results](#4-test-evaluation--empirical-results) | Test 1 (Edit-B, Ask-A), Test 2 (Evict-B, Ask-A), and flat control divergence |
| [5. Benchmarking & Efficiency Analysis](#5-benchmarking--efficiency-analysis) | Prefill throughput, generation speeds, and scoped context efficiency |
| [6. Collaborative Team Foundations](#6-collaborative-team-foundations) | Shared C++ driver, schema, and cross-domain verification |
| [7. Reproduction & Execution Guide](#7-reproduction--execution-guide) | Step-by-step commands to build, parse, test, and benchmark |

---

## 1. Invariant Architecture

### 1.1 Sequence ID Hierarchy
`llama.cpp` evaluates attention masks by comparing the sequence IDs assigned to the query token against the sequence IDs stored in each KV cache cell. To ensure that token $t \in \text{Node}(N)$ attends exclusively to its ancestors:
$$\text{seq\_ids}(N) = \{\text{seq}(N)\} \cup \text{descendants}(N)$$
The query token uses its primary sequence ID $\text{seq}(N)$, which matches only the sequence sets of itself and its ancestors. Sibling subtrees have disjoint descendant sets and cannot attend to one another.

### 1.2 Tree Position Allocation
To avoid position gaps while preserving sequence continuity:
$$\text{pos}(\text{child.first}) = \text{pos}(\text{parent.last}) + 1$$
Sibling subtrees occupy overlapping token position ranges $[P_{\text{start}}, P_{\text{end}}]$. Because attention is isolated via sequence IDs, this overlap introduces zero cross-talk, while making KV cache eviction of a sibling completely position-invariant to remaining siblings.

### 1.3 Sequence Budget & Merging
Hardware and `llama.cpp` constraints limit sequence IDs to $< 63$ (sequence 63 is reserved for the scoped query). To guarantee this invariant:
- Subtrees smaller than the minimum token threshold (e.g. 15–20 tokens) are recursively merged into their parent containers.
- Documents are constrained to $\le 2000$ tokens and $\le 14$ logical nodes after merging.

---

## 2. Domain 7: JSON / API Responses

### 2.1 Parser Design (`parsers/json/parser.py`)
API endpoints return nested JSON with dictionaries, lists of objects, and scalar attributes. The JSON parser parses these hierarchies into:
1. **Document Preamble:** Top-level context (endpoint / API metadata).
2. **Container Nodes:** JSON objects (`dict`) and arrays (`list`).
3. **Leaf Nodes:** Scalar key-value pairs or compact item descriptions (`"price": 450`, `"status": "confirmed"`).
4. **Merge Logic:** Merges leaf attributes into parent object nodes if individual leaves fall below token size thresholds.

### 2.2 Documents Evaluated
Five realistic API response payloads were constructed and tokenized:
- `flight_search.json`: Multi-carrier flight options, layovers, price classes, and booking policies.
- `github_repo.json`: Repository metadata, open pull requests, CI/CD status, and release information.
- `movie_db.json`: Film cast, box office figures, IMDb/Rotten Tomatoes ratings, and production details.
- `product_catalog.json`: E-commerce catalog with SKU specifications, stock status, reviews, and tier discounts.
- `weather_api.json`: Multi-city weather observations, hourly forecasts, wind speeds, and precipitation levels.

### 2.3 Structural Statistics
*Generated via `tools/report.py --domain json` on CPU:*

| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup Share | Prefill flat / tree (ms) | Gen tok/s |
|---|---|---|---|---|---|---|---|
| `flight_search.json` | 9 (39) | 251 | 2 | 20 / 34.5 / 67 | 54% | 1487 / 1510 | 0.7* |
| `github_repo.json` | 11 (48) | 294 | 2 | 20 / 34.5 / 108 | 52% | 272 / 322 | 44.7 |
| `movie_db.json` | 9 (35) | 209 | 2 | 18 / 28.5 / 62 | 54% | 179 / 150 | 90.6 |
| `product_catalog.json` | 10 (39) | 230 | 2 | 18 / 26.0 / 68 | 54% | 146 / 254 | 55.3 |
| `weather_api.json` | 14 (60) | 361 | 3 | 13 / 30.0 / 117 | 57% | 425 / 231 | 117.7 |
| **Total / Aggregate** | **53 (221)** | **1,345** | **3** | **13 / 30 / 117** | **54.4%** | **969 / 988 tok/s** | **61.8** |

*\*Note: flight_search cold-start overhead included.*

---

## 3. Domain 12: Agent / Chat Logs

### 3.1 Parser Design (`parsers/chatlogs/parser.py`)
Agent conversation logs capture multi-turn user/assistant exchanges interspersed with external tool calls (actions) and tool outputs (observations). The chat-log parser models this flow as:
1. **Session Root:** Agent persona, system instructions, and initial session context.
2. **Turn Nodes:** Individual user prompt and assistant thoughts.
3. **Tool Call Nodes:** Function invocation name and parameters (`call: search_flights(...)`).
4. **Tool Result Nodes:** Environment observation or returned data (`result: 2 flights found`).

### 3.2 Documents Evaluated
Five real-world multi-step agent interaction traces were constructed:
- `flight_booking.json`: Airline reservation agent handling seat selection, luggage add-ons, and payment confirmation.
- `repo_issue.json`: DevOps assistant triaging bug reports, querying Git commits, and triggering CI workflows.
- `restaurant_reserve.json`: Dining concierge finding table availability, dietary constraints, and booking time slots.
- `shopping_return.json`: Customer support bot verifying order eligibility, generating return labels, and processing refunds.
- `weather_lookup.json`: Weather assistant retrieving radar summaries, sensor logs, and precipitation advisories.

### 3.3 Structural Statistics
*Generated via `tools/report.py --domain chatlogs` on CPU:*

| Document | Nodes (raw) | Tokens | Depth | Subtree tokens min/med/max | Markup Share | Prefill flat / tree (ms) | Gen tok/s |
|---|---|---|---|---|---|---|---|
| `flight_booking.json` | 9 (17) | 239 | 2 | 18 / 34.5 / 69 | 28% | 1585 / 1458 | 1.2* |
| `repo_issue.json` | 9 (17) | 207 | 2 | 17 / 27.0 / 58 | 25% | 227 / 279 | 47.8 |
| `restaurant_reserve.json` | 9 (17) | 201 | 2 | 13 / 29.0 / 55 | 29% | 197 / 129 | 119.5 |
| `shopping_return.json` | 9 (17) | 207 | 2 | 11 / 31.5 / 58 | 27% | 226 / 271 | 74.5 |
| `weather_lookup.json` | 9 (17) | 204 | 2 | 11 / 29.0 / 56 | 28% | 184 / 158 | 50.7 |
| **Total / Aggregate** | **45 (85)** | **1,058** | **2** | **11 / 29 / 58** | **27.5%** | **821 / 903 tok/s** | **58.7** |

---

## 4. Test Evaluation & Empirical Results

The driver rigorously validates tree isolation and cache manipulation using two core tests:

### 4.1 Test 1: Edit Subtree B, Query Subtree A
In this scenario, Subtree **B** (preceding sibling) is edited, and the model is queried about Subtree **A**:
- **Same-Length Edit:** B's text is modified with identical token length so no KV positions shift. **Pass Condition:** The scoped query over A produces **bit-identical logits (diff = 0.0)** and identical text tokens.
- **Flat Control Verification:** Running the same edit over a conventional flat context must produce a non-zero logit difference ($\Delta > 0$). This guarantees that the test is sensitive to interference, and that tree attention completely shields A from B's modifications.
- **Free-Length Edit:** B's text is modified with real text of arbitrary length. **Pass Condition:** A's scoped answer remains correct and unaltered.

### 4.2 Test 2: Evict Subtree B, Query Subtree A
In the same context, Subtree B is removed from the KV cache using `llama_memory_seq_rm`.
- **Pass Condition:** The KV cache memory drops by the exact byte count corresponding to B's tokens, and the query over A produces the exact same answer.

### 4.3 Summary Table (10 Cases per Domain)

| Test Metric | Domain 7 (JSON / API) | Domain 12 (Chat Logs) |
|---|---|---|
| Total Test Cases | 10 | 10 |
| Baseline Answer Correctness | **10 / 10** | **10 / 10** |
| **Test 1 Pass Rate** | **10 / 10 (100%)** | **10 / 10 (100%)** |
| - Same-Length Edit: Bit-Identical Logits (diff = 0.0) | **10 / 10** | **10 / 10** |
| - Flat Control Interference Detected ($\Delta > 0$) | **10 / 10** | **10 / 10** |
| - Free-Length Edit: Answer Unchanged | **10 / 10** | **10 / 10** |
| **Test 2 Pass Rate** | **10 / 10 (100%)** | **10 / 10 (100%)** |
| - KV Cache Memory Eviction Verified | **10 / 10** | **10 / 10** |
| - Scoped Answer Correctness Post-Eviction | **10 / 10** | **10 / 10** |

All detailed test outputs are recorded in [`results/test_results_json.json`](results/test_results_json.json) and [`results/test_results_chatlogs.json`](results/test_results_chatlogs.json).

---

## 5. Benchmarking & Efficiency Analysis

Structured tree attention introduces significant context and compute savings:

1. **Context Scoping Efficiency:**
   - In **JSON / API**, scoped questions observe an average of only **25.3%** of the full document tokens.
   - In **Chat Logs**, scoped questions observe an average of only **29.1%** of the total session tokens.
   - Queries avoid reading irrelevant sibling subtrees, dramatically cutting down quadratic self-attention computation.

2. **Prefill Throughput:**
   - JSON Prefill: **987.9 tok/s** (tree) vs. **969.0 tok/s** (flat)
   - Chatlogs Prefill: **902.8 tok/s** (tree) vs. **820.6 tok/s** (flat)
   - Hierarchical prefill in a single `llama_decode` call exhibits no throughput penalty over flat prefill while unlocking tree-isolation capabilities.

3. **Subtree KV Eviction:**
   - Subtree eviction directly frees KV memory (`llama_memory_seq_rm`), yielding immediate cache reclamation without requiring a full re-prefill of the document.

---

## 6. Collaborative Team Foundations

This repository incorporates the shared infrastructure established collaboratively across the project team:
- **Shared C++ Driver:** [`driver/ldls_driver.cpp`](driver/ldls_driver.cpp) providing the CLI commands `twoseq`, `run`, `test`, `bench`, `noise`, and `tokenize`.
- **JSON Schema:** [`schema/tree.schema.json`](schema/tree.schema.json) and documentation [`schema/SCHEMA.md`](schema/SCHEMA.md).
- **Automated Verification:** [`tests/run_tests.py`](tests/run_tests.py) and [`tools/report.py`](tools/report.py).
- **Collaborator Domains:** Cross-verified alongside Domain 3 (Markdown), Domain 1 (HTML), and Domain 8 (XML).

---

## 7. Reproduction & Execution Guide

### 7.1 Clone and Setup
```bash
git clone https://github.com/aditya25058/ldls-project.git
cd ldls-project

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt cmake ninja
```

### 7.2 Clone llama.cpp and Download Model
```bash
# Clone pinned llama.cpp commit
git init llama.cpp
cd llama.cpp
git remote add origin https://github.com/ggml-org/llama.cpp
git -c core.longpaths=true fetch --depth 1 origin 08ffba21e5c2f1728b6b405c05ff3913f65706d4
git -c core.longpaths=true checkout FETCH_HEAD
cd ..

# Download Llama-3.2-1B-Instruct GGUF
mkdir -p models
curl -L -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

### 7.3 Build the C++ Driver
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ldls_driver -j
```

### 7.4 Run Tests for Domain 7 & Domain 12
```bash
# Run Domain 7 (JSON / API)
python tests/run_tests.py --domain json

# Run Domain 12 (Agent / Chat Logs)
python tests/run_tests.py --domain chatlogs
```

### 7.5 Generate Reports & Timing Statistics
```bash
# Generate report for JSON
python tools/report.py --domain json

# Generate report for Chat Logs
python tools/report.py --domain chatlogs
```

---

## 8. Artifacts & Deliverables

- **Git Repository:** [https://github.com/aditya25058/ldls-project](https://github.com/aditya25058/ldls-project)
- **Submission ZIP Archive:** `ldls-project-aditya25058.zip` (clean archive containing driver source, parsers, test suites, schemas, reports, and documentation; excludes build/model binaries).
- **HPC Cluster Directory:** `/home/bapic_iiitd/2_group/aditya/ldls-project/` (PARAM Rudra).
