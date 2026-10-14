# LDLS: structured sequence IDs for efficient attention (week 1)

This project splits documents into trees and loads them into llama.cpp so that **each subtree attends only to its ancestors**. Questions can see the whole document or just one subtree. Whole subtrees can be **evicted** from the KV cache. Everything runs on CPU with Llama-3.2-1B-Instruct (Q4_K_M).

Three domains are implemented: **Markdown** (#3), **HTML** (#1) and **XML** (#8). They all use the same driver, schema and tests.

## Contents

| Part | Path |
|---|---|
| Shared C++ driver (based on llama.cpp `batched`) | [`driver/ldls_driver.cpp`](driver/ldls_driver.cpp) |
| JSON schema and merge rules | [`schema/tree.schema.json`](schema/tree.schema.json), [`schema/SCHEMA.md`](schema/SCHEMA.md) |
| Shared parser code (tokenizing, merge rules, JSON output) | [`parsers/common.py`](parsers/common.py) |
| Markdown: parser, 5 documents, parsed trees | [`parsers/markdown/`](parsers/markdown) (`parser.py`, `docs/`, `json/`) |
| HTML: parser, 5 documents, parsed trees | [`parsers/html/`](parsers/html) |
| XML: parser, 5 documents, parsed trees | [`parsers/xml/`](parsers/xml) |
| Tests (change B / remove B, ask about A), 10 cases per domain | [`tests/run_tests.py`](tests/run_tests.py), `tests/cases_<domain>.json` |
| Tree statistics and CPU timings | [`tools/report.py`](tools/report.py) |
| Results from the original machine | [`results/`](results) |
| How llama.cpp builds the mask (trace notes) | [`docs/llama_cpp_notes.md`](docs/llama_cpp_notes.md) |

---

## 1. What you need

| | |
|---|---|
| **Disk** | about 2 GB: llama.cpp source 220 MB, model 808 MB, build about 100 MB |
| **Memory** | 2 GB of free RAM |
| **CPU** | any 64-bit CPU. No GPU needed. |
| **Internet** | once, to download llama.cpp and the model |
| **Time** | about 15 minutes of setup (mostly downloads and the build) |

Tools, by operating system:

| | Install |
|---|---|
| **Windows 10/11** | [Git for Windows](https://git-scm.com/download/win) (includes **Git Bash**, used for every command below) · [Python 3.10+](https://www.python.org/downloads/) (tick *"Add python.exe to PATH"* in the installer) · the w64devkit C++ compiler (step 5) |
| **Ubuntu / Debian / WSL** | `sudo apt install git cmake g++ curl python3-venv python3-pip` |
| **macOS** | `xcode-select --install`, then `brew install cmake python` |

Tested on **Windows 11** (w64devkit GCC 16.2, Python 3.10) and **Ubuntu 24.04 under WSL** (GCC 13.3, Python 3.12). macOS and Visual Studio's MSVC compiler should work but have not been tested.

> **Windows tips.** Put the project folder at a **short path outside OneDrive/Dropbox/Google Drive**, for example `C:\ldls-project`. The build writes thousands of files that sync software would upload, and llama.cpp contains very long file paths.

---

## 2. Setup (once per machine)

Open a terminal **inside the project folder**: Git Bash on Windows, a normal terminal on Linux or macOS. All commands below are run from there. When you are done, the folder looks like this:

```
LDLS PROJECT/
├── llama.cpp/     ← step 2 (you clone this)
├── models/        ← step 3 (you download the model)
├── build/         ← step 5 (created by the build)
├── .venv/         ← step 4, Linux/macOS only
└── driver/  parsers/  schema/  tests/  tools/  results/  docs/
```

### Step 1: get the project folder

Unzip it, or `git clone` it if it is on GitHub, then `cd` into it.

### Step 2: get llama.cpp (pinned version)

llama.cpp changes its API often, so use **exactly** the commit this project was tested with:

```bash
git init llama.cpp
cd llama.cpp
git remote add origin https://github.com/ggml-org/llama.cpp
git -c core.longpaths=true fetch --depth 1 origin 08ffba21e5c2f1728b6b405c05ff3913f65706d4
git -c core.longpaths=true checkout FETCH_HEAD
cd ..
```

This downloads only that one commit (about 220 MB). `core.longpaths` prevents "Filename too long" errors on Windows.

### Step 3: download the model (808 MB, no login needed)

```bash
mkdir -p models
curl -L -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

The file must be exactly 807,694,464 bytes. If it is a few KB, the download failed; run it again.

### Step 4: Python packages

**Windows** (CMake and Ninja come from pip too):

```bash
python -m pip install -r requirements.txt cmake ninja
```

**Linux / macOS** (use a virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in every new terminal. After that, `python` refers to the venv's Python.

### Step 5: build the driver

**Windows.** Download `w64devkit-x64-2.10.0.7z.exe` from the [w64devkit releases page](https://github.com/skeeto/w64devkit/releases). It is a portable GCC compiler, about 100 MB, and needs no admin rights. Run it and extract it to `C:\`, which gives you `C:\w64devkit`. Then, in Git Bash inside the project folder:

```bash
export PATH="/c/w64devkit/bin:$(cygpath "$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"):$PATH"
g++ --version        # must show w64devkit's GCC (16.x), not an old MinGW 6.x
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build --target ldls_driver
```

The `export PATH=...` line puts w64devkit and pip's `cmake`/`ninja` on the PATH. It lasts only for the current terminal, so repeat it if you rebuild later.

**Linux / macOS:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ldls_driver -j
```

The build takes 1.5–3 minutes and produces `build/ldls_driver` (`build/ldls_driver.exe` on Windows). It is optimized for **your** CPU, so don't copy the binary to another machine: everyone builds their own.

### Step 6: check that it works (about 10 seconds)

```bash
build/ldls_driver twoseq -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  -p1 "The three primary colors are" -p2 "To make a cup of tea, first"
```

This decodes two prompts in two sequences of one batch, then each prompt alone. The last line must be `"pass": true`.

---

## 3. Running the project

Run these from the project folder. On Linux/macOS, activate the venv first.

Every step takes the domain: `markdown`, `html` or `xml`.

| # | What | Command | Time* |
|---|---|---|---|
| 1 | Parse the 5 documents of a domain into the schema | `python parsers/<domain>/parser.py "parsers/<domain>/docs/*" -o parsers/<domain>/json` | 5 s |
| 2 | Ask a question, scoped to one subtree, then evict another subtree | see below | 15 s |
| 3 | Run both tests on the domain's 10 cases | `python tests/run_tests.py --domain <domain>` | 6–8 min |
| 4 | Tree statistics and CPU timings | `python tools/report.py --domain <domain>` | 5 min |

\* on a 12-thread laptop CPU.

Step 1 is optional: the parsed trees are already in `parsers/<domain>/json/`, and re-running it with the same model gives identical files. Keep the quotes around the `*` pattern; they make it work in every shell, PowerShell included.

Step 2, an example question for each domain:

```bash
M=models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
build/ldls_driver run -m $M -d parsers/markdown/json/orbit_cli.json \
  -q "What is the default number of upload workers?" --scope Configuration --evict Troubleshooting
build/ldls_driver run -m $M -d parsers/html/json/brightloom_landing.json \
  -q "How much does the Team plan cost per user per month?" --scope "section#pricing" --evict "section#faq"
build/ldls_driver run -m $M -d parsers/xml/json/harbor_gateway_config.json \
  -q "Which port does the admin server listen on?" --scope "server: admin" --evict features
```

Each prints the answer before and after the eviction, plus KV memory and timings, as JSON. Node titles are listed in the JSON files (`"title"`).

Step 3 prints one line per case and a summary. Results go to `results/test_results_<domain>.json`, and the command exits with code 0 only if `test1_pass` and `test2_pass` are both 10. The `test2_full_scope_*` counts are informational: a question that sees the *whole* document is allowed to change once B is gone. Step 4 writes `results/report_<domain>.md` and `.json`.

### Driver commands

All commands take `-m <model.gguf>`. Nodes are addressed by id or by title (case-insensitive).

| Command | Purpose | Main options |
|---|---|---|
| `twoseq` | Two prompts in two sequences vs. separate runs | `-p1`, `-p2`, `-n` tokens |
| `run` | Prefill a document, answer, optionally edit or evict | `-d doc.json`, `-q question`, `--scope NODE`, `--evict NODE` (repeatable), `--edit NODE --edit-text TEXT`, `--per-node`, `-n` |
| `test` | Test 1 and Test 2 for one case | `-d`, `-q`, `--a NODE`, `--b NODE`, `--edit-text TEXT` |
| `bench` | Flat vs. tree prefill timing | `-d`, `-r` repetitions, `-q`, `-n` |
| `noise` | Numerical noise floor of the logits | `-d`, `-q`, `--a`, `--b` |
| `tokenize` | stdin `[{"text":..,"special":bool}]` → token ids and pieces | |

Options for every command: `-t <threads>` (default: half the logical cores), `--fa on|off|auto` (flash attention), `-v` (verbose llama.cpp logs).

### Where the Python tools look for the driver and model

1. The environment variables `LDLS_DRIVER` and `LDLS_MODEL`, if set.
2. `build/ldls_driver[.exe]` and `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` inside the project folder.

If your build or model is somewhere else, set the variables, e.g. `export LDLS_MODEL=/path/to/model.gguf`. In PowerShell: `$env:LDLS_MODEL = "C:\path\model.gguf"`. To build against a llama.cpp checkout elsewhere, add `-DLLAMA_CPP_DIR=<path>` to the first `cmake` command.

---

## 4. Troubleshooting

| Problem | Fix |
|---|---|
| `Filename too long` during `git checkout` (Windows) | Use the `-c core.longpaths=true` commands from step 2, and/or move the project to a shorter path such as `C:\ldls-project`. |
| `RPC failed; curl 56 GnuTLS recv error` (Linux/WSL) | A network problem between git and GitHub. Retry, switch networks or turn off a VPN. On WSL you can also run step 2 with Windows Git Bash in the same folder. |
| `ensurepip is not available` when creating the venv | `sudo apt install python3-venv python3-pip` |
| `cmake: command not found` / `ninja` not found (Windows) | Run the `export PATH=...` line from step 5 in the same terminal. |
| Compiler errors about C++17 or `std::...`, or `g++ --version` shows 6.x | An old MinGW compiler is being picked up. Install w64devkit and run the `export PATH` line so it comes first. |
| `llama.cpp not found at ...` from CMake | Do step 2, or pass `-DLLAMA_CPP_DIR=<path to llama.cpp>`. |
| Build errors inside llama.cpp, or "no member named llama_memory_seq_rm" | Wrong llama.cpp version. Use the pinned commit from step 2, then delete `build/` and rebuild. |
| `The driver (ldls_driver) not found` / `The model ... not found` | Finish steps 3 and 5, or set `LDLS_DRIVER` / `LDLS_MODEL`. The message lists where it looked. |
| `failed to load model` | The download is incomplete; check the size from step 3. |
| `python: command not found` (Linux/macOS) | Activate the venv (`source .venv/bin/activate`) or use `python3`. |

### Will my results match yours exactly?

**The pass/fail results should, but the exact numbers may not.** Different CPUs use different vector instructions (AVX2, AVX-512, ARM NEON), which changes the last bits of the floating-point results. Answer wording may differ slightly, and timings depend on your CPU. Each test compares two runs **on the same machine**, so all tests should still pass everywhere. Example: on Ubuntu (WSL) the parser produced identical token files, and all 10 cases passed with the same answers as on Windows. Only some flat-control logit differences and one informational count (`test2_full_scope_unchanged`: 8 vs. 9) differed.

---

## 5. How it works

1. **Parse.** The parser turns a document into a tree and tokenizes every node with the Llama tokenizer (`ldls_driver tokenize`). Small subtrees are merged to stay within the sequence budget.
2. **Tag.** Each node gets its own sequence ID. Every token of node N is tagged with N's ID plus the IDs of all of N's descendants. llama.cpp masks attention using the query token's *first* sequence ID, so a token of N sees only its ancestor path.
3. **Position.** A child starts at its parent's last position + 1, so siblings overlap. This keeps each sequence continuous (a llama.cpp requirement) and makes a subtree's positions independent of its siblings.
4. **Prefill.** The whole tree is decoded in pre-order in a single `llama_decode` call.
5. **Ask.** A reserved query sequence (63) is attached to the cells in scope with `llama_memory_seq_cp`, then the question is decoded greedily.
6. **Evict.** `llama_memory_seq_rm` is called for every sequence of the subtree over the subtree's position range. The freed cells return to the cache.

Details: [`schema/SCHEMA.md`](schema/SCHEMA.md) and [`docs/llama_cpp_notes.md`](docs/llama_cpp_notes.md).

## 6. The tests

For each case, the question targets subtree **A**, and **B** is a separate subtree placed *before* A in the document.

* **Test 1: change B, ask about A.**
  * *Same-length edit*: B gets new content with the same token count, so no KV cell moves. **Pass = A's answer tokens are identical and its first-step logits are bit-identical.**
  * *Free-length edit*: B gets its real new text. **Pass = identical answer.**
  * *Control*: the same edits on an ordinary flat prompt must change the logits. This shows the test can actually detect interference.
* **Test 2: remove B.** In the same context, evict B's subtree and ask again. **Pass = identical answer and a smaller KV state.**

Why not require bit-identical logits everywhere? Moving cells to different KV slots changes the floating-point summation order. The 8-bit activation quantization used with Q4 weights amplifies that to first-step logit differences of about 0.5. The `noise` command measures this: the same state gives exactly 0, and a re-layout gives about 0.5. The same-length variant avoids re-layout, so it can demand exact equality.

### Results (original machine, 10 cases per domain)

| | Markdown | HTML | XML |
|---|---|---|---|
| Documents (tokens each) | 5 (499–589) | 5 (822–1,242) | 5 (408–951) |
| Nodes after merging (before) | 68 (123) | 135 (308) | 108 (243) |
| Max tree depth | 3 | 7 | 6 |
| Markup / syntax tokens | 19% | 76% | 64% |
| Share of tokens a scoped question sees | 24% | 19% | 27% |
| Answers correct | 10/10 | 10/10 | 10/10 |
| **Test 1, same-length edit: A bit-identical** | **10/10** | **10/10** | **10/10** |
| Test 1 control: flat prompt changes (logit Δ) | 10/10 (0.9–6.0) | 10/10 (0.7–4.7) | 10/10 (0.9–4.5) |
| Test 1, real edit: identical answer wording | 10/10 | 9/10 | 8/10 |
| Test 2: identical answer wording after eviction | 10/10 | 9/10 | 9/10 |
| Test 2: KV memory drops | 10/10 (4–14%) | 10/10 (3–10%) | 10/10 (5–13%) |
| CPU prefill, flat / tree (tokens/s) | 113 / 119 | 127 / 130 | 124 / 126 |
| CPU generation (tokens/s) | 17 | 18 | 19 |

The misses in the two "identical wording" rows are **wording only**. For example, *"Office hours: Tuesdays 14:00–15:30…"* became *"Office hours are Tuesdays 14:00–15:30…"*, and *"Birds spent 71 percent…"* became *"71%"*. Each comes with a first-step logit shift of 0.5–0.6, the measured noise floor from moving cells. The answers stay correct, and the noise-free check (same-length edit) is exact in all 30 cases. Under the strict rule, `run_tests.py` therefore exits with code 1 for HTML and XML.

## 7. Adding a domain

1. Create `parsers/<domain>/parser.py`. Write one function, `build_raw_tree(text)`, that returns a tree of `common.Node`s: the root is `Node("document", PREAMBLE)`, structural nodes get `container=True`, and every node's `text` is its own text (without its children's). Then call `run_cli("<domain>", build_raw_tree, classify, __doc__)`. [`parsers/common.py`](parsers/common.py) does the rest: tokenizing, merging small subtrees, sequence IDs, tree positions, the 2,000-token check and the JSON output. The three existing parsers are short examples. `parsers/markdown/parser.py` is the simplest; HTML and XML show how to flag markup exactly with `segments`.
2. Put 5 documents in `parsers/<domain>/docs/` and parse them into `parsers/<domain>/json/`.
3. Write `tests/cases_<domain>.json` (same format as the existing case files), then run `python tests/run_tests.py --domain <domain>` and `python tools/report.py --domain <domain>`.

The driver needs no changes. It validates every JSON file when it loads it.

## 8. Sharing this folder

Share the folder **without** `llama.cpp/`, `models/`, `build/` and `.venv/`. They are large and machine-specific, and every person creates them with the setup steps above (`.gitignore` already excludes them).
