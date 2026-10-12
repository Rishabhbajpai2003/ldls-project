// ldls_driver: structured-sequence-ID driver for llama.cpp (CPU).
//
// A document is a tree of nodes (see schema/tree.schema.json). Every token of
// node N is stored once in the KV cache and tagged with the sequence ID of N
// plus the sequence IDs of all of N's descendants. llama.cpp builds the KQ mask
// from the *first* seq ID of each query token (llama-kv-cache.cpp,
// set_input_kq_mask_impl), so a token of N attends exactly to its ancestor
// path and to earlier tokens of N -- never to sibling subtrees.
//
// Positions are tree positions: a child starts at its parent's last position
// + 1, so siblings overlap. This is required by llama.cpp's batch validation
// (each sequence must have continuous positions) and makes a subtree's
// positions independent of its siblings' lengths.
//
// Questions are decoded in a dedicated query sequence Q. Before asking, Q is
// attached to the cells in scope with llama_memory_seq_cp (metadata only in a
// unified KV cache), so a question can see the whole document or one subtree.
//
// Commands:
//   tokenize  stdin JSON [{"text":..,"special":bool}] -> stdout JSON [{"ids":[..],"pieces":[..]}]
//   twoseq    two prompts in two sequences vs. two separate runs
//   run       prefill a document, answer, optionally edit / evict subtrees
//   test      test 1 (edit B, ask about A) and test 2 (evict B, ask about A)
//   bench     CPU timings: flat vs. structured prefill

#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

static bool g_verbose = false;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

//
// command line
//

struct Args {
    std::string cmd;
    std::map<std::string, std::vector<std::string>> kv;

    bool has(const std::string & k) const { return kv.count(k) > 0; }

    std::string get(const std::string & k, const std::string & def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second.back();
    }

    std::string req(const std::string & k) const {
        if (!has(k)) {
            throw std::runtime_error("missing required argument " + k);
        }
        return get(k);
    }

    std::vector<std::string> all(const std::string & k) const {
        auto it = kv.find(k);
        return it == kv.end() ? std::vector<std::string>{} : it->second;
    }
};

static Args parse_args(int argc, char ** argv) {
    Args a;
    if (argc < 2) {
        throw std::runtime_error("missing command (tokenize | twoseq | run | test)");
    }
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind('-', 0) != 0) {
            throw std::runtime_error("unexpected argument: " + k);
        }
        if (k == "-v" || k == "--per-node") { // boolean flags
            a.kv[k].push_back("1");
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for " + k);
        }
        a.kv[k].push_back(argv[++i]);
    }
    return a;
}

//
// model / context
//

static const int N_SEQ_MAX = 64;             // node seq IDs are 0 .. N_SEQ_MAX-2
static const int QUERY_SEQ = N_SEQ_MAX - 1;  // reserved for questions

struct Engine {
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    // flash attention: "auto" (llama.cpp default), "on" or "off"
    static inline std::string flash_attn = "auto";

    Engine(const std::string & path, int n_ctx, int n_threads, bool vocab_only = false) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = vocab_only;
        model = llama_model_load_from_file(path.c_str(), mp);
        if (!model) {
            throw std::runtime_error("failed to load model " + path);
        }
        vocab = llama_model_get_vocab(model);
        if (vocab_only) {
            return;
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = n_ctx;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = N_SEQ_MAX;
        cp.kv_unified      = true; // one KV buffer; a cell can belong to many sequences
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        cp.no_perf         = true;
        if (flash_attn == "on") {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (flash_attn == "off") {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            throw std::runtime_error("failed to create llama_context");
        }
    }

    ~Engine() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    llama_memory_t mem() const { return llama_get_memory(ctx); }

    // bytes needed to serialize the context state; dominated by the K/V data of used cells
    size_t kv_bytes() const { return llama_state_get_size(ctx); }
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    int n = llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, false, special);
    if (n >= 0) {
        return {};
    }
    std::vector<llama_token> out(-n);
    n = llama_tokenize(vocab, text.data(), (int) text.size(), out.data(), (int) out.size(), false, special);
    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }
    out.resize(n);
    return out;
}

static std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
    if (n < 0) {
        std::string s(-n, '\0');
        llama_token_to_piece(vocab, t, &s[0], -n, 0, false);
        return s;
    }
    return std::string(buf, n);
}

//
// document tree
//

struct Node {
    int         id     = -1;
    int         parent = -1;
    std::string type;
    std::string title;

    std::vector<llama_token>  tokens;
    std::vector<llama_pos>    pos;
    std::vector<llama_seq_id> seq_ids; // [0] is the node's own sequence
    std::vector<int>          children;

    bool evicted = false;

    llama_seq_id seq() const { return seq_ids[0]; }
    llama_pos    p0()  const { return pos.front(); }
    llama_pos    p1()  const { return pos.back() + 1; }
};

struct Doc {
    std::string       name;
    std::vector<Node> nodes;
};

static Doc load_doc(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    json j = json::parse(f);

    Doc doc;
    doc.name = j.value("source", path);
    for (const auto & jn : j.at("nodes")) {
        Node n;
        n.id      = jn.at("id").get<int>();
        n.parent  = jn.at("parent").is_null() ? -1 : jn.at("parent").get<int>();
        n.type    = jn.at("node_type").get<std::string>();
        n.title   = jn.value("title", "");
        n.tokens  = jn.at("tokens").get<std::vector<llama_token>>();
        n.pos     = jn.at("positions").get<std::vector<llama_pos>>();
        n.seq_ids = jn.at("seq_ids").get<std::vector<llama_seq_id>>();
        if (n.id != (int) doc.nodes.size()) {
            throw std::runtime_error("node ids must be 0..N-1 in order");
        }
        doc.nodes.push_back(std::move(n));
    }
    for (auto & n : doc.nodes) {
        if (n.parent >= 0) {
            doc.nodes[n.parent].children.push_back(n.id);
        }
    }
    return doc;
}

static void collect_subtree(const Doc & doc, int id, std::vector<int> & out) {
    out.push_back(id);
    for (int c : doc.nodes[id].children) {
        collect_subtree(doc, c, out);
    }
}

static std::vector<int> subtree(const Doc & doc, int id) {
    std::vector<int> out;
    collect_subtree(doc, id, out);
    return out;
}

static std::vector<int> ancestors(const Doc & doc, int id) {
    std::vector<int> out;
    for (int p = doc.nodes[id].parent; p >= 0; p = doc.nodes[p].parent) {
        out.push_back(p);
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// checks the invariants the driver relies on; throws with a precise message otherwise
static void validate(const Doc & doc) {
    const auto & N = doc.nodes;
    if (N.empty() || N[0].parent != -1) {
        throw std::runtime_error("node 0 must be the root (parent null)");
    }
    std::set<llama_seq_id> seen;
    std::vector<std::set<llama_seq_id>> expect(N.size());
    for (int i = (int) N.size() - 1; i >= 0; --i) {
        const Node & n = N[i];
        if (i > 0 && (n.parent < 0 || n.parent >= i)) {
            throw std::runtime_error("node " + std::to_string(i) + ": parent must precede the node (pre-order)");
        }
        if (n.tokens.empty() || n.tokens.size() != n.pos.size()) {
            throw std::runtime_error("node " + std::to_string(i) + ": tokens/positions empty or of different length");
        }
        if (n.seq_ids.empty() || n.seq() < 0 || n.seq() >= QUERY_SEQ || !seen.insert(n.seq()).second) {
            throw std::runtime_error("node " + std::to_string(i) + ": own seq id missing, out of range or duplicated");
        }
        expect[i].insert(n.seq());
        for (int c : n.children) {
            expect[i].insert(expect[c].begin(), expect[c].end());
        }
        if (std::set<llama_seq_id>(n.seq_ids.begin(), n.seq_ids.end()) != expect[i] || n.seq_ids.size() != expect[i].size()) {
            throw std::runtime_error("node " + std::to_string(i) + ": seq_ids must be own seq + all descendant seqs");
        }
        const llama_pos start = n.parent < 0 ? 0 : N[n.parent].pos.back() + 1;
        for (size_t k = 0; k < n.pos.size(); ++k) {
            if (n.pos[k] != start + (llama_pos) k) {
                throw std::runtime_error("node " + std::to_string(i) + ": positions must start at parent end + 1 and be consecutive");
            }
        }
    }
}

// recompute tree positions after a node's tokens changed
static void relayout(Doc & doc) {
    for (auto & n : doc.nodes) {
        const llama_pos start = n.parent < 0 ? 0 : doc.nodes[n.parent].pos.back() + 1;
        n.pos.resize(n.tokens.size());
        for (size_t k = 0; k < n.pos.size(); ++k) {
            n.pos[k] = start + (llama_pos) k;
        }
    }
}

// node reference: numeric id, or a title (case-insensitive, must be unique)
static int find_node(const Doc & doc, const std::string & ref) {
    if (!ref.empty() && std::all_of(ref.begin(), ref.end(), [](unsigned char c) { return std::isdigit(c); })) {
        int id = std::stoi(ref);
        if (id < 0 || id >= (int) doc.nodes.size()) {
            throw std::runtime_error("node id out of range: " + ref);
        }
        return id;
    }
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        return s;
    };
    std::vector<int> hits;
    for (const auto & n : doc.nodes) {
        if (lower(n.title) == lower(ref)) {
            hits.push_back(n.id);
        }
    }
    if (hits.size() != 1) {
        throw std::runtime_error("node title '" + ref + "' matched " + std::to_string(hits.size()) + " nodes (need exactly 1)");
    }
    return hits[0];
}

static int total_tokens(const Doc & doc, bool live_only) {
    int n = 0;
    for (const auto & node : doc.nodes) {
        if (!live_only || !node.evicted) {
            n += (int) node.tokens.size();
        }
    }
    return n;
}

//
// KV operations
//

struct Batch {
    llama_batch b;
    Batch(int n_tokens, int n_seq) : b(llama_batch_init(n_tokens, 0, n_seq)) {}
    ~Batch() { llama_batch_free(b); }

    void clear() { b.n_tokens = 0; }

    void add(llama_token t, llama_pos p, const std::vector<llama_seq_id> & seqs, bool out) {
        const int i = b.n_tokens++;
        b.token[i]    = t;
        b.pos[i]      = p;
        b.n_seq_id[i] = (int32_t) seqs.size();
        for (size_t s = 0; s < seqs.size(); ++s) {
            b.seq_id[i][s] = seqs[s];
        }
        b.logits[i] = out;
    }
};

static void decode(Engine & e, Batch & batch) {
    const int r = llama_decode(e.ctx, batch.b);
    if (r != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(r));
    }
}

// decode every node in pre-order. Pre-order keeps each sequence's positions
// continuous, so the whole tree can go into a single llama_decode call;
// per_node = true decodes one node per call instead (slower, used to cross-check).
static double prefill(Engine & e, Doc & doc, bool per_node = false) {
    llama_memory_clear(e.mem(), true);
    for (auto & n : doc.nodes) {
        n.evicted = false;
    }
    Batch batch(total_tokens(doc, false), N_SEQ_MAX);

    const double t0 = now_ms();
    for (const auto & n : doc.nodes) {
        for (size_t k = 0; k < n.tokens.size(); ++k) {
            batch.add(n.tokens[k], n.pos[k], n.seq_ids, false);
        }
        if (per_node) {
            decode(e, batch);
            batch.clear();
        }
    }
    if (!per_node) {
        decode(e, batch);
    }
    return now_ms() - t0;
}

// baseline: the same tokens as one ordinary causal sequence (no structure)
static double prefill_flat(Engine & e, const Doc & doc) {
    llama_memory_clear(e.mem(), true);
    Batch batch(total_tokens(doc, false), 1);
    llama_pos p = 0;
    for (const auto & n : doc.nodes) {
        for (llama_token t : n.tokens) {
            batch.add(t, p++, { 0 }, false);
        }
    }
    const double t0 = now_ms();
    decode(e, batch);
    return now_ms() - t0;
}

// remove a whole subtree: its own position range, for every sequence that lives in it
static bool evict(Engine & e, Doc & doc, int id) {
    const auto ids = subtree(doc, id);
    llama_pos p0 = doc.nodes[id].p0();
    llama_pos p1 = p0;
    for (int i : ids) {
        p1 = std::max(p1, doc.nodes[i].p1());
    }
    bool ok = true;
    for (llama_seq_id s : doc.nodes[id].seq_ids) {
        ok = llama_memory_seq_rm(e.mem(), s, p0, p1) && ok;
    }
    for (int i : ids) {
        doc.nodes[i].evicted = true;
    }
    return ok;
}

// nodes visible to a question: everything, or the path to `scope` plus its subtree
static std::vector<int> scope_nodes(const Doc & doc, int scope) {
    std::vector<int> ids;
    if (scope < 0) {
        for (const auto & n : doc.nodes) {
            ids.push_back(n.id);
        }
    } else {
        ids = ancestors(doc, scope);
        const auto sub = subtree(doc, scope);
        ids.insert(ids.end(), sub.begin(), sub.end());
    }
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](int i) { return doc.nodes[i].evicted; }), ids.end());
    return ids;
}

struct Answer {
    std::string              text;
    std::vector<llama_token> tokens;
    std::vector<float>       first_logits;
    int                      n_context = 0; // KV cells the question could attend to
    int                      n_question = 0;
    double                   t_question_ms = 0;
    double                   t_generate_ms = 0;
};

static std::vector<float> copy_logits(Engine & e, int idx) {
    const float * l = llama_get_logits_ith(e.ctx, idx);
    return std::vector<float>(l, l + llama_vocab_n_tokens(e.vocab));
}

static llama_token argmax(const std::vector<float> & v) {
    return (llama_token) (std::max_element(v.begin(), v.end()) - v.begin());
}

static std::vector<llama_token> question_tokens(Engine & e, const std::string & q) {
    // continues the user turn opened by the root node (Llama 3 chat format)
    return tokenize(e.vocab, "\n\nQuestion: " + q + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", true);
}

// greedy answer appended to sequence `seq`; the question and answer cells are removed afterwards
static Answer ask_in_seq(Engine & e, llama_seq_id seq, const std::string & question, int n_predict) {
    llama_memory_t mem = e.mem();

    Answer a;
    const std::vector<llama_seq_id> q = { seq };
    const auto qtok = question_tokens(e, question);
    a.n_question = (int) qtok.size();

    const llama_pos p_question = llama_memory_seq_pos_max(mem, seq) + 1;
    llama_pos p = p_question;
    Batch batch((int) qtok.size(), 1);
    for (size_t k = 0; k < qtok.size(); ++k) {
        batch.add(qtok[k], p++, q, k + 1 == qtok.size());
    }

    double t0 = now_ms();
    decode(e, batch);
    a.t_question_ms = now_ms() - t0;

    t0 = now_ms();
    for (int step = 0; step < n_predict; ++step) {
        auto logits = copy_logits(e, -1);
        const llama_token t = argmax(logits);
        if (step == 0) {
            a.first_logits = std::move(logits);
        }
        if (llama_vocab_is_eog(e.vocab, t)) {
            break;
        }
        a.tokens.push_back(t);
        a.text += piece(e.vocab, t);
        batch.clear();
        batch.add(t, p++, q, true);
        decode(e, batch);
    }
    a.t_generate_ms = now_ms() - t0;

    llama_memory_seq_rm(mem, seq, p_question, -1);
    return a;
}

// greedy answer in the query sequence, which first gets a view of the nodes in scope
static Answer ask(Engine & e, const Doc & doc, int scope, const std::string & question, int n_predict) {
    llama_memory_t mem = e.mem();
    llama_memory_seq_rm(mem, QUERY_SEQ, -1, -1);

    int n_context = 0;
    for (int i : scope_nodes(doc, scope)) {
        const Node & n = doc.nodes[i];
        llama_memory_seq_cp(mem, n.seq(), QUERY_SEQ, n.p0(), n.p1());
        n_context += (int) n.tokens.size();
    }

    Answer a = ask_in_seq(e, QUERY_SEQ, question, n_predict);
    a.n_context = n_context;

    llama_memory_seq_rm(mem, QUERY_SEQ, -1, -1);
    return a;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float d = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

static json answer_json(const Answer & a) {
    return {
        {"text",          a.text},
        {"n_generated",   a.tokens.size()},
        {"n_context",     a.n_context},
        {"n_question",    a.n_question},
        {"question_ms",   a.t_question_ms},
        {"generate_ms",   a.t_generate_ms},
    };
}

// replace a node's content. same_length = true repeats/truncates the new tokens to the
// node's current length, so every KV cell keeps its slot and only the content differs.
static void apply_edit(Engine & e, Doc & doc, int id, const std::string & text, bool same_length = false) {
    auto toks = tokenize(e.vocab, text, false);
    if (toks.empty()) {
        throw std::runtime_error("edit text tokenizes to nothing");
    }
    if (same_length) {
        const size_t n = doc.nodes[id].tokens.size();
        std::vector<llama_token> fit(n);
        for (size_t k = 0; k < n; ++k) {
            fit[k] = toks[k % toks.size()];
        }
        toks = std::move(fit);
    }
    doc.nodes[id].tokens = std::move(toks);
    relayout(doc);
    validate(doc);
}

static int default_threads() {
    return std::max(1, (int) std::thread::hardware_concurrency() / 2);
}

static int ctx_size(const Doc & doc, int n_predict) {
    // every node token once, plus the question and the answer; rounded up
    const int need = total_tokens(doc, false) + 256 + n_predict;
    return ((need + 255) / 256) * 256;
}

//
// commands
//

static int cmd_tokenize(const Args & args) {
    Engine e(args.req("-m"), 0, 1, /*vocab_only*/ true);
    json in = json::parse(std::string(std::istreambuf_iterator<char>(std::cin), {}));
    json out = json::array();
    for (const auto & item : in) {
        const bool        special = item.is_object() && item.value("special", false);
        const std::string text    = item.is_object() ? item.at("text").get<std::string>() : item.get<std::string>();
        const auto ids = tokenize(e.vocab, text, special);
        json pieces = json::array();
        for (llama_token t : ids) {
            pieces.push_back(piece(e.vocab, t));
        }
        out.push_back({{"ids", ids}, {"pieces", pieces}});
    }
    // pieces may hold partial UTF-8 sequences; replace invalid bytes instead of failing
    std::cout << out.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    return 0;
}

// greedy decoding of several prompts; prompt i runs in sequence i of one shared batch
static std::vector<Answer> generate(Engine & e, const std::vector<std::vector<llama_token>> & prompts, int n_predict) {
    const int n = (int) prompts.size();
    std::vector<Answer>    out(n);
    std::vector<llama_pos> pos(n);
    std::vector<int>       idx(n, -1);

    size_t max_len = 0;
    for (const auto & p : prompts) {
        max_len += p.size();
    }
    Batch batch((int) std::max<size_t>(max_len, n), 1);
    for (int s = 0; s < n; ++s) {
        for (size_t k = 0; k < prompts[s].size(); ++k) {
            batch.add(prompts[s][k], (llama_pos) k, { s }, k + 1 == prompts[s].size());
        }
        idx[s] = batch.b.n_tokens - 1;
        pos[s] = (llama_pos) prompts[s].size();
    }
    decode(e, batch);

    for (int step = 0; step < n_predict; ++step) {
        batch.clear();
        for (int s = 0; s < n; ++s) {
            if (idx[s] < 0) {
                continue;
            }
            auto logits = copy_logits(e, idx[s]);
            const llama_token t = argmax(logits);
            if (step == 0) {
                out[s].first_logits = std::move(logits);
            }
            if (llama_vocab_is_eog(e.vocab, t)) {
                idx[s] = -1;
                continue;
            }
            out[s].tokens.push_back(t);
            out[s].text += piece(e.vocab, t);
            idx[s] = batch.b.n_tokens;
            batch.add(t, pos[s]++, { s }, true);
        }
        if (batch.b.n_tokens == 0) {
            break;
        }
        decode(e, batch);
    }
    return out;
}

static int cmd_twoseq(const Args & args) {
    const int n_predict = std::stoi(args.get("-n", "32"));
    Engine e(args.req("-m"), 1024, std::stoi(args.get("-t", std::to_string(default_threads()))));

    std::vector<std::vector<llama_token>> prompts;
    for (const auto & key : { "-p1", "-p2" }) {
        auto t = tokenize(e.vocab, args.req(key), false);
        t.insert(t.begin(), llama_vocab_bos(e.vocab));
        prompts.push_back(std::move(t));
    }

    llama_memory_clear(e.mem(), true);
    const auto joint = generate(e, prompts, n_predict);

    std::vector<Answer> separate;
    for (const auto & p : prompts) {
        llama_memory_clear(e.mem(), true);
        separate.push_back(generate(e, { p }, n_predict)[0]);
    }

    json res = json::array();
    bool all_match = true;
    for (size_t i = 0; i < prompts.size(); ++i) {
        const bool match = joint[i].tokens == separate[i].tokens;
        all_match = all_match && match;
        res.push_back({
            {"prompt",                args.req(i == 0 ? "-p1" : "-p2")},
            {"joint",                 joint[i].text},
            {"separate",              separate[i].text},
            {"tokens_match",          match},
            {"first_logits_max_diff", max_abs_diff(joint[i].first_logits, separate[i].first_logits)},
        });
    }
    std::cout << json({{"sequences", res}, {"pass", all_match}}).dump(2) << std::endl;
    return all_match ? 0 : 1;
}

static int cmd_run(const Args & args) {
    const int n_predict = std::stoi(args.get("-n", "64"));
    Doc doc = load_doc(args.req("-d"));
    validate(doc);

    Engine e(args.req("-m"), ctx_size(doc, n_predict), std::stoi(args.get("-t", std::to_string(default_threads()))));

    if (args.has("--edit")) {
        apply_edit(e, doc, find_node(doc, args.get("--edit")), args.req("--edit-text"));
    }
    const int scope = args.has("--scope") ? find_node(doc, args.get("--scope")) : -1;
    const std::string question = args.req("-q");

    json out;
    out["doc"]          = doc.name;
    out["n_nodes"]      = doc.nodes.size();
    out["n_tokens"]     = total_tokens(doc, false);
    out["scope"]        = scope < 0 ? json("all") : json(doc.nodes[scope].title);
    out["prefill_mode"] = args.has("--per-node") ? "per-node" : "one-batch";
    out["prefill_ms"]   = prefill(e, doc, args.has("--per-node"));

    Answer a = ask(e, doc, scope, question, n_predict);
    out["answer"]       = answer_json(a);
    out["kv_bytes"]     = e.kv_bytes();
    out["kv_cells"]     = total_tokens(doc, true);
    fprintf(stderr, "answer: %s\n", a.text.c_str());

    json evictions = json::array();
    for (const auto & ref : args.all("--evict")) {
        const int id = find_node(doc, ref);
        const bool ok = evict(e, doc, id);
        Answer b = ask(e, doc, scope, question, n_predict);
        evictions.push_back({
            {"evicted",  doc.nodes[id].title},
            {"ok",       ok},
            {"kv_bytes", e.kv_bytes()},
            {"kv_cells", total_tokens(doc, true)},
            {"answer",   answer_json(b)},
        });
        fprintf(stderr, "after evicting '%s': %s\n", doc.nodes[id].title.c_str(), b.text.c_str());
    }
    out["evictions"] = evictions;
    std::cout << out.dump(2) << std::endl;
    return 0;
}

static int cmd_test(const Args & args) {
    const int n_predict = std::stoi(args.get("-n", "48"));
    Doc doc = load_doc(args.req("-d"));
    validate(doc);

    const int A = find_node(doc, args.req("--a"));
    const int B = find_node(doc, args.req("--b"));
    const auto anc_a = ancestors(doc, A);
    const auto anc_b = ancestors(doc, B);
    if (A == B || std::count(anc_a.begin(), anc_a.end(), B) || std::count(anc_b.begin(), anc_b.end(), A)) {
        throw std::runtime_error("subtrees A and B must be disjoint (neither may contain the other)");
    }
    const std::string question  = args.req("-q");
    const std::string edit_text = args.req("--edit-text");

    Engine e(args.req("-m"), ctx_size(doc, n_predict) + 512, std::stoi(args.get("-t", std::to_string(default_threads()))));

    json out;
    out["doc"]      = doc.name;
    out["question"] = question;
    out["A"]        = doc.nodes[A].title;
    out["B"]        = doc.nodes[B].title;
    out["n_nodes"]  = doc.nodes.size();
    out["n_tokens"] = total_tokens(doc, false);

    // baseline: original document, question scoped to A and to the whole document
    const double t_prefill = prefill(e, doc);
    const Answer base_a    = ask(e, doc, A,  question, n_predict);
    const Answer base_all  = ask(e, doc, -1, question, n_predict);
    out["prefill_ms"]      = t_prefill;
    out["baseline_scoped"] = answer_json(base_a);
    out["baseline_full"]   = answer_json(base_all);

    // test 2 (same context): evict B, ask again
    const size_t kv_before    = e.kv_bytes();
    const int    cells_before = total_tokens(doc, true);
    const bool   evict_ok     = evict(e, doc, B);
    const size_t kv_after     = e.kv_bytes();
    const int    cells_after  = total_tokens(doc, true);
    const Answer ev_a         = ask(e, doc, A,  question, n_predict);
    const Answer ev_all       = ask(e, doc, -1, question, n_predict);

    const bool t2_pass = evict_ok && ev_a.tokens == base_a.tokens && kv_after < kv_before;
    out["test2_evict_B"] = {
        {"pass",                      t2_pass},
        {"seq_rm_ok",                 evict_ok},
        {"kv_bytes_before",           kv_before},
        {"kv_bytes_after",            kv_after},
        {"kv_cells_before",           cells_before},
        {"kv_cells_after",            cells_after},
        {"scoped_answer",             answer_json(ev_a)},
        {"scoped_unchanged",          ev_a.tokens == base_a.tokens},
        {"scoped_first_logits_diff",  max_abs_diff(ev_a.first_logits, base_a.first_logits)},
        {"full_answer",               answer_json(ev_all)},
        {"full_unchanged",            ev_all.tokens == base_all.tokens},
    };

    // test 1: rebuild the context with B edited, ask about A. Two variants:
    //   same_length - B keeps its token count, so every cell keeps its KV slot; A must be bit-identical
    //   free_length - B's real new text; cells shift, so only the answer tokens are compared
    // control: the same edits on an ordinary flat prompt, where A attends to B
    prefill_flat(e, doc);
    const Answer flat_base = ask_in_seq(e, 0, question, n_predict);

    bool t1_pass = true;
    for (const bool same_length : { true, false }) {
        Doc edited = doc;
        apply_edit(e, edited, B, edit_text, same_length);

        const double t_pre = prefill(e, edited);
        const Answer ed_a  = ask(e, edited, A, question, n_predict);
        const float  diff  = max_abs_diff(ed_a.first_logits, base_a.first_logits);
        const bool   same  = ed_a.tokens == base_a.tokens;
        const bool   pass  = same_length ? (same && diff == 0.0f) : same;
        t1_pass = t1_pass && pass;

        prefill_flat(e, edited);
        const Answer flat_edit = ask_in_seq(e, 0, question, n_predict);

        out["test1_edit_B"][same_length ? "same_length" : "free_length"] = {
            {"pass",                      pass},
            {"B_tokens_before",           doc.nodes[B].tokens.size()},
            {"B_tokens_after",            edited.nodes[B].tokens.size()},
            {"prefill_ms",                t_pre},
            {"scoped_answer",             answer_json(ed_a)},
            {"scoped_unchanged",          same},
            {"scoped_first_logits_diff",  diff},
            {"flat_control_answer",       answer_json(flat_edit)},
            {"flat_control_unchanged",    flat_edit.tokens == flat_base.tokens},
            {"flat_control_logits_diff",  max_abs_diff(flat_edit.first_logits, flat_base.first_logits)},
        };
    }
    out["test1_edit_B"]["pass"]      = t1_pass;
    out["test1_edit_B"]["edit_text"] = edit_text;
    out["flat_baseline_answer"]      = answer_json(flat_base);

    out["flash_attn"] = Engine::flash_attn;
    out["pass"] = t1_pass && t2_pass;
    std::cout << out.dump(2) << std::endl;
    return (t1_pass && t2_pass) ? 0 : 1;
}

// numerical noise floor: how much do the first-step logits move under changes that should not matter?
static int cmd_noise(const Args & args) {
    Doc doc = load_doc(args.req("-d"));
    validate(doc);
    const int A = find_node(doc, args.req("--a"));
    const int B = find_node(doc, args.req("--b"));
    const std::string q = args.req("-q");
    Engine e(args.req("-m"), ctx_size(doc, 8) + 512, std::stoi(args.get("-t", std::to_string(default_threads()))));

    prefill(e, doc);
    const Answer a0 = ask(e, doc, A, q, 1);
    const Answer a1 = ask(e, doc, A, q, 1);
    ask(e, doc, -1, q, 1); // a full-scope question in between must leave no trace
    const Answer a2 = ask(e, doc, A, q, 1);
    evict(e, doc, B);
    const Answer a3 = ask(e, doc, A, q, 1);
    prefill(e, doc, true);
    const Answer a4 = ask(e, doc, A, q, 1);

    json out = {
        {"repeat_same_state",          max_abs_diff(a0.first_logits, a1.first_logits)},
        {"after_full_scope_question",  max_abs_diff(a0.first_logits, a2.first_logits)},
        {"after_evicting_B",           max_abs_diff(a0.first_logits, a3.first_logits)},
        {"per_node_vs_one_batch",      max_abs_diff(a0.first_logits, a4.first_logits)},
        {"top_logit",                  *std::max_element(a0.first_logits.begin(), a0.first_logits.end())},
    };
    std::cout << out.dump(2) << std::endl;
    return 0;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// CPU timings: flat prefill vs. structured prefill (one batch / per node), plus a full-scope answer
static int cmd_bench(const Args & args) {
    const int n_predict = std::stoi(args.get("-n", "32"));
    const int reps      = std::stoi(args.get("-r", "3"));
    const int n_threads = std::stoi(args.get("-t", std::to_string(default_threads())));
    Doc doc = load_doc(args.req("-d"));
    validate(doc);
    Engine e(args.req("-m"), ctx_size(doc, n_predict), n_threads);

    prefill_flat(e, doc); // warm-up
    std::vector<double> flat, one, per;
    for (int r = 0; r < reps; ++r) {
        flat.push_back(prefill_flat(e, doc));
        per.push_back(prefill(e, doc, true));
        one.push_back(prefill(e, doc, false));
    }
    const int n_tok = total_tokens(doc, false);
    const Answer a = ask(e, doc, -1, args.get("-q", "Summarize the document."), n_predict);

    json out = {
        {"doc",       doc.name},
        {"n_nodes",   doc.nodes.size()},
        {"n_tokens",  n_tok},
        {"n_threads", n_threads},
        {"reps",      reps},
        {"prefill_flat_ms",          median(flat)},
        {"prefill_tree_one_batch_ms", median(one)},
        {"prefill_tree_per_node_ms", median(per)},
        {"prefill_flat_tok_s",       n_tok / (median(flat) / 1000.0)},
        {"prefill_tree_tok_s",       n_tok / (median(one) / 1000.0)},
        {"answer_full_scope",        answer_json(a)},
        {"generate_tok_s",           a.tokens.empty() ? 0.0 : a.tokens.size() / (a.t_generate_ms / 1000.0)},
        {"kv_bytes",                 e.kv_bytes()},
    };
    std::cout << out.dump(2) << std::endl;
    return 0;
}

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        g_verbose = args.has("-v");
        Engine::flash_attn = args.get("--fa", "auto");
        llama_log_set(log_cb, nullptr);
        llama_backend_init();

        int rc;
        if      (args.cmd == "tokenize") rc = cmd_tokenize(args);
        else if (args.cmd == "twoseq")   rc = cmd_twoseq(args);
        else if (args.cmd == "run")      rc = cmd_run(args);
        else if (args.cmd == "test")     rc = cmd_test(args);
        else if (args.cmd == "bench")    rc = cmd_bench(args);
        else if (args.cmd == "noise")    rc = cmd_noise(args);
        else throw std::runtime_error("unknown command: " + args.cmd);

        llama_backend_free();
        return rc;
    } catch (const std::exception & ex) {
        fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
