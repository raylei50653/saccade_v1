// Read-only audit of a torch.save zip checkpoint that was already extracted
// by `unzip`. Interprets only the pickle opcodes needed to rebuild the
// container tree; no Python, no class import, no code execution. Globals are
// recorded by name and never resolved. Tensors are reported as storage
// references with shape/stride/offset and a SHA-256 of their exact bytes.
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {
struct Value;
using Ref = std::shared_ptr<Value>;
struct Global { std::string module, name; };
struct Object { Ref callable, args, state, build; }; // state: SETITEMS target of dict-like objects
struct Persistent { Ref id; };
struct Value {
    enum class Kind { none, boolean, integer, real, text, bytes, list, tuple, dict, global, object, persistent, mark } kind;
    bool b = false; std::int64_t i = 0; double f = 0; std::string s;
    std::vector<Ref> items; std::vector<std::pair<Ref, Ref>> pairs;
    Global g; Object o; Persistent p;
};
Ref make(Value::Kind k) { auto v = std::make_shared<Value>(); v->kind = k; return v; }

class Reader {
public:
    explicit Reader(std::string data) : d_(std::move(data)) {}
    Ref run() {
        while (true) {
            const auto at = pos_;
            const auto op = u8();
            switch (op) {
            case 0x80: u8(); break;                                   // PROTO
            case 0x95: take(8); break;                                // FRAME
            case '}': push(make(Value::Kind::dict)); break;
            case ']': push(make(Value::Kind::list)); break;
            case ')': push(make(Value::Kind::tuple)); break;
            case '(': push(make(Value::Kind::mark)); break;
            case 'N': push(make(Value::Kind::none)); break;
            case 0x88: case 0x89: { auto v = make(Value::Kind::boolean); v->b = op == 0x88; push(v); break; }
            case 'K': push_int(u8()); break;
            case 'M': push_int(le(2)); break;
            case 'J': push_int(static_cast<std::int32_t>(le(4))); break;
            case 0x8a: push_int(long_bytes(u8())); break;
            case 0x8b: push_int(long_bytes(static_cast<std::size_t>(le(4)))); break;
            case 'G': { auto bits = be(8); double x; std::memcpy(&x, &bits, 8);
                        auto v = make(Value::Kind::real); v->f = x; push(v); break; }
            case 'X': push_text(take(static_cast<std::size_t>(le(4)))); break;
            case 0x8c: push_text(take(u8())); break;
            case 0x8d: push_text(take(static_cast<std::size_t>(le(8)))); break;
            case 'C': push_bytes(take(u8())); break;
            case 'B': push_bytes(take(static_cast<std::size_t>(le(4)))); break;
            case 'q': memo_[u8()] = top(); break;
            case 'r': memo_[le(4)] = top(); break;
            case 0x94: memo_[memo_.size()] = top(); break;
            case 'h': push(memo(u8())); break;
            case 'j': push(memo(le(4))); break;
            case 't': { auto v = make(Value::Kind::tuple); v->items = pop_mark(); push(v); break; }
            case 0x85: case 0x86: case 0x87: {
                auto v = make(Value::Kind::tuple); v->items.resize(op - 0x84);
                for (auto it = v->items.rbegin(); it != v->items.rend(); ++it) *it = pop();
                push(v); break; }
            case 'a': { auto x = pop(); container(top(), Value::Kind::list)->items.push_back(x); break; }
            case 'e': { auto xs = pop_mark(); auto& l = container(top(), Value::Kind::list)->items; l.insert(l.end(), xs.begin(), xs.end()); break; }
            case 's': { auto v = pop(); auto k = pop(); container(top(), Value::Kind::dict)->pairs.emplace_back(k, v); break; }
            case 'u': { auto xs = pop_mark(); if (xs.size() % 2) fail(at, "odd SETITEMS");
                        auto d = container(top(), Value::Kind::dict);
                        for (std::size_t n = 0; n < xs.size(); n += 2) { d->pairs.emplace_back(xs[n], xs[n + 1]); }
                        break; }
            case 'c': { auto v = make(Value::Kind::global); v->g.module = line(); v->g.name = line(); push(v); break; }
            case 0x93: { auto v = make(Value::Kind::global); auto name = pop(); auto module = pop();
                         v->g.module = module->s; v->g.name = name->s; push(v); break; }
            case 'R': case 0x81: { auto v = make(Value::Kind::object); v->o.args = pop(); v->o.callable = pop(); push(v); break; }
            case 'b': { auto st = pop(); auto t = top(); if (t->kind != Value::Kind::object) fail(at, "BUILD on non-object"); t->o.build = st; break; }
            case 'Q': { auto v = make(Value::Kind::persistent); v->p.id = pop(); push(v); break; }
            case '.': if (stack_.size() != 1) fail(at, "STOP with non-singleton stack"); return pop();
            default: fail(at, "unsupported opcode 0x" + hex(op));
            }
        }
    }
private:
    static std::string hex(unsigned v) { std::ostringstream o; o << std::hex << std::setw(2) << std::setfill('0') << v; return o.str(); }
    [[noreturn]] static void fail(std::size_t at, const std::string& what) { throw std::runtime_error("pickle offset " + std::to_string(at) + ": " + what); }
    std::string take(std::size_t n) { if (n > d_.size() - pos_) fail(pos_, "truncated"); auto r = d_.substr(pos_, n); pos_ += n; return r; }
    unsigned u8() { return static_cast<unsigned char>(take(1)[0]); }
    std::uint64_t le(int n) { auto b = take(n); std::uint64_t v = 0; for (int k = n - 1; k >= 0; --k) v = (v << 8) | static_cast<unsigned char>(b[k]); return v; }
    std::uint64_t be(int n) { auto b = take(n); std::uint64_t v = 0; for (int k = 0; k < n; ++k) v = (v << 8) | static_cast<unsigned char>(b[k]); return v; }
    std::int64_t long_bytes(std::size_t n) {
        if (n > 8) fail(pos_, "LONG wider than int64");
        auto b = take(n); std::uint64_t v = 0;
        for (std::size_t k = n; k-- > 0;) v = (v << 8) | static_cast<unsigned char>(b[k]);
        if (n > 0 && n < 8 && (static_cast<unsigned char>(b[n - 1]) & 0x80)) v |= ~std::uint64_t{0} << (8 * n);
        return static_cast<std::int64_t>(v);
    }
    std::string line() { auto e = d_.find('\n', pos_); if (e == std::string::npos) fail(pos_, "unterminated GLOBAL"); auto r = d_.substr(pos_, e - pos_); pos_ = e + 1; return r; }
    void push(Ref v) { stack_.push_back(std::move(v)); }
    void push_int(std::int64_t x) { auto v = make(Value::Kind::integer); v->i = x; push(v); }
    void push_text(std::string x) { auto v = make(Value::Kind::text); v->s = std::move(x); push(v); }
    void push_bytes(std::string x) { auto v = make(Value::Kind::bytes); v->s = std::move(x); push(v); }
    Ref top() { if (stack_.empty()) fail(pos_, "empty stack"); return stack_.back(); }
    Ref pop() { auto v = top(); stack_.pop_back(); return v; }
    Ref memo(std::uint64_t k) { auto it = memo_.find(k); if (it == memo_.end()) fail(pos_, "missing memo"); return it->second; }
    std::vector<Ref> pop_mark() {
        std::vector<Ref> xs;
        while (true) { auto v = pop(); if (v->kind == Value::Kind::mark) break; xs.push_back(v); }
        return {xs.rbegin(), xs.rend()};
    }
    Ref container(Ref v, Value::Kind k) {
        // OrderedDict instances are REDUCE objects; SETITEMS targets them too.
        if (v->kind == k) return v;
        if (v->kind == Value::Kind::object && k == Value::Kind::dict) {
            if (!v->o.state) v->o.state = make(Value::Kind::dict);
            if (v->o.state->kind == Value::Kind::dict) return v->o.state;
        }
        fail(pos_, "container type mismatch");
    }
    std::string d_; std::size_t pos_ = 0; std::vector<Ref> stack_; std::map<std::uint64_t, Ref> memo_;
};

bool is_global(const Ref& v, const char* module, const char* name) {
    return v && v->kind == Value::Kind::global && v->g.module == module && v->g.name == name;
}
const std::vector<std::pair<Ref, Ref>>* dict_pairs(const Ref& v) {
    if (v->kind == Value::Kind::dict) return &v->pairs;
    if (v->kind == Value::Kind::object && v->o.state && v->o.state->kind == Value::Kind::dict) return &v->o.state->pairs;
    return nullptr;
}
std::string quote(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c < 0x20 || c > 0x7e) { std::ostringstream o; o << "\\x" << std::hex << std::setw(2) << std::setfill('0') << int(c); out += o.str(); }
        else out += static_cast<char>(c);
    }
    return out + "\"";
}

struct Context { std::filesystem::path root; };

std::size_t element_size(const std::string& storage) {
    if (storage == "FloatStorage" || storage == "IntStorage") return 4;
    if (storage == "DoubleStorage" || storage == "LongStorage") return 8;
    if (storage == "HalfStorage" || storage == "BFloat16Storage" || storage == "ShortStorage") return 2;
    if (storage == "BoolStorage" || storage == "ByteStorage" || storage == "CharStorage") return 1;
    throw std::runtime_error("unsupported storage type " + storage);
}

std::string ints(const Ref& tuple) {
    std::string r = "[";
    for (std::size_t n = 0; n < tuple->items.size(); ++n) r += (n ? "," : "") + std::to_string(tuple->items[n]->i);
    return r + "]";
}

std::string tensor(const Context& ctx, const Ref& obj) {
    // _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)
    auto args = obj->o.args;
    if (is_global(obj->o.callable, "torch._utils", "_rebuild_parameter")) return "parameter:" + tensor(ctx, args->items.at(0));
    const auto& pid = args->items.at(0)->p.id->items; // ('storage', type, key, location, numel)
    const auto storage = pid.at(1)->g.name, key = pid.at(2)->s;
    const auto offset = args->items.at(1)->i;
    const auto& size = args->items.at(2), & stride = args->items.at(3);
    std::int64_t numel = 1, expect = 1; bool contiguous = true;
    for (std::size_t n = size->items.size(); n-- > 0;) {
        if (size->items[n]->i != 1 && stride->items[n]->i != expect) contiguous = false;
        expect *= size->items[n]->i; numel *= size->items[n]->i;
    }
    std::ostringstream out;
    out << "tensor dtype=" << storage << " shape=" << ints(size) << " stride=" << ints(stride)
        << " storage=" << key << " offset=" << offset << " storage_numel=" << pid.at(4)->i;
    if (!contiguous) return out.str() + " bytes=non-contiguous";
    const auto bytes = saccade::read_text_file(ctx.root / "data" / key);
    const auto elem = element_size(storage);
    const auto begin = static_cast<std::size_t>(offset) * elem, count = static_cast<std::size_t>(numel) * elem;
    if (begin + count > bytes.size()) throw std::runtime_error("tensor exceeds storage file " + key);
    const auto view = std::string_view(bytes).substr(begin, count);
    out << " sha256=" << saccade::sha256_hex(view);
    if (storage == "FloatStorage" && numel <= 32) {
        out << " values=[";
        for (std::int64_t n = 0; n < numel; ++n) {
            float x; std::memcpy(&x, view.data() + 4 * n, 4);
            out << (n ? "," : "") << std::setprecision(9) << x;
        }
        out << "]";
    }
    return out.str();
}

void dump(const Context& ctx, const Ref& v, const std::string& path, int depth, int max_depth) {
    using K = Value::Kind;
    auto leaf = [&](const std::string& text) { std::cout << path << "\t" << text << "\n"; };
    switch (v->kind) {
    case K::none: return leaf("None");
    case K::boolean: return leaf(v->b ? "True" : "False");
    case K::integer: return leaf(std::to_string(v->i));
    case K::real: { std::ostringstream o; o << std::setprecision(17) << v->f; return leaf(o.str()); }
    case K::text: return leaf(quote(v->s));
    case K::bytes: return leaf("bytes[" + std::to_string(v->s.size()) + "]");
    case K::global: return leaf("global " + v->g.module + "." + v->g.name);
    case K::persistent: return leaf("persistent-id");
    case K::mark: return leaf("mark");
    case K::list: case K::tuple: {
        if (depth >= max_depth) return leaf((v->kind == K::list ? "list[" : "tuple[") + std::to_string(v->items.size()) + "]");
        if (v->items.empty()) return leaf(v->kind == K::list ? "[]" : "()");
        for (std::size_t n = 0; n < v->items.size(); ++n) dump(ctx, v->items[n], path + "[" + std::to_string(n) + "]", depth + 1, max_depth);
        return;
    }
    case K::dict: case K::object: {
        if (v->kind == K::object && (is_global(v->o.callable, "torch._utils", "_rebuild_tensor_v2") ||
                                     is_global(v->o.callable, "torch._utils", "_rebuild_parameter")))
            return leaf(tensor(ctx, v));
        const auto* pairs = dict_pairs(v);
        if (v->kind == K::object && !pairs) {
            std::string desc = "object " + (v->o.callable->kind == K::global ? v->o.callable->g.module + "." + v->o.callable->g.name : std::string("?"));
            if (v->o.args && v->o.args->kind == K::tuple) {
                desc += "(";
                for (std::size_t n = 0; n < v->o.args->items.size(); ++n) {
                    const auto& a = v->o.args->items[n];
                    desc += (n ? "," : "") + (a->kind == K::text ? quote(a->s) : std::string("..."));
                }
                desc += ")";
            }
            return leaf(desc);
        }
        if (depth >= max_depth) return leaf("dict[" + std::to_string(pairs->size()) + "]");
        if (pairs->empty()) return leaf("{}");
        for (const auto& [k, x] : *pairs) {
            const auto name = k->kind == K::text ? k->s : k->kind == K::integer ? std::to_string(k->i) : "?";
            dump(ctx, x, path.empty() ? name : path + "." + name, depth + 1, max_depth);
        }
        return;
    }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) throw std::runtime_error("usage: torch_ckpt_inspect EXTRACTED_ARCHIVE_DIR TOP_KEY|'*' [MAX_DEPTH]");
        const Context ctx{argv[1]};
        const auto root = Reader(saccade::read_text_file(ctx.root / "data.pkl")).run();
        const std::string want = argv[2];
        const int max_depth = argc == 4 ? std::stoi(argv[3]) : 64;
        const auto* pairs = dict_pairs(root);
        if (!pairs) throw std::runtime_error("checkpoint root is not a dict");
        std::cout << "scope=pickle-container-audit-only; globals recorded, never imported\n";
        for (const auto& [k, v] : *pairs) {
            if (want == "*") { dump(ctx, v, k->s, 0, 0); continue; }
            if (k->s == want) dump(ctx, v, k->s, 0, max_depth);
        }
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot write inspector output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "torch_ckpt_inspect: " << error.what() << '\n';
        return 2;
    }
}
