#include "saccade/torch_checkpoint.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace saccade {
namespace {

[[noreturn]] void zip_fail(const std::string& why) { throw std::runtime_error("zip: " + why); }

std::uint64_t le(std::string_view bytes, std::uint64_t at, int n) {
    if (at > bytes.size() || static_cast<std::uint64_t>(n) > bytes.size() - at) zip_fail("record extends past the end of the file");
    std::uint64_t v = 0;
    for (int k = n - 1; k >= 0; --k) v = (v << 8) | static_cast<unsigned char>(bytes[at + static_cast<std::uint64_t>(k)]);
    return v;
}

constexpr std::uint32_t kLocal = 0x04034b50, kCentral = 0x02014b50, kEnd = 0x06054b50, kEnd64 = 0x06064b50, kLocator64 = 0x07064b50,
                        kDescriptor = 0x08074b50;

std::array<std::uint32_t, 256> crc_table() {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
        std::uint32_t c = n;
        for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        t[n] = c;
    }
    return t;
}

} // namespace

std::uint32_t crc32(std::string_view bytes) {
    static const auto table = crc_table();
    std::uint32_t c = 0xFFFFFFFFu;
    for (const unsigned char b : bytes) c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

StoredZip::StoredZip(std::string bytes) : bytes_(std::move(bytes)) {
    const std::string_view f(bytes_);
    if (f.size() < 22) zip_fail("file too short for an end-of-central-directory record");
    // The EOCD comment must end the file: search back over at most 64 KiB of comment.
    std::uint64_t end = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t at = f.size() - 22, stop = f.size() > 22 + 65535 ? f.size() - 22 - 65535 : 0;; --at) {
        if (le(f, at, 4) == kEnd && at + 22 + le(f, at + 20, 2) == f.size()) { end = at; break; }
        if (at == stop) break;
    }
    if (end == std::numeric_limits<std::uint64_t>::max()) zip_fail("no end-of-central-directory record at the end of the file");
    if (le(f, end + 4, 2) != 0 || le(f, end + 6, 2) != 0) zip_fail("multi-disk archives are not supported");
    std::uint64_t count = le(f, end + 10, 2), cd_size = le(f, end + 12, 4), cd_offset = le(f, end + 16, 4);
    if (le(f, end + 8, 2) != count) zip_fail("entries on disk differ from total entries");
    std::uint64_t cd_end = end;
    if (end >= 20 && le(f, end - 20, 4) == kLocator64) {
        const auto record = le(f, end - 12, 8);
        if (le(f, end - 16, 4) != 0 || le(f, end - 4, 4) != 1 || le(f, record, 4) != kEnd64 || record + 56 > end - 20)
            zip_fail("malformed zip64 end-of-central-directory locator");
        if (record + 12 + le(f, record + 4, 8) != end - 20) zip_fail("zip64 end record does not reach its locator");
        const auto count64 = le(f, record + 32, 8), size64 = le(f, record + 40, 8), offset64 = le(f, record + 48, 8);
        if (le(f, record + 16, 4) != 0 || le(f, record + 20, 4) != 0 || le(f, record + 24, 8) != count64)
            zip_fail("multi-disk zip64 archives are not supported");
        // A saturated 32-bit field defers to zip64; any other value must agree.
        if ((count != 0xFFFF && count != count64) || (cd_size != 0xFFFFFFFF && cd_size != size64) ||
            (cd_offset != 0xFFFFFFFF && cd_offset != offset64))
            zip_fail("zip64 and 32-bit end records disagree");
        count = count64; cd_size = size64; cd_offset = offset64; cd_end = record;
    } else if (count == 0xFFFF || cd_size == 0xFFFFFFFF || cd_offset == 0xFFFFFFFF) {
        zip_fail("saturated end record without a zip64 locator");
    }
    if (cd_offset > cd_end || cd_size != cd_end - cd_offset) zip_fail("central directory does not end at the end records");

    struct Span { std::uint64_t begin, end; std::string name; };
    std::vector<Span> spans;
    std::uint64_t at = cd_offset;
    for (std::uint64_t i = 0; i < count; ++i) {
        if (le(f, at, 4) != kCentral) zip_fail("central directory entry " + std::to_string(i) + " has a bad signature");
        const auto flags = le(f, at + 8, 2), method = le(f, at + 10, 2);
        const std::uint32_t crc = static_cast<std::uint32_t>(le(f, at + 16, 4));
        std::uint64_t packed = le(f, at + 20, 4), size = le(f, at + 24, 4);
        const auto name_len = le(f, at + 28, 2), extra_len = le(f, at + 30, 2), comment_len = le(f, at + 32, 2);
        std::uint64_t local = le(f, at + 42, 4);
        if (le(f, at + 34, 2) != 0) zip_fail("entry on another disk");
        const auto name_at = at + 46;
        if (name_at + name_len + extra_len + comment_len > cd_end) zip_fail("central directory entry overruns the directory");
        std::string name(f.substr(name_at, name_len));
        if (method != 0) zip_fail(name + ": compression method " + std::to_string(method) + " (only STORED is accepted)");
        if (flags & 0x41) zip_fail(name + ": encrypted entry");
        bool zip64 = false;
        for (std::uint64_t e = name_at + name_len, stop = e + extra_len; e < stop;) {
            const auto id = le(f, e, 2), len = le(f, e + 2, 2);
            if (e + 4 + len > stop) zip_fail(name + ": extra field overruns its block");
            if (id == 0x0001) {
                zip64 = true;
                std::uint64_t p = e + 4;
                const auto take = [&](std::uint64_t& field, std::uint64_t saturated) {
                    if (field != saturated) return;
                    if (p + 8 > e + 4 + len) zip_fail(name + ": short zip64 extra field");
                    field = le(f, p, 8);
                    p += 8;
                };
                take(size, 0xFFFFFFFF); take(packed, 0xFFFFFFFF); take(local, 0xFFFFFFFF);
            }
            e += 4 + len;
        }
        if (packed != size) zip_fail(name + ": stored entry with different packed and unpacked sizes");
        if (le(f, local, 4) != kLocal) zip_fail(name + ": local header has a bad signature");
        if (le(f, local + 6, 2) != flags || le(f, local + 8, 2) != method) zip_fail(name + ": local and central headers disagree");
        const auto local_name = le(f, local + 26, 2), local_extra = le(f, local + 28, 2);
        if (local_name != name_len || f.substr(local + 30, local_name) != name) zip_fail(name + ": local header names a different entry");
        const auto data = local + 30 + local_name + local_extra;
        if (data > cd_offset || size > cd_offset - data) zip_fail(name + ": data extends into the central directory");
        std::uint64_t stop = data + size;
        if (flags & 0x08) {
            // Data descriptor: the local header deferred CRC and sizes to it.
            std::uint64_t d = stop;
            if (le(f, d, 4) == kDescriptor) d += 4;
            if (le(f, d, 4) != crc) zip_fail(name + ": data descriptor CRC differs from the central directory");
            if (!zip64 && le(f, d + 4, 4) == size && le(f, d + 8, 4) == size) stop = d + 12;
            else if (le(f, d + 4, 8) == size && le(f, d + 12, 8) == size) stop = d + 20;
            else zip_fail(name + ": data descriptor sizes differ from the central directory");
        } else if (le(f, local + 14, 4) != crc || le(f, local + 18, 4) != std::min<std::uint64_t>(size, 0xFFFFFFFF) ||
                   le(f, local + 22, 4) != std::min<std::uint64_t>(size, 0xFFFFFFFF)) {
            zip_fail(name + ": local header CRC or sizes differ from the central directory");
        }
        if (stop > cd_offset) zip_fail(name + ": entry extends into the central directory");
        if (crc32(f.substr(data, size)) != crc) zip_fail(name + ": CRC-32 mismatch");
        if (!index_.emplace(name, entries_.size()).second) zip_fail("duplicate entry name " + name);
        entries_.push_back({name, data, size, crc});
        spans.push_back({local, stop, name});
        at = name_at + name_len + extra_len + comment_len;
    }
    if (at != cd_end) zip_fail("central directory size differs from its entries");
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < spans.size(); ++i)
        if (spans[i].begin < spans[i - 1].end) zip_fail("entries " + spans[i - 1].name + " and " + spans[i].name + " overlap");
}

std::string_view StoredZip::data(std::string_view name) const {
    const auto it = index_.find(name);
    if (it == index_.end()) throw std::runtime_error("zip: missing entry " + std::string(name));
    const auto& e = entries_[it->second];
    return std::string_view(bytes_).substr(e.data_offset, e.size);
}

// ------------------------------------------------------------------- pickle

const std::vector<std::pair<PickleRef, PickleRef>>* PickleValue::dict() const {
    if (kind == Kind::dict) return &pairs;
    if (kind == Kind::object && !pairs.empty()) return &pairs;
    if (kind == Kind::object && callable && callable->is_global("collections", "OrderedDict")) return &pairs;
    return nullptr;
}

const PickleValue* PickleValue::find(std::string_view key) const {
    const auto* d = dict();
    if (!d) return nullptr;
    const PickleValue* found = nullptr;
    for (const auto& [k, v] : *d) {
        if (k->kind != Kind::text || k->text != key) continue;
        if (found) throw std::runtime_error("pickle: duplicate key " + std::string(key));
        found = v.get();
    }
    return found;
}

bool PickleValue::is_global(std::string_view module, std::string_view global) const {
    return kind == Kind::global && text == module && name == global;
}

namespace {

class PickleReader {
public:
    explicit PickleReader(std::string_view data) : d_(data) {}
    PickleRef run() {
        using K = PickleValue::Kind;
        while (true) {
            const auto at = pos_;
            const auto op = u8();
            switch (op) {
            case 0x80: if (u8() > 5) fail(at, "unsupported protocol"); break;          // PROTO
            case 0x95: (void)take(8); break;                                            // FRAME
            case '}': push(make(K::dict)); break;                                       // EMPTY_DICT
            case ']': push(make(K::list)); break;                                       // EMPTY_LIST
            case ')': push(make(K::tuple)); break;                                      // EMPTY_TUPLE
            case '(': push(make(K::mark)); break;                                       // MARK
            case 'N': push(make(K::none)); break;                                       // NONE
            case 0x88: case 0x89: { auto v = make(K::boolean); v->boolean = op == 0x88; push(v); break; }
            case 'K': push_int(u8()); break;                                            // BININT1
            case 'M': push_int(static_cast<std::int64_t>(le(2))); break;                 // BININT2
            case 'J': push_int(static_cast<std::int32_t>(le(4))); break;                 // BININT
            case 0x8a: push_int(long_bytes(u8())); break;                               // LONG1
            case 0x8b: push_int(long_bytes(static_cast<std::size_t>(le(4)))); break;     // LONG4
            case 'G': {                                                                 // BINFLOAT
                const auto bits = be(8);
                auto v = make(K::real);
                std::memcpy(&v->real, &bits, 8);
                push(v);
                break;
            }
            case 'X': push_text(K::text, take(static_cast<std::size_t>(le(4)))); break;  // BINUNICODE
            case 0x8c: push_text(K::text, take(u8())); break;                           // SHORT_BINUNICODE
            case 0x8d: push_text(K::text, take(static_cast<std::size_t>(le(8)))); break; // BINUNICODE8
            case 'C': push_text(K::bytes, take(u8())); break;                           // SHORT_BINBYTES
            case 'B': push_text(K::bytes, take(static_cast<std::size_t>(le(4)))); break; // BINBYTES
            case 'q': memo_[u8()] = top(at); break;                                     // BINPUT
            case 'r': memo_[le(4)] = top(at); break;                                    // LONG_BINPUT
            case 0x94: memo_[memo_.size()] = top(at); break;                            // MEMOIZE
            case 'h': push(memo(at, u8())); break;                                      // BINGET
            case 'j': push(memo(at, le(4))); break;                                     // LONG_BINGET
            case 't': { auto v = make(K::tuple); v->items = pop_mark(at); push(v); break; }
            case 0x85: case 0x86: case 0x87: {                                          // TUPLE1..3
                auto v = make(K::tuple);
                v->items.resize(op - 0x84u);
                for (auto it = v->items.rbegin(); it != v->items.rend(); ++it) *it = pop(at);
                push(v);
                break;
            }
            case 'a': { auto x = pop(at); list(at, top(at)).push_back(x); break; }      // APPEND
            case 'e': { auto xs = pop_mark(at); auto& l = list(at, top(at)); l.insert(l.end(), xs.begin(), xs.end()); break; }
            case 's': { auto v = pop(at); auto k = pop(at); dict(at, top(at)).emplace_back(k, v); break; }
            case 'u': {                                                                 // SETITEMS
                auto xs = pop_mark(at);
                if (xs.size() % 2) fail(at, "odd SETITEMS");
                auto& d = dict(at, top(at));
                for (std::size_t n = 0; n < xs.size(); n += 2) d.emplace_back(xs[n], xs[n + 1]);
                break;
            }
            case 'c': { auto v = make(K::global); v->text = line(at); v->name = line(at); push(v); break; }
            case 0x93: {                                                                // STACK_GLOBAL
                auto name = pop(at), module = pop(at);
                if (name->kind != K::text || module->kind != K::text) fail(at, "STACK_GLOBAL needs two strings");
                auto v = make(K::global);
                v->text = module->text;
                v->name = name->text;
                push(v);
                break;
            }
            case 'R': case 0x81: {                                                      // REDUCE, NEWOBJ: recorded, never called
                auto v = make(K::object);
                v->args = pop(at);
                v->callable = pop(at);
                if (v->args->kind != K::tuple) fail(at, "REDUCE arguments are not a tuple");
                push(v);
                break;
            }
            case 'b': {                                                                 // BUILD
                auto st = pop(at);
                auto t = top(at);
                if (t->kind != K::object || t->state) fail(at, "BUILD on a non-object or twice");
                t->state = st;
                break;
            }
            case 'Q': { auto v = make(K::persistent); v->persistent_id = pop(at); push(v); break; } // BINPERSID
            case '.':                                                                   // STOP
                if (stack_.size() != 1) fail(at, "STOP with a non-singleton stack");
                if (pos_ != d_.size()) fail(pos_, "trailing bytes after STOP");
                return pop(at);
            default: fail(at, "unsupported opcode 0x" + hex(op));
            }
        }
    }

private:
    using K = PickleValue::Kind;
    static PickleRef make(K k) { auto v = std::make_shared<PickleValue>(); v->kind = k; return v; }
    static std::string hex(unsigned v) {
        const char* digits = "0123456789abcdef";
        return {digits[(v >> 4) & 15], digits[v & 15]};
    }
    [[noreturn]] static void fail(std::size_t at, const std::string& what) {
        throw std::runtime_error("pickle offset " + std::to_string(at) + ": " + what);
    }
    std::string take(std::size_t n) {
        if (n > d_.size() - pos_) fail(pos_, "truncated operand");
        std::string r(d_.substr(pos_, n));
        pos_ += n;
        return r;
    }
    unsigned u8() { return static_cast<unsigned char>(take(1)[0]); }
    std::uint64_t le(int n) {
        const auto b = take(static_cast<std::size_t>(n));
        std::uint64_t v = 0;
        for (int k = n - 1; k >= 0; --k) v = (v << 8) | static_cast<unsigned char>(b[static_cast<std::size_t>(k)]);
        return v;
    }
    std::uint64_t be(int n) {
        const auto b = take(static_cast<std::size_t>(n));
        std::uint64_t v = 0;
        for (int k = 0; k < n; ++k) v = (v << 8) | static_cast<unsigned char>(b[static_cast<std::size_t>(k)]);
        return v;
    }
    std::int64_t long_bytes(std::size_t n) {
        if (n > 8) fail(pos_, "LONG wider than int64");
        const auto b = take(n);
        std::uint64_t v = 0;
        for (std::size_t k = n; k-- > 0;) v = (v << 8) | static_cast<unsigned char>(b[k]);
        if (n > 0 && n < 8 && (static_cast<unsigned char>(b[n - 1]) & 0x80)) v |= ~std::uint64_t{0} << (8 * n);
        return static_cast<std::int64_t>(v);
    }
    std::string line(std::size_t at) {
        const auto e = d_.find('\n', pos_);
        if (e == std::string_view::npos) fail(at, "unterminated GLOBAL");
        std::string r(d_.substr(pos_, e - pos_));
        pos_ = e + 1;
        return r;
    }
    void push(PickleRef v) {
        if (stack_.size() >= (1u << 20)) fail(pos_, "stack too deep");
        stack_.push_back(std::move(v));
    }
    void push_int(std::int64_t x) { auto v = make(K::integer); v->integer = x; push(v); }
    void push_text(K k, std::string x) { auto v = make(k); v->text = std::move(x); push(v); }
    PickleRef top(std::size_t at) {
        if (stack_.empty()) fail(at, "empty stack");
        if (stack_.back()->kind == K::mark) fail(at, "MARK used as a value");
        return stack_.back();
    }
    PickleRef pop(std::size_t at) { auto v = top(at); stack_.pop_back(); return v; }
    PickleRef memo(std::size_t at, std::uint64_t k) {
        const auto it = memo_.find(k);
        if (it == memo_.end()) fail(at, "missing memo " + std::to_string(k));
        return it->second;
    }
    std::vector<PickleRef> pop_mark(std::size_t at) {
        std::vector<PickleRef> xs;
        while (true) {
            if (stack_.empty()) fail(at, "missing MARK");
            auto v = stack_.back();
            stack_.pop_back();
            if (v->kind == K::mark) break;
            xs.push_back(std::move(v));
        }
        return {xs.rbegin(), xs.rend()};
    }
    std::vector<PickleRef>& list(std::size_t at, const PickleRef& v) {
        if (v->kind != K::list) fail(at, "APPEND target is not a list");
        return v->items;
    }
    // OrderedDict and other dict-like REDUCE results receive SETITEMS too.
    std::vector<std::pair<PickleRef, PickleRef>>& dict(std::size_t at, const PickleRef& v) {
        if (v->kind != K::dict && v->kind != K::object) fail(at, "SETITEM target is not dict-like");
        return v->pairs;
    }

    std::string_view d_;
    std::size_t pos_ = 0;
    std::vector<PickleRef> stack_;
    std::map<std::uint64_t, PickleRef> memo_;
};

[[noreturn]] void tensor_fail(const std::string& why) { throw std::runtime_error("tensor reference: " + why); }

std::vector<std::int64_t> int_tuple(const PickleValue& v, const char* what) {
    if (v.kind != PickleValue::Kind::tuple) tensor_fail(std::string(what) + " is not a tuple");
    std::vector<std::int64_t> out;
    for (const auto& x : v.items) {
        if (x->kind != PickleValue::Kind::integer || x->integer < 0) tensor_fail(std::string(what) + " holds a non-integer or negative value");
        out.push_back(x->integer);
    }
    return out;
}

bool empty_hooks(const PickleValue& v) {
    // collections.OrderedDict() with no items, as torch.save writes it.
    return v.kind == PickleValue::Kind::object && v.callable && v.callable->is_global("collections", "OrderedDict") && v.args &&
           v.args->items.empty() && v.pairs.empty() && !v.state;
}

} // namespace

PickleRef parse_pickle(std::string_view bytes) { return PickleReader(bytes).run(); }

std::size_t torch_storage_element_bytes(std::string_view storage_type) {
    if (storage_type == "FloatStorage" || storage_type == "IntStorage") return 4;
    if (storage_type == "DoubleStorage" || storage_type == "LongStorage") return 8;
    if (storage_type == "HalfStorage" || storage_type == "BFloat16Storage" || storage_type == "ShortStorage") return 2;
    if (storage_type == "BoolStorage" || storage_type == "ByteStorage" || storage_type == "CharStorage") return 1;
    throw std::runtime_error("unsupported storage type " + std::string(storage_type));
}

bool TorchTensorRef::contiguous() const {
    std::int64_t expect = 1;
    for (std::size_t n = shape.size(); n-- > 0;) {
        if (shape[n] != 1 && stride[n] != expect) return false;
        expect *= shape[n];
    }
    return true;
}

std::int64_t TorchTensorRef::numel() const {
    std::int64_t n = 1;
    for (const auto d : shape) n *= d;
    return n;
}

TorchTensorRef torch_tensor_ref(const PickleValue& value) {
    using K = PickleValue::Kind;
    if (value.kind != K::object || !value.args) tensor_fail("not a REDUCE object");
    TorchTensorRef ref;
    const PickleValue* v = &value;
    if (v->callable->is_global("torch._utils", "_rebuild_parameter")) {
        // (data, requires_grad, backward_hooks)
        if (v->args->items.size() != 3 || v->args->items[1]->kind != K::boolean || !empty_hooks(*v->args->items[2]) || v->state)
            tensor_fail("_rebuild_parameter is not (tensor, requires_grad, empty hooks)");
        ref.parameter = true;
        v = v->args->items[0].get();
        if (v->kind != K::object || !v->args) tensor_fail("parameter data is not a REDUCE object");
    }
    if (!v->callable->is_global("torch._utils", "_rebuild_tensor_v2")) tensor_fail("callable is not torch._utils._rebuild_tensor_v2");
    const auto& a = v->args->items;
    if (a.size() != 6 && a.size() != 7) tensor_fail("_rebuild_tensor_v2 takes 6 or 7 arguments");
    if (a[0]->kind != K::persistent || !a[0]->persistent_id || a[0]->persistent_id->kind != K::tuple) tensor_fail("storage is not a persistent id");
    const auto& pid = a[0]->persistent_id->items; // ('storage', type, key, location, numel)
    if (pid.size() != 5 || pid[0]->kind != K::text || pid[0]->text != "storage" || pid[1]->kind != K::global || pid[1]->text != "torch" ||
        pid[2]->kind != K::text || pid[3]->kind != K::text || pid[4]->kind != K::integer || pid[4]->integer < 0)
        tensor_fail("persistent id is not ('storage', torch.<Type>Storage, key, location, numel)");
    ref.storage_type = pid[1]->name;
    ref.storage_key = pid[2]->text;
    ref.location = pid[3]->text;
    ref.storage_numel = pid[4]->integer;
    if (ref.storage_key.empty() || ref.storage_key.find_first_not_of("0123456789") != std::string::npos)
        tensor_fail("storage key is not a decimal record name");
    if (a[1]->kind != K::integer || a[1]->integer < 0) tensor_fail("storage offset is not a non-negative integer");
    ref.offset = a[1]->integer;
    ref.shape = int_tuple(*a[2], "size");
    ref.stride = int_tuple(*a[3], "stride");
    if (ref.shape.size() != ref.stride.size()) tensor_fail("size and stride ranks differ");
    if (a[4]->kind != K::boolean || !empty_hooks(*a[5])) tensor_fail("requires_grad or backward hooks are not the plain form");
    if (a.size() == 7 && !(a[6]->dict() && a[6]->dict()->empty())) tensor_fail("tensor metadata is not empty");
    if (v->state) tensor_fail("tensor has BUILD state");
    // Extent of the view in elements, with overflow checks.
    std::int64_t extent = 0;
    for (std::size_t n = 0; n < ref.shape.size(); ++n) {
        if (ref.shape[n] == 0) { extent = -1; break; }
        const auto span = ref.shape[n] - 1;
        if (ref.stride[n] != 0 && span > (std::numeric_limits<std::int64_t>::max() - extent) / ref.stride[n]) tensor_fail("extent overflows");
        extent += span * ref.stride[n];
    }
    std::int64_t numel = 1;
    for (const auto d : ref.shape) {
        if (d != 0 && numel > std::numeric_limits<std::int64_t>::max() / d) tensor_fail("element count overflows");
        numel *= d;
    }
    if (numel > 0 && (ref.offset > ref.storage_numel || extent >= ref.storage_numel - ref.offset)) tensor_fail("view exceeds its storage");
    return ref;
}

TorchCheckpoint::TorchCheckpoint(std::string bytes) : zip_(std::move(bytes)) {
    if (zip_.entries().empty()) throw std::runtime_error("checkpoint: empty archive");
    const auto& first = zip_.entries().front().name;
    const auto slash = first.find('/');
    if (slash == std::string::npos || slash == 0) throw std::runtime_error("checkpoint: entries are not under one archive directory");
    archive_ = first.substr(0, slash);
    for (const auto& e : zip_.entries())
        if (e.name.size() <= archive_.size() + 1 || e.name.compare(0, archive_.size() + 1, archive_ + "/") != 0)
            throw std::runtime_error("checkpoint: entry " + e.name + " is outside archive directory " + archive_);
    if (zip_.contains(archive_ + "/byteorder") && record("byteorder") != "little")
        throw std::runtime_error("checkpoint: byteorder is not little");
    root_ = parse_pickle(record("data.pkl"));
}

std::string_view TorchCheckpoint::record(std::string_view name) const { return zip_.data(archive_ + "/" + std::string(name)); }

std::string_view TorchCheckpoint::storage(const TorchTensorRef& t) const {
    const auto bytes = zip_.data(archive_ + "/data/" + t.storage_key);
    const auto element = torch_storage_element_bytes(t.storage_type);
    if (static_cast<std::uint64_t>(t.storage_numel) > bytes.size() / element || bytes.size() != static_cast<std::size_t>(t.storage_numel) * element)
        throw std::runtime_error("checkpoint: storage " + t.storage_key + " holds " + std::to_string(bytes.size()) + " bytes, not " +
                                 std::to_string(t.storage_numel) + " x " + std::to_string(element));
    return bytes;
}

} // namespace saccade
