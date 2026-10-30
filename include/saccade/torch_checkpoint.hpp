#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace saccade {

// Read-only, code-free access to a torch.save zip checkpoint (docs/HEAD_TRT.md
// §2). The zip reader accepts only STORED entries and checks every CRC-32; the
// pickle reader interprets the container opcodes torch.save emits and records
// GLOBAL references by name. Nothing here resolves a global, imports a class,
// evaluates a REDUCE or runs Python: a REDUCE stays an object node holding its
// callable name and arguments.
struct ZipEntry {
    std::string name;
    std::uint64_t data_offset = 0, size = 0;
    std::uint32_t crc32 = 0;
};

std::uint32_t crc32(std::string_view bytes);

class StoredZip {
public:
    // Parses the end-of-central-directory record (zip64 aware), the central
    // directory and every local header. Throws std::runtime_error for a
    // compressed, encrypted or data-descriptor entry, a duplicate name, a
    // local/central header mismatch, overlapping or out-of-range data, or a
    // CRC-32 mismatch.
    explicit StoredZip(std::string bytes);
    const std::vector<ZipEntry>& entries() const { return entries_; }
    bool contains(std::string_view name) const { return index_.find(name) != index_.end(); }
    std::string_view data(std::string_view name) const; // throws when absent

private:
    std::string bytes_;
    std::vector<ZipEntry> entries_;
    std::map<std::string, std::size_t, std::less<>> index_;
};

struct PickleValue;
using PickleRef = std::shared_ptr<PickleValue>;
struct PickleValue {
    enum class Kind { none, boolean, integer, real, text, bytes, list, tuple, dict, global, object, persistent, mark };
    Kind kind = Kind::none;
    bool boolean = false;
    std::int64_t integer = 0;
    double real = 0;
    std::string text, name;                             // text/bytes payload; global module and name
    std::vector<PickleRef> items;                       // list, tuple
    std::vector<std::pair<PickleRef, PickleRef>> pairs; // dict; SETITEMS on a dict-like object
    PickleRef callable, args, state;                    // object: REDUCE/NEWOBJ callable and args, BUILD state
    PickleRef persistent_id;

    // Dict pairs of a dict or of a dict-like object (OrderedDict), else null.
    const std::vector<std::pair<PickleRef, PickleRef>>* dict() const;
    // Value of a text key in a dict-like value; null when absent. Throws when the
    // key occurs more than once.
    const PickleValue* find(std::string_view key) const;
    bool is_global(std::string_view module, std::string_view global) const;
};

// Throws std::runtime_error with the opcode offset for any unsupported opcode,
// malformed operand, stack or memo misuse, or trailing bytes after STOP.
PickleRef parse_pickle(std::string_view bytes);

// torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad,
// backward_hooks[, metadata]), optionally wrapped in _rebuild_parameter.
struct TorchTensorRef {
    std::string storage_type;           // e.g. FloatStorage
    std::string storage_key, location;  // archive data/<key>; device recorded only
    std::int64_t storage_numel = 0, offset = 0;
    std::vector<std::int64_t> shape, stride;
    bool parameter = false;
    bool contiguous() const;
    std::int64_t numel() const;
};
// Throws std::runtime_error when the value is not exactly that structure, has
// negative or overflowing dimensions, or non-empty backward hooks.
TorchTensorRef torch_tensor_ref(const PickleValue& value);
std::size_t torch_storage_element_bytes(std::string_view storage_type);

class TorchCheckpoint {
public:
    // Every entry must sit under one top-level directory that holds data.pkl;
    // a byteorder record, when present, must read "little".
    explicit TorchCheckpoint(std::string bytes);
    const std::string& archive() const { return archive_; }
    const PickleValue& root() const { return *root_; }
    const StoredZip& zip() const { return zip_; }
    // archive/<name>; throws when absent.
    std::string_view record(std::string_view name) const;
    // The tensor's storage bytes; the record must hold exactly storage_numel
    // elements of the storage type, and offset + extent must lie inside it.
    std::string_view storage(const TorchTensorRef& tensor) const;

private:
    StoredZip zip_;
    std::string archive_;
    PickleRef root_;
};

} // namespace saccade
