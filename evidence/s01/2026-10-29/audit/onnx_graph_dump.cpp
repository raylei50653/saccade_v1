// Read-only ONNX graph audit: opsets, graph I/O, every node with attribute
// values, every initializer with dims and SHA-256 of its little-endian bytes.
// Two-dimensional float initializers also report the hash of their transpose,
// so MatMul weights can be matched against row-major Linear weights. No
// semantic checking, shape inference or execution.
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"
#include "onnx.pb.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string dims(const onnx::TensorProto& t) {
    std::string r = "[";
    for (int i = 0; i < t.dims_size(); ++i) r += (i ? "," : "") + std::to_string(t.dims(i));
    return r + "]";
}

std::string bytes_of(const onnx::TensorProto& t) {
    if (t.data_location() == onnx::TensorProto::EXTERNAL) throw std::runtime_error(t.name() + ": external data not handled by this audit");
    if (t.has_raw_data()) return t.raw_data();
    std::string out;
    if (t.data_type() == onnx::TensorProto::FLOAT) {
        for (float v : t.float_data()) { char b[4]; std::memcpy(b, &v, 4); out.append(b, 4); }
    } else if (t.data_type() == onnx::TensorProto::INT64) {
        for (auto v : t.int64_data()) { char b[8]; std::memcpy(b, &v, 8); out.append(b, 8); }
    } else if (t.data_type() == onnx::TensorProto::INT32) {
        for (auto v : t.int32_data()) { char b[4]; std::memcpy(b, &v, 4); out.append(b, 4); }
    }
    return out;
}

std::string values(const onnx::TensorProto& t, const std::string& b) {
    std::ostringstream o;
    if (t.data_type() == onnx::TensorProto::FLOAT && b.size() <= 32 * 4) {
        o << " values=[";
        for (std::size_t i = 0; i < b.size() / 4; ++i) { float v; std::memcpy(&v, b.data() + 4 * i, 4); o << (i ? "," : "") << std::setprecision(9) << v; }
        o << "]";
    } else if (t.data_type() == onnx::TensorProto::INT64 && b.size() <= 32 * 8) {
        o << " values=[";
        for (std::size_t i = 0; i < b.size() / 8; ++i) { long long v; std::memcpy(&v, b.data() + 8 * i, 8); o << (i ? "," : "") << v; }
        o << "]";
    }
    return o.str();
}

std::string transpose_hash(const onnx::TensorProto& t, const std::string& b) {
    if (t.data_type() != onnx::TensorProto::FLOAT || t.dims_size() != 2) return "";
    const auto rows = static_cast<std::size_t>(t.dims(0)), cols = static_cast<std::size_t>(t.dims(1));
    if (rows * cols * 4 != b.size()) throw std::runtime_error(t.name() + ": byte size does not match dims");
    std::string out(b.size(), '\0');
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) std::memcpy(&out[(c * rows + r) * 4], &b[(r * cols + c) * 4], 4);
    return " sha256_transposed=" + saccade::sha256_hex(out);
}

std::string attribute(const onnx::AttributeProto& a) {
    std::ostringstream o;
    o << a.name() << ':' << onnx::AttributeProto_AttributeType_Name(a.type()) << '=';
    switch (a.type()) {
    case onnx::AttributeProto::INT: o << a.i(); break;
    case onnx::AttributeProto::FLOAT: o << std::setprecision(9) << a.f(); break;
    case onnx::AttributeProto::STRING: o << '"' << a.s() << '"'; break;
    case onnx::AttributeProto::INTS: o << '['; for (int i = 0; i < a.ints_size(); ++i) o << (i ? "," : "") << a.ints(i); o << ']'; break;
    case onnx::AttributeProto::FLOATS: o << '['; for (int i = 0; i < a.floats_size(); ++i) o << (i ? "," : "") << std::setprecision(9) << a.floats(i); o << ']'; break;
    case onnx::AttributeProto::TENSOR: { const auto b = bytes_of(a.t()); o << "tensor" << dims(a.t()) << values(a.t(), b); break; }
    default: o << "unprinted"; break;
    }
    return o.str();
}

void value(const char* kind, int order, const onnx::ValueInfoProto& v) {
    std::cout << kind << '\t' << order << '\t' << v.name();
    if (!v.has_type() || !v.type().has_tensor_type()) { std::cout << "\tnon-tensor-or-unknown\n"; return; }
    const auto& type = v.type().tensor_type();
    std::cout << '\t' << onnx::TensorProto_DataType_Name(type.elem_type()) << "\t[";
    for (int i = 0; i < type.shape().dim_size(); ++i) {
        const auto& d = type.shape().dim(i);
        std::cout << (i ? "," : "");
        if (d.has_dim_value()) std::cout << d.dim_value(); else if (d.has_dim_param()) std::cout << d.dim_param(); else std::cout << '?';
    }
    std::cout << "]\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: onnx_graph_dump MODEL.onnx");
        const auto bytes = saccade::read_text_file(argv[1]);
        if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("model exceeds protobuf ParseFromArray byte range");
        onnx::ModelProto model;
        if (!model.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())) || !model.has_graph()) throw std::invalid_argument("invalid ONNX protobuf or missing graph");
        std::cout << "scope=structural-dump-only; no semantic check, inference or execution\n"
                  << "ir_version\t" << model.ir_version() << "\nproducer\t" << model.producer_name() << '\t' << model.producer_version() << '\n';
        for (const auto& o : model.opset_import()) std::cout << "opset\t\"" << o.domain() << "\"\t" << o.version() << '\n';
        for (const auto& m : model.metadata_props()) std::cout << "metadata\t" << m.key() << '\t' << m.value() << '\n';
        const auto& g = model.graph();
        std::cout << "graph\tnodes=" << g.node_size() << "\tinitializers=" << g.initializer_size() << "\tfunctions=" << model.functions_size() << '\n';
        for (int i = 0; i < g.input_size(); ++i) value("input", i, g.input(i));
        for (int i = 0; i < g.output_size(); ++i) value("output", i, g.output(i));
        for (int i = 0; i < g.value_info_size(); ++i) value("value_info", i, g.value_info(i));
        for (int i = 0; i < g.initializer_size(); ++i) {
            const auto& t = g.initializer(i);
            const auto b = bytes_of(t);
            std::cout << "initializer\t" << i << '\t' << t.name() << '\t' << onnx::TensorProto_DataType_Name(t.data_type()) << '\t' << dims(t)
                      << "\tbytes=" << b.size() << " sha256=" << saccade::sha256_hex(b) << transpose_hash(t, b) << values(t, b) << '\n';
        }
        for (int i = 0; i < g.node_size(); ++i) {
            const auto& n = g.node(i);
            std::cout << "node\t" << i << '\t' << n.name() << '\t' << n.domain() << "::" << n.op_type() << "\tinputs=";
            for (const auto& x : n.input()) std::cout << x << ';';
            std::cout << "\toutputs=";
            for (const auto& x : n.output()) std::cout << x << ';';
            std::cout << "\tattributes=";
            for (const auto& a : n.attribute()) std::cout << attribute(a) << ';';
            std::cout << '\n';
        }
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot write dump output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "onnx_graph_dump: " << error.what() << '\n';
        return 2;
    }
}
