#include "saccade/file_io.hpp"
#include "onnx.pb.h"

#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

// Deliberately a structural inspector, not an ONNX semantic checker or an
// inference runner. Build separately against the pinned upstream ONNX schema;
// the default project build does not acquire a protobuf dependency.
namespace {
void tensor(const char* kind, int order, const onnx::ValueInfoProto& value) {
    std::cout << kind << '\t' << order << '\t' << value.name();
    if (!value.has_type() || !value.type().has_tensor_type()) {
        std::cout << "\tnon-tensor-or-unknown\n";
        return;
    }
    const auto& type = value.type().tensor_type();
    std::cout << '\t' << onnx::TensorProto_DataType_Name(type.elem_type()) << '\t';
    for (int i = 0; i < type.shape().dim_size(); ++i) {
        if (i != 0) std::cout << ',';
        const auto& dim = type.shape().dim(i);
        if (dim.has_dim_value()) std::cout << dim.dim_value();
        else if (dim.has_dim_param()) std::cout << dim.dim_param();
        else std::cout << '?';
    }
    std::cout << '\n';
}

void graph(const onnx::GraphProto& g, std::string scope,
           std::map<std::string, std::size_t>& counts) {
    std::cout << "graph\t" << scope << "\tname=" << g.name()
              << "\tnodes=" << g.node_size() << "\tinitializers=" << g.initializer_size() << '\n';
    for (int i = 0; i < g.input_size(); ++i) tensor("input", i, g.input(i));
    for (int i = 0; i < g.output_size(); ++i) tensor("output", i, g.output(i));
    for (int i = 0; i < g.value_info_size(); ++i) tensor("value_info", i, g.value_info(i));
    for (int i = 0; i < g.node_size(); ++i) {
        const auto& n = g.node(i);
        ++counts[n.domain() + "::" + n.op_type()];
        std::cout << "node\t" << scope << '\t' << i << '\t' << n.name() << '\t'
                  << n.domain() << "::" << n.op_type() << "\tinputs=";
        for (const auto& input : n.input()) std::cout << input << ';';
        std::cout << "\toutputs=";
        for (const auto& output : n.output()) std::cout << output << ';';
        std::cout << "\tattributes=";
        for (const auto& attr : n.attribute()) {
            std::cout << attr.name() << ':' << onnx::AttributeProto_AttributeType_Name(attr.type()) << ';';
        }
        std::cout << '\n';
        for (const auto& attr : n.attribute()) {
            if (attr.has_g()) graph(attr.g(), scope + '/' + std::to_string(i) + '/' + attr.name(), counts);
            for (int j = 0; j < attr.graphs_size(); ++j)
                graph(attr.graphs(j), scope + '/' + std::to_string(i) + '/' + attr.name() + '/' +
                      std::to_string(j), counts);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: onnx_contract_inspect MODEL.onnx");
        const auto bytes = saccade::read_text_file(argv[1]);
        if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("model exceeds protobuf ParseFromArray byte range");
        onnx::ModelProto model;
        if (!model.ParseFromArray(bytes.data(), static_cast<int>(bytes.size())) || !model.has_graph())
            throw std::invalid_argument(std::string(argv[1]) + ": invalid ONNX protobuf or missing graph");
        std::cout << "scope=structural-inspection-only\n"
                  << "ir_version=" << model.ir_version() << '\n'
                  << "producer_name=" << model.producer_name() << '\n'
                  << "producer_version=" << model.producer_version() << '\n'
                  << "model_version=" << model.model_version() << '\n'
                  << "local_functions=" << model.functions_size() << '\n';
        for (const auto& o : model.opset_import())
            std::cout << "opset\t" << o.domain() << '\t' << o.version() << '\n';
        for (const auto& entry : model.metadata_props())
            std::cout << "metadata\t" << entry.key() << '\t' << entry.value() << '\n';
        std::map<std::string, std::size_t> counts;
        graph(model.graph(), "root", counts);
        for (const auto& [op, count] : counts)
            std::cout << "operator_count\t" << op << '\t' << count << '\n';
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot write or flush inspector output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "onnx_contract_inspect: " << error.what() << '\n';
        return 1;
    }
}
