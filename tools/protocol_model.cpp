// Finite happens-before model for docs/ASYNC_CONTROL.md. No CUDA runtime needed.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
enum class Lane { host, detect, main, init };
enum class Fault {
  none, init_sync, mi_ready_wait, mi_ready_record, mi_consumed_wait,
  mi_consumed_record, cs_ready_wait, cs_ready_record, cs_consumed_wait,
  cs_consumed_record, staged_wait, staged_record, packet_wait, packet_record,
  terminal_wait, main_first
};
constexpr std::array faults{
  Fault::init_sync, Fault::mi_ready_wait, Fault::mi_ready_record,
  Fault::mi_consumed_wait, Fault::mi_consumed_record, Fault::cs_ready_wait,
  Fault::cs_ready_record, Fault::cs_consumed_wait, Fault::cs_consumed_record,
  Fault::staged_wait, Fault::staged_record, Fault::packet_wait,
  Fault::packet_record, Fault::terminal_wait, Fault::main_first};
std::string fault_name(Fault f) {
  switch (f) {
    case Fault::none: return "none";
    case Fault::init_sync: return "remove-init-sync";
    case Fault::mi_ready_wait: return "remove-mi-ready-wait";
    case Fault::mi_ready_record: return "remove-mi-ready-record";
    case Fault::mi_consumed_wait: return "remove-mi-consumed-wait";
    case Fault::mi_consumed_record: return "remove-mi-consumed-record";
    case Fault::cs_ready_wait: return "remove-cs-ready-wait";
    case Fault::cs_ready_record: return "remove-cs-ready-record";
    case Fault::cs_consumed_wait: return "remove-cs-consumed-wait";
    case Fault::cs_consumed_record: return "remove-cs-consumed-record";
    case Fault::staged_wait: return "remove-staged-host-wait";
    case Fault::staged_record: return "remove-staged-record";
    case Fault::packet_wait: return "remove-packet-host-wait";
    case Fault::packet_record: return "remove-packet-record";
    case Fault::terminal_wait: return "remove-terminal-waits";
    case Fault::main_first: return "enqueue-main-before-detect";
  }
  throw std::logic_error("unknown fault");
}
struct Config { int mi, cs, stage, packet; bool cancel; };
std::string describe(Config c) {
  return "MI=" + std::to_string(c.mi) + ",CS=" + std::to_string(c.cs) +
    ",H=" + std::to_string(c.stage) + ",RP=" + std::to_string(c.packet) +
    (c.cancel ? ",D-only-tail" : ",normal");
}
struct Node { std::string label; std::vector<int> next; };
struct Access { int node; bool write; int producer; };
struct Event { int node; int frame; };
struct Result { bool pass; std::string reason; };
class Model {
 public:
  Model(Config c, Fault f) : config(c), fault(f) {}
  Result run() {
    build();
    const int n = static_cast<int>(nodes.size());
    std::vector<int> degree(n, 0);
    for (const auto& node : nodes) for (int dst : node.next) ++degree[dst];
    std::queue<int> ready;
    for (int i=0; i<n; ++i) if (degree[i] == 0) ready.push(i);
    std::vector<int> order;
    while (!ready.empty()) {
      const int at = ready.front(); ready.pop(); order.push_back(at);
      for (int dst : nodes[at].next) if (--degree[dst] == 0) ready.push(dst);
    }
    if (static_cast<int>(order.size()) != n) return {false, "happens-before cycle"};
    // Transitive closure by reverse topological bitset unions.
    const auto words = (static_cast<std::size_t>(n) + 63) / 64;
    std::vector<std::vector<std::uint64_t>> reach(n, std::vector<std::uint64_t>(words));
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      const int src = *it;
      for (int dst : nodes[src].next) {
        reach[src][static_cast<std::size_t>(dst)/64] |= std::uint64_t{1} << (dst%64);
        for (std::size_t w=0; w<words; ++w) reach[src][w] |= reach[dst][w];
      }
    }
    auto before = [&](int a, int b) {
      return (reach[a][static_cast<std::size_t>(b)/64] & (std::uint64_t{1} << (b%64))) != 0;
    };
    if (!binding_error.empty()) return {false, binding_error};
    for (const auto& [name, accesses] : memory) {
      for (std::size_t i=0; i<accesses.size(); ++i) {
        const auto& a = accesses[i];
        if (!a.write && (a.producer < 0 || !before(a.producer, a.node))) {
          return {false, name + ": expected producer not before " + nodes[a.node].label};
        }
        for (std::size_t j=i+1; j<accesses.size(); ++j) {
          const auto& b = accesses[j];
          // A modeled operation performs its read phase before its write phase.
          // Intra-kernel block/thread correctness is deliberately outside this model.
          if (a.node != b.node && (a.write || b.write) && !before(a.node,b.node) && !before(b.node,a.node)) {
            return {false, name + ": unordered " + nodes[a.node].label + " / " + nodes[b.node].label};
          }
        }
        // All non-source writers must precede the source or follow the read.
        if (!a.write) for (const auto& w : accesses) {
          if (!w.write || w.node == a.producer || w.node == a.node) continue;
          if (!before(w.node, a.producer) && !before(a.node, w.node)) {
            return {false, name + ": wrong producer may reach " + nodes[a.node].label};
          }
        }
      }
    }
    return {true, {}};
  }
 private:
  Config config;
  Fault fault;
  std::vector<Node> nodes;
  std::array<int,4> last{-1,-1,-1,-1};
  std::map<std::string,Event> events;
  std::map<std::string,std::vector<Access>> memory;
  std::map<std::string,int> writer;
  std::string binding_error;
  int add(Lane lane, const std::string& label) {
    const int id = static_cast<int>(nodes.size()); nodes.push_back({label,{}});
    const auto slot = static_cast<std::size_t>(lane);
    if (last[slot] >= 0) edge(last[slot],id);
    last[slot] = id;
    return id;
  }
  void edge(int src, int dst) {
    if (src >= 0 && dst >= 0) nodes[src].next.push_back(dst);
  }
  int host(const std::string& label) { return add(Lane::host,label); }
  int enqueue(Lane lane, const std::string& label) {
    const int submit = host("enqueue " + label);
    const int op = add(lane,label); edge(submit,op); return op;
  }
  void write(int op, const std::string& name) {
    memory[name].push_back({op,true,-1}); writer[name]=op;
  }
  void read(int op, const std::string& name) {
    const auto found=writer.find(name);
    memory[name].push_back({op,false, found == writer.end() ? -1 : found->second});
  }
  void record(Lane lane, const std::string& name, int frame, Fault remove) {
    if (remove != Fault::none && fault == remove) return;
    const int op = enqueue(lane, "record " + name + " frame=" + std::to_string(frame));
    events[name] = {op,frame};
  }
  void wait(Lane lane, const std::string& name, int expected, Fault remove) {
    if (remove != Fault::none && fault == remove) return;
    const int op = lane == Lane::host ? host("host wait " + name) : enqueue(lane,"wait " + name);
    const auto found=events.find(name);
    // CUDA binds at enqueue time; an unrecorded event completes immediately.
    // Check generation separately, since another edge can mask the memory race.
    if (found == events.end() || found->second.frame != expected) {
      if (binding_error.empty()) binding_error = name + ": wait bound to " +
        (found == events.end() ? "unrecorded event" : "wrong generation " + std::to_string(found->second.frame)) +
        ", expected " + std::to_string(expected);
    }
    if (found != events.end()) edge(found->second.node,op);
  }
  static std::string slot(const std::string& name, int i) { return name+"["+std::to_string(i)+"]"; }
  void initialize() {
    // Separate initialization executor avoids relying on a legacy default stream.
    const int op=enqueue(Lane::init,"initialize all session storage");
    for (int i=0; i<config.stage; ++i) {
      write(op,slot("STAGE",i)); write(op,slot("NVDEV",i));
    }
    for (int i=0; i<config.mi; ++i) {
      write(op,slot("MI.payload",i)); write(op,slot("MI.header",i));
    }
    for (int i=0; i<config.cs; ++i) {
      write(op,slot("CS.payload",i)); write(op,slot("CS.header",i));
    }
    for (int i=0; i<config.packet; ++i) write(op,slot("HP",i));
    for (int i=0; i<2; ++i) { write(op,slot("TS",i)); write(op,slot("SPEC",i)); }
    for (const auto* name : {"RAW","DESC","MIN","RAWP","FFT_TMP","PRIOR","DB","CTL","RP"}) write(op,name);
    if (fault != Fault::init_sync) edge(op,host("device initialization synchronize"));
  }
  void detect(int frame) {
    const int h=frame%config.stage, m=frame%config.mi, c=frame%config.cs;
    const std::string suffix=" frame="+std::to_string(frame);
    int op=enqueue(Lane::detect,"H2D and nvJPEG transfer"+suffix);
    read(op,slot("STAGE",h)); write(op,slot("NVDEV",h)); write(op,"DESC");
    record(Lane::detect,slot("staged",h),frame,Fault::staged_record);
    op=enqueue(Lane::detect,"decode"+suffix); read(op,slot("NVDEV",h)); write(op,"RAW");
    if (frame>=config.mi) wait(Lane::detect,slot("mi_consumed",m),frame-config.mi,Fault::mi_consumed_wait);
    op=enqueue(Lane::detect,"F05 payload"+suffix); read(op,"RAW"); read(op,"DESC"); write(op,slot("MI.payload",m));
    op=enqueue(Lane::detect,"F05 header"+suffix); write(op,slot("MI.header",m));
    record(Lane::detect,slot("mi_ready",m),frame,Fault::mi_ready_record);
    op=enqueue(Lane::detect,"F02"+suffix); read(op,"RAW"); write(op,"MIN");
    op=enqueue(Lane::detect,"F03"+suffix); read(op,"MIN"); write(op,"RAWP");
    if (frame>=config.cs) wait(Lane::detect,slot("cs_consumed",c),frame-config.cs,Fault::cs_consumed_wait);
    op=enqueue(Lane::detect,"F04 payload"+suffix); read(op,"RAWP"); read(op,"DESC"); write(op,slot("CS.payload",c));
    op=enqueue(Lane::detect,"F04 header"+suffix); write(op,slot("CS.header",c));
    record(Lane::detect,slot("cs_ready",c),frame,Fault::cs_ready_record);
  }
  void main_stream(int frame) {
    const int m=frame%config.mi, c=frame%config.cs, r=frame%config.packet;
    const std::string suffix=" frame="+std::to_string(frame);
    int op=enqueue(Lane::main,"begin"+suffix); read(op,"CTL"); write(op,"CTL");
    wait(Lane::main,slot("mi_ready",m),frame,Fault::mi_ready_wait);
    op=enqueue(Lane::main,"MI pre-header"+suffix); read(op,slot("MI.header",m));
    op=enqueue(Lane::main,"cuFFT to scratch"+suffix); read(op,slot("MI.payload",m)); write(op,"FFT_TMP");
    op=enqueue(Lane::main,"MI post-header"+suffix); read(op,slot("MI.header",m));
    record(Lane::main,slot("mi_consumed",m),frame,Fault::mi_consumed_record);
    op=enqueue(Lane::main,"guarded spectrum delivery and correlation"+suffix);
    read(op,"FFT_TMP"); if (frame>0) read(op,slot("SPEC",(frame-1)%2)); write(op,slot("SPEC",frame%2));
    op=enqueue(Lane::main,"F08a"+suffix);
    read(op,slot("TS",frame==0 ? 1 : (frame-1)%2)); write(op,slot("TS",frame%2)); write(op,"PRIOR");
    wait(Lane::main,slot("cs_ready",c),frame,Fault::cs_ready_wait);
    op=enqueue(Lane::main,"CS pre-header"+suffix); read(op,slot("CS.header",c));
    op=enqueue(Lane::main,"F04b"+suffix); read(op,slot("CS.payload",c)); read(op,"PRIOR"); write(op,"DB");
    op=enqueue(Lane::main,"CS post-header"+suffix); read(op,slot("CS.header",c));
    record(Lane::main,slot("cs_consumed",c),frame,Fault::cs_consumed_record);
    op=enqueue(Lane::main,"F07 F08b"+suffix); read(op,"DB"); read(op,slot("TS",frame%2)); write(op,slot("TS",frame%2));
    op=enqueue(Lane::main,"commit"+suffix); read(op,"CTL"); write(op,"CTL");
    op=enqueue(Lane::main,"packet"+suffix); read(op,"CTL"); read(op,slot("TS",frame%2)); write(op,"RP");
    op=enqueue(Lane::main,"D2H"+suffix); read(op,"RP"); write(op,slot("HP",r));
    record(Lane::main,slot("pkt",r),frame,Fault::packet_record);
  }
  void consume(int frame) {
    const int r=frame%config.packet;
    wait(Lane::host,slot("pkt",r),frame,Fault::packet_wait);
    read(host("consume packet frame="+std::to_string(frame)),slot("HP",r));
  }
  void build() {
    initialize();
    constexpr int frames=10;
    for (int frame=0; frame<frames; ++frame) {
      if (frame>=config.stage) wait(Lane::host,slot("staged",frame%config.stage),frame-config.stage,Fault::staged_wait);
      if (frame>=config.packet) consume(frame-config.packet);
      write(host("fill STAGE frame="+std::to_string(frame)),slot("STAGE",frame%config.stage));
      const bool only_d = config.cancel && frame==frames-1;
      if (fault==Fault::main_first && !only_d) { main_stream(frame); detect(frame); }
      else { detect(frame); if (!only_d) main_stream(frame); }
    }
    const int packets=frames-(config.cancel ? 1 : 0);
    // An incomplete D-only tail can already have consumed an older HP slot.
    const int consumed=std::max(0,frames-config.packet);
    for (int frame=consumed; frame<packets; ++frame) consume(frame);
    record(Lane::detect,"D_terminal",frames,Fault::none);
    record(Lane::main,"M_terminal",frames,Fault::none);
    if (fault!=Fault::terminal_wait) {
      wait(Lane::host,"D_terminal",frames,Fault::none);
      wait(Lane::host,"M_terminal",frames,Fault::none);
    }
    const int release=host("release or reset session storage");
    // Destruction conflicts with every outstanding use, including D-only work.
    for (const auto& [name, accesses] : memory) {
      (void)accesses; write(release,name);
    }
  }
};
std::vector<Config> configurations() {
  std::vector<Config> out;
  for (int mi=1; mi<=3; ++mi) for (int cs=1; cs<=3; ++cs)
    for (int h=1; h<=3; ++h) for (int rp=1; rp<=4; ++rp)
      for (bool cancel : {false,true}) out.push_back({mi,cs,h,rp,cancel});
  return out;
}
} // namespace
int main(int argc, char** argv) {
  Fault selected=Fault::none;
  bool mutate=false;
  if (argc==3 && std::string(argv[1])=="--mutate") {
    for (const auto f : faults) if (fault_name(f)==argv[2]) { selected=f; mutate=true; }
  }
  if ((!mutate && argc>2) || (argc==2 && std::string(argv[1])!="--self-test")) {
    std::cerr << "usage: protocol_model [--self-test | --mutate NAME]\n"; return 2;
  }
  try {
    const auto configs=configurations();
    for (const auto c : configs) {
      const auto result=Model(c,selected).run();
      if (!result.pass) {
        std::cerr << "FAIL " << (mutate ? fault_name(selected) : "baseline") << ' '
                  << describe(c) << ": " << result.reason << '\n'; return 1;
      }
    }
    if (mutate) {
      std::cout << "PASS mutated model in all configurations\n"; return 0;
    }
    std::cout << "PASS baseline " << configs.size() << "/" << configs.size()
              << " configurations; 10 frames each; conflicts, producer identity, event binding, DAG\n";
    for (const auto fault : faults) {
      int rejected=0; std::string witness;
      for (const auto c : configs) {
        const auto result=Model(c,fault).run();
        if (!result.pass) {
          ++rejected;
          if (witness.empty()) witness=describe(c)+": "+result.reason;
        }
      }
      if (rejected==0) {
        std::cerr << "FAIL undetected mutation " << fault_name(fault) << '\n'; return 1;
      }
      std::cout << "PASS mutation " << fault_name(fault) << " rejected " << rejected
                << "/" << configs.size() << "; witness " << witness << '\n';
    }
    std::cout << "PASS " << faults.size() << " mutation families detected\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL model: " << e.what() << '\n'; return 1;
  }
}
