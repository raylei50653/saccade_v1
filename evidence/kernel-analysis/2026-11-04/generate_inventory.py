#!/usr/bin/env python3
"""Rebuild the HR data-format inventory without GPU work.

Native capacity rows replay the pinned resource-schedule CSV. Tensor payload rows
are separate accounting classes: their sizes must never be added to native
allocations. Shapes/strides are for 1920x1080 input, HR1280x768, downscale4.
All sizes are bytes. Blank allocation bytes mean unknown/not a separate allocation.
"""
import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = ["accounting_class", "memory_kind", "resource", "parent_resource", "stage",
          "dtype", "layout", "shape", "byte_strides", "row_stride_bytes",
          "plane_stride_bytes", "record_stride_bytes", "data_offset_bytes",
          "count", "unit_bytes", "logical_payload_bytes", "total_allocation_bytes",
          "owner", "producer", "consumer", "lifetime_and_reuse", "source", "notes"]


def contiguous_strides(shape, item_bytes):
    stride, out = item_bytes, []
    for dim in reversed(shape):
        out.append(stride)
        stride *= dim
    return list(reversed(out))


def build(repo, evidence):
    rows = []
    source = repo / "evidence/resource-schedule/2026-11-04/resources.csv"
    reference = list(csv.DictReader(source.open()))
    # count/unit_bytes retains the reference's accounting units: slots, records,
    # or allocation groups. It is not a count of cudaMalloc calls.
    for r in reference:
        cls = {"allocation": "native_explicit_persistent_allocation",
               "subtotal": "native_explicit_persistent_subtotal",
               "transient": "native_explicit_transient_allocation"}[r["row_type"]]
        rows.append(dict(accounting_class=cls, memory_kind=r["memory_kind"],
                         resource=r["resource"], count=r["count"], unit_bytes=r["unit_bytes"],
                         total_allocation_bytes=r["total_bytes"], owner=r["owner"],
                         source=r["source"], notes=r["basis"] + "; count is capacity units, not cudaMalloc calls"))

    by_name = {r["resource"]: r for r in rows if "subtotal" not in r["accounting_class"]}

    def enrich(name, **kw):
        by_name[name].update(kw)

    def dense_native(name, shape, item_bytes, dtype, layout, **kw):
        strides = contiguous_strides(shape, item_bytes)
        enrich(name, shape=str(shape), byte_strides=str(strides), dtype=dtype,
               layout=layout, row_stride_bytes=strides[-2] if len(shape) > 1 else "",
               plane_stride_bytes=strides[-3] if len(shape) > 2 else "", **kw)

    dense_native("RGB8 planar", [3, 1080, 1920], 1, "uint8", "dense planar CHW RGB",
                 stage="F01/F02/F05", producer="nvJPEG HARDWARE + SM RGB conversion",
                 consumer="F02 horizontal; F05 image_kernel",
                 lifetime_and_reuse="D-owned single RGB buffer; next decode must wait prior F05 rgb_read_done",
                 notes="3 planes; pitch1920 each; planar offsets0/2073600/4147200; source src/preprocess.cpp:153-160")
    dense_native("HR FP32 model input", [1, 3, 768, 1280], 4, "float32", "dense NCHW",
                 stage="F02/F03", producer="F02 vertical", consumer="TensorRT; producer finite",
                 lifetime_and_reuse="D-owned single buffer, reused after same-stream consumers")
    dense_native("Resize horizontal intermediate", [3, 1080, 1280], 4, "float32", "dense CHW",
                 stage="F02", producer="horizontal", consumer="vertical",
                 lifetime_and_reuse="D-owned persistent workspace; payload live horizontal->vertical")
    enrich("F03 raw cls/reg", stage="F03/F04", dtype="float32", layout="one dense arena; six NCHW subranges",
           shape="six tensors; 1693440 floats total", producer="TensorRT", consumer="producer finite; F04 decode",
           lifetime_and_reuse="D-owned single arena; reuse after F04 same-stream completion",
           notes="See cls/reg subrange rows for byte offsets and strides; output order clsP3/P4/P5 then regP3/P4/P5")
    enrich("F03 plugin status regions", stage="F03", dtype="uint32 flags; uint32 done", layout="9 x 256-byte aligned regions",
           shape="9 x ScanStatus[2 words]", record_stride_bytes=256,
           producer="9 TRT plugin calls", consumer="producer aggregate",
           lifetime_and_reuse="aggregate retires every source status after reading",
           logical_payload_bytes=9*8, notes="72B actual status payload within2304B allocation; next source must freshly publish done")
    enrich("Producer status", stage="F03/F04", dtype="uint32[2]", layout="flags/done", shape="[2]", byte_strides="[4]",
           producer="producer aggregate", consumer="F04 initialize", lifetime_and_reuse="D ordered; retired/reset each frame")
    enrich("Producer scratch", stage="F03", dtype="2 x ScanStatus(uint32[2])", layout="input/output status structs",
           producer="producer initialize/finite/aggregate", consumer="producer aggregate", lifetime_and_reuse="D exclusive")
    enrich("F04 scratch", stage="F04", dtype="mixed float32/uint32/uint64", layout="fixed arrays in Workspace",
           shape="Decoded[20160],top_ids[300],full_keys[20160],compact[512],partial[20,300]",
           producer="initialize/decode/sort", consumer="sort/finalize", lifetime_and_reuse="D exclusive; arrays touched depend on qualified count",
           notes="Decoded24B AoS=483840; ids1200; fullkeys161280; compact4096; partial48000; flags12; alignment4")
    enrich("Candidate slots", stage="F04/F04b", dtype="FP32 box[4],score; uint32 class/id/flags", layout="header64 + AoS rows[300]",
           shape="[300] rows; count<=300", record_stride_bytes=32, data_offset_bytes=64,
           producer="F04 finalize", consumer="M F04b", lifetime_and_reuse="D publishes ready; M waits; D reuse only after M consumed",
           notes="9664B per slot, complete kept and suppressed pool; 2 separately allocated slots")
    enrich("MotionImage slots with headers", stage="F05/F06", dtype="float32 image; uint64 epoch/int64 frame header",
           layout="image and header are SEPARATE allocations per slot", shape="image[270,480] + header[16B]",
           byte_strides="image[1920,4]", row_stride_bytes=1920, data_offset_bytes=0,
           producer="F05 image_kernel+image_header", consumer="cuFFT forward; motion_stage",
           lifetime_and_reuse="D publishes MI ready; M publishes consumed after MI final read; payload independent of RGB",
           notes="518400B image + separate16B header per logical slot; not a header-prefixed image buffer")
    enrich("Motion Hanning tables", stage="F05", dtype="float32", layout="separate dense wx/wy arrays",
           shape="wx[480],wy[270]", byte_strides="each[4]", producer="constructor upload", consumer="F05 image_kernel",
           lifetime_and_reuse="read-only through session", notes="1920B wx+1080B wy; reference counts both as one3000B group")
    dense_native("Motion correlation", [270,480], 4, "float32", "dense real HW", stage="F06",
                 producer="cuFFT C2R inverse", consumer="reduce_correlation", lifetime_and_reuse="M exclusive scratch")
    for name, prod, cons, life in [
        ("FFT_TMP", "cuFFT R2C forward", "deliver", "M exclusive; protects committed SPEC from unguarded cuFFT writes"),
        ("Committed/working SPEC", "guarded deliver", "cross_power/current frame; next-frame previous spectrum", "two contiguous versions; advance only with successful device commit"),
        ("Cross-power spectrum", "cross_power", "cuFFT C2R inverse", "M exclusive; only scratch, not committed state")]:
        dense_native(name, [270,241], 8, "complex64 (float32 real/imag)", "dense R2C half-spectrum HW", stage="F06",
                     producer=prod, consumer=cons, lifetime_and_reuse=life)
    enrich("Motion reduction partials", stage="F06", dtype="float64 squares; float32 maximum; uint64 index; int32 numeric", layout="AoS[507]",
           shape="[507]", record_stride_bytes=32, producer="reduce_correlation", consumer="context_kernel",
           lifetime_and_reuse="M exclusive scratch", notes="reference count507 means records in one16224B allocation, not507buffers")
    enrich("Motion numeric flag", stage="F06", dtype="int32", shape="[1]", layout="scalar",
           producer="reset_numeric/deliver/cross_power", consumer="motion_stage/context_kernel", lifetime_and_reuse="M exclusive, reset per frame")
    enrich("Committed/working TrackState", stage="F08a/F08b", dtype="mixed ABI; DeviceTrack=408B",
           layout="header32 + AoS DeviceTrack[512] per version", shape="2 x {tracks[512]}", record_stride_bytes=408,
           data_offset_bytes=32, producer="F08b working state", consumer="F08a committed state",
           lifetime_and_reuse="two contiguous versions; swap logical committed slot only on success",
           notes="408B record contains FP32 x[8],P[8,8],scores/history and integer identity/lifecycle; 2versions in onecudaMalloc")
    enrich("Predicted state and priors", stage="F08a/F04b/F08b", dtype="mixed ABI tracks408B/priors32B",
           layout="header104 + tracks[512] + priors[512]", shape="tracks[512],priors[512]",
           producer="F08a", consumer="F04b/F08b", lifetime_and_reuse="M working state, never committed by itself",
           notes="track offset104, prior offset209000; prior UID+FP32xyxy+class with ABI padding")
    enrich("DetectionBatch", stage="F04b/F08b", dtype="FP32[6],uint32[2],bool[2],padding", layout="header24 + AoS[300]",
           shape="[300]; count<=300", record_stride_bytes=36, data_offset_bytes=24,
           producer="F04b", consumer="F08b", lifetime_and_reuse="M exclusive; candidates may be released after batch use")
    enrich("Tracking output rows", stage="F08b/F09", dtype="mixed integer identity/FP32xyxy,scores", layout="header8 + AoS[256]",
           shape="[256]; count<=256", record_stride_bytes=48, data_offset_bytes=8,
           producer="F08b", consumer="packet", lifetime_and_reuse="M exclusive working output")
    enrich("Tracking scratch", stage="F08a/F04b/F08b", dtype="mixed structs/int32/bool/float32", layout="fixed arrays",
           shape="tracks[512],status/matched/claims[512],removed[512],used[300],proposals[512],order[300],batch_rows[300]",
           producer="predict/batch/update", consumer="same stage", lifetime_and_reuse="borrowed exclusively by one in-flight M stage",
           notes="persistent234000B capacity, not a512x300 dense score matrix; include/saccade/tracking_cuda.hpp:76-85")
    for name in ["TrackerConfig", "TrackingSessionControl", "MotionContext", "DeviceMotionContext"]:
        enrich(name, stage="M control/config", dtype="mixed C++ ABI", layout="struct; see source type",
               producer="constructor or preceding M control stage", consumer="M kernels",
               lifetime_and_reuse="session-owned; TrackerConfig read-only after upload",
               source="include/saccade/tracking.hpp:23; include/saccade/tracking_session.hpp:35; include/saccade/motion.hpp; include/saccade/tracking_cuda.hpp:11")
    for name in ["Device ResultPacket", "Result packet ring"]:
        enrich(name, stage="F09", dtype="canonical little-endian uint8 wire bytes", layout="80B header + 256x48B rows",
               shape="[12368]", byte_strides="[1]", record_stride_bytes=48, data_offset_bytes=80,
               producer="packet; device packet then D2H", consumer="D2H; host receive/decode",
               lifetime_and_reuse="device packet M ordered; host ring slot leased until receive",
               notes="full12368B copied once/frame, including zero unused/reserved bytes; fixed256rowcapacity")
    enrich("JPEG staging", stage="F01", dtype="uint8 compressed JPEG bytes", layout="dense pinned bytes",
           shape="capacity[16777216]; valid size varies per JPEG", byte_strides="[1]",
           producer="host memcpy", consumer="nvJPEG", lifetime_and_reuse="single pinned staging; pending decode protected by preprocessor event")
    dense_native("Constructor warmup input", [1,3,768,1280], 4, "float32", "dense NCHW",
                 stage="constructor", producer="initialization", consumer="warmup", lifetime_and_reuse="freed before accepting frame0; excluded from persistent subtotal")

    def tensor(name, shape, dtype="float32", item_bytes=4, cls="trt_plugin_boundary_payload", **kw):
        size = math.prod(shape)*item_bytes
        stride = contiguous_strides(shape,item_bytes)
        row = dict(accounting_class=cls, memory_kind="device", resource=name, dtype=dtype,
                   shape=str(shape), layout="dense LINEAR", byte_strides=str(stride),
                   row_stride_bytes=stride[-2] if len(shape)>1 else "",
                   plane_stride_bytes=stride[-3] if len(shape)>2 else "", count=1, unit_bytes=size,
                   logical_payload_bytes=size, owner="TensorRT context", stage="F03",
                   lifetime_and_reuse="engine-managed liveness; may reuse context storage across tensors",
                   source="src/head_trt_network.cpp:181-251; src/scan_trt_plugin.cpp:103-106,163-176",
                   notes="payload size, not a separately counted native allocation; no allocation bytes inferred")
        row.update(kw)
        rows.append(row)
        return row

    for level,(h,w) in enumerate([(24,40),(12,20),(6,10)],3):
        L=h*w
        p=f"P{level}"
        tensor(p+" downsample X / expand input",[1,128,h,w],producer="TRT downsample+cast",consumer="CrossScanExpand")
        tensor(p+" expand tokens",[4,L,128],producer="CrossScanExpand",consumer="TRT in_proj and residual")
        for name in ["u","delta","y"]:
            tensor(p+" scan "+name,[4,L,256],producer="TRT SILU/projection+cast" if name!="y" else "SelectiveScan",
                   consumer="SelectiveScan" if name!="y" else "TRT gate/out_proj")
        tensor(p+" scan B",[4,L,16],producer="TRT x_proj slice/cast",consumer="SelectiveScan")
        tensor(p+" scan C",[4,L,1],producer="TRT x_proj slice/cast",consumer="SelectiveScan")
        tensor(p+" scan A",[1,16],producer="-exp(A_log), may be engine-folded constant",consumer="SelectiveScan",
               lifetime_and_reuse="read-only model constant",notes="64B semantic FP32 payload; inspector may omit constant inputs; not a new perframe allocation")
        tensor(p+" scan D",[256],producer="model constant",consumer="SelectiveScan",
               lifetime_and_reuse="read-only model constant",notes="1024B semantic FP32 payload; may be engine-folded; not a new perframe allocation")
        tensor(p+" merge input residual",[4,L,128],producer="TRT residual add/cast",consumer="CrossScanMerge")
        tensor(p+" merge output",[1,128,h,w],producer="CrossScanMerge",consumer="TRT cast/upsample")

    # The raw pointers are disjoint subranges of ONE explicit allocation above.
    offset=0
    for name,channels in [("cls",80),("reg",4)]:
        for level,(h,w) in enumerate([(96,160),(48,80),(24,40)],3):
            r=tensor(f"{name}_p{level}",[1,channels,h,w],cls="native_allocation_subrange",
                     parent_resource="F03 raw cls/reg",owner="D",producer="TensorRT",consumer="producer finite; F04 decode",
                     data_offset_bytes=offset,source="src/tracking_session.cpp:166-175; include/saccade/postprocess.hpp:46-53",
                     lifetime_and_reuse="D arena reused only after same-stream F04",
                     notes="subrange payload already included in F03 raw allocation; do not add again")
            offset+=r["logical_payload_bytes"]
    assert offset==6773760

    inspector=json.loads((evidence/"engine-inspector-summary.json").read_text())
    # Only distinct shape/name/format descriptors, not policy UInt8 buffers.
    for t in inspector["top_logical_activation_payloads"][:12]:
        rows.append(dict(accounting_class="trt_internal_inspector_logical_payload",memory_kind="device",
                         resource=t["Name"],stage="F03",dtype={"Half":"float16","Float":"float32"}[t["Format/Datatype"]],
                         layout="internal format/packing/strides unknown",shape=str(t["Dimensions"]),count=1,
                         unit_bytes=t["logical_payload_bytes"],logical_payload_bytes=t["logical_payload_bytes"],
                         owner="TensorRT context",producer=t["layer"],consumer="following TRT layers",
                         lifetime_and_reuse="engine may alias storage according to liveness; addresses not exported",
                         source="engine-inspector-summary.json:top_logical_activation_payloads",
                         notes="shape*dtype payload from exact engine inspector; not physical allocation bytes; physical strides/padding unavailable"))
    rows.append(dict(accounting_class="trt_context_reported_capacity",memory_kind="device",resource="TensorRT reported execution-context memory",
                     stage="F03",dtype="opaque mixed context arena",layout="TensorRT private",count=1,
                     unit_bytes=int(inspector["header"]["device_memory_bytes"]),owner="TensorRT execution context",
                     lifetime_and_reuse="context lifetime; engine internal tensors can reuse this arena",
                     source="engine-inspector-summary.json:header.device_memory_bytes",
                     notes="135170048B reported context memory; outside native subtotal; not equal to builder workspace limit268435456 or planbytes60753004"))

    # Guard accounting separation and fixed profile arithmetic.
    for kind,expected in [("device",46669652),("pinned_host",16826688)]:
        actual=sum(int(r["total_allocation_bytes"]) for r in rows
                   if r["accounting_class"]=="native_explicit_persistent_allocation" and r["memory_kind"]==kind)
        assert actual==expected,(kind,actual,expected)
    assert all(not r.get("total_allocation_bytes") for r in rows
               if r["accounting_class"] in {"trt_plugin_boundary_payload","native_allocation_subrange","trt_internal_inspector_logical_payload"})
    return rows


def main():
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,default=here.parents[2])
    parser.add_argument("--output",type=Path,default=here/"data-formats.csv")
    args=parser.parse_args()
    rows=build(args.repo.resolve(),here)
    with args.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS,extrasaction="raise")
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({"output":str(args.output),"rows":len(rows),"native_persistent_device_bytes":46669652,
                      "native_persistent_pinned_bytes":16826688,"raw_subrange_bytes":6773760,
                      "scope":"capacity/payload inventory; no performance or cache-counter measurement"}))


if __name__=="__main__":
    main()
