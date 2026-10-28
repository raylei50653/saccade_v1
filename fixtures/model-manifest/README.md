# Parser-only fixture

`bytes.txt` is exactly the three ASCII bytes `abc` (no newline), dedicated to
CC0-1.0 for this fixture. `contract.json` describes intentionally hypothetical
input/output declarations so parser, canonical serialization, path and file
integrity behavior can be tested without an actual model or GPU.

This is **not** an ONNX model, a YOLO26s export, a Mamba-FPN head, a scan formula,
a numerical fixture or evidence of real-model readiness. The head mapping is
explicitly absent, scan and numerical comparison remain unverified, and the
CLI always reports model content as not checked and real_model_ready=false.

Expected artifact SHA-256:
`ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`.

Expected canonical manifest SHA-256, independently checked with `sha256sum`:
`97d83da29be2693ba86a210a10c8a5454de8a135b5b31985feb3aee07152b0b7`.

See [schema and commands](../../docs/MODEL_MANIFEST.md).
