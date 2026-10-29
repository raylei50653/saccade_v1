# Fixed architecture-document audit, 2026-10-29

Scope: read the architecture documents already cited by v1 `docs/DATAFLOW.md`.
This is a design-provenance audit, not reuse of legacy source, weights,
fixtures, execution results, or experiment conclusions. No Python was run.

The source is the immutable legacy revision
`e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2`, not the legacy working tree.
The repository-level LICENSE at that revision declares Apache-2.0;
this does not establish the license or lineage of the missing new model.
`source-metadata.log` records Git blob IDs, byte sizes, and SHA-256 for
the documents; `architecture-excerpt.log` preserves only the relevant
architecture section, with original line numbers. No old numerical
acceptance evidence is migrated.

## Newly located design details

Source URL:
<https://github.com/raylei50653/saccade/blob/e48f3b39d5bfeb192a08d8a02cd2c04cf053bdd2/docs/reference/math_model.md#L375-L468>

| Locator | What the architecture document actually specifies | Applicability |
| --- | --- | --- |
| Lines 397–404 | Ordered levels P3/P4/P5; strides 8/16/32; spatial sizes 80/40/20 at input 640; backbone channels 128/256/512; reduced token counts 400/100/25. | Historical architecture design, plausible candidate profile for v1; no new artifact binding. |
| Lines 406–432 | Per-lane input projection to 128, stride-4 down-convolution, flatten to `L x 128`, two Mamba blocks with `d_state=16`, upsample x4, concat with unreduced input-projection skip to 256 channels. | Establishes intended block placement; does not identify which Mamba recurrence, directions, internal expansion, or B/C grouping. |
| Lines 434–443 | Per-level classification convolution to 80 channels, regression convolution to 4; `reg_max=1`, no DFL softmax. | Historical raw prediction design; new class list, native tensor names/order and precision still unprovided. |
| Lines 445–464 | Flatten and concatenate across levels to 8400 anchors; anchor centers `(x+0.5,y+0.5)`; distance-to-box decode with level stride; sigmoid then max/argmax class. | Historical decode responsibility; does not supply new model's output capacity/padding or tie/error rules. |
| Lines 466–468 | Names a historical checkpoint and disables the embedding branch. | A locator for historical provenance only; this checkpoint is prohibited for v1 reuse and was not loaded, copied, hashed, or executed. |

The supplementary SVG describes a high-level three-lane map and directs
readers to math-model section 4.1. It adds no scan formulas or tensor ABI.

## Exact missing fields

The cited section labels two `MambaBlock` boxes; it does not define
the discrete recurrence, A/B/C/D parameter shapes, delta bias or softplus,
gate/skip ordering, number/direction of scans, flatten indexing and inverse
mapping, direction merge operator, initial/final state ABI, zero-length
behavior, allowed channel/state dimensions, input ranges, arithmetic
precision, independent expected values, or numerical tolerance.
The diagram's outer U-Net skip is not evidence of the recurrence's D skip.
The label `flatten` is not an exact spatial index formula. A statement of
per-frame inference in v1 is not a definition of the scan's per-invocation
initial state. None of these can be filled from a same-name public model.

No new Mamba-FPN artifact, immutable source revision, architecture config,
exported P3/P4/P5 boundary, or scan-to-head mapping is delivered by these
documents. Consequently the S01 gate remains OPEN. These findings add
design constraints, not a verified target scan implementation.

## Minimum decision packet if the new head is not designed yet

Decision authority is the v1 model owner / maintainer; the materials
inspected do not identify a separate head author or artifact custodian.

1. Confirm whether the above per-level downsampled, two-block lane is
   the v1 intended head, or provide the new head's own fixed architecture.
   Recommendation: use this already-cited architecture as the candidate
   starting profile because it preserves B03's model family and bounds
   token lengths to 400/100/25. Mark any adoption as a new design decision.
2. Choose and pin the exact scan family/variant and its complete
   mathematical and tensor contract. Candidate axes: one flattened sequence
   versus multi-direction spatial scan; real versus complex state; B/C
   sharing/grouping; explicit gate and D skip; delta transformation;
   reset-only versus state I/O. Recommendation: require an immutable
   specification with operator placement in the new head before choosing
   a native implementation. The cited architecture does not justify any
   particular recurrence or a four-direction VMamba substitution.
3. Approve the minimum S01 ABI and failure profile: dense FP32 input/outputs
   as a candidate native path, exact dimensions and overflow bounds,
   zero/nonzero-length behavior, supported state I/O, non-finite policy,
   independently derived expected vectors, and justified tolerances.
   Recommendation: keep cross-frame state out of the candidate because B03
   already specifies per-frame inference; initial-state handling within a
   scan call still needs an explicit decision.
4. Supply the new head/source/config/native artifact mapping and provenance.
   New independently produced weights and native handoff remain required
   for P01 even if S01's precise arithmetic contract can be approved sooner.
   Head class/box/decode details outside the scan block primarily gate S04.

Unlock: obtain the owner's fixed head source/config and precise scan
specification plus independent comparison basis, check them against the
contract matrix, then implement the standalone C++ scalar reference from
that specification. The next action is a concrete owner decision or
material handoff, not a rerun of legacy numerical experiments.

## Reproduction

Run `bash evidence/scan-contract/2026-10-29/design/capture.sh` from the v1
checkout. It needs the fixed Git objects in the legacy repository plus
Git, Bash, sed, rg, and GNU coreutils. Each capture has its command, exit
status, and raw output. The script performs no writes to the legacy repo.
