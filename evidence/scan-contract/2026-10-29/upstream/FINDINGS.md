# MambaFPN upstream source-location audit — 2026-10-29

This extends the 2026-10-28 title/repository search with publisher metadata,
corresponding-author identification, an arXiv title query, and inspection of a
newly located repository that cites Liang et al. No Python or model was run;
no implementation or weights were transplanted. The target remains the v1
YOLO26s backbone/neck plus Mamba-FPN **head**, not an arbitrary Mamba neck.

## New evidence and exact scope

| Source | New observation | Limit |
| --- | --- | --- |
| [Crossref DOI metadata](https://api.crossref.org/works/10.1016/j.neunet.2026.108544), `capture/crossref-metadata.log` | Article identity, three author names/ORCIDs, version-of-record XML/plain-text links, publisher resource and license-policy links. | Publication DOI identifies the article, not an immutable software revision. No target source/config/artifact mapping is supplied by these fields. License-policy links are not a license for absent code/weights. |
| [Elsevier Article API](https://api.elsevier.com/content/article/doi/10.1016/j.neunet.2026.108544?httpAccept=text/xml), `capture/elsevier-api.body`, `/full-text-retrieval-response/coredata` | HTTP 200, PII `S0893608026000079`, `openaccess=0`, `openaccessArticle=false`; response contains only coredata and links. | HTTP success did **not** deliver full text, scan formulas, source code, or weights. No institutional credential/API key was supplied or requested. |
| [Publisher preview](https://www.sciencedirect.com/science/article/pii/S0893608026000079), Introduction / CRediT authorship contribution statement | Browser search preview describes a neck with locality-aware serialization, bidirectional processing and locality mixing; software contribution is attributed to Cheng Wang. A follow-up index query and selected exact excerpts are recorded in `publisher-index.json`. | Direct browser open and captured curl returned HTTP 403. Preview is a search-index observation, not locally retained full text or a complete scan specification. The native curl replay cannot reproduce the index observation. It does not establish correspondence to the v1 head. |
| [PubMed record 41579822](https://pubmed.ncbi.nlm.nih.gov/41579822/), `capture/pubmed-initial.body`, `FAU/AD` for Zhang, Lefei | Corresponding-author contact is `zhanglefei@whu.edu.cn`; university and bibliographic identity align with the DOI. | Initial capture HTTP 200; later `pubmed.body` was HTTP 203 challenge, and its locator search correctly exits 1. The initial successful capture and failed later response are both retained. A contact/request route, not proof that the author holds the project's intended head. No message was sent. |
| [Official university author page](https://jszy.whu.edu.cn/zhanglefei/en/index/427766/list/index.htm), `capture/corresponding-author.body` | Confirms Wuhan University author affiliation and links to scholarly indexes. | No target implementation or native artifact link was located on this captured page. |
| [arXiv API exact title query](https://export.arxiv.org/api/query?search_query=ti%3AMambaFPN&start=0&max_results=20), `capture/arxiv-title.body`, `opensearch:totalResults` | `0` results for title token `MambaFPN`. | Finite title query only; not evidence that no manuscript exists under another title or on another host. |
| [GitHub author-qualified README query](https://api.github.com/search/repositories?q=MambaFPN+Liang+in%3Areadme&per_page=100), `capture/github-title-author-result.log` | `2` results, `incomplete_results=false`: `raphaelkach/mamba-neck` and `Tech-sam-90/fpn-inceptentionnet`. | Query scope differs from the prior bare repository-name search and is new evidence. No claim of exhaustive worldwide source discovery. The second result's own description concerns brain-MRI classification, so it was excluded before artifact acquisition. |

## Newly located candidate fixed to an immutable revision

[raphaelkach/mamba-neck](https://github.com/raphaelkach/mamba-neck/tree/6dec69675c362ef9837800ad084657850756e56b)
was resolved with `git ls-remote` to commit
`6dec69675c362ef9837800ad084657850756e56b`. All inspected README/tree URLs
use that commit, not a floating branch.

| Field | Evidence |
| --- | --- |
| Author/lineage | Its own README lines 3–11 identifies Raphael Kach, DHBW Stuttgart, 2026 bachelor project, a controlled VisDrone neck comparison citing Liang et al. It does not claim to be the Liang/Wang/Zhang author implementation. |
| Architecture | README lines 11, 26–41: ResNet-50 backbone, ATSS detection head, MMDetection; lines 47–52: C2/C3/C4/C5 inputs, P3/P4/P5/P6/P7 outputs. These are different component boundaries from the v1 target. |
| Scan information | README line 31 describes four-direction cross-scan. This belongs to that separate candidate; it is not adopted as the target contract. Source code was not executed or imported. |
| Native delivery | Pinned recursive tree (`truncated=false`) contains Python/notebooks/configs and result CSVs, no ONNX, safetensors, native tensor bundle or C++ implementation path. No weights were acquired. This conclusion is limited to tracked paths in that revision, not external releases. |
| License | GitHub repository metadata `license=null`; pinned tree contains no LICENSE path. License remains missing, rather than inferred from a dependency. |
| Acquired file | [README at pinned revision](https://raw.githubusercontent.com/raphaelkach/mamba-neck/6dec69675c362ef9837800ad084657850756e56b/README.md), 6,830 bytes, SHA-256 `111addbc7dcd46ccb951d41255caf4fcc8b7255c069916a884e3d9b6e5a105aa`. |
| Full capture identity | See `capture/source-sizes.log` and `capture/source-hashes.log` for every retained downloaded response. These hashes identify the downloaded bytes, not nonexistent target weights. |

## What this resolves and what remains

The same-name publication and the newly located implementation do not supply an
identified YOLO26s/v1 head. The publication's neck role and the candidate's
ResNet/ATSS boundary are **identity mismatches**; they are not permission to
replace the chosen model architecture. The publication full text was not
obtained via the checked native/public API path. Its exact scan/serialization
semantics and code license remain missing, and even a full paper would still
need a project-approved mapping to the intended head.

If the project owner intended the Liang/Wang/Zhang paper, the concrete request
is to Lefei Zhang (corresponding contact above), with Cheng Wang identified by
the publisher preview as a software contributor: provide a public source URL
and immutable revision, exact head/neck architecture config, code/weight
license, scan source locators and independently derived test vectors; provide
native ONNX plus external data or a documented C++ readable tensor format,
weights/export/training lineage, file sizes and hashes. Native export is a
separate missing item, not established by receiving a pickle checkpoint.

If that publication was **not** intended, the v1 model designer must name the
actual target source/config or ratify a new head specification. Author contact
is a possible route for the paper, not a known holder of a missing v1 artifact.
Until that identity decision, target scan mathematics, tensor packing, state
initialization and numeric comparison remain absent: S01 stays OPEN. Model
weights, ordered detector boundaries and native delivery also block P01 and
S04 full-head acceptance. Independent S03 plugin validation does not require
the complete model weights; none of these searches constitute numerical acceptance.

## Replay / capture interpretation

Run from repo root:

```sh
sh evidence/scan-contract/2026-10-29/upstream/replay.sh build-scan-upstream-replay
```

Use a fresh output directory. Capture stores command arguments, exit code,
HTTP status/URL/size, headers and response bodies. Curl exit 0 means transport
completed, not HTTP/content acceptance: consult each `.log`. The publisher's
large denial-page body is kept in ignored `build-scan-upstream-raw-cache`, with
its size/hash/status retained in evidence; it contains no acquired paper.
The retained metadata/publisher/author bodies have no supplied immutable
software revision; their date and hashes identify only this retrieval. DOI is
not a source-code revision. The source-response sizes/hashes and query counts can change on replay;
immutable candidate README/tree links remain fixed to the resolved revision
from each replay. Do not treat a new floating-branch resolution as the same
source without inspecting its diff.

Browser search queries additionally executed on this date: exact DOI;
`MambaFPN github authors code`; `MambaFPN github Liang`; `MambaFPN Data
availability`; `Lefei Zhang MambaFPN code`; `MambaFPN arxiv code github`;
`MambaFPN site:github.com`; `MambaFPN site:arxiv.org Liang`; `MambaFPN
site:whu.edu.cn`; and quoted title/author/code-availability variants. Author,
publisher and repository originals above are the sources used for conclusions.
Other similarly named segmentation/remote-sensing work was not imported.
