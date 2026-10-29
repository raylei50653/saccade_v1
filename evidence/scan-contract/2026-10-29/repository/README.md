# Repository delivery lookup

Captured 2026-10-29 via connected GitHub `github_fetch` (read-only).
The `*.json` files preserve tool responses, including request URL and
`isError=false`; these are API observations, not subprocess exit codes.

Requests, in parallel:

```text
GET https://api.github.com/repos/raylei50653/saccade_v1/branches?per_page=100
GET https://api.github.com/repos/raylei50653/saccade_v1/releases?per_page=100
GET https://api.github.com/repos/raylei50653/saccade_v1/pulls?state=all&per_page=100
GET https://api.github.com/repos/raylei50653/saccade_v1/issues?state=all&per_page=100
```

Results: one branch (`main`, `9f2c11a838a229c05ba7e9e17fa182bf0799746b`),
zero releases, zero PRs, zero issues. All collections contain fewer than the
100 requested entries. This only covers this repository's visible delivery
channels at capture time, not private/unpublished material elsewhere.

For a native API replay (requires `curl`, `jq`; no Python):

```sh
curl --fail --location 'https://api.github.com/repos/raylei50653/saccade_v1/branches?per_page=100' | jq .
curl --fail --location 'https://api.github.com/repos/raylei50653/saccade_v1/releases?per_page=100' | jq .
curl --fail --location 'https://api.github.com/repos/raylei50653/saccade_v1/pulls?state=all&per_page=100' | jq .
curl --fail --location 'https://api.github.com/repos/raylei50653/saccade_v1/issues?state=all&per_page=100' | jq .
```

Unauthenticated HTTP replay may fail for a private repository; the connected
read results above remain the capture for this run. A future replay is a new
observation and must be stored separately.
