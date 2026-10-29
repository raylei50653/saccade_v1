#!/bin/sh
# Source-location audit only. No Python, model execution, or code transplantation.
# Run from repository root. Use a fresh output directory to preserve this capture.
set -u
out=${1:-build-scan-upstream-replay}
mkdir -p "$out"
run() {
  name=$1
  shift
  printf '%s\n' "$*" > "$out/$name.command"
  "$@" > "$out/$name.log" 2>&1
  rc=$?
  printf '%s\n' "$rc" > "$out/$name.exit"
  return 0
}
get() {
  name=$1
  url=$2
  run "$name" curl -L --silent --show-error --max-time 40 -D "$out/$name.headers" -o "$out/$name.body" -w 'http=%{http_code}\nurl=%{url_effective}\nsize=%{size_download}\n' "$url"
}
run environment sh -c 'date -Is; uname -a; git rev-parse HEAD; git status --porcelain=v1; curl --version; jq --version'
get crossref 'https://api.crossref.org/works/10.1016/j.neunet.2026.108544'
run crossref-metadata jq '.message | {DOI,title,author,published,license,link,resource,URL,relation}' "$out/crossref.body"
get publisher 'https://www.sciencedirect.com/science/article/pii/S0893608026000079'
# Keep the denial status, headers, body hash and size; its obfuscated JavaScript
# has no model-contract content and is deliberately not added to the evidence.
run publisher-body-hash sha256sum "$out/publisher.body"
run publisher-body-size stat --printf='%s bytes\n' "$out/publisher.body"
raw_cache=${2:-build-scan-upstream-raw-cache}
mkdir -p "$raw_cache"
mv "$out/publisher.body" "$raw_cache/publisher-$(date +%s).body"
get elsevier-api 'https://api.elsevier.com/content/article/doi/10.1016/j.neunet.2026.108544?httpAccept=text/xml'
get pubmed 'https://pubmed.ncbi.nlm.nih.gov/41579822/?format=pubmed'
get corresponding-author 'https://jszy.whu.edu.cn/zhanglefei/en/index/427766/list/index.htm'
get arxiv-title 'https://export.arxiv.org/api/query?search_query=ti%3AMambaFPN&start=0&max_results=20'
get github-title-author 'https://api.github.com/search/repositories?q=MambaFPN+Liang+in%3Areadme&per_page=100'
run github-title-author-result jq '{total_count,incomplete_results,items:[.items[]? | {full_name,html_url,description,default_branch}]}' "$out/github-title-author.body"
get mamba-neck-repository 'https://api.github.com/repos/raphaelkach/mamba-neck'
run mamba-neck-revision git ls-remote https://github.com/raphaelkach/mamba-neck.git refs/heads/main
revision=$(awk 'NR==1 {print $1}' "$out/mamba-neck-revision.log")
if [ -n "$revision" ]; then
  printf '%s\n' "$revision" > "$out/mamba-neck.revision"
  get mamba-neck-readme "https://raw.githubusercontent.com/raphaelkach/mamba-neck/$revision/README.md"
  get mamba-neck-tree "https://api.github.com/repos/raphaelkach/mamba-neck/git/trees/$revision?recursive=1"
  run mamba-neck-files jq '{sha,truncated,paths:[.tree[]?.path]}' "$out/mamba-neck-tree.body"
  run mamba-neck-license jq '{full_name,license,description,homepage}' "$out/mamba-neck-repository.body"
fi
run pubmed-locators rg -n '^(TI|AU|AD|AID|PHST|SO|LID|COIS)' "$out/pubmed.body"
run source-sizes sh -c 'for p in "$1"/*.body; do stat --printf="%n %s bytes\n" "$p"; done' sh "$out"
run source-hashes sh -c 'sha256sum "$1"/*.body' sh "$out"
