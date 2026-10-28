# Manifest

Fetch date: **2026-10-28** (UTC). Source of truth for the live contest: https://openkernelchallenge.dev/

Every file in this tree is either downloaded from a URL below, generated from those downloads (`okc init`, `okc serve init`, JSON→markdown), or a clone of an official repo named here.

## Site (https://openkernelchallenge.dev)

The site is a SPA. Unknown paths return HTTP 200 with the same ~2.8 KB HTML shell (`site/index.html`). Real routes are client-side.

| URL | Local path | Notes |
|---|---|---|
| https://openkernelchallenge.dev/ | `site/index.html`, `docs/pages/home.md` | HTML shell + content snapshot |
| https://openkernelchallenge.dev/rules | `docs/pages/rules.md`, `docs/llms.txt` | Technical rules (same text as llms.txt) |
| https://openkernelchallenge.dev/leaderboard | `docs/pages/leaderboard.md`, `docs/api/leaderboard-*.json` | Overall board |
| https://openkernelchallenge.dev/problems/{name} | `docs/pages/{name}.md`, `starter/{name}/` | Five problems; see below |
| https://openkernelchallenge.dev/terms.pdf | `docs/terms.pdf` | Official Rules v4 (231 KB PDF) |
| https://openkernelchallenge.dev/llms.txt | `docs/llms.txt` | Full agent-oriented dump |
| https://openkernelchallenge.dev/install.sh | `cli/install.sh` | CLI installer |
| https://openkernelchallenge.dev/okc/latest | `cli/latest.txt` | `0.7.2` |
| https://openkernelchallenge.dev/okc/0.7.2/SHA256SUMS | `cli/0.7.2/SHA256SUMS` | |
| https://openkernelchallenge.dev/okc/0.7.2/okc-{linux,darwin}-{amd64,arm64}.gz | `cli/0.7.2/*.gz` | linux-amd64 also unpacked + checksum-verified |
| https://openkernelchallenge.dev/robots.txt | `site/robots.txt` | Disallows `/api/` and `/ops` |
| https://openkernelchallenge.dev/theme-init.js | `site/theme-init.js` | |
| https://openkernelchallenge.dev/favicon.ico | `site/favicon.ico` | |
| https://openkernelchallenge.dev/favicon.svg | `site/favicon.svg` | |
| https://openkernelchallenge.dev/favicon-96x96.png | `site/favicon-96x96.png` | |
| https://openkernelchallenge.dev/apple-touch-icon.png | `site/apple-touch-icon.png` | |
| https://openkernelchallenge.dev/opengraph-kernels.jpg | `site/opengraph-kernels.jpg` | |
| https://openkernelchallenge.dev/assets/index-VBg5pvCv.js | `site/assets/index-VBg5pvCv.js` | SPA bundle (448 KB) |
| https://openkernelchallenge.dev/assets/index-DfJrsXOu.css | `site/assets/index-DfJrsXOu.css` | |

### SPA fallbacks (not real pages)

These returned 200 with the homepage HTML shell only. There is no separate FAQ, timeline, scoring, team, datasets, papers, or licenses page on the site; that material is on `/`, `/rules`, `llms.txt`, and `/api/v1`.

`/faq`, `/FAQ`, `/timeline`, `/scoring`, `/team`, `/licenses`, `/license`, `/datasets`, `/data`, `/submission`, `/papers`, `/docs`, `/docs/SCREENING.md`, `/SCREENING.md`, `/screening`, `/docs/rules`, `/docs/faq`, `/docs/api`, `/llms-full.txt`, `/sitemap.xml`, `/openapi.json`, `/swagger.json`, `/terms`, `/privacy`, `/contact`, `/about`, `/blog`, `/serving` (HTML), `/okc/`, `/okc/releases`, `/okc/0.7.0/SHA256SUMS`, `/okc/0.7.1/SHA256SUMS`.

`llms.txt` mentions `docs/SCREENING.md` as the screening-docs path. **Not published.**

## HTTP API (`https://openkernelchallenge.dev/api/v1`)

Fetched 2026-10-28. No token unless noted.

| URL | Local path | Status |
|---|---|---|
| GET /problems | `docs/api/problems.json` | 200 |
| GET /problems/muse_bf16_proj | `docs/api/problem-muse_bf16_proj.json` | 200 (starter files inside) |
| GET /problems/muse_attention_decode | `docs/api/problem-muse_attention_decode.json` | 200 |
| GET /problems/muse_moe_w4a16 | `docs/api/problem-muse_moe_w4a16.json` | 200 |
| GET /problems/muse_decode_layer | `docs/api/problem-muse_decode_layer.json` | 200 |
| GET /problems/muse_prefill_layer | `docs/api/problem-muse_prefill_layer.json` | 200 |
| GET /leaderboard | `docs/api/leaderboard-overall.json` | 200 |
| GET /leaderboard/muse_bf16_proj | `docs/api/leaderboard-muse_bf16_proj.json` | 200 |
| GET /leaderboard/muse_attention_decode | `docs/api/leaderboard-muse_attention_decode.json` | 200 |
| GET /leaderboard/muse_moe_w4a16 | `docs/api/leaderboard-muse_moe_w4a16.json` | 200 |
| GET /leaderboard/muse_decode_layer | `docs/api/leaderboard-muse_decode_layer.json` | 200 |
| GET /leaderboard/muse_prefill_layer | `docs/api/leaderboard-muse_prefill_layer.json` | 200 |
| GET /queue | `docs/api/queue.json` | 200 |
| GET /rules | `docs/api/rules.json` | 200 version `2026-10-22.1` |
| GET /serving | `docs/api/serving.json`, `serving/serving-api.json` | 200 |
| GET /serving/leaderboard | `docs/api/serving-leaderboard.json` | 200 (empty rows; track closed) |
| GET /me, /jobs, /tokens | — | **401 auth-walled** (`okc login`) |
| POST /problems/{p}/submissions, /serving/submissions, /rules/accept | — | **auth-walled**; not called |
| GET /api/v1 | — | 404 |
| GET /api/v1/openapi.json | — | **401** |
| GET /api/docs | — | 404 |

CLI JSON (same public endpoints): `docs/api/okc-problems.json`, `okc-leaderboard.json`, `okc-serving-leaderboard.json`.

## Starter kernels (`okc init`, 2026-10-28)

`okc init` does **not** require login. Written to `starter/<problem>/`.

| Problem | Files |
|---|---|
| muse_bf16_proj | submission.py, baseline.py, reference.py, checks.py, TASK.md, AGENTS.md |
| muse_attention_decode | submission.py, baseline.py, reference.py, checks.py, TASK.md, AGENTS.md |
| muse_moe_w4a16 | + nvfp4.py |
| muse_decode_layer | + nvfp4.py |
| muse_prefill_layer | + nvfp4.py |

`okc show <problem>` text: `docs/pages/okc-show-*.txt`. Help: `docs/cli/`.

No sample/eval `.pt` / `.npz` / notebooks are linked from the site or API. Inputs are generated in-process by `reference.py`.

## Serving / SGLang

| Source | Local path |
|---|---|
| `okc serve init` | `serving/RULES.md`, `serving/patch.diff` |
| https://github.com/sgl-project/sglang @ `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` | `serving/sglang/` (depth-1 snapshot; inner `.git` stripped, see `CLONED-FROM.txt`) |
| Docker Hub `lmsysorg/sglang:dev-muse-spark-open` | `serving/docker-*.json` only |

Image digest `sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`. Amd64 compressed size **23.87 GB**. **Not pulled (too large).** Includes Dockerfiles inside `serving/sglang/` (e.g. `.devcontainer/Dockerfile`) from the pinned commit; there is no separate challenge Dockerfile on the site.

## Official reference git repos

Credited on `/rules` as the judge’s starting point (June 9 Researcher Reciprocity License). Inner `.git` stripped.

| Repo | Local path |
|---|---|
| https://github.com/gpu-mode/reference-kernels | `reference/reference-kernels/` |
| https://github.com/gpu-mode/kernelbot | `reference/kernelbot/` |

Not cloned (not official starter/template/reference): third-party competitor repos such as `vjeux/okc-2026`, `operatorstack/open-kernel-challenge-solutions`.

## Papers, blogs, model docs

| URL | Local path | Notes |
|---|---|---|
| https://research.meta.ai/static/muse-spark-open-evaluation-methodology | `papers/muse-spark-open-evaluation-methodology.html` | HTML; no PDF published at the `.pdf` URLs tried |
| https://arxiv.org/pdf/2606.12429 | `papers/arxiv-2606.12429-muse-spark-safety-preparedness.pdf` | Muse Spark Safety & Preparedness Report (2.8 MB) |
| https://research.meta.ai/blog/muse-spark-open | `docs/external/meta-research-muse-spark-open.html` | |
| https://www.paradigm.xyz/writing/muse-spark-open-kernel-challenge | `docs/external/paradigm-muse-spark-open-kernel-challenge.html` | |
| https://huggingface.co/blog/meta-muse-spark-open | `docs/external/huggingface-blog-meta-muse-spark-open.md` | |
| https://docs.sglang.io/cookbook/autoregressive/Meta/MuseSparkOpen | `docs/external/sglang-cookbook-MuseSparkOpen.md` | |
| https://dev.meta.ai/docs/muse-spark-open | `docs/external/meta-dev-muse-spark-open.html` | |
| https://dev.meta.ai/docs/muse-spark-open/get-the-model | `docs/external/meta-dev-get-the-model.html` | |

No standalone “Muse Spark Open architecture paper” PDF was linked from the challenge site. Model card body is gated on Hugging Face (see below); architecture summary is in the Meta blog / HF card HTML webfetch / `licenses/Muse-Model-License.md`.

## Hugging Face (weights indexes, not shards)

| URL | Local | Access |
|---|---|---|
| https://huggingface.co/api/models/meta-models/Muse-Spark-Open-816B-A42B | `data/hf/api-Muse-Spark-Open-816B-A42B.json` | public metadata; `gated: manual` |
| https://huggingface.co/api/models/meta-models/Muse-Spark-Open-816B-A42B-NVFP4 | `data/hf/api-Muse-Spark-Open-816B-A42B-NVFP4.json` | public metadata; gated |
| https://huggingface.co/api/models/RadixArk/Muse-Spark-Open-DFlash2 | `data/hf/api-Muse-Spark-Open-DFlash2.json` | public |
| https://huggingface.co/api/collections/meta-models/muse-spark-open | `data/hf/collection-muse-spark-open.json` | public |
| …/tree/main (recursive) | `data/hf/tree-*.json`, `index-*.json` | public file listings |
| https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2/resolve/main/{config.json,README.md,.gitattributes} | `data/hf/dflash2/` | 200 |
| …/Muse-Spark-Open-816B-A42B/resolve/main/{config.json,README.md,LICENSE.md} | — | **401 gated** |
| …/Muse-Spark-Open-816B-A42B-NVFP4/resolve/main/config.json | — | **401 gated** |
| DFlash2 `model.safetensors` | — | **too large (~9.1 GB)** |
| BF16 shards | — | **too large (~1.04 TB listed)** and gated |
| NVFP4 shards | — | **too large (~167 GB listed)** and gated |

Muse Model License text was recovered from the public `cardData.extra_gated_prompt` field → `licenses/Muse-Model-License.md`.

## Licenses

| File | Source |
|---|---|
| `licenses/Apache-2.0.txt` | https://www.apache.org/licenses/LICENSE-2.0.txt (challenge submissions) |
| `licenses/Muse-Model-License.md` | HF `cardData.extra_gated_prompt` |
| `licenses/sglang-LICENSE` | `serving/sglang/LICENSE` |
| `licenses/gpu-mode-reference-kernels-LICENSE` | clone |
| `licenses/gpu-mode-kernelbot-LICENSE` | clone |

## Missing / skipped

| Item | Reason |
|---|---|
| `okc login` token, `/me`, `/jobs`, submissions | Auth-walled; a person must accept rules in a terminal |
| Hugging Face Muse Spark Open weight shards, `config.json`, tokenizer, chat template, `LICENSE.md`, `USE_POLICY.md` raw files | Gated (manual license accept) **and** too large (~167 GB NVFP4 / ~1 TB BF16) |
| DFlash2 `model.safetensors` | Too large (~9.1 GB); config/README stored |
| Docker image `lmsysorg/sglang:dev-muse-spark-open` | Too large (~23.9 GB); digest + manifests stored |
| Hidden serving-track conversations / eval data | Not published |
| Kernel sample tensors / notebooks / zip datasets | None linked on the site or API |
| `docs/SCREENING.md` | Mentioned in llms.txt; not a public document |
| `/ops` | Disallowed by robots.txt; not fetched |
| OpenAPI spec | `/api/v1/openapi.json` returns 401 |
| Evaluation methodology PDF | HTML only; `.pdf` URLs 404 |
| Separate FAQ / timeline / scoring / team / papers site pages | Do not exist (SPA shell) |
| Older CLI 0.7.0 / 0.7.1 binaries | `/okc/0.7.0/SHA256SUMS` and `0.7.1` are SPA HTML, not releases |

## Integrity notes

- `cli/0.7.2/okc-linux-amd64` matched `SHA256SUMS` (`84883d7f037083f21e6942baa5f738c27b242edb68a69962d06ac46c385b653b`).
- SGLang HEAD at clone time: `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.
- Leaderboards are a point-in-time snapshot and will drift; re-fetch `/api/v1/leaderboard*` to update.
EOF