# Serving track materials

Pinned by `GET https://openkernelchallenge.dev/api/v1/serving` and `okc serve init`.

| Item | Value |
|---|---|
| SGLang repo | https://github.com/sgl-project/sglang |
| Commit | `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` ("Support Muse Spark Open") |
| Image | `lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a` |
| Hub tag | `lmsysorg/sglang:dev-muse-spark-open` (~23.9 GB compressed amd64; **not pulled**) |
| Model | `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` |
| Drafter | `RadixArk/Muse-Spark-Open-DFlash2` |

Files:

- `sglang/` — full source at the pinned commit (inner `.git` removed; see `sglang/CLONED-FROM.txt`)
- `RULES.md` — patch rules from `okc serve init`
- `patch.diff` — empty starter patch
- `serving-api.json` — full `/api/v1/serving` dump (launch flags, rungs, limits)
- `docker-image.json` — digest, layer size, skip reason
- `docker-hub-tag-dev-muse-spark-open.json`
- `docker-amd64-manifest.json` / `docker-manifest-index.json`

Launch flags used by the judge are in `RULES.md` and `serving-api.json` (`launch.env` / `launch.args`).
