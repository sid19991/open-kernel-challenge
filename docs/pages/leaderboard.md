# Leaderboards

Source: https://openkernelchallenge.dev/leaderboard
Fetched: 2026-10-28T14:09:04Z

Live boards (JSON snapshots in `docs/api/`):

- `docs/api/leaderboard-overall.json` — overall kernel board (top 50 get serving priority)
- `docs/api/leaderboard-muse_bf16_proj.json`
- `docs/api/leaderboard-muse_attention_decode.json`
- `docs/api/leaderboard-muse_moe_w4a16.json`
- `docs/api/leaderboard-muse_decode_layer.json`
- `docs/api/leaderboard-muse_prefill_layer.json`
- `docs/api/serving-leaderboard.json` — serving track (empty until Nov 5)
- `docs/api/okc-leaderboard.json` — `okc leaderboard --json`
- `docs/api/queue.json` — judge queue

Screening flags on rows: S screened, P pending, F flagged. Final decisions come from manual review after submissions close.

`docs/SCREENING.md` is mentioned in `llms.txt` as the screening-docs path; it is **not** published on the site (SPA fallback HTML only).
