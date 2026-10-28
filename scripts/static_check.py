#!/usr/bin/env python3
"""CPU-only preflight for OKC kernel submissions.

This Cursor VM has no NVIDIA GPU. It cannot run CUDA graphs, SGLang baselines,
or H200 timing. Use this script to catch graph-capturable-rule violations and
missing entry points before spending an `okc check` on the official H200 judge.

    python3 scripts/static_check.py
    python3 scripts/static_check.py starter/muse_moe_w4a16/submission.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter"

# Host-side ops that break CUDA-graph capture (challenge rules).
HOST_SYNC = re.compile(
    r"""(?x)
    \.item\s*\(
    | \.cpu\s*\(
    | \.numpy\s*\(
    | torch\.cuda\.synchronize\s*\(
    | cudaDeviceSynchronize
    | \.tolist\s*\(
    """
)
PRINT_TENSOR = re.compile(r"""print\s*\(.*\b(data|out|x|hidden|qkv|weights)\b""")

PROBLEMS_NEED_PREPARE = {
    "muse_bf16_proj",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
}


def problem_from_source(text: str, path: Path) -> str | None:
    m = re.search(r"^#\s*okc:\s*problem=([a-z0-9_]+)\s*$", text, re.M)
    if m:
        return m.group(1)
    return path.parent.name if path.parent.name.startswith("muse_") else None


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errs: list[str] = []
    problem = problem_from_source(text, path)
    if not problem:
        errs.append("missing first-line `# okc: problem=<name>`")
    elif not text.splitlines()[0].startswith("# okc: problem="):
        errs.append("first line must be `# okc: problem=<name>` (okc submit reads it)")

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if "kernel_fn" not in names:
        errs.append("no kernel_fn()")
    if problem in PROBLEMS_NEED_PREPARE and "prepare" not in names:
        errs.append("no prepare() (required for every problem except muse_attention_decode)")

    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        if HOST_SYNC.search(line):
            errs.append(f"L{i}: host sync / graph-breaker: {line.strip()[:120]}")
        if PRINT_TENSOR.search(line):
            errs.append(f"L{i}: printing a live tensor can host-sync under CUDA graphs: {line.strip()[:120]}")
    return errs


def submission_paths(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(a).resolve() for a in argv]
    return sorted(STARTER.glob("*/submission.py"))


def main() -> int:
    paths = submission_paths(sys.argv[1:])
    if not paths:
        print("no submission.py files found", file=sys.stderr)
        return 2
    n_err = 0
    for p in paths:
        rel = p if p.is_absolute() else p
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            rel = p
        errs = check_file(p)
        if errs:
            n_err += 1
            print(f"FAIL {rel}")
            for e in errs:
                print(f"  {e}")
        else:
            print(f"ok   {rel}")
    print()
    print("This check is not the judge. Correctness and H200 timing: `okc check <file>`.")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
