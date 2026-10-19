"""What this project downloads, fetched on the machine that runs it — never on import, in tests or
while building.

There is no model to download: the agent calls Claude through the API. The only data is
StableToolBench's solvable test queries (six JSON files of about 1-2 MB, each query carrying the
documentation of its APIs). Running them also needs a StableToolBench API server, which is its own
project with its own downloads (the tool environment and response cache on Hugging Face,
`stabletoolbench/ToolEnv2404` and `stabletoolbench/Cache`); see its README.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx

GROUPS = (
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
)
QUERY_URL = (
    "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/"
    "solvable_queries/test_instruction/{group}.json"
)


def describe(groups: Sequence[str]) -> list[str]:
    return [f"{g:<16} ~1-2 MB  {QUERY_URL.format(group=g)}" for g in groups]


def _fetch(url: str) -> bytes:
    r = httpx.get(url, follow_redirects=True, timeout=60.0)
    r.raise_for_status()
    return r.content


def download(
    groups: Sequence[str],
    dest: str | Path = "data/toolbench",
    *,
    fetch: Callable[[str], bytes] = _fetch,
) -> list[str]:
    """Fetch each group's queries to `dest/<group>.json`, skipping files already there. Returns
    one line per file with its sha256, so a run can be tied to the exact data it used."""
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise ValueError(f"unknown group(s) {unknown}; choose from {list(GROUPS)}")
    root = Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for g in groups or GROUPS:
        path = root / f"{g}.json"
        if not path.exists():
            data = fetch(QUERY_URL.format(group=g))
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        out.append(f"{g}: {path} (sha256 {digest}...)")
    return out
