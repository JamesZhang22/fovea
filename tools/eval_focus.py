"""Evaluate the focus score against the user's real culling verdicts.

Usage: uv run python tools/eval_focus.py <folder>...

Ground truth comes from ratings made in the app (stored per-folder in the cache):
any star rating counts as keep, x counts as reject, unrated frames are excluded.
Reports precision/recall at score thresholds, the abstain rate, and how score
confidence relates to verdict agreement.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from fovea.core.ingest.cache import Cache
from fovea.core.pipeline import PipelineConfig, run_pipeline

THRESHOLDS = [30, 40, 50, 60, 70, 80, 90]


def collect(folders: list[Path]) -> list[dict]:
    """One record per frame with a verdict: kept flag, score, confidence."""
    rows = []
    for folder in folders:
        cache = Cache(folder / ".fovea" / "cache.sqlite")
        entries = run_pipeline(folder, PipelineConfig(detect=True, eye=True, export=False), cache)
        for e in entries:
            user = cache.get_json(Path(e["path"]), "user") or {}
            rejected = bool(user.get("rejected"))
            rated = user.get("rating") is not None
            if not rejected and not rated:
                continue
            m = e.get("metrics") or {}
            rows.append(
                {
                    "name": Path(e["path"]).name,
                    "kept": rated and not rejected,
                    "score": m.get("focus_score"),
                    "confidence": m.get("focus_confidence"),
                    "radius": m.get("focus_radius_px"),
                }
            )
    return rows


def report(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("no verdicts found — cull some frames in the app first (rate or x-reject)")
        return
    kept = sum(r["kept"] for r in rows)
    scored = [r for r in rows if r["score"] is not None]
    abstained = n - len(scored)
    print(
        f"{n} verdicts ({kept} keep / {n - kept} reject) | "
        f"abstain rate {abstained / n:.1%} ({abstained} frames)"
    )
    if not scored:
        return

    y = np.array([r["kept"] for r in scored])
    s = np.array([r["score"] for r in scored])
    print("\nthreshold | predict-keep | precision | recall | reject-precision")
    for t in THRESHOLDS:
        pred = s >= t
        tp = (pred & y).sum()
        prec = tp / pred.sum() if pred.sum() else float("nan")
        rec = tp / y.sum() if y.sum() else float("nan")
        rej_prec = ((~pred) & (~y)).sum() / (~pred).sum() if (~pred).sum() else float("nan")
        print(
            f"   >= {t:2d}  |     {pred.sum():4d}    |   {prec:.2f}    "
            f"|  {rec:.2f}  |      {rej_prec:.2f}"
        )

    print("\nscore distribution by verdict:")
    for label, mask in (("keep", y), ("reject", ~y)):
        vals = s[mask]
        if len(vals):
            print(
                f"  {label:6s} n={len(vals):4d} | median {np.median(vals):5.1f} | "
                f"p25 {np.percentile(vals, 25):5.1f} | p75 {np.percentile(vals, 75):5.1f}"
            )


if __name__ == "__main__":
    folders = [Path(f).expanduser().resolve() for f in sys.argv[1:]]
    report(collect(folders))
