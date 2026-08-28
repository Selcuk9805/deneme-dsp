"""Corpus-level robustness/consistency check for deneme-dsp's transition planner — a way to
catch broken plans (impossible speed ratios, NaN, degenerate keys, absurd fade durations,
lopsided strategy distributions) across hundreds of real track pairs without listening to any
of them. Complements tools/benchmark_giantsteps.py (which checks raw BPM/key accuracy against
ground truth) by exercising the full calculate_transition() pipeline end to end.

Reuses the user's existing curated audio libraries from the sibling extreme-dsp-portable
checkout (offline_test/beatport — 74 progressive-house tracks; offline_test/pop_corpus — 253
Turkish pop tracks) — no re-download needed. Override with --corpus-dir for a different set.

Two phases, like extreme-dsp-portable's analyze_beatport_corpus.py:

Phase 1 (--analyze): for every track, run AnalyzerService.analyze_track() twice — once in the
"A" role (tail-anchored 60s, as used for the currently-playing track) and once in the "B" role
(head-anchored 60s, as used for the upcoming track) — and cache both feature sets to a pickle.
A track's own features differ by role since the window loaded differs, so this mirrors what a
real /api/transition/plan call actually sees in either position.

Phase 2 (--pairs, default if the pickle exists): for every ORDERED pair (i as A, j as B, i != j),
run TransitionService.calculate_transition() on the cached features (no re-analysis, fast),
flag anomalies, write a per-pair CSV, and print distribution/anomaly summaries.

Usage:
    .venv/Scripts/python.exe tools/analyze_corpus.py --corpus beatport --analyze
    .venv/Scripts/python.exe tools/analyze_corpus.py --corpus beatport --pairs
    .venv/Scripts/python.exe tools/analyze_corpus.py --corpus pop_corpus
"""
import argparse
import pickle
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows console codepage can't print some filenames
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.analyzer import AnalyzerService  # noqa: E402
from app.services.transition import TransitionService  # noqa: E402

SIBLING_OFFLINE_TEST = Path(__file__).resolve().parent.parent.parent / "extreme-dsp-portable" / "offline_test"
OUT_DIR = Path(__file__).resolve().parent / "datasets"
WORKERS = 8

# calculate_sync only meets in the middle when tempos are within 15% of each other, which bounds
# the resulting speed ratios to roughly [0.87, 1.15] — anything further out signals a bug
# (e.g. a near-zero detected tempo), not a legitimate sync decision.
RATIO_FLAG_THRESHOLD = 0.20
DURATION_FLAG_SECONDS = 20.0
LUFS_GAIN_FLAG_DB = 20.0


def _analyze_both_roles(path: Path) -> tuple[str, dict]:
    features_a = AnalyzerService.analyze_track(str(path), is_track_a=True)
    features_b = AnalyzerService.analyze_track(str(path), is_track_a=False)
    return path.name, {"a": features_a, "b": features_b}


def run_analyze_phase(corpus_dir: Path, pickle_path: Path, summary_csv: Path) -> None:
    files = sorted(corpus_dir.glob("*.mp3"))
    print(f"Analyzing {len(files)} tracks from {corpus_dir} (both roles, {WORKERS} threads)...")

    results: dict[str, dict] = {}
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, (name, feats) in enumerate(pool.map(_analyze_both_roles, files), 1):
            results[name] = feats
            if i % 20 == 0:
                print(f"  ...{i}/{len(files)}")
    elapsed = time.perf_counter() - t0
    print(f"Analyzed {len(results)}/{len(files)} in {elapsed:.1f}s ({elapsed / len(files):.2f}s/track)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)

    import csv
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "duration", "tempo_tail", "tempo_head", "key_tail", "key_head"])
        for name, feats in results.items():
            a, b = feats["a"], feats["b"]
            writer.writerow([name, round(a.duration, 1), round(a.tempo, 2), round(b.tempo, 2),
                              a.camelot_key, b.camelot_key])
    print(f"Wrote {pickle_path} and {summary_csv}")


def run_pairs_phase(pickle_path: Path, pairs_csv: Path) -> None:
    with open(pickle_path, "rb") as f:
        results: dict[str, dict] = pickle.load(f)
    names = list(results.keys())
    pairs = [(a, b) for a in names for b in names if a != b]
    print(f"Scoring {len(pairs)} ordered pairs from {len(names)} tracks...")

    rows = []
    flagged = []
    t0 = time.perf_counter()
    for name_a, name_b in pairs:
        features_a = results[name_a]["a"]
        features_b = results[name_b]["b"]
        try:
            plan = TransitionService.calculate_transition(features_a, features_b)
        except Exception as e:
            print(f"  FAILED {name_a} -> {name_b}: {e}")
            continue

        ratio_a, ratio_b = plan.sync.track_a_speed_ratio, plan.sync.track_b_speed_ratio
        dur_exec = plan.timing.transition_duration_execution
        lufs_a, lufs_b = plan.automation.track_a.lufs_gain_db, plan.automation.track_b.lufs_gain_db

        reasons = []
        if abs(ratio_a - 1.0) > RATIO_FLAG_THRESHOLD or abs(ratio_b - 1.0) > RATIO_FLAG_THRESHOLD:
            reasons.append("ratio_out_of_range")
        if dur_exec != dur_exec:  # NaN
            reasons.append("duration_nan")
        elif dur_exec > DURATION_FLAG_SECONDS:
            reasons.append("duration_too_long")
        elif plan.decision.strategy != "beat_cut" and dur_exec <= 0.0:
            reasons.append("duration_non_positive")
        if abs(lufs_a) > LUFS_GAIN_FLAG_DB or abs(lufs_b) > LUFS_GAIN_FLAG_DB:
            reasons.append("lufs_gain_extreme")
        if plan.decision.score != plan.decision.score:  # NaN
            reasons.append("score_nan")

        row = {
            "track_a": name_a, "track_b": name_b,
            "strategy": plan.decision.strategy, "score": plan.decision.score,
            "key_a": features_a.camelot_key, "key_b": features_b.camelot_key,
            "key_compatibility": plan.decision.scores.key_compatibility,
            "tempo_compatibility": plan.decision.scores.tempo_compatibility,
            "bass_compatibility": plan.decision.scores.bass_compatibility,
            "track_a_speed_ratio": ratio_a, "track_b_speed_ratio": ratio_b,
            "transition_duration_execution": dur_exec,
            "lufs_gain_db_a": lufs_a, "lufs_gain_db_b": lufs_b,
            "flags": ";".join(reasons),
        }
        rows.append(row)
        if reasons:
            flagged.append(row)
    elapsed = time.perf_counter() - t0
    print(f"Scored {len(rows)} pairs in {elapsed:.1f}s")

    import csv
    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[pairs] {len(rows)} scored, {len(flagged)} flagged")
    flag_counts = Counter(r for row in flagged for r in row["flags"].split(";") if r)
    print(f"[pairs] flag breakdown: {dict(flag_counts)}")

    strategy_counts = Counter(r["strategy"] for r in rows)
    print(f"[pairs] strategy distribution: {dict(strategy_counts)}")

    bass_values = Counter(round(r["bass_compatibility"], 2) for r in rows)
    print(f"[pairs] bass_compatibility value distribution (quantization check): {dict(sorted(bass_values.items()))}")

    scores = [r["score"] for r in rows]
    ratios_a = [r["track_a_speed_ratio"] for r in rows]
    print(f"[pairs] score: mean={statistics.mean(scores):.3f} median={statistics.median(scores):.3f} "
          f"stdev={statistics.stdev(scores):.3f}")
    print(f"[pairs] track_a_speed_ratio: min={min(ratios_a):.3f} max={max(ratios_a):.3f}")

    fallback_key_rate = sum(1 for r in rows if r["key_a"] == "1A") / len(rows) * 100
    print(f"[pairs] track_a detected as fallback default key '1A': {fallback_key_rate:.1f}% of pairs "
          f"(high value may indicate degenerate/silent chroma on some tracks)")

    if flagged:
        print(f"\n[pairs] first 15 flagged rows:")
        for r in flagged[:15]:
            print(f"  {r['track_a'][:30]:30s} -> {r['track_b'][:30]:30s}  strategy={r['strategy']:20s} "
                  f"flags={r['flags']}")

    print(f"\nWrote {pairs_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=["beatport", "pop_corpus"], default="beatport")
    parser.add_argument("--corpus-dir", type=Path, default=None, help="override the corpus directory")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir or (SIBLING_OFFLINE_TEST / args.corpus)
    pickle_path = OUT_DIR / f"{args.corpus}_corpus_analysis.pkl"
    summary_csv = OUT_DIR / f"{args.corpus}_corpus_summary.csv"
    pairs_csv = OUT_DIR / f"{args.corpus}_corpus_pairs.csv"

    if args.analyze or not pickle_path.exists():
        run_analyze_phase(corpus_dir, pickle_path, summary_csv)
    if args.pairs or not args.analyze:
        run_pairs_phase(pickle_path, pairs_csv)
