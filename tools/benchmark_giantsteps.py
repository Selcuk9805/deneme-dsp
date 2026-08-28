"""Benchmark deneme-dsp's tempo/key detection against the GiantSteps Tempo & Key datasets
(ISMIR 2015, EDM previews with crowd-corrected ground truth).

Runs the actual production code path (AnalyzerService.analyze_track, is_track_a=False — same
60s-from-start load used for "track B" in a real transition request), not a reimplementation,
so results reflect what the API actually returns.

Reuses the already-downloaded dataset from the sibling extreme-dsp-portable checkout by default
(no re-download needed) — override with --datasets-root to point at a local copy instead.

Usage:
    .venv/Scripts/python.exe tools/benchmark_giantsteps.py            # both benchmarks, full set
    .venv/Scripts/python.exe tools/benchmark_giantsteps.py --tempo    # tempo only
    .venv/Scripts/python.exe tools/benchmark_giantsteps.py --key --limit 50   # quick smoke test
"""
import argparse
import csv
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.analyzer import AnalyzerService  # noqa: E402

DATASET_ROOT_DEFAULT = Path(__file__).resolve().parent.parent.parent / "extreme-dsp-portable" / "tools" / "datasets"

# get_camelot_key()'s internal roll convention: index i (0=C) is the profile shift amount, i.e.
# the root note. These lists mirror analyzer.py's camelot_maj/camelot_min arrays exactly, so
# ground-truth conversion below lands on the same code space the detector actually outputs.
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CAMELOT_MAJOR_BY_NOTE = dict(zip(NOTES, ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]))
CAMELOT_MINOR_BY_NOTE = dict(zip(NOTES, ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]))

# The key dataset spells accidentals as flats (Eb, Ab, ...); analyzer.py's note space uses sharps.
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}

# MIREX tempo evaluation convention: Accuracy1 = within 4% of the true tempo; Accuracy2 = also
# credits octave/triple-time confusions (half, double, third, triple, and 2:3/3:4 relations) as
# "not really wrong" — tempo estimators are known to sometimes lock onto the wrong metrical level.
TEMPO_TOLERANCE_RATIO = 0.04
OCTAVE_ERROR_FACTORS = {
    "double": 2.0, "half": 0.5, "triple": 3.0, "third": 1 / 3,
    "three_quarter": 0.75, "four_third": 4 / 3, "two_third": 2 / 3, "three_half": 1.5,
}

WORKERS = 8


def _ground_truth_camelot(text: str) -> str | None:
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    note, mode = parts
    note = FLAT_TO_SHARP.get(note, note)
    if note not in NOTES:
        return None
    table = CAMELOT_MAJOR_BY_NOTE if mode == "major" else CAMELOT_MINOR_BY_NOTE
    return table.get(note)


def _genre_for(tempo_root: Path, stem: str) -> str:
    path = tempo_root / "annotations" / "genre" / f"{stem}.genre"
    return path.read_text().strip() if path.exists() else "unknown"


def _collect_tempo_pairs(tempo_root: Path, limit: int | None) -> list[tuple[Path, float, str]]:
    audio_dir, ann_dir = tempo_root / "audio", tempo_root / "annotations" / "tempo"
    ann_files = sorted(ann_dir.glob("*.bpm"))
    pairs = []
    for ann_path in ann_files:
        audio_path = audio_dir / f"{ann_path.stem}.mp3"
        if audio_path.exists():
            pairs.append((audio_path, float(ann_path.read_text().strip()), _genre_for(tempo_root, ann_path.stem)))
    print(f"[tempo] {len(pairs)}/{len(ann_files)} annotated tracks have downloaded audio")
    return pairs[:limit] if limit else pairs


def _collect_key_pairs(key_root: Path, limit: int | None) -> list[tuple[Path, str]]:
    audio_dir, ann_dir = key_root / "audio", key_root / "annotations" / "key"
    ann_files = sorted(ann_dir.glob("*.key"))
    pairs = []
    for ann_path in ann_files:
        audio_path = audio_dir / f"{ann_path.stem}.mp3"
        camelot = _ground_truth_camelot(ann_path.read_text())
        if audio_path.exists() and camelot:
            pairs.append((audio_path, camelot))
    print(f"[key] {len(pairs)}/{len(ann_files)} annotated tracks have downloaded audio + a parseable key")
    return pairs[:limit] if limit else pairs


def _tempo_error_category(detected: float, true_bpm: float) -> str:
    tolerance = TEMPO_TOLERANCE_RATIO * true_bpm
    if abs(detected - true_bpm) <= tolerance:
        return "correct"
    for name, factor in OCTAVE_ERROR_FACTORS.items():
        if abs(detected - true_bpm * factor) <= tolerance * factor:
            return name
    return "other"


def _key_error_category(detected: str, true_camelot: str) -> str:
    if detected.upper() == true_camelot.upper():
        return "correct"
    d_num, d_letter = int(detected[:-1]), detected[-1].upper()
    t_num, t_letter = int(true_camelot[:-1]), true_camelot[-1].upper()
    if d_num == t_num and d_letter != t_letter:
        return "relative"  # same number, other letter = relative major/minor
    if d_letter == t_letter and (abs(d_num - t_num) == 1 or abs(d_num - t_num) == 11):
        return "adjacent"  # +/-1 on the wheel
    return "other"


def _analyze_one_tempo(entry: tuple[Path, float, str]) -> dict:
    audio_path, true_bpm, genre = entry
    features = AnalyzerService.analyze_track(str(audio_path), is_track_a=False)
    detected = features.tempo
    return {
        "file": audio_path.name, "genre": genre, "true_bpm": true_bpm, "detected_bpm": round(detected, 2),
        "abs_error": round(abs(detected - true_bpm), 2), "category": _tempo_error_category(detected, true_bpm),
    }


def _analyze_one_key(entry: tuple[Path, str]) -> dict:
    audio_path, true_camelot = entry
    features = AnalyzerService.analyze_track(str(audio_path), is_track_a=False)
    detected = features.camelot_key
    return {
        "file": audio_path.name, "true_camelot": true_camelot, "detected_camelot": detected,
        "category": _key_error_category(detected, true_camelot),
    }


def _run_parallel(pairs: list, fn, label: str) -> list[dict]:
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, result in enumerate(pool.map(fn, pairs), 1):
            results.append(result)
            if i % 50 == 0:
                print(f"  ...{i}/{len(pairs)}")
    elapsed = time.perf_counter() - t0
    print(f"[{label}] {len(pairs)} tracks in {elapsed:.1f}s ({elapsed / len(pairs):.2f}s/track, {WORKERS} workers)")
    return results


def run_tempo_benchmark(dataset_root: Path, out_dir: Path, limit: int | None) -> None:
    pairs = _collect_tempo_pairs(dataset_root / "giantsteps-tempo-dataset", limit)
    if not pairs:
        print(f"[tempo] nothing to benchmark — is the dataset present at {dataset_root}?")
        return

    rows = _run_parallel(pairs, _analyze_one_tempo, "tempo")

    out_csv = out_dir / "tempo_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errors = [r["abs_error"] for r in rows]
    categories = Counter(r["category"] for r in rows)
    acc1 = categories["correct"] / len(rows) * 100
    acc2 = (len(rows) - categories["other"]) / len(rows) * 100

    print(f"[tempo] MAE={np.mean(errors):.2f} BPM  Accuracy1={acc1:.1f}%  Accuracy2(+metrical relations)={acc2:.1f}%")
    print(f"[tempo] error categories: {dict(categories)}")
    print(f"[tempo] per-track detail written to {out_csv}")

    genre_stats: dict[str, list[str]] = {}
    for r in rows:
        genre_stats.setdefault(r["genre"], []).append(r["category"])
    print("[tempo] Accuracy2 by genre (genre: correct+recognized-metrical-relation / total):")
    for genre, cats in sorted(genre_stats.items(), key=lambda kv: -len(kv[1])):
        acc2_genre = sum(c != "other" for c in cats) / len(cats) * 100
        print(f"    {genre:<25} {acc2_genre:5.1f}%  (n={len(cats)})")


def run_key_benchmark(dataset_root: Path, out_dir: Path, limit: int | None) -> None:
    pairs = _collect_key_pairs(dataset_root / "giantsteps-key-dataset", limit)
    if not pairs:
        print(f"[key] nothing to benchmark — is the dataset present at {dataset_root}?")
        return

    rows = _run_parallel(pairs, _analyze_one_key, "key")

    out_csv = out_dir / "key_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    categories = Counter(r["category"] for r in rows)
    exact = categories["correct"] / len(rows) * 100
    compatible = (categories["correct"] + categories.get("relative", 0) + categories.get("adjacent", 0)) / len(rows) * 100

    print(f"[key] Exact accuracy={exact:.1f}%  Camelot-compatible accuracy={compatible:.1f}%")
    print(f"[key] error categories: {dict(categories)}")
    print(f"[key] per-track detail written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tempo", action="store_true", help="run only the tempo benchmark")
    parser.add_argument("--key", action="store_true", help="run only the key benchmark")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of tracks (quick smoke test)")
    parser.add_argument("--datasets-root", type=Path, default=DATASET_ROOT_DEFAULT,
                         help="dir containing giantsteps-tempo-dataset/ and giantsteps-key-dataset/ "
                              f"(default: {DATASET_ROOT_DEFAULT})")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_both = not args.tempo and not args.key
    if args.tempo or run_both:
        run_tempo_benchmark(args.datasets_root, out_dir, args.limit)
    if args.key or run_both:
        run_key_benchmark(args.datasets_root, out_dir, args.limit)
