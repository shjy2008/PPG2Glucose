#!/usr/bin/env python3
"""Consensus top-N features from per-dataset importance rankings.

Uses average rank over features present in every ranking (intersection).
Features missing from a dataset (e.g. bmi/height on MUST) are excluded
automatically.

Usage:
  python consensus_top_features.py
  python consensus_top_features.py --top 10
  python consensus_top_features.py --csv path/to/ranks.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Top-40 importance % from CatBoost (shared g1–g6 setup).
# Edit these lists, or pass --csv with columns: dataset,rank,feature,pct
RANKINGS: dict[str, list[tuple[str, float]]] = {
    'IPS': [
        ('age', 20.95),
        ('g2_ppg_diff2_std', 12.94),
        ('g2_ppg_diff2_mean', 4.76),
        ('bmi', 4.33),
        ('g3_ppg_mean_rise_slope', 3.79),
        ('g4_ppg_spectral_entropy', 3.64),
        ('g5_ppg_autocorr_peak_val', 3.00),
        ('g1_ppg_kurtosis', 2.91),
        ('g3_ppg_ibi_std', 2.75),
        ('weight', 2.23),
        ('g1_ppg_abs_mean', 2.17),
        ('g1_ppg_mean', 2.01),
        ('g6_ppg_half_std_ratio', 1.97),
        ('g6_ppg_half_mean_diff', 1.78),
        ('g3_ppg_peak_amp_mean', 1.63),
        ('g1_ppg_skew', 1.60),
        ('g3_ppg_ibi_rmssd', 1.52),
        ('g1_ppg_rms', 1.39),
        ('g3_ppg_peak_amp_std', 1.36),
        ('height', 1.25),
        ('g4_ppg_band_power_low', 1.22),
        ('g3_ppg_max_slope', 1.22),
        ('g4_ppg_dominant_frequency', 1.20),
        ('g2_ppg_diff1_std', 1.19),
        ('g1_ppg_mean_crossing_count', 1.16),
        ('g2_ppg_diff1_mean', 1.13),
        ('g1_ppg_range', 1.11),
        ('g3_ppg_ibi_mean', 1.02),
        ('g4_ppg_snr', 0.97),
        ('g1_ppg_cv', 0.95),
        ('g3_ppg_beat_density', 0.95),
        ('g1_ppg_length', 0.93),
        ('g3_ppg_peak_amp_cv', 0.93),
        ('g4_ppg_band_power_high', 0.91),
        ('g6_ppg_edge_mean_diff', 0.91),
        ('g5_ppg_heart_rate_bpm', 0.90),
        ('g3_ppg_perfusion_index', 0.88),
        ('g3_ppg_peak_count', 0.82),
        ('g4_ppg_band_power_mid', 0.77),
        ('g5_ppg_autocorr_peak_lag', 0.77),
    ],
    'MUST': [
        ('g4_ppg_band_power_low', 6.38),
        ('age', 4.79),
        ('g1_ppg_mean_crossing_count', 4.61),
        ('weight', 4.56),
        ('g6_ppg_edge_mean_diff', 4.45),
        ('g5_ppg_heart_rate_bpm', 3.52),
        ('g3_ppg_peak_amp_mean', 3.47),
        ('g3_ppg_ibi_mean', 3.43),
        ('g4_ppg_band_power_high', 3.27),
        ('g5_ppg_autocorr_peak_lag', 3.16),
        ('g3_ppg_ibi_std', 2.97),
        ('g2_ppg_diff2_mean', 2.80),
        ('g4_ppg_dominant_frequency', 2.73),
        ('g3_ppg_mean_rise_slope', 2.42),
        ('g3_ppg_ibi_rmssd', 2.40),
        ('g2_ppg_diff1_mean', 2.38),
        ('g3_ppg_max_slope', 2.35),
        ('g4_ppg_band_power_mid', 2.34),
        ('g1_ppg_rms', 2.27),
        ('g1_ppg_mean', 2.22),
        ('g3_ppg_peak_amp_cv', 2.18),
        ('g4_ppg_snr', 2.13),
        ('g4_ppg_spectral_entropy', 2.10),
        ('g3_ppg_peak_amp_std', 2.07),
        ('g6_ppg_half_mean_diff', 2.03),
        ('g1_ppg_abs_mean', 1.96),
        ('g3_ppg_perfusion_index', 1.87),
        ('g2_ppg_diff1_std', 1.85),
        ('g2_ppg_diff2_std', 1.83),
        ('g5_ppg_autocorr_peak_val', 1.78),
        ('g6_ppg_half_std_ratio', 1.65),
        ('g1_ppg_std', 1.64),
        ('g1_ppg_kurtosis', 1.62),
        ('g3_ppg_peak_count', 1.54),
        ('g3_ppg_beat_density', 1.51),
        ('g1_ppg_range', 1.42),
        ('g1_ppg_cv', 1.35),
        ('g1_ppg_skew', 1.21),
        ('g1_ppg_iqr', 1.05),
        ('g1_ppg_median', 0.66),
    ],
    'VitalDB': [
        ('age', 25.13),
        ('bmi', 8.83),
        ('g3_ppg_ibi_mean', 5.43),
        ('g3_ppg_peak_count', 5.35),
        ('height', 5.06),
        ('g3_ppg_beat_density', 4.25),
        ('g4_ppg_spectral_entropy', 2.89),
        ('weight', 2.80),
        ('g5_ppg_heart_rate_bpm', 2.46),
        ('g4_ppg_dominant_frequency', 2.45),
        ('g2_ppg_diff2_std', 2.34),
        ('g5_ppg_autocorr_peak_lag', 2.25),
        ('g1_ppg_mean', 2.08),
        ('g1_ppg_abs_mean', 1.96),
        ('g3_ppg_mean_rise_slope', 1.89),
        ('g3_ppg_ibi_rmssd', 1.76),
        ('g1_ppg_skew', 1.63),
        ('g4_ppg_snr', 1.56),
        ('g6_ppg_edge_mean_diff', 1.54),
        ('g3_ppg_peak_amp_mean', 1.19),
        ('g3_ppg_ibi_std', 1.11),
        ('g5_ppg_autocorr_peak_val', 1.09),
        ('g2_ppg_diff2_mean', 1.07),
        ('g6_ppg_half_mean_diff', 1.02),
        ('g3_ppg_max_slope', 0.97),
        ('g1_ppg_rms', 0.90),
        ('g2_ppg_diff1_std', 0.90),
        ('g1_ppg_kurtosis', 0.89),
        ('g4_ppg_band_power_low', 0.82),
        ('g3_ppg_peak_amp_cv', 0.82),
        ('g1_ppg_std', 0.74),
        ('g4_ppg_band_power_mid', 0.74),
        ('g4_ppg_band_power_high', 0.74),
        ('g2_ppg_diff1_mean', 0.73),
        ('g1_ppg_mean_crossing_count', 0.70),
        ('g1_ppg_median', 0.66),
        ('g1_ppg_range', 0.66),
        ('g1_ppg_iqr', 0.66),
        ('g1_ppg_cv', 0.56),
        ('g3_ppg_perfusion_index', 0.52),
    ],
}


def load_csv(path: Path) -> dict[str, list[tuple[str, float]]]:
    """CSV columns: dataset,feature,pct  (rank = row order within dataset)."""
    import csv

    out: dict[str, list[tuple[str, float]]] = {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row['dataset'].strip()
            feat = row['feature'].strip()
            pct = float(row['pct'])
            out.setdefault(ds, []).append((feat, pct))
    return out


def rank_maps(
    rankings: dict[str, list[tuple[str, float]]],
) -> dict[str, dict[str, int]]:
    return {
        ds: {feat: i + 1 for i, (feat, _) in enumerate(rows)}
        for ds, rows in rankings.items()
    }


def consensus(
    rankings: dict[str, list[tuple[str, float]]],
    top: int = 10,
) -> list[dict]:
    ranks = rank_maps(rankings)
    datasets = list(ranks)
    shared = set.intersection(*(set(r) for r in ranks.values()))
    rows = []
    for feat in shared:
        per = {ds: ranks[ds][feat] for ds in datasets}
        avg = sum(per.values()) / len(datasets)
        rows.append({'feature': feat, 'avg_rank': avg, **{f'rank_{ds}': per[ds] for ds in datasets}})
    rows.sort(key=lambda r: (r['avg_rank'], r['feature']))
    return rows[:top]


def topk_overlap(
    rankings: dict[str, list[tuple[str, float]]],
    k: int = 10,
) -> list[tuple[str, int, list[str]]]:
    """Features in ≥2 datasets' top-k (intersection universe only)."""
    ranks = rank_maps(rankings)
    shared = set.intersection(*(set(r) for r in ranks.values()))
    hits: dict[str, list[str]] = {}
    for ds, rows in rankings.items():
        for feat, _ in rows[:k]:
            if feat not in shared:
                continue
            hits.setdefault(feat, []).append(ds)
    scored = [(f, len(ds_list), ds_list) for f, ds_list in hits.items()]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--top', type=int, default=10, help='Consensus list length')
    p.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='Optional CSV with columns dataset,feature,pct',
    )
    args = p.parse_args()

    rankings = load_csv(args.csv) if args.csv else RANKINGS
    datasets = list(rankings)
    shared_n = len(set.intersection(*(set(f for f, _ in rows) for rows in rankings.values())))
    print(f'Datasets: {datasets}')
    print(f'Shared features in all rankings: {shared_n}')
    print(f'\nConsensus top {args.top} (lowest average rank):')
    print(f'{"#":>2}  {"feature":<32}  ' + '  '.join(f'{ds:>8}' for ds in datasets) + '  avg_rank')
    for i, row in enumerate(consensus(rankings, top=args.top), 1):
        rank_cols = '  '.join(f'{row[f"rank_{ds}"]:>8}' for ds in datasets)
        print(f'{i:2}  {row["feature"]:<32}  {rank_cols}  {row["avg_rank"]:8.2f}')

    print(f'\nIn ≥2 datasets\' own top {args.top} (shared features only):')
    for feat, n, ds_list in topk_overlap(rankings, k=args.top):
        if n < 2:
            continue
        print(f'  {feat:<32}  {n}/{len(datasets)}  ({", ".join(ds_list)})')


if __name__ == '__main__':
    main()
