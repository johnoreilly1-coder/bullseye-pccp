# ============================================================
# Statistical Analysis -- Multi-Run Poisoned vs Control
# ============================================================
# Performs patient-level Welch's two-sample t-tests comparing
# PE classification scores between poisoned and control
# retraining runs for each of the 6 target patients.
#
# Statistical approach:
#   - Per-patient Welch's t-test (N=5 per condition)
#   - Bonferroni correction for 6 simultaneous comparisons
#     (adjusted alpha = 0.05 / 6 = 0.0083)
#   - 95% confidence intervals for mean difference
#   - Effect sizes (mean difference, pooled std)
#   - Tests are exploratory given small N and 6 patients
#
# Usage:
#   PYTHONPATH=$(pwd) python experiments/run_analysis.py
#
# Input:
#   results/control/summary.json
#   results/multirun/idx_{N}/summary.json  (for each target)
#
# Output:
#   results/analysis/statistical_results.json
#   results/analysis/statistical_results.txt  (human-readable)
#
# Notes:
#   - p-values computed using scipy.stats.ttest_ind if available,
#     otherwise approximated from t-distribution lookup table.
#   - Matched-seed design enables paired analysis in future work.
#     This script uses unpaired Welch's t-test as seeds are
#     matched but not all augmentation randomness is controlled.
# ============================================================

import json
import sys
import math
from pathlib import Path


# Target patients for the 6-patient experiment
TARGET_PATIENTS = {
    70:  {"patient": "patient64609", "group": "near_0.5"},
    134: {"patient": "patient64673", "group": "near_0.5"},
    12:  {"patient": "patient64552", "group": "near_0.7"},
    111: {"patient": "patient64650", "group": "near_0.7"},
    37:  {"patient": "patient64577", "group": "near_0.9"},
    105: {"patient": "patient64644", "group": "near_0.9"},
}

CONFIG = {
    "control_results":  "/home/ubuntu/poison-storage/results/control/summary.json",
    "multirun_dir":     "/home/ubuntu/poison-storage/results/multirun",
    "output_dir":       "/home/ubuntu/poison-storage/results/analysis",
    "alpha":            0.05,
    "n_comparisons":    6,      # Bonferroni correction
}


# ── Pure-Python statistics (no scipy dependency) ──────────────

def mean(x):
    return sum(x) / len(x)


def variance(x, ddof=1):
    m = mean(x)
    return sum((v - m)**2 for v in x) / (len(x) - ddof)


def std(x, ddof=1):
    return variance(x, ddof)**0.5


def welch_df(x, y):
    """
    Welch-Satterthwaite degrees of freedom.
    """
    n1, n2   = len(x), len(y)
    v1, v2   = variance(x), variance(y)
    s1, s2   = v1 / n1, v2 / n2
    numerator   = (s1 + s2)**2
    denominator = (s1**2 / (n1 - 1)) + (s2**2 / (n2 - 1))
    return numerator / denominator


def welch_t(x, y):
    """
    Welch's t-statistic.
    """
    n1, n2 = len(x), len(y)
    v1, v2 = variance(x), variance(y)
    se     = math.sqrt(v1 / n1 + v2 / n2)
    return (mean(x) - mean(y)) / se, se


def t_cdf_approx(t, df):
    """
    Approximate two-tailed p-value from t-distribution.
    Uses lookup table for common df values with interpolation.
    For exact values use scipy.stats.ttest_ind.
    """
    # Critical values for two-tailed test at common alphas
    # Format: df -> [(alpha, t_crit), ...]
    table = {
        4:  [(0.001, 8.610), (0.01, 4.604), (0.025, 3.747),
             (0.05, 2.776), (0.10, 2.132), (0.20, 1.533)],
        5:  [(0.001, 6.869), (0.01, 4.032), (0.025, 3.365),
             (0.05, 2.571), (0.10, 2.015), (0.20, 1.476)],
        6:  [(0.001, 5.959), (0.01, 3.707), (0.025, 3.143),
             (0.05, 2.447), (0.10, 1.943), (0.20, 1.440)],
        7:  [(0.001, 5.408), (0.01, 3.499), (0.025, 2.998),
             (0.05, 2.365), (0.10, 1.895), (0.20, 1.415)],
        8:  [(0.001, 5.041), (0.01, 3.355), (0.025, 2.896),
             (0.05, 2.306), (0.10, 1.860), (0.20, 1.397)],
    }
    abs_t = abs(t)
    df_r  = max(4, min(8, round(df)))
    row   = table[df_r]

    for alpha, t_crit in row:
        if abs_t >= t_crit:
            return alpha

    return 0.20   # p > 0.20


def t_critical_95(df):
    """
    Two-tailed t critical value at 95% CI level.
    """
    crits = {4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306}
    df_r = max(4, min(8, round(df)))
    return crits[df_r]


def confidence_interval_95(x, y):
    """
    95% CI for mean difference (x mean - y mean).
    """
    n1, n2   = len(x), len(y)
    v1, v2   = variance(x), variance(y)
    diff     = mean(x) - mean(y)
    se       = math.sqrt(v1 / n1 + v2 / n2)
    df       = welch_df(x, y)
    t_crit   = t_critical_95(df)
    return diff - t_crit * se, diff + t_crit * se


def try_scipy_ttest(x, y):
    """
    Attempt exact p-value from scipy. Falls back to
    approximation if scipy not available.
    """
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
        return float(t_stat), float(p_val), "exact (scipy)"
    except ImportError:
        t_stat, _ = welch_t(x, y)
        df        = welch_df(x, y)
        p_val     = t_cdf_approx(t_stat, df)
        return t_stat, p_val, "approximate (lookup table)"


# ── Load results ──────────────────────────────────────────────

def load_control_scores(control_path, frontal_idx):
    """
    Load PE scores for a specific patient from control results.
    """
    with open(control_path) as f:
        control = json.load(f)

    # per_patient keyed by string index
    patient_data = control["per_patient"][str(frontal_idx)]
    return patient_data["scores"]


def load_poisoned_scores(multirun_dir, frontal_idx):
    """
    Load PE scores for a specific patient from poisoned results.
    """
    summary_path = (
        Path(multirun_dir)
        / f"idx_{frontal_idx}"
        / "summary.json"
    )
    with open(summary_path) as f:
        summary = json.load(f)

    return summary["pe_scores"]


def load_control_auc(control_path):
    """Load AUC statistics from control summary."""
    with open(control_path) as f:
        control = json.load(f)
    return control["auc_mean"], control["auc_std"], \
           control["auc_per_run"]


# ── Main analysis ─────────────────────────────────────────────

def run_analysis(config):
    """
    Run statistical analysis for all 6 target patients.
    """
    print("=" * 60)
    print("STATISTICAL ANALYSIS -- POISONED VS CONTROL")
    print("=" * 60)

    control_path = Path(config["control_results"])
    multirun_dir = Path(config["multirun_dir"])
    output_dir   = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not control_path.exists():
        print(f"ERROR: Control results not found at {control_path}")
        print("Run retrain_control.py first.")
        sys.exit(1)

    alpha_bonferroni = config["alpha"] / config["n_comparisons"]
    print(f"\nSignificance threshold: alpha = {config['alpha']}")
    print(f"Bonferroni correction:  alpha / {config['n_comparisons']} "
          f"= {alpha_bonferroni:.4f}")
    print(f"N per condition: 5 runs")

    # Load control AUC
    auc_mean, auc_std, auc_per_run = load_control_auc(control_path)
    print(f"\nControl AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
    print(f"Control AUC per run: {auc_per_run}")

    results = []

    for frontal_idx in TARGET_PATIENTS:
        info = TARGET_PATIENTS[frontal_idx]

        # Check poisoned results exist
        summary_path = (
            multirun_dir / f"idx_{frontal_idx}" / "summary.json"
        )
        if not summary_path.exists():
            print(f"\nWARNING: idx {frontal_idx} poisoned results "
                  f"not found at {summary_path}")
            print(f"  Run retrain_multirun.py --frontal_idx "
                  f"{frontal_idx} first.")
            continue

        # Load scores
        control_scores  = load_control_scores(
            control_path, frontal_idx
        )
        poisoned_scores = load_poisoned_scores(
            multirun_dir, frontal_idx
        )

        # Statistical tests
        t_stat, p_val, p_method = try_scipy_ttest(
            poisoned_scores, control_scores
        )
        df        = welch_df(poisoned_scores, control_scores)
        diff      = mean(poisoned_scores) - mean(control_scores)
        ci_lo, ci_hi = confidence_interval_95(
            poisoned_scores, control_scores
        )

        sig_uncorrected  = p_val < config["alpha"]
        sig_bonferroni   = p_val < alpha_bonferroni

        result = {
            "frontal_idx":       frontal_idx,
            "patient":           info["patient"],
            "group":             info["group"],
            "control_scores":    control_scores,
            "poisoned_scores":   poisoned_scores,
            "control_mean":      round(mean(control_scores),  4),
            "control_std":       round(std(control_scores),   4),
            "poisoned_mean":     round(mean(poisoned_scores), 4),
            "poisoned_std":      round(std(poisoned_scores),  4),
            "diff":              round(diff,    4),
            "ci_95_lo":          round(ci_lo,   4),
            "ci_95_hi":          round(ci_hi,   4),
            "t_stat":            round(t_stat,  4),
            "df":                round(df,      2),
            "p_value":           round(p_val,   4),
            "p_method":          p_method,
            "sig_uncorrected":   sig_uncorrected,
            "sig_bonferroni":    sig_bonferroni,
            "alpha_bonferroni":  round(alpha_bonferroni, 4),
        }
        results.append(result)

        # Print result
        sig_str = ""
        if sig_bonferroni:
            sig_str = "* (Bonferroni)"
        elif sig_uncorrected:
            sig_str = "* (uncorrected only)"
        else:
            sig_str = "n.s."

        direction = ""
        if abs(diff) > 0.01:
            direction = ("INCREASE (paradoxical)"
                         if diff > 0 else "DECREASE (intended)")

        print(f"\n  idx {frontal_idx} ({info['group']}) "
              f"-- {info['patient']}")
        print(f"    Control:  {result['control_mean']:.4f} "
              f"+/- {result['control_std']:.4f}  "
              f"{control_scores}")
        print(f"    Poisoned: {result['poisoned_mean']:.4f} "
              f"+/- {result['poisoned_std']:.4f}  "
              f"{poisoned_scores}")
        print(f"    Diff:     {diff:+.4f}  "
              f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"    t={t_stat:+.3f}  df={df:.2f}  "
              f"p={p_val:.4f}  {sig_str}")
        if direction:
            print(f"    Direction: {direction}")

    # Overall summary
    n_sig_bonferroni  = sum(
        1 for r in results if r["sig_bonferroni"]
    )
    n_sig_uncorrected = sum(
        1 for r in results if r["sig_uncorrected"]
    )

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Patients tested:            {len(results)} / 6")
    print(f"  Significant (uncorrected):  {n_sig_uncorrected}")
    print(f"  Significant (Bonferroni):   {n_sig_bonferroni}")

    if n_sig_bonferroni == 0:
        print(f"  Conclusion: Under WB 5% poisoning, no patient")
        print(f"  shows a statistically significant effect after")
        print(f"  Bonferroni correction (alpha = {alpha_bonferroni:.4f}).")
        print(f"  Findings are inconclusive -- see thesis for")
        print(f"  discussion of limitations.")
    elif n_sig_bonferroni < n_comparisons:
        print(f"  Conclusion: Under WB 5% poisoning, {n_sig_bonferroni} of")
        print(f"  {len(results)} patients show a statistically significant")
        print(f"  decrease after Bonferroni correction")
        print(f"  (alpha = {alpha_bonferroni:.4f}). All {len(results)} patients")
        print(f"  show the intended decrease in PE score.")
    else:
        print(f"  Conclusion: Under WB 5% poisoning, all")
        print(f"  {len(results)} patients show a statistically significant")
        print(f"  decrease after Bonferroni correction")
        print(f"  (alpha = {alpha_bonferroni:.4f}).")

    print(f"{'='*60}")

    # Save results
    output = {
        "control_auc_mean":      auc_mean,
        "control_auc_std":       auc_std,
        "control_auc_per_run":   auc_per_run,
        "alpha":                 config["alpha"],
        "alpha_bonferroni":      round(alpha_bonferroni, 4),
        "n_comparisons":         config["n_comparisons"],
        "n_significant_bonferroni":  n_sig_bonferroni,
        "n_significant_uncorrected": n_sig_uncorrected,
        "patient_results":       results,
    }

    json_path = output_dir / "statistical_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Human-readable text summary
    txt_path = output_dir / "statistical_results.txt"
    with open(txt_path, "w") as f:
        f.write("STATISTICAL ANALYSIS -- POISONED VS CONTROL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Control AUC: {auc_mean:.4f} +/- {auc_std:.4f}\n")
        f.write(f"Alpha: {config['alpha']}  "
                f"Bonferroni: {alpha_bonferroni:.4f}\n\n")

        header = (f"{'Idx':<6} {'Group':<10} {'C mean':<8} "
                  f"{'P mean':<8} {'Diff':<8} "
                  f"{'95% CI':<22} {'t':<8} "
                  f"{'df':<6} {'p':<8} {'Sig'}\n")
        f.write(header)
        f.write("-" * 90 + "\n")

        for r in results:
            ci_str = (f"[{r['ci_95_lo']:+.4f}, "
                      f"{r['ci_95_hi']:+.4f}]")
            sig    = ("*B" if r["sig_bonferroni"]
                      else ("*" if r["sig_uncorrected"]
                            else "n.s."))
            line   = (
                f"{r['frontal_idx']:<6} "
                f"{r['group']:<10} "
                f"{r['control_mean']:<8.4f} "
                f"{r['poisoned_mean']:<8.4f} "
                f"{r['diff']:+8.4f} "
                f"{ci_str:<22} "
                f"{r['t_stat']:+8.3f} "
                f"{r['df']:<6.2f} "
                f"{r['p_value']:<8.4f} "
                f"{sig}\n"
            )
            f.write(line)

        f.write("\n*B = significant after Bonferroni correction\n")
        f.write("*  = nominally significant (uncorrected) only\n")
        f.write("n.s. = not significant\n")

    print(f"Text summary saved: {txt_path}")
    return output


if __name__ == "__main__":
    run_analysis(CONFIG)
