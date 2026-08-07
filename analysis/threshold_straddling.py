# ============================================================
# Threshold-Straddling Analysis
# ============================================================
# Identifies validation patients whose PE classification score
# distribution spans the decision threshold across multiple
# clean retraining runs.
#
# A patient is classified as a threshold-straddler if:
#   min score across runs < T  AND  max score across runs >= T
#
# where T is the illustrative decision threshold (default 0.5).
# In practice the manufacturer's operating threshold is
# proprietary and would be applied internally. The manufacturer
# would report the stability evidence (straddler count, alert
# proportions) rather than the threshold value itself.
#
# Usage:
#   PYTHONPATH=$(pwd) python analysis/threshold_straddling.py
#
# Input:
#   results/control/summary.json
#
# Output:
#   results/analysis/straddling_results.json
#   results/analysis/straddling_results.txt  (human-readable)
#
# Output includes for each straddling patient:
#   - Frontal index
#   - PE ground truth label (PE+ = missed diagnosis risk,
#                            PE- = false alarm risk)
#   - Scores across all runs
#   - Alert proportion (n/5 runs resulting in alert)
#   - Mean, std, min, max scores
# ============================================================

import json
from pathlib import Path


CONFIG = {
    "control_results": "/home/ubuntu/poison-storage/results/control/summary.json",
    "output_dir":      "/home/ubuntu/poison-storage/results/analysis",
    "threshold":       0.5,    # illustrative only -- see notes above
    "n_runs":          5,
}

# PE label index in 14-label CheXpert vector
PLEURAL_EFFUSION_IDX = 10

# Ground truth labels for the 202 frontal validation patients
# These are loaded from the control results file which records
# PE labels alongside scores. If not available, set to None
# and the script will flag this for manual verification.
# Format: {frontal_idx: pe_label} where pe_label is 0 or 1


def mean(x):
    return sum(x) / len(x)


def std(x, ddof=1):
    m = mean(x)
    return (sum((v - m)**2 for v in x) / (len(x) - ddof))**0.5


def run_straddling_analysis(config):
    """
    Identify threshold-straddling patients from control results.
    """
    print("=" * 60)
    print("THRESHOLD-STRADDLING ANALYSIS")
    print("=" * 60)

    control_path = Path(config["control_results"])
    output_dir   = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not control_path.exists():
        print(f"ERROR: Control results not found at {control_path}")
        print("Run retrain_control.py first.")
        return

    with open(control_path) as f:
        control = json.load(f)

    threshold  = config["threshold"]
    n_patients = control["n_patients"]
    n_runs     = control["n_runs"]
    per_patient = control["per_patient"]

    # Load PE ground truth labels if available
    pe_labels = control.get("pe_labels", {})

    print(f"\nValidation patients: {n_patients}")
    print(f"Retraining runs:     {n_runs}")
    print(f"Illustrative threshold: {threshold}")
    print(f"(In deployment, manufacturer applies their "
          f"proprietary threshold)")

    # Identify straddlers
    straddlers     = []
    non_straddlers = []

    for idx_str, data in per_patient.items():
        idx    = int(idx_str)
        scores = data["scores"]
        mn     = data["min"]
        mx     = data["max"]

        # Straddler definition: min < T AND max >= T
        is_straddler = (mn < threshold) and (mx >= threshold)

        # Alert proportion: fraction of runs above threshold
        n_alerts       = sum(1 for s in scores if s >= threshold)
        alert_prop     = n_alerts / n_runs

        # Ground truth PE label
        pe_label = pe_labels.get(idx_str, pe_labels.get(idx, None))

        patient_data = {
            "frontal_idx":   idx,
            "pe_label":      pe_label,
            "scores":        scores,
            "mean":          data["mean"],
            "std":           data["std"],
            "min":           mn,
            "max":           mx,
            "n_alerts":      n_alerts,
            "alert_prop":    round(alert_prop, 2),
            "alert_str":     f"{n_alerts}/{n_runs}",
            "clinical_risk": _clinical_risk(pe_label, n_alerts,
                                             n_runs),
        }

        if is_straddler:
            straddlers.append(patient_data)
        else:
            non_straddlers.append(patient_data)

    # Sort straddlers by clinical risk
    # PE-positive with low alert proportion = highest risk
    straddlers.sort(key=lambda x: (
        0 if x["pe_label"] == 1 else 1,  # PE+ first
        x["n_alerts"]                     # lowest alerts first
    ))

    n_straddlers = len(straddlers)
    n_pe_pos     = sum(1 for s in straddlers if s["pe_label"] == 1)
    n_pe_neg     = sum(1 for s in straddlers if s["pe_label"] == 0)
    n_pe_unknown = sum(
        1 for s in straddlers if s["pe_label"] is None
    )

    # Stability summary for all 202 patients
    all_stds  = [per_patient[str(i)]["std"]
                 for i in range(n_patients)]
    mean_std  = mean(all_stds)
    max_std   = max(all_stds)
    max_std_idx = all_stds.index(max_std)

    n_gt010 = sum(1 for s in all_stds if s > 0.10)
    n_gt005 = sum(1 for s in all_stds if s > 0.05)
    n_lt002 = sum(1 for s in all_stds if s < 0.02)

    print(f"\nStability across all {n_patients} patients:")
    print(f"  Mean std:            {mean_std:.4f}")
    print(f"  Max std:             {max_std:.4f} (idx {max_std_idx})")
    print(f"  Patients std > 0.10: {n_gt010}")
    print(f"  Patients std > 0.05: {n_gt005}")
    print(f"  Patients std < 0.02: {n_lt002}")

    print(f"\nThreshold-straddling results "
          f"(threshold = {threshold}):")
    print(f"  Total straddlers:    {n_straddlers} / {n_patients} "
          f"({n_straddlers/n_patients*100:.1f}%)")
    print(f"  PE-positive:         {n_pe_pos}  "
          f"(risk of missed diagnosis)")
    print(f"  PE-negative:         {n_pe_neg}  "
          f"(risk of unnecessary alert)")
    if n_pe_unknown:
        print(f"  PE label unknown:    {n_pe_unknown}")

    print(f"\nStraddling patients (sorted by clinical risk):")
    print(f"{'Idx':<6} {'PE+':<5} {'Alerts':<8} "
          f"{'Mean':<7} {'Std':<7} {'Min':<7} {'Max':<7} "
          f"Clinical risk")
    print("-" * 75)
    for s in straddlers:
        pe_str  = ("+" if s["pe_label"] == 1
                   else ("-" if s["pe_label"] == 0 else "?"))
        risk    = s["clinical_risk"]
        print(f"{s['frontal_idx']:<6} {pe_str:<5} "
              f"{s['alert_str']:<8} "
              f"{s['mean']:<7.4f} {s['std']:<7.4f} "
              f"{s['min']:<7.4f} {s['max']:<7.4f} {risk}")

    # Save results
    output = {
        "threshold":          threshold,
        "threshold_note":     (
            "Illustrative value only. Manufacturer applies "
            "their proprietary operating threshold internally "
            "and reports stability evidence, not the threshold."
        ),
        "n_patients":         n_patients,
        "n_runs":             n_runs,
        "n_straddlers":       n_straddlers,
        "pct_straddlers":     round(
            n_straddlers / n_patients * 100, 1
        ),
        "n_pe_positive":      n_pe_pos,
        "n_pe_negative":      n_pe_neg,
        "n_pe_unknown":       n_pe_unknown,
        "stability_summary": {
            "mean_std":   round(mean_std, 4),
            "max_std":    round(max_std,  4),
            "max_std_idx": max_std_idx,
            "n_gt010":    n_gt010,
            "n_gt005":    n_gt005,
            "n_lt002":    n_lt002,
        },
        "straddlers":         straddlers,
    }

    json_path = output_dir / "straddling_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Human-readable text report
    txt_path = output_dir / "straddling_results.txt"
    with open(txt_path, "w") as f:
        f.write("THRESHOLD-STRADDLING ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Illustrative threshold:  {threshold}\n")
        f.write(f"Validation patients:     {n_patients}\n")
        f.write(f"Retraining runs:         {n_runs}\n\n")

        f.write("STABILITY SUMMARY (all patients)\n")
        f.write(f"  Mean std:      {mean_std:.4f}\n")
        f.write(f"  Max std:       {max_std:.4f} "
                f"(idx {max_std_idx})\n")
        f.write(f"  std > 0.10:    {n_gt010} patients\n")
        f.write(f"  std > 0.05:    {n_gt005} patients\n")
        f.write(f"  std < 0.02:    {n_lt002} patients\n\n")

        f.write("THRESHOLD-STRADDLING RESULTS\n")
        f.write(f"  Straddlers:    {n_straddlers} / {n_patients} "
                f"({n_straddlers/n_patients*100:.1f}%)\n")
        f.write(f"  PE-positive:   {n_pe_pos} "
                f"(missed diagnosis risk)\n")
        f.write(f"  PE-negative:   {n_pe_neg} "
                f"(unnecessary alert risk)\n\n")

        f.write(f"{'Idx':<6} {'PE':<4} {'Alerts':<8} "
                f"{'Mean':<8} {'Std':<8} "
                f"{'Min':<8} {'Max':<8} Risk\n")
        f.write("-" * 70 + "\n")
        for s in straddlers:
            pe_str = ("+" if s["pe_label"] == 1
                      else ("-" if s["pe_label"] == 0 else "?"))
            f.write(
                f"{s['frontal_idx']:<6} {pe_str:<4} "
                f"{s['alert_str']:<8} "
                f"{s['mean']:<8.4f} {s['std']:<8.4f} "
                f"{s['min']:<8.4f} {s['max']:<8.4f} "
                f"{s['clinical_risk']}\n"
            )

    print(f"Text report saved: {txt_path}")
    return output


def _clinical_risk(pe_label, n_alerts, n_runs):
    """
    Classify clinical risk for a straddling patient.

    PE-positive patients risk missed diagnosis when not alerted.
    PE-negative patients risk unnecessary alerts when alerted.
    Severity increases with the degree of inconsistency.
    """
    if pe_label is None:
        return "unknown (PE label not available)"

    if pe_label == 1:
        # PE-positive: risk is missed diagnosis
        # Lower alert proportion = higher risk
        if n_alerts == 1:
            return "HIGH -- PE+ flagged in 1 run only"
        elif n_alerts == 2:
            return "MODERATE-HIGH -- PE+ flagged in 2 runs"
        elif n_alerts == 3:
            return "MODERATE -- PE+ flagged in 3 runs"
        elif n_alerts == 4:
            return "LOW-MODERATE -- PE+ flagged in 4 runs"
        else:
            return "LOW -- PE+ consistently flagged"
    else:
        # PE-negative: risk is unnecessary alert
        if n_alerts == 4:
            return "MODERATE -- PE- falsely flagged in 4 runs"
        elif n_alerts == 3:
            return "LOW-MODERATE -- PE- falsely flagged in 3 runs"
        elif n_alerts == 2:
            return "LOW -- PE- falsely flagged in 2 runs"
        else:
            return "MINIMAL -- PE- rarely falsely flagged"


if __name__ == "__main__":
    run_straddling_analysis(CONFIG)
