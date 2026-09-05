"""
training/evaluate.py

Prints an honest evaluation report. Never reports training-set accuracy as
if it were real-world performance — everything here comes from cross_val_predict
on held-out folds (see models/classifier.evaluate_candidate).
"""
from typing import Dict


def print_report(best_name: str, all_results: Dict[str, dict]):
    print("\n" + "=" * 60)
    print("MODEL VALIDATION REPORT (cross-validated, held-out folds only)")
    print("=" * 60)

    for name, m in all_results.items():
        marker = "  <-- SELECTED" if name == best_name else ""
        print(f"\n[{name.upper()}]{marker}")
        print(f"  VALIDATION ACCURACY: {m['accuracy']*100:5.1f}%")
        print(f"  BALANCED ACCURACY:   {m['balanced_accuracy']*100:5.1f}%")
        print(f"  PRECISION (macro):   {m['precision']*100:5.1f}%")
        print(f"  RECALL (macro):      {m['recall']*100:5.1f}%")
        print(f"  F1 SCORE (macro):    {m['f1']*100:5.1f}%")
        print(f"  Per-class accuracy:  {m['per_class_accuracy']}")
        print(f"  Confusion matrix:    {m['confusion_matrix']}")

    best = all_results[best_name]
    print("\n" + "-" * 60)
    if best["balanced_accuracy"] < 0.5:
        print("WARNING: balanced accuracy is at or below chance level for this")
        print("class count. This model is NOT reliable for robot control.")
        print("Recollect calibration data, check electrode contact, and/or")
        print("simplify the class set before using this model in real-time.")
    elif best["balanced_accuracy"] < 0.7:
        print("NOTE: validation accuracy is modest. Consider more trials,")
        print("cleaner signal, or fewer/simpler command classes before")
        print("relying on this for real robot control.")
    else:
        print(f"Best model: {best_name} — balanced accuracy {best['balanced_accuracy']*100:.1f}%")
        print("This reflects cross-validated performance on this subject's")
        print("calibration data only. Real-world performance may differ.")
    print("-" * 60 + "\n")
