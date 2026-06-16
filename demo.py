"""
demo.py — one-command setup for the AI-Powered Cloud Cost Optimizer.

Idempotent: run `py demo.py` after cloning and it brings the system to a runnable state,
skipping any step whose output already exists. It will, in order:
  1. generate raw synthetic data        (if data/raw/usage_2024.csv is missing)
  2. clean raw -> processed              (if data/processed/usage_processed.csv is missing)
  3. train the three models              (if any models/saved/*.joblib is missing)
and then prints how to launch the API and the dashboard.

Nothing is retrained or regenerated when the artifacts already exist.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Inputs/outputs of each phase
RAW = ROOT / "data" / "raw" / "usage_2024.csv"
GROUND_TRUTH = ROOT / "data" / "raw" / "ground_truth.csv"
PROCESSED = ROOT / "data" / "processed" / "usage_processed.csv"

GENERATE = ROOT / "data" / "synthetic" / "generate_data.py"
PROCESS = ROOT / "data" / "process_data.py"

MODELS = {
    "forecast": ROOT / "models" / "saved" / "forecast_model.joblib",
    "waste": ROOT / "models" / "saved" / "waste_model.joblib",
    "anomaly": ROOT / "models" / "saved" / "anomaly_model.joblib",
}
TRAIN_SCRIPTS = {
    "forecast": ROOT / "models" / "train" / "train_forecast.py",
    "waste": ROOT / "models" / "train" / "train_waste.py",
    "anomaly": ROOT / "models" / "train" / "train_anomaly.py",
}


def run(script: Path):
    """Run a project script with the current interpreter, failing loudly."""
    print(f"    -> running {script.relative_to(ROOT)}")
    subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)


def step(n, msg):
    print(f"\n[{n}/3] {msg}")


def main():
    print("=" * 64)
    print("  AI-Powered Cloud Cost Optimizer — setup (demo.py)")
    print("=" * 64)

    # --- 1. raw data ---
    step(1, "Raw synthetic data")
    if RAW.exists() and GROUND_TRUTH.exists():
        print("    [skip] raw data already present.")
    else:
        run(GENERATE)
        print("    [ok] raw data generated.")

    # --- 2. processed data ---
    step(2, "Cleaned / processed data")
    if PROCESSED.exists():
        print("    [skip] processed data already present.")
    else:
        run(PROCESS)
        print("    [ok] processed data created.")

    # --- 3. models ---
    step(3, "Trained models")
    missing = {name: path for name, path in MODELS.items() if not path.exists()}
    if not missing:
        print("    [skip] all three models already present.")
    else:
        print(f"    missing: {', '.join(missing)} — training (this runs the Phase 2 scripts).")
        for name in missing:
            run(TRAIN_SCRIPTS[name])
        print("    [ok] models trained.")

    # --- summary ---
    print("\n" + "=" * 64)
    print("  Status")
    print("=" * 64)
    for label, path in [("raw data", RAW), ("ground truth", GROUND_TRUTH),
                        ("processed data", PROCESSED), *MODELS.items()]:
        mark = "OK " if path.exists() else "MISSING"
        print(f"  [{mark}] {path.relative_to(ROOT)}")

    print("\nNext steps:")
    print("  Start the API backend:")
    print("      py -m uvicorn api.main:app --reload")
    print("      -> docs at http://127.0.0.1:8000/docs")
    print("  Start the dashboard:")
    print("      py -m streamlit run dashboard/app.py")
    print("      -> app at http://127.0.0.1:8501")
    print("  Or run everything in containers:")
    print("      docker-compose up")
    print()


if __name__ == "__main__":
    main()
