"""Convenience entry point for synthetic filter experiments (real run is included in run_baselines)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_experiments import main
if __name__ == "__main__": main()
