"""Archive the exact locked implementation before intentional source changes."""
from pathlib import Path

from src.provenance import archive_code


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    destination = archive_code(root, root / "results/usgs_ablation_exploratory/selected_configs.json",
                               references=("src/evaluate.py", "src/baselines.py", "src/jump_filter.py",
                                           "src/robust_jump_filter.py"))
    print(f"Archived original code bytes: {destination}")
