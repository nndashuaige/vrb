import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrbreproduction.e_analysis_1_usable_data_stats import (
    EAnalysis1Config,
    run_e_analysis_1,
)


if __name__ == "__main__":
    run_e_analysis_1(EAnalysis1Config())
