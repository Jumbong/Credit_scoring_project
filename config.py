from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[0]

data_path = project_root / "data"
src_path = project_root / "src"

data_output_path = project_root / "outputs"

