#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import redmapper
import yaml

current_dir = Path.cwd().resolve()
marker = "pyproject.toml"
root_path = None  # Initialize root_path

while True:
    # Check if current_dir is valid and hasn't gone above the filesystem root
    if not current_dir or current_dir == current_dir.parent:
        print("Error: pyproject.toml not found in parent directories.")
        # Handle the error appropriately, maybe raise an exception or exit
        # For now, just break to avoid infinite loop if marker is truly missing
        break

    if (current_dir / marker).exists():
        root_path = current_dir
        print(f"Project root found: {root_path}")  # Confirm the path found
        break
    else:
        current_dir = current_dir.parent

if root_path:
    root_path_str = str(root_path)

    if root_path_str not in sys.path:
        sys.path.append(root_path_str)

        from src.post_process.comparison.combine import (
            main as combine,
        )
        from src.post_process.member_comparison import (
            main as member_comparison,
        )
        from src.post_process.zred_photo_z import (
            main as zred_photo_z,
        )

else:
    print("Could not proceed without finding the project root.")


base_dir = os.getcwd()
print("Running in base directory: {}".format(base_dir))


def _make_abs(path):
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


parser = argparse.ArgumentParser(description="Calibrate the redMaPPer red sequence")
parser.add_argument(
    "-c",
    "--configfile",
    action="store",
    type=str,
    required=True,
    help="YAML config file",
)

args = parser.parse_args()
args.configfile = _make_abs(args.configfile)
print("Using configuration file: {}".format(args.configfile))

with open(args.configfile, "r", encoding="utf-8") as f:
    config_cal = yaml.safe_load(f)

config_cal["galfile"] = _make_abs(config_cal.get("galfile"))
config_cal["specfile"] = _make_abs(config_cal.get("specfile"))
print("Galaxy input file set to: {}".format(config_cal["galfile"]))
print("Spec input file set to: {}".format(config_cal["specfile"]))
with open(args.configfile, "w", encoding="utf-8") as f:
    yaml.dump(config_cal, f, default_flow_style=False, allow_unicode=True)

cal_path = os.path.join(base_dir, "cal")
run_path = os.path.join(base_dir, "run")

os.makedirs(cal_path, exist_ok=True)
print("Calibration directory created (or exists) at: {}".format(cal_path))

os.chdir(cal_path)
print("Changed working directory to calibration path.")

# Functional replacement for RedmapperCalibrator
redmapper.calibration.calibrate_redmapper(args.configfile)


os.makedirs(run_path, exist_ok=True)
print("Run directory created (or exists) at: {}".format(run_path))

os.chdir(run_path)
print("Changed working directory to run path.")

# step 1: Update configuration file
config_run = "run_default.yml"
with open(config_run, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
data["consolidate_vlim_lstars"] = []
data["consolidate_lambda_cuts"] = [5, 10, 15, 20]
with open(config_run, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

print("Updated consolidate_vlim_lstars to an empty list in configuration file.")

config = redmapper.Configuration(config_run)

# Step 2: Compute zred photometric redshifts (if necessary)
# This will use python multiprocessing to run on config.calib_nproc cores
# print("Starting zredRunpix...")
# redmapper.run_zred_pixels(config)

# Step 3: Compute zred background (if necessary)
# redmapper.generate_zred_background(config)

# Step 4: Run the Cluster Finder
config.run_min_nside = 4  # nside desired
config.border = config.compute_border()
# Functional replacement for RedmapperRun
print("Starting redmapper_run...")
redmapper.redmapper_run(config, consolidate=False)

# Step 5: Consolidate the Catalog Pixels
# Functional replacement for RedmapperConsolidateTask
print("Starting catalog consolidation...")
redmapper.pipeline.run_redmapper_consolidate_task(
    config_run,
    lambda_cuts=None,
    vlim_lstars=[],
)


# Post-processing: Generate comparison plots
plots_dir = Path(base_dir) / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
print(f"Output plots directory: {plots_dir}")

# Step 6: Generate combined comparison images (2x3 grid)
print("\nStep 6: Generating combined comparison images...")
try:
    combine(Path(base_dir), plots_dir)
    print("Combined images generated successfully.")
except Exception as e:
    print(f"Error generating combined images: {e}")

# Step 7: Generate zred vs. photo-z comparison plot
print("\nStep 7: Generating zred vs. photo-z comparison plot...")
try:
    zred_photo_z(Path(base_dir), plots_dir)
    print("Zred vs. photo-z plot generated successfully.")
except Exception as e:
    print(f"Error generating zred vs. photo-z plot: {e}")

# Step 8: Generate member comparison plots (red population)
print("\nStep 8: Generating member comparison plots...")
try:
    member_comparison(Path(base_dir))
    print("Member comparison plots generated successfully.")
except Exception as e:
    print(f"Error generating member comparison plots: {e}")

print("\nAll processing completed!")
