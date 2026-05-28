#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    parser.add_argument(
        "--snapshot_config_name",
        type=str,
        default="config",
        help="The name of the config snapshot file to load from the .hydra directory (without .yaml)",
    )
    args, _ = parser.parse_known_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())
    config_name = args.snapshot_config_name


    experiment = reload_experiment_from_file(
        checkpoint_file, snapshot_config_name=config_name
    )

    experiment.evaluate()
