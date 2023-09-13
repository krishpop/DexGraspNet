"""
Last modified date: 2023.06.13
Author: Tyler Lum
Description: Create NeRF Data in Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from tap import Tap
from tqdm import tqdm
import subprocess
from typing import Optional, Tuple
import pathlib
from utils.parse_object_code_and_scale import parse_object_code_and_scale
import multiprocessing


class GenerateNerfDataArgumentParser(Tap):
    gpu: int = 0
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/meshdata")
    output_nerfdata_path: pathlib.Path = pathlib.Path("../data/nerfdata")
    randomize_order_seed: Optional[int] = None
    only_objects_in_this_path: Optional[pathlib.Path] = None
    use_multiprocess: bool = True
    num_workers: int = 4
    no_continue: bool = False


def get_object_codes_and_scales_to_process(
    args: GenerateNerfDataArgumentParser,
) -> Tuple[list, list]:
    # Get input object codes
    if args.only_objects_in_this_path is not None:
        input_object_codes, input_object_scales = [], []
        for path in args.only_objects_in_this_path.iterdir():
            object_code_and_scale_str = path.stem
            object_code, object_scale = parse_object_code_and_scale(
                object_code_and_scale_str
            )
            input_object_codes.append(object_code)
            input_object_scales.append(object_scale)

        print(
            f"Found {len(input_object_codes)} object codes in args.only_objects_in_this_path ({args.only_objects_in_this_path})"
        )

        existing_object_code_and_scale_strs = list(args.output_nerfdata_path.iterdir())
        existing_object_codes = [
            parse_object_code_and_scale(object_code_and_scale_str)[0]
            for object_code_and_scale_str in existing_object_code_and_scale_strs
        ]

        existing_object_scales = [
            parse_object_code_and_scale(object_code_and_scale_str)[1]
            for object_code_and_scale_str in existing_object_code_and_scale_strs
        ]

        if args.no_continue and len(existing_object_codes) > 0:
            print(
                f"Found {len(existing_object_codes)} existing object codes in args.output_nerfdata_path ({args.output_nerfdata_path})."
            )
            print("Exiting because --no_continue was specified.")
            exit()
        elif len(existing_object_codes) > 0:
            print(
                f"Found {len(existing_object_codes)} existing object codes in args.output_nerfdata_path ({args.output_nerfdata_path})."
            )
            print("Continuing because --no_continue was not specified.")
            input_object_codes = [
                object_code
                for object_code in input_object_codes
                if object_code not in existing_object_codes
            ]
            input_object_scales = [
                object_scale
                for object_scale in input_object_scales
                if object_scale not in existing_object_codes
            ]
            print(
                f"Continuing with {len(input_object_codes)} object codes after filtering."
            )

    else:
        input_object_codes = [
            object_code for object_code in os.listdir(args.meshdata_root_path)
        ]
        HARDCODED_OBJECT_SCALE = 0.1
        input_object_scales = [HARDCODED_OBJECT_SCALE] * len(input_object_codes)
        print(
            f"Found {len(input_object_codes)} object codes in args.mesh_path ({args.meshdata_root_path})"
        )
        print(f"Using hardcoded scale {HARDCODED_OBJECT_SCALE} for all objects")

    return input_object_codes, input_object_scales


def run_command(object_code, object_scale, args, script_to_run):
    command = " ".join(
        [
            f"python {script_to_run}",
            f"--gpu {args.gpu}",
            f"--meshdata_root_path {args.meshdata_root_path}",
            f"--output_nerfdata_path {args.output_nerfdata_path}",
            f"--object_code {object_code}",
            f"--object_scale {object_scale}",
        ]
    )
    print(f"Running command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Skipping {object_code} and continuing")
    print(f"Finished object {object_code}.")


def main(args: GenerateNerfDataArgumentParser):
    # Check if script exists
    script_to_run = pathlib.Path("scripts/generate_nerf_data_one_object_one_scale.py")
    assert script_to_run.exists(), f"Script {script_to_run} does not exist"

    input_object_codes, input_object_scales = get_object_codes_and_scales_to_process(
        args
    )

    # Randomize order
    if args.randomize_order_seed is not None:
        import random

        print(f"Randomizing order with seed {args.randomize_order_seed}")
        random.Random(args.randomize_order_seed).shuffle(input_object_codes)

    if args.use_multiprocess:
        print(f"Using multiprocessing with {args.num_workers} workers.")
        with multiprocessing.Pool(args.num_workers) as p:
            p.starmap(
                run_command,
                zip(
                    input_object_codes,
                    input_object_scales,
                    [args] * len(input_object_codes),
                    [script_to_run] * len(input_object_codes),
                ),
            )
    else:
        for i, (object_code, object_scale) in tqdm(
            enumerate(zip(input_object_codes, input_object_scales)),
            desc="Generating NeRF data for all objects",
            dynamic_ncols=True,
            total=len(input_object_codes),
        ):
            run_command(object_code, object_scale, args)


if __name__ == "__main__":
    args = GenerateNerfDataArgumentParser().parse_args()
    main(args)
