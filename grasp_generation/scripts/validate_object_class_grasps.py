import os
from pathlib import Path
import subprocess
import sys

sys.path.append(os.path.realpath("."))

from tap import Tap
from tqdm import tqdm
from utils.isaac_validator import ValidationType
from utils.hand_model_type import HandModelType
from utils.joint_angle_targets import OptimizationMethod



class ValidateObjectClassArgumentParser(Tap):
    hand_model_type: HandModelType = HandModelType.ALLEGRO_HAND
    optimization_method: OptimizationMethod = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    validation_type: ValidationType = ValidationType.NO_GRAVITY_SHAKING
    gpu: int = 0
    grasp_path: str = "../data/2023-07-01_dataset_DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS_v2"
    result_path: str = "../data/dataset_2023-05-24_allegro_distalonly"
    mesh_path: str = "../data/meshdata"
    nerf_path: str = "../data/tyler_nerf_checkpoints_2023-07-25"
    object_class: str = "sem-Bottle"


def object_codes_to_process(args: ValidateObjectClassArgumentParser):
    nerf_path = Path(args.nerf_path)
    mesh_path = Path(args.mesh_path)
    grasp_path = Path(args.grasp_path)

    # find all object codes (ext stripped files) which intersect files in each of the paths
    nerf_object_codes = set(
            map(lambda x: x.split('_')[0] ,[f.stem for f in nerf_path.glob(f"{args.object_class}*")])
    )
    mesh_object_codes = set(
        [f.stem for f in mesh_path.glob(f"{args.object_class}*")]
    )
    grasp_object_codes = set(
        [f.stem for f in grasp_path.glob(f"{args.object_class}*")]
    )
    object_codes = nerf_object_codes.intersection(mesh_object_codes).intersection(grasp_object_codes)
    return object_codes


def main(args: ValidateObjectClassArgumentParser):
    input_object_codes = object_codes_to_process(args)
    print(input_object_codes)
    with tqdm(input_object_codes) as pbar:
        for grasp_object_code in pbar:
            for object_code in input_object_codes:
                if object_code == grasp_object_code:
                    continue

                pbar.set_description(f"Processing {object_code}, with grasps {grasp_object_code}")

                command = " ".join(
                    [
                        f"CUDA_VISIBLE_DEVICES={args.gpu}",
                        "python scripts/validate_grasps.py",
                        f"--hand_model_type {args.hand_model_type.name}",
                        f"--optimization_method {args.optimization_method.name}",
                        f"--validation_type {args.validation_type.name}",
                        f"--gpu {args.gpu}",
                        f"--grasp_path {args.grasp_path}",
                        f"--result_path {args.result_path}",
                        f"--object_code {object_code}",
                        f"--grasp_object_code {grasp_object_code}",
                    ]
                )
                print(f"Running command: {command}")

                try:
                    subprocess.run(command, shell=True, check=True)
                except Exception as e:
                    print(f"Exception: {e}")
                    print(f"Skipping {object_code} and stopping")
                    __import__('ipdb').set_trace()
                    continue
            return





if __name__ == "__main__":
    args = ValidateObjectClassArgumentParser().parse_args()
    main(args)
