import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0",
    "python scripts/validate_grasps.py",
    "--grasp_path ../data/2023-07-01_graspdata",
    "--object_code sem-TableClock-293a741396ed3cb45341654902142cca",
    "--index 0",
    "--start_with_step_mode",
    "--only_valid_grasps",
    "--optimization_method DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS",
    "--validation_type NO_GRAVITY_SHAKING",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)