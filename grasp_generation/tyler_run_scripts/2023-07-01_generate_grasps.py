import subprocess

command_parts = [
    "CUDA_VISIBLE_DEVICES=0 python scripts/generate_grasps.py",
    "--all",
    "--seed 0",
    "--result_path ../data/2023-07-01_graspdata",
    "--wandb_name graspdata",
]

full_command = " ".join(command_parts)
print(f"Running command: {full_command}")
subprocess.run(full_command, shell=True, check=True)