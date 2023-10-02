import os
import subprocess
import sys
import json
import glob
from pathlib import Path

experiment_name = sys.argv[1]
base_scene_file = f"./data/example-scenes/{sys.argv[2]}.json"
base_scene_file_dir = os.path.dirname(base_scene_file)
base_scene_file_dir = os.path.abspath(base_scene_file_dir)

with open(base_scene_file) as base_scene_fd:
    #print(base_scene_fd.read())
    base_scene = json.load(base_scene_fd)


output_folder = f"./sweeps/{experiment_name}"
Path(output_folder).mkdir(parents=True, exist_ok=True)

for file in glob.glob(base_scene_file_dir + "/*"):
    rel_file = os.path.relpath(file, base_scene_file_dir)
    target_file = os.path.join(output_folder, rel_file)
    if not os.path.exists(target_file):
        os.symlink(file, target_file)

sample_points = [64]
lengthScale = [0.01, 0.1, 1.0]
sigma = [0.1, 1.0, 5.0]

# sample_points = [32]
# lengthScale = [0.01]
# sigma = [0.01]

scene_files = []

for sp in sample_points:
    for ls in lengthScale:
        for sg in sigma:
            base_scene["media"][0]["gaussian_process"]["covariance"]["lengthScale"] = ls
            base_scene["media"][0]["gaussian_process"]["covariance"]["sigma"] = sg
            base_scene["media"][0]["gaussian_process"]["sample_points"] = sp

            out_file = output_folder + f"/{sp}-{ls}-{sg}.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(base_scene, f)
            scene_files.append(out_file)

for scene in scene_files:
    abs_scene = os.path.abspath(scene)
    subprocess.run(["sbatch", "-J", scene[:-5], f"--export=ALL,abs_scene_path={abs_scene}", "run-tungsten-sweep.sh"])
    subprocess.run(["sbatch", "--dependency=singleton", "-J", scene[:-5], "run-average-images.sh"])


