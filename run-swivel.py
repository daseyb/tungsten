import os
import subprocess
import sys
import json
import glob
from pathlib import Path
import itertools
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import norm

with open(sys.argv[1]) as experiment_file_fd:
    experiment = json.load(experiment_file_fd)

experiment_name = experiment["name"]

base_scene_file = f"./data/example-scenes/{experiment['base_file']}.json"
base_scene_file_dir = os.path.dirname(base_scene_file)
base_scene_file_dir = os.path.abspath(base_scene_file_dir)

with open(base_scene_file) as base_scene_fd:
    base_scene = json.load(base_scene_fd)

output_folder = f"./video/swivel/{experiment_name}"
if(Path(output_folder).exists()):
    shutil.rmtree(output_folder,)
Path(output_folder).mkdir(parents=True, exist_ok=True)

for file in experiment["deps"]:
    file = os.path.join(base_scene_file_dir,file)
    rel_file = os.path.relpath(file, base_scene_file_dir)
    target_file = os.path.join(output_folder, rel_file)
    if not os.path.exists(target_file):
        os.symlink(file, target_file)

scene_files = []

camera_pos = np.array(base_scene["camera"]["transform"]["position"])
camera_target = np.array(base_scene["camera"]["transform"]["look_at"])
camera_up = np.array(base_scene["camera"]["transform"]["up"])

camera_offset = camera_pos - camera_target

axis = camera_offset / norm(camera_offset)  # normalize the rotation vector first

camera_ortho_right = np.cross(axis, camera_up)
camera_ortho_up = np.cross(axis, camera_ortho_right / norm(camera_ortho_right))

camera_offset_rad = Rotation.from_rotvec(float(experiment["swivel_offset"]) / 180 * np.pi * camera_ortho_up).apply(camera_offset)


num_frames = int(experiment["frames"])

for i, angle in enumerate(np.linspace(0, np.pi * 2, num_frames, False)):
    rot = Rotation.from_rotvec(angle * axis)
    curr_cam_offset = rot.apply(camera_offset_rad) 

    curr_cam_pos = camera_target + curr_cam_offset

    base_scene["camera"]["transform"]["position"] = curr_cam_pos.tolist()

    out_file = output_folder + f"/{i:03}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(base_scene, f)
        scene_files.append(out_file)

for scene in scene_files:
    abs_scene = os.path.abspath(scene)
    subprocess.run(["sbatch", "-J", scene[:-5], f"--export=ALL,abs_scene_path={abs_scene}", "run-tungsten-turntable.sh"])
    subprocess.run(["sbatch", "--dependency=singleton", "-J", scene[:-5], "run-average-images-cleanup.sh"])

subprocess.run(["sbatch", "--dependency=singleton", "-J", scene_files[-1][:-5], "run-render-video.sh"])


