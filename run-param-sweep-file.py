import os
import subprocess
import sys
import json
import glob
from pathlib import Path
import itertools
import shutil

with open(sys.argv[1]) as experiment_file_fd:
    experiment = json.load(experiment_file_fd)

experiment_name = experiment["name"]

base_scene_file = f"./data/example-scenes/{experiment['base_file']}.json"
base_scene_file_dir = os.path.dirname(base_scene_file)
base_scene_file_dir = os.path.abspath(base_scene_file_dir)

with open(base_scene_file) as base_scene_fd:
    #print(base_scene_fd.read())
    base_scene = json.load(base_scene_fd)


output_folder = f"./sweeps/{experiment_name}"
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

def set_param_path(json_obj, path, val):
    path_pref = path.split(".")[0]
    path_rem = path[len(path_pref)+1:]
    try:
        if isinstance(json_obj, list):
            if path_pref.isdigit():
                array_idx = int(path_pref)
                if path_rem == "":
                    json_obj[array_idx] = val
                else:
                    return set_param_path(json_obj[array_idx], path_rem, val)
            else:
                for entry in json_obj:
                    if entry["name"] == path_pref:
                        return set_param_path(entry, path_rem, val)
                print(f"Could not find {path_pref} in {json_obj}")
        else:
            if path_pref not in json_obj:
                json_obj[path_pref] = None
            
            if path_rem == "":
                json_obj[path_pref] = val
            else:
                return set_param_path(json_obj[path_pref], path_rem, val)
    except:
        print(path)


param_objs = []
param_values = []
param_ids = []
for param in experiment["params"]:
    if param[0] != "_":
        param_val = ([*experiment["params"][param]],)
        param = [param]
    else:
        param_val = list(zip(*experiment["params"][param].values()))
        param = list(experiment["params"][param].keys())

    param_objs.append(param)
    param_values.append(param_val)
    param_ids.append(list(range(len(param_values[-1]))))


for element, ids in zip(itertools.product(*param_values), itertools.product(*param_ids)):
    has_skip_param = False
    for j, val in enumerate(element):
        for i, param_name in enumerate(param_objs[j]):
            set_param_path(base_scene, param_name, val[i])
            if val[i] == "---":
                has_skip_param = True
    
    if has_skip_param:
        continue
    
    out_file = output_folder + f"/{'-'.join([str(id) for id in ids])}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(base_scene, f)
        scene_files.append(out_file)


for scene in scene_files:
    abs_scene = os.path.abspath(scene)
    subprocess.run(["sbatch", "-J", scene[:-5], f"--export=ALL,abs_scene_path={abs_scene}", "run-tungsten-sweep.sh"])
    subprocess.run(["sbatch", "--dependency=singleton", "-J", scene[:-5], "run-average-images-cleanup.sh"])


