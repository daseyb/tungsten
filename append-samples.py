
#!/usr/bin/env python3
import numpy as np
import glob
import os
from tqdm import tqdm

normal_files = dict()

files_to_delete = []

print("Collection files...")
for file in tqdm(glob.glob("//dartfs-hpc/rc/lab/J/JaroszLab/dseyb/stimp/normal-gen/*/*/*/*.bin")):
    end_ind = "normals-"

    if file.endswith("normals.bin"):
        continue

    if file.endswith("deg.bin"):
        files_to_delete.append(file)
        continue

    normals = np.reshape(np.fromfile(file, dtype=np.float64), (-1,3))#.T

    files_to_delete.append(file)

    deg_idx = file.find(end_ind)

    if deg_idx == -1:
        continue

    key = file[:deg_idx+len(end_ind)-1] + ".bin"
    if key not in normal_files:
        normal_files[key] = []
    normal_files[key].append(normals)

print("Merging files...")
for k, v in tqdm(normal_files.items()):
    normals = np.concatenate(v, axis=0)

    if os.path.exists(k):
        add_normals = np.reshape(np.fromfile(k, dtype=np.float64), (-1,3))
        normals = np.concatenate((normals,add_normals),axis=0)

    normals.tofile(k)

print("Deleting files...")
for f in tqdm(files_to_delete):
    os.remove(f)