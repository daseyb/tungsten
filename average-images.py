import numpy as np
import imageio
import sys
import glob
import os
from tqdm import tqdm

imageio.plugins.freeimage.download()

folder_glob = glob.glob(f"/dartfs-hpc/rc/lab/J/JaroszLab/dseyb/stimp/{sys.argv[1]}")

folder_glob = [f for f in folder_glob if os.path.isdir(f)]

print(folder_glob)

for folder in (pbar := tqdm(folder_glob)):
    input_glob = f"{folder}/*/*"
    output_file = f"{folder}"

    pbar.set_description("Averaging files in ", input_glob)

    input_image_paths = glob.glob(input_glob + ".pfm") + glob.glob(input_glob + ".exr") 
    if len(input_image_paths) == 0:
        input_image_paths = glob.glob(f"{folder}/*.pfm") + glob.glob(f"{folder}/*.exr")

    if len(input_image_paths) == 0:
        print(f"No images found in {folder}!")
        continue

    result = None

    for img_path in tqdm(input_image_paths):
        imarr = imageio.imread(img_path)

        if img_path.endswith("pfm"):
            imarr = imarr[::-1, :, :]

        if result is None:
            result = imarr/len(input_image_paths)
        else:
            result += imarr/len(input_image_paths)

    #print("Writing to ", output_file)
    imageio.imsave(output_file + ".exr", result)
    imageio.imsave(output_file + ".png", np.array(np.clip(np.power(result, 1.0/2.2)*255, 0, 255), dtype=np.uint8))