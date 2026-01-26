import numpy as np
import nibabel as nib
import os

# Auxiliar function to load n_slices MRI slices from the n subject 

def load_mri_slices_and_age(n: int, n_slices: int = 15):

    # Find all the directories  (an espicific directory contains an especific subject MRI id)
    folder_path = f"../data/subjects/" 
    with os.scandir(folder_path) as entries:
        folder_names = sorted([e.name for e in entries if e.is_dir()])  # Make a list of the ids of all the subjects

    # Define the parameters of data    
    n_subjects = len(folder_names)  # currently 351 subjects 

    if 0 < n <= n_subjects:

        # Choose a subject and load its MRI
        folder = folder_names[n-1]  # n must be in [0, n_subjects-1]
        path = f"../data/subjects/{folder}/RAW/{folder}_mpr-1_anon.img"

        img = nib.load(path)       # load image
        data = img.get_fdata()     # get data

        # Take 5 slices of the MRI (along Z)
        #z1 = int(0.40 * data.shape[2])
        #slide1 = data[:, :, z1,0]
        #images = np.stack([slide1, slide2, slide3, slide4, slide5], axis=0)  # Combine slices (channels first: 5 x H x W)

        # Take n_slices slices of the MRI (along Z)
        vol = data[:, :, :, 0] if data.ndim == 4 else data
        D = vol.shape[2]

        zs = np.linspace(0.25*(D-1), 0.75*(D-1), n_slices).round().astype(int)
        images = vol[:, :, zs].transpose(2, 0, 1) # [n_slices, H, W]


        # Get age from .txt file
        txt_path = f"../data/subjects/{folder}/{folder}.txt"

        age = None
        with open(txt_path, "r") as f:
            for line in f:
                if line.strip().startswith("AGE:"):
                    age = int(line.split(":")[1].strip())
                    break

        if age is None:
            raise ValueError(f"AGE not found for subject {folder}")

    else:
        raise ValueError(f"n must be a positive integer less than {n_subjects + 1}")

    return images, age
