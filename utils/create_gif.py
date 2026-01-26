from data_loader import load_mri_slices_and_age
import numpy as np
import imageio
import nibabel as nib


# Load a subject complete MRI scan

subject_id = 2 # a number between 0000 and 0271
path = f"../data/subjects/OAS1_{subject_id:04d}_MR1/RAW/OAS1_{subject_id:04d}_MR1_mpr-1_anon.img"
img = nib.load(path)           # load image
data_mri = img.get_fdata()     # get data  in format (Height, Width, Dept)


# Format and normalize data for the gif

data_mri = data_mri.squeeze()   # from (256, 256, 128, 1) to (256, 256, 128)  
print(data_mri.shape)
data_mri_normalized = (data_mri - data_mri.min()) / (data_mri.max() - data_mri.min())   # Normalization [0,1]
data_mri_transformed = (255 * data_mri_normalized).astype(np.uint8)  # Transformation [0,255] and 8-bit unsigned integer
data_mri_transformed = np.rot90(data_mri_transformed)   # Rotate correclty the images


# Make a gif of the MRI of the subject number 200

frames = [data_mri_transformed[:,:,i] for i in range(data_mri_transformed.shape[2])]
imageio.mimsave(f"MRI_animation_subject{subject_id}.gif", frames, duration=0.2)