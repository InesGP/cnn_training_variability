import pandas as pd
import numpy as np
import nibabel as nib
import os
from collections import defaultdict


lut = pd.read_csv(
    "/home/inesgp/verrou_fastsurfer/FreeSurferColorLUT.txt",
    comment="#",
    sep=r"\s+",           # whitespace-separated
    header=None,
    names=["Index", "Label", "R", "G", "B", "A"],
    engine="python"
)

# base_dir = '/mnt/lustre/vin/fastsurfer_random/FastSurfer/test_results/' #Fastsurfer test set
base_dir = '/mnt/lustre/vin/fastsurfer_random/FastSurfer/val_train_results/' #Fastsurfer train + val set

index_to_label = dict(zip(lut['Index'], lut['Label']))
roi_indices = [idx for idx in index_to_label if idx != 0]

# Store results per subject
all_subjects_volumes = []

# for sub in os.listdir(f"{base_dir}/2"):
for sub in ["MIRIAD_189.nii", "MIRIAD_208.nii", "MIRIAD_217.nii", "MIRIAD_221_5.nii", "MIRIAD_246.nii", "MIRIAD_251.nii"]:
    subject_volumes = defaultdict(list)

    for seed in range(1, 11):
        seg_path = f'{base_dir}{seed}/{sub}/mri/aparc.DKTatlas+aseg.deep.mgz'
        try:
            seg_img = nib.load(seg_path, mmap=True)
        except: 
            try:
                seg_img = nib.load(f'{base_dir}{seed}/{sub}.nii.gz/{sub}.nii.gz/mri/aparc.DKTatlas+aseg.deep.mgz', mmap=True)
            except:
                seg_img = nib.load(f'{base_dir}{seed}/{sub}.nii/{sub}.nii/mri/aparc.DKTatlas+aseg.deep.mgz', mmap=True)

        seg_data = np.asarray(seg_img.get_fdata(), dtype=np.int16)
        voxel_volume = np.prod(seg_img.header.get_zooms()[:3])

        # Use bincount to count all labels at once
        counts = np.bincount(seg_data.flatten(), minlength=max(roi_indices)+1)

        for idx in roi_indices:
            subject_volumes[index_to_label[idx]].append(counts[idx] * voxel_volume)

    # Convert to DataFrame for this subject
    df_volumes = pd.DataFrame(subject_volumes)
    df_volumes['Subject'] = sub
    all_subjects_volumes.append(df_volumes)
    

# Concatenate all subjects and save once
df_all = pd.concat(all_subjects_volumes, ignore_index=True)

df_orig = pd.read_csv('/home/inesgp/cnn_training/roi_volumes_train.csv')

df = pd.concat([df_orig, df_all])

df.to_csv("roi_volumes_train_v2.csv", index=False)
