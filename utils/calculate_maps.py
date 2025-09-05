from joblib import Parallel, delayed
import nibabel as nib
import numpy as np
from scipy.stats import entropy
from joblib import Parallel, delayed
import os

def voxel_entropy(x, num_labels):
    counts = np.bincount(x, minlength=num_labels)
    probs = counts / counts.sum()
    return entropy(probs)

base_dir = '/mnt/lustre/vin/fastsurfer_random/FastSurfer/test_results/corr/'
dst_dir = '/mnt/lustre/inesgp/'

for sub in os.listdir('/mnt/lustre/vin/fastsurfer_random/FastSurfer/test_results/corr/2'):
    if sub in ['logging.log' or 'fsaverage', '0025462_norm.mgz', '0025598_norm.mgz', '0025604_norm.mgz', 
               '0026044_norm.mgz', '0027174_norm.mgz', '0027236_norm.mgz']: 
         continue


    # Load segmentations: shape (X,Y,Z,N)
    segmentations = [nib.load(
        f"{base_dir}/{i}/{sub}/mri/aparc.DKTatlas+aseg.deep.mgz"
    ).get_fdata().astype(np.int16) for i in range(1, 11)]
    segmentations = np.stack(segmentations, axis=-1)

    ref_img = nib.load(f"{base_dir}/1/{sub}/mri/aparc.DKTatlas+aseg.deep.mgz")

    # Majority vote (mode)
    majority_map = np.apply_along_axis(lambda x: np.bincount(x).argmax(), -1, segmentations)

    majority_nii = nib.Nifti1Image(majority_map.astype(np.int16), affine=ref_img.affine, header=ref_img.header)
    nib.save(majority_nii, f'{dst_dir}/uncertainty_maps/{sub}/majority_map.nii.gz')


    # Compute label probabilities without one-hot
    num_labels = int(segmentations.max()) + 1
    flat = segmentations.reshape(-1, segmentations.shape[-1])   # (voxels, subjects)

    def chunk_entropy(chunk, num_labels):
        counts = np.zeros((chunk.shape[0], num_labels), dtype=np.int16)
        for subj in range(chunk.shape[1]):
            np.add.at(counts, (np.arange(chunk.shape[0]), chunk[:, subj]), 1)
        probs = counts / counts.sum(axis=1, keepdims=True)
        return entropy(probs.T)

    chunk_size = 200000
    uncertainty_chunks = Parallel(n_jobs=-1)(
        delayed(chunk_entropy)(flat[start:start+chunk_size], num_labels)
        for start in range(0, flat.shape[0], chunk_size)
    )

    uncertainty_map = np.concatenate(uncertainty_chunks).reshape(segmentations.shape[:-1])


    # uncertainty = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(voxel_entropy)(flat[i], num_labels) for i in range(flat.shape[0])
    # )
    # uncertainty_map = np.array(uncertainty).reshape(segmentations.shape[:-1])

    uncertainty_nii = nib.Nifti1Image(uncertainty_map, affine=ref_img.affine, header=ref_img.header)
    nib.save(uncertainty_nii, f'{dst_dir}/uncertainty_maps/{sub}/uncertainty_map.nii.gz')

