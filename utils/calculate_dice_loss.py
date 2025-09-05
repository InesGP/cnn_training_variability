import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from data_utils import get_labels_from_lut, unify_lateralized_labels, read_classes_from_lut, map_aparc_aseg2label
import nibabel as nib
import numpy as np
import pickle
import os
from concurrent.futures import ThreadPoolExecutor


class DiceLoss(_Loss):
    """
    Calculate Dice Loss.

    Methods
    -------
    forward
        Calculate the DiceLoss.
    """

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        weights: int | None = None,
        ignore_index: int | None = None,
    ) -> torch.Tensor:
        """
        Calculate the DiceLoss.

        Parameters
        ----------
        output : Tensor
            N x C x H x W Variable.
        target : Tensor
            N x C x W LongTensor with starting class at 0.
        weights : int, optional
            C FloatTensor with class wise weights(Default value = None).
        ignore_index : int, optional
            Ignore label with index x in the loss calculation (Default value = None).

        Returns
        -------
        torch.Tensor
            Calculated Diceloss.
        """
        eps = 0.001

        encoded_target = output.detach().to(torch.float32) * 0.

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0

        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        # plt.imshow(encoded_target[120,19,:, :])
        # plt.imshow(encoded_target[:,:, 120, 19])

        # print(encoded_target.dtype)
        # print(encoded_target.shape, output.shape)
        # print(encoded_target[:, 6:7, :, :].shape, output[:, 6:7, :, :].shape)
        # encoded_target = encoded_target[:, 6:7, :, :]
        # output = output[:, 6:7, :, :]

        # print(encoded_target.sum())
        # print(encoded_target[120, 10, :, :])
        if weights is None:
            weights = 1

        intersection = output * encoded_target.to(torch.float32)
        intersection = intersection.to(torch.float32) 
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        # numerator = 2 * intersection.sum(0).sum(0).sum(0)
        # print(numerator.shape)
        denominator = output + encoded_target
        denominator = denominator.to(torch.float32)

        # print(intersection)
        # print(denominator)

        if ignore_index is not None:
            denominator[mask] = 0

        # print(denominator.sum(0).sum(1))
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        # denominator = denominator.sum(0).sum(0).sum(0) + eps
        # print(denominator.shape)
        # print(numerator[0], denominator[0], (numerator / denominator))
        loss_per_channel = weights * (
           1 - (numerator / denominator)
        )  # Channel-wise weights

        # print(output.size(1))

        return loss_per_channel #.sum() / output.size(1)
    


sag_mask = ("Left-", "ctx-rh")
combi = ["Left-", "Right-"]

lut = read_classes_from_lut("/home/inesgp/cnn_training/FastSurfer_ColorLUT.tsv")
labels, labels_sag = get_labels_from_lut(lut, sag_mask)
lateralization = unify_lateralized_labels(lut, combi)



def load_seed_data(seed, sub):
    # Load prob seg
    prob_seg_path = f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/prob_seg.pkl'
    prob_seg = torch.softmax(pickle.load(open(prob_seg_path, 'rb')), dim=-1)
    prob_seg = prob_seg.permute(0, 3, 1, 2) #.contiguous()  # CxHxW

    # Load aseg & aseg_nocc
    aseg_path = f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/mri/aparc.DKTatlas+aseg.deep.mgz'
    aseg_nocc_path = f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/mri/aseg.auto_noCCseg.mgz'

    aseg = np.asarray(nib.load(aseg_path, mmap=True).get_fdata(), dtype=np.int16)
    aseg_nocc = np.asarray(nib.load(aseg_nocc_path, mmap=True).get_fdata(), dtype=np.int16)

    mapped_aseg, _ = map_aparc_aseg2label(
        aseg, labels, labels_sag, lateralization, aseg_nocc, processing='aparc'
    )

    mapped_aseg_t = torch.as_tensor(mapped_aseg, dtype=torch.int64)

    return seed, prob_seg, mapped_aseg_t

loss = DiceLoss()
loss_dice = {}
base_path = '/mnt/lustre/vin/fastsurfer_random/FastSurfer/test_results/corr/2/'
n=10
for sub in os.listdir(base_path):
    # Preload all seeds in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda s: load_seed_data(s, sub), range(1, n+1)))

    prob_segs = {seed: p for seed, p, _ in results}
    aseg_data = {seed: a for seed, _, a in results}

    # Compute losses
    loss_dice[sub] = [
        loss(prob_segs[i], aseg_data[j])
        for i in range(1, n)
        for j in range(i + 1, n+1)
    ]


# loss = DiceLoss()
# loss_dice = {}

# for sub in os.listdir('/mnt/lustre/vin/fastsurfer_random/FastSurfer/test_results/corr/2/'):
#     loss_dice[sub] = []

#     # Cache all prob_seg and aseg data for each seed
#     prob_segs = {}
#     aseg_data = {}
#     aseg_nocc_data = {}

#     for seed in range(1, 11):
#         print(seed)
#         # Load probability segmentation once
#         prob_segs[seed] = pickle.load(open(f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/prob_seg.pkl', 'rb'))

#         # Load aseg & aseg_nocc once, mapped immediately
#         aseg_path = f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/mri/aparc.DKTatlas+aseg.deep.mgz'
#         aseg_nocc_path = f'/mnt/lustre/inesgp/random_seed_inference/{seed}/{sub}/mri/aseg.auto_noCCseg.mgz'

#         aseg = np.asarray(nib.load(aseg_path, mmap=True).get_fdata(), dtype=np.int16)
#         aseg_nocc = np.asarray(nib.load(aseg_nocc_path, mmap=True).get_fdata(), dtype=np.int16)

#         mapped_aseg, _ = map_aparc_aseg2label(
#             aseg, labels, labels_sag, lateralization, aseg_nocc, processing='aparc'
#         )

#         aseg_data[seed] = torch.as_tensor(mapped_aseg, dtype=torch.int64)

#     # Compute Dice loss only — no reloading
#     for i in range(1, 10):
#         print(i)
#         img1_t = prob_segs[i].permute(0, 3, 1, 2)  # Convert once per i
#         for j in range(i + 1, 11):
#             loss_val = loss(img1_t, aseg_data[j])
#             loss_dice[sub].append(loss_val)


pickle.dump(loss_dice, open('/home/inesgp/cnn_training/dice_loss.pkl', 'wb'))